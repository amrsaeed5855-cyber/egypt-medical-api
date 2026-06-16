"""
fix_prices.py — Clean suspicious drug prices in egypt_drugs_cleaned_utf8.csv
=============================================================================
Source data (egyptdwa.com) includes crowd-submitted prices — some are wrong
(e.g. 5,558,555 EGP for a common antibiotic). This script:

1. Keeps original prices in `price_egp_raw`
2. Detects outliers using peer groups (ingredient + form + strength)
3. Falls back to broader groups when peers are sparse (brand, ingredient-only)
4. Imputes missing/zero/outlier prices from robust group median
5. Writes cleaned `price_egp` + `price_corrected` flag

Usage:
    python fix_prices.py
    python fix_prices.py --dry-run
    python fix_prices.py --input my.csv --output my_fixed.csv
"""

import argparse
import os
import re
import shutil
from datetime import datetime

import numpy as np
import pandas as pd

DEFAULT_INPUT = os.getenv("EGYPT_DRUGS_CSV", "egypt_drugs_cleaned_utf8.csv")
MIN_VALID_PRICE = 0.5
ABSOLUTE_MAX_PRICE = 250_000

# Soft caps for common OTC forms — prices above 2× cap are almost always crowd errors.
FORM_SOFT_CAP = {
    "tablet": 3_000,
    "capsule": 3_000,
    "caplet": 3_000,
    "syrup": 1_500,
    "suspension": 1_500,
    "drops": 800,
    "cream": 2_000,
    "ointment": 2_000,
    "gel": 2_000,
    "sachet": 500,
    "suppository": 500,
    "inhaler": 3_000,
    "spray": 1_500,
    "vial": 15_000,
    "ampoule": 5_000,
    "injection": 15_000,
}

INGREDIENT_ALIASES = (
    (r"amox.*clav|clav.*amox", "amoxicillin-clavulanate"),
    (r"paracetamol|acetaminophen", "paracetamol"),
    (r"ibuprofen", "ibuprofen"),
    (r"omeprazole", "omeprazole"),
    (r"metformin", "metformin"),
    (r"cetirizine", "cetirizine"),
    (r"loratadine", "loratadine"),
)


def _normalize_ingredient(raw: str) -> str:
    s = re.sub(r"\s+", " ", str(raw or "").strip().lower())
    s = re.sub(r"[+/_\-]+", " ", s)
    if not s:
        return "unknown"
    for pattern, canonical in INGREDIENT_ALIASES:
        if re.search(pattern, s):
            return canonical
    return s[:60]


def _normalize_form(raw: str) -> str:
    s = str(raw or "").strip().lower()
    if not s:
        return "unknown"
    if "tab" in s:
        return "tablet"
    if "cap" in s and "susp" not in s:
        return "capsule"
    if "susp" in s or "syrup" in s or "شراب" in s or "معلق" in s:
        return "suspension"
    if "drop" in s or "قطرة" in s:
        return "drops"
    if "cream" in s or "كريم" in s:
        return "cream"
    if "vial" in s or "فيال" in s:
        return "vial"
    if "inj" in s or "amp" in s:
        return "injection"
    return s[:30]


def _extract_strength(row) -> str:
    for field in ("dosage", "name_en", "name_ar"):
        text = str(row.get(field) or "")
        m = re.search(r"(\d+(?:\.\d+)?)\s*(?:mg|gm|g|mcg|iu|%)\b", text, re.I)
        if m:
            return m.group(1)
    return ""


def _brand_name(row) -> str:
    name = str(row.get("name_en") or row.get("name_ar") or "").strip().lower()
    if not name:
        return "unknown"
    token = re.split(r"[\s\-]+", name)[0]
    return re.sub(r"[^a-z0-9]", "", token)[:30] or "unknown"


def _group_keys(row) -> list:
    ing = _normalize_ingredient(row.get("ingredient_clean") or row.get("active_ingredient") or "")
    form = _normalize_form(row.get("form_clean") or row.get("form") or "")
    strength = _extract_strength(row)
    brand = _brand_name(row)
    keys = []
    if strength:
        keys.append(f"ing:{ing}|{form}|{strength}")
    keys.append(f"ing:{ing}|{form}")
    keys.append(f"brand:{brand}|{form}")
    keys.append(f"ing:{ing}")
    return keys


def _robust_prices(prices: list) -> list:
    """Drop crowd-sourced garbage before median/IQR (e.g. 5,558,555 EGP)."""
    valid = sorted(p for p in prices if p and p >= MIN_VALID_PRICE)
    if len(valid) < 4:
        return valid
    arr = np.array(valid, dtype=float)
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = max(q3 - q1, 1.0)
    upper = min(ABSOLUTE_MAX_PRICE, q3 + 3.0 * iqr)
    lower = max(MIN_VALID_PRICE, q1 - 1.5 * iqr)
    robust = [p for p in valid if lower <= p <= upper]
    return robust if len(robust) >= 2 else valid


def _reference_prices(row, group_map: dict) -> list:
    seen = set()
    merged = []
    for key in _group_keys(row):
        for price in group_map.get(key, []):
            if price not in seen and price and price >= MIN_VALID_PRICE:
                seen.add(price)
                merged.append(price)
    return merged


def _form_cap(form: str):
    return FORM_SOFT_CAP.get(_normalize_form(form))


def _needs_fix(price, ref_prices: list, form: str) -> bool:
    if pd.isna(price) or price <= 0:
        return True

    cap = _form_cap(form)
    if cap and price > cap * 2:
        return True

    robust = _robust_prices(ref_prices)
    if not robust:
        return price > ABSOLUTE_MAX_PRICE

    if len(robust) < 4:
        med = float(np.median(robust))
        if price < med * 0.2 or price > med * 5:
            return True
        if cap and price > cap:
            return True
        return False

    arr = np.array(robust, dtype=float)
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = max(q3 - q1, 1.0)
    lower = max(MIN_VALID_PRICE, q1 - 1.5 * iqr)
    upper = min(ABSOLUTE_MAX_PRICE, q3 + 3.0 * iqr)
    med = float(np.median(robust))

    if price < lower or price > upper:
        return True
    if price < med * 0.2 or price > med * 5:
        return True
    return False


def _impute_price(ref_prices: list):
    robust = _robust_prices(ref_prices)
    if not robust:
        return None
    return round(float(np.median(robust)), 2)


def clean_prices(df: pd.DataFrame) -> tuple:
    df = df.copy()
    df["price_egp"] = pd.to_numeric(df["price_egp"], errors="coerce")
    if "price_egp_raw" not in df.columns:
        df["price_egp_raw"] = df["price_egp"]

    group_map: dict[str, list] = {}
    row_keys: list[list[str]] = []

    for _, row in df.iterrows():
        keys = _group_keys(row)
        row_keys.append(keys)
        price = row["price_egp"]
        if pd.isna(price):
            continue
        for key in keys:
            group_map.setdefault(key, []).append(float(price))

    corrected_flags = []
    new_prices = []

    for i, (_, row) in enumerate(df.iterrows()):
        keys = row_keys[i]
        price = row["price_egp"]
        form = str(row.get("form_clean") or row.get("form") or "")

        ref = []
        seen = set()
        for key in keys:
            for p in group_map.get(key, []):
                if p not in seen:
                    seen.add(p)
                    ref.append(p)

        fix = _needs_fix(price, ref, form)

        if fix:
            imputed = _impute_price(ref)
            new_prices.append(imputed)
            corrected_flags.append(True)
        else:
            new_prices.append(round(float(price), 2) if not pd.isna(price) else np.nan)
            corrected_flags.append(False)

    df["price_egp"] = new_prices
    df["price_corrected"] = corrected_flags
    stats = {
        "total": len(df),
        "corrected": int(sum(corrected_flags)),
        "still_missing": int(pd.isna(df["price_egp"]).sum()),
        "price_gt_10k": int((pd.to_numeric(df["price_egp"], errors="coerce") > 10_000).sum()),
    }
    return df, stats


def main():
    parser = argparse.ArgumentParser(description="Clean drug prices in Egyptian drugs CSV")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=None, help="Defaults to overwrite --input")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()

    inp = args.input
    out = args.output or inp

    if not os.path.exists(inp):
        raise SystemExit(f"File not found: {inp}")

    print(f"Loading {inp} ...")
    df = pd.read_csv(inp)
    before = pd.to_numeric(df["price_egp"], errors="coerce")
    print(f"  rows: {len(df)}")
    print(f"  price=0: {(before == 0).sum()}")
    print(f"  price>10000: {(before > 10000).sum()}")

    cleaned, stats = clean_prices(df)
    print("\nAfter cleaning:")
    print(f"  corrected rows: {stats['corrected']}")
    print(f"  still missing:  {stats['still_missing']}")
    print(f"  price>10000:    {stats['price_gt_10k']}")

    if "price_egp_raw" in cleaned.columns:
        fixes = cleaned[cleaned["price_corrected"]].head(10)
        for _, r in fixes.iterrows():
            name = str(r.get("name_en") or r.get("name_ar") or "")[:45]
            print(f"  FIX: {name} | {r['price_egp_raw']} -> {r['price_egp']}")

    if args.dry_run:
        print("\n[dry-run] No file written.")
        return

    if out == inp and not args.no_backup:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = f"{inp}.backup_{ts}"
        shutil.copy2(inp, backup)
        print(f"\nBackup: {backup}")

    cleaned.to_csv(out, index=False, encoding="utf-8")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
