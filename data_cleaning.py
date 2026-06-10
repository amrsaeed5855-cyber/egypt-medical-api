# data_cleaning.py — Startup-only dataset cleaning, column pruning, ingredient normalization.
# Runs once at boot; all structures cached in memory (zero per-request cost).

from __future__ import annotations

import re
from typing import Dict, FrozenSet, List, Set, Tuple

import pandas as pd

from trade_name_utils import (
    generate_search_variants,
    normalize_text,
    resolve_trade_alias,
    strip_form_noise,
)

# Canonical ingredient aliases (explicit map — do not guess beyond this list).
INGREDIENT_ALIAS_MAP: Dict[str, str] = {
    "acetaminophen": "paracetamol",
    "apap": "paracetamol",
    "panadol": "paracetamol",
    "tylenol": "paracetamol",
    "ibuprofen": "ibuprofen",
    "brufen": "ibuprofen",
    "advil": "ibuprofen",
    "diclofenac": "diclofenac",
    "voltaren": "diclofenac",
    "cataflam": "diclofenac",
    "devarol": "diclofenac",
    "ديفارول": "diclofenac",
    "paracetamol": "paracetamol",
    "cetirizine": "cetirizine",
    "zyrtec": "cetirizine",
    "loratadine": "loratadine",
    "claritin": "claritin",
    "omeprazole": "omeprazole",
    "losec": "omeprazole",
    "amoxicillin": "amoxicillin",
    "augmentin": "amoxicillin clavulanic",
    "pseudo ephedrine": "pseudoephedrine",
    "pseudoephedrine": "pseudoephedrine",
    "phenyl ephedrine": "phenylephrine",
    "phenylephrine": "phenylephrine",
    "vitamin c": "ascorbic acid",
    "ascorbic acid": "ascorbic acid",
    "ascorbic": "ascorbic acid",
}

GENERIC_INGREDIENT_WORDS: FrozenSet[str] = frozenset({
    "vitamin", "enzyme", "mineral", "herbal", "extract", "oil", "supplement",
})

FORM_LABELS: Dict[str, str] = {
    "tablet": "أقراص",
    "capsule": "كبسولات",
    "cream": "كريم",
    "syrup": "شراب",
    "suspension": "معلق",
    "vial": "أمبول",
    "ampoule": "أمبول",
    "sachet": "أكياس",
    "gel": "جل",
    "drop": "قطرة",
    "spray": "بخاخ",
    "injection": "حقن",
    "other": "أخرى",
}

# Populated by clean_dataframe()
KNOWN_BRAND_TOKENS: Set[str] = set()
SUBSTITUTE_ELIGIBLE: List[bool] = []
TUNED_MIN_TRADE_RERANK: float = 0.85
TUNED_MIN_SEMANTIC: float = 0.32


def is_empty_value(val) -> bool:
    s = str(val or "").strip().lower()
    return s in ("", "nan", "none", "null")


def column_inventory(df: pd.DataFrame) -> pd.DataFrame:
    """Return name, dtype, null/empty %, unique count for every column."""
    rows = []
    n = max(len(df), 1)
    for col in df.columns:
        empty = df[col].apply(is_empty_value).sum()
        rows.append({
            "column": col,
            "dtype": str(df[col].dtype),
            "null_empty_pct": round(empty / n * 100, 1),
            "unique": df[col].nunique(dropna=False),
        })
    return pd.DataFrame(rows)


def is_generic_ingredient(ingredient: str) -> bool:
    """Rows with vague single-token ingredients must not drive substitute matching."""
    raw = (ingredient or "").strip().lower()
    if not raw or len(raw) < 5:
        return True
    tokens = [t for t in re.split(r"[+/\s,\-()]+", raw) if t.strip()]
    if len(tokens) != 1:
        return False
    token = tokens[0]
    if len(token) < 5:
        return True
    return token in GENERIC_INGREDIENT_WORDS


def normalize_ingredient_canonical(ingredient: str) -> str:
    """Lowercase, strip spaces, map known aliases to one canonical form."""
    raw = re.sub(r"\s+", " ", (ingredient or "").strip().lower())
    if not raw:
        return ""
    # Parenthetical aliases e.g. paracetamol(acetaminophen)
    raw = raw.replace("(", " ").replace(")", " ")
    raw = re.sub(r"\s+", " ", raw).strip()
    parts = re.split(r"[+/\s,]+", raw)
    normalized_parts: List[str] = []
    for part in parts:
        p = part.strip()
        if not p or len(p) < 2:
            continue
        p = INGREDIENT_ALIAS_MAP.get(p, p)
        if p not in normalized_parts:
            normalized_parts.append(p)
    return "+".join(normalized_parts) if normalized_parts else raw


def _drop_low_value_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    dropped: List[str] = []
    keep = []
    n = max(len(df), 1)
    for col in df.columns:
        empty_pct = df[col].apply(is_empty_value).sum() / n * 100
        uniq = df[col].nunique(dropna=False)
        if empty_pct > 60 or uniq <= 1:
            dropped.append(col)
        else:
            keep.append(col)
    # Always keep core retrieval columns even if sparse
    for required in ("name_ar", "name_en", "price_egp", "combined", "dosage_clean", "form_clean", "ingredient_clean"):
        if required in df.columns and required not in keep:
            keep.append(required)
            if required in dropped:
                dropped.remove(required)
    return df[keep].copy(), dropped


def _build_known_brand_tokens(df: pd.DataFrame) -> Set[str]:
    tokens: Set[str] = set()
    for _, row in df.iterrows():
        for key in ("name_ar", "name_en"):
            val = normalize_text(str(row.get(key, "")))
            val = strip_form_noise(val)
            if len(val) >= 3:
                tokens.add(val)
                first = val.split()[0]
                if len(first) >= 3:
                    tokens.add(first)
    for alias in INGREDIENT_ALIAS_MAP:
        if len(alias) >= 3:
            tokens.add(normalize_text(alias))
    return tokens


def is_known_brand_query(query: str) -> bool:
    """True when query resolves to a brand token present in the dataset or alias map."""
    if not query:
        return False
    for variant in generate_search_variants(query):
        nv = strip_form_noise(normalize_text(variant))
        resolved = strip_form_noise(normalize_text(resolve_trade_alias(variant)))
        for token in (nv, resolved):
            if token in KNOWN_BRAND_TOKENS:
                return True
            if len(token) >= 4:
                for known in KNOWN_BRAND_TOKENS:
                    if len(known) >= 4 and (known == token or known.startswith(token + " ") or token.startswith(known + " ")):
                        return True
    return False


def tune_rerank_thresholds(df: pd.DataFrame, ingredient_col: str) -> Tuple[float, float]:
    """
    Estimate trade-name rerank cutoffs from known vs gibberish queries (lexical-only).
    Uses a row sample to keep startup fast on large CSVs.
    """
    from retrieval import DrugRetrievalEngine

    sample_df = df.head(3000) if len(df) > 3000 else df
    engine = DrugRetrievalEngine(
        df=sample_df,
        index=None,
        ingredient_col=ingredient_col,
        get_embed_model=lambda: None,
        enable_semantic=False,
    )
    known_scores: List[float] = []
    for name in sample_df["name_en"].dropna().astype(str).tolist()[:12]:
        brand = strip_form_noise(normalize_text(name.split()[0] if name else ""))
        if len(brand) < 4:
            continue
        cands = engine.hybrid_candidates(brand, mode="trade_name", top_k=20)
        reranked = engine.rerank_candidates(cands, brand, mode="trade_name")
        if reranked:
            known_scores.append(reranked[0].rerank)

    unknown_scores: List[float] = []
    for gibberish in ("toblaxiel", "xyzabc999", "qqqwwweee"):
        cands = engine.hybrid_candidates(gibberish, mode="trade_name", top_k=20)
        reranked = engine.rerank_candidates(cands, gibberish, mode="trade_name")
        if reranked:
            unknown_scores.append(reranked[0].rerank)

    if known_scores and unknown_scores:
        cutoff = (min(known_scores) + max(unknown_scores)) / 2
        cutoff = max(0.78, min(0.95, cutoff + 0.05))
    elif known_scores:
        cutoff = max(0.78, min(known_scores) * 0.85)
    else:
        cutoff = 0.85

    return round(cutoff, 3), TUNED_MIN_SEMANTIC


def clean_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    One-time startup cleaning. Returns (cleaned_df, metadata dict).
    """
    global KNOWN_BRAND_TOKENS, SUBSTITUTE_ELIGIBLE, TUNED_MIN_TRADE_RERANK, TUNED_MIN_SEMANTIC

    inventory = column_inventory(df)
    print("CSV column inventory:")
    for _, row in inventory.iterrows():
        print(f"   {row['column']:20s}  {row['dtype']:10s}  empty={row['null_empty_pct']:5.1f}%  unique={row['unique']}")

    cleaned, dropped_cols = _drop_low_value_columns(df)
    if dropped_cols:
        print(f"Dropped columns (>60% empty or single value): {', '.join(dropped_cols)}")

    ingredient_col = "ingredient_clean" if "ingredient_clean" in cleaned.columns else "active_ingredient"
    if ingredient_col in cleaned.columns:
        cleaned[ingredient_col] = cleaned[ingredient_col].apply(normalize_ingredient_canonical)
        if "active_ingredient" in cleaned.columns:
            cleaned["active_ingredient"] = cleaned[ingredient_col]

    SUBSTITUTE_ELIGIBLE = [
        not is_generic_ingredient(str(row.get(ingredient_col, "")))
        for _, row in cleaned.iterrows()
    ]
    generic_count = len(SUBSTITUTE_ELIGIBLE) - sum(SUBSTITUTE_ELIGIBLE)
    if generic_count:
        print(f"Flagged {generic_count} rows with generic active_ingredient (excluded from substitutes)")

    if "combined" not in cleaned.columns or cleaned["combined"].apply(is_empty_value).all():
        cleaned["combined"] = (
            cleaned.get("name_ar", pd.Series([""] * len(cleaned))).astype(str)
            + " "
            + cleaned.get("name_en", pd.Series([""] * len(cleaned))).astype(str)
            + " "
            + cleaned.get(ingredient_col, pd.Series([""] * len(cleaned))).astype(str)
        ).str.strip()

    KNOWN_BRAND_TOKENS = _build_known_brand_tokens(cleaned)
    TUNED_MIN_TRADE_RERANK, TUNED_MIN_SEMANTIC = tune_rerank_thresholds(cleaned, ingredient_col)
    print(f"Tuned trade-name rerank threshold: {TUNED_MIN_TRADE_RERANK}")

    meta = {
        "inventory": inventory.to_dict("records"),
        "dropped_columns": dropped_cols,
        "ingredient_col": ingredient_col,
        "known_brands": len(KNOWN_BRAND_TOKENS),
        "substitute_eligible_rows": sum(SUBSTITUTE_ELIGIBLE),
        "tuned_min_trade_rerank": TUNED_MIN_TRADE_RERANK,
    }
    return cleaned.reset_index(drop=True), meta


def display_form(row: dict) -> str:
    """Map form_clean to a display label; empty → omit upstream."""
    raw = str(row.get("form_clean") or row.get("form") or "").strip().lower()
    if is_empty_value(raw):
        return ""
    return FORM_LABELS.get(raw, raw)
