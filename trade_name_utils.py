# trade_name_utils.py — Trade-name query normalization and extraction.
# Changed: multi-drug split, ambiguous follow-up detection, name variant collection for substitute exclusion.

"""
trade_name_utils.py — Trade-name query normalization and extraction.

Supports exact/normalized matching, Arabic↔English aliases, Franco-Arabic,
and drug-name extraction from common pharmacy question patterns.
"""

from __future__ import annotations

import re
from typing import List, Optional, Set

# Franco-Arabic (Arabizi) → Arabic letter mapping (common Egyptian pharmacy usage)
FRANCO_MAP = {
    "2": "ا", "3": "ع", "3a": "غ", "3'": "غ", "5": "خ", "6": "ط",
    "7": "ح", "8": "ج", "9": "ص", "sh": "ش", "kh": "خ", "gh": "غ",
    "aa": "ا", "ee": "ي", "oo": "و",
}

# Well-known trade names: any alias → canonical English search term
TRADE_NAME_ALIASES: dict[str, str] = {
    "بانادول": "panadol",
    "بنادول": "panadol",
    "panadol": "panadol",
    "بانادول اكسترا": "panadol extra",
    "panadol extra": "panadol extra",
    "اوجمنتين": "augmentin",
    "أوجمنتين": "augmentin",
    "augmentin": "augmentin",
    "اوجمنتين": "augmentin",
    "بروفين": "brufen",
    "brufen": "brufen",
    "ادفيل": "advil",
    "advil": "advil",
    "فولتارين": "voltaren",
    "voltaren": "voltaren",
    "فلانزا": "flansa",
    "flansa": "flansa",
    "كونتاجيون": "augmentin",
    "كتافلام": "cataflam",
    "cataflam": "cataflam",
    "اسبرين": "aspirin",
    "aspirin": "aspirin",
    "كلاريتين": "claritin",
    "claritin": "claritin",
    "زيرتك": "zyrtec",
    "zyrtec": "zyrtec",
}

# Tokens stripped for secondary fuzzy pass (dosage/form noise)
STRIP_TOKENS_RE = re.compile(
    r"\b(?:gm?|mg|ml|tab(?:let)?s?|cap(?:sule)?s?|sachets?|cream|gel|syrup|susp(?:ension)?|"
    r"قرص|أقراص|اقراص|كبسول|شراب|كريم|جم|مل|علبة|عبوة)\b",
    re.IGNORECASE,
)

AR_NUMS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

PRODUCT_INFO_MARKERS = [
    "سعر", "بكام", "كام", "price", "بديل", "بدائل", "بدل", "مكان",
    "substitute", "alternative", "المادة الفعالة", "active ingredient",
    "متوفر", "availability", "تفاصيل", "بيعمل ايه", "ايه استخدام",
    "ايه فايده", "ايه فائده", "what does", "dosage form", "الشكل",
    "التركيز", "composition", "مكونات",
]

SYMPTOM_TREATMENT_MARKERS = [
    "دواء ل", "عاوز دواء", "عايز دواء", "علاج", "للصداع", "للكحة",
    "للحرارة", "للزكام", "للألم", "للام", "للمغص", "للاسهال",
    "للإسهال", "medicine for", "treatment for",
]

SUBSTITUTE_MARKERS = ["بديل", "بدائل", "بدل", "مكان", "substitute", "alternative", "generic for"]

AMBIGUOUS_FOLLOWUP_MARKERS = [
    "فين", "كمّل", "كمml", "وبعدين", "إيه", "ايه", "تاني", "والنتيجة",
    "النتيجه", "و بعدين", "فين النتيجة", "فين النتيجه", "more", "continue",
]

SHOW_MORE_MARKERS = [
    "اكتر", "أكتر", "more", "زود", "باقي", "الباقي", "نتائج تانية", "نتائج أخرى",
]

MULTI_DRUG_SPLIT_RE = re.compile(r"\s*\+\s*|\s*,\s*|\s*،\s*|\s*&\s*|\s+و\s+|\s+و(?=\S)|(?<=\S)و\s+")

def collect_name_variants(row: dict) -> Set[str]:
    """All normalized trade-name tokens for a row (for substitute self-exclusion)."""
    variants: Set[str] = set()
    for key in ("name_ar", "name_en"):
        raw = str(row.get(key, "") or "")
        norm = strip_form_noise(trade_normalize := normalize_text(raw))
        if len(norm) >= 2:
            variants.add(norm)
            for part in re.split(r"[\(\)/\-]", raw):
                p = strip_form_noise(normalize_text(part))
                if len(p) >= 3:
                    variants.add(p)
            first = norm.split()[0]
            if len(first) >= 3:
                variants.add(first)
    return variants


def is_ambiguous_followup(query: str) -> bool:
    norm = normalize_text(query)
    words = norm.split()
    if len(words) <= 3 and any(m in norm for m in AMBIGUOUS_FOLLOWUP_MARKERS):
        return True
    return len(norm) <= 12 and norm in {normalize_text(m) for m in AMBIGUOUS_FOLLOWUP_MARKERS}


def is_show_more_request(query: str) -> bool:
    norm = normalize_text(query)
    return any(m in norm for m in SHOW_MORE_MARKERS)


def split_multi_drug_names(text: str) -> List[str]:
    """Split multi-drug queries on و / , / + and return cleaned drug name fragments."""
    raw = (text or "").strip()
    if not raw:
        return []
    norm = normalize_text(raw)
    for pattern in DRUG_EXTRACT_PATTERNS:
        m = re.search(pattern, norm, re.IGNORECASE)
        if m:
            raw = m.group(1).strip()
            norm = normalize_text(raw)
            break
    for marker in ("سعر", "بكام", "كام", "بديل", "بدائل", "price"):
        if marker in norm:
            parts = re.split(rf"{re.escape(marker)}\s*(?:لـ?|ل)?\s*", norm, maxsplit=1)
            if len(parts) > 1 and parts[1].strip():
                norm = parts[1].strip()
                break
    norm = re.sub(r"[؟?!.]+$", "", norm).strip()
    parts = [p.strip() for p in MULTI_DRUG_SPLIT_RE.split(norm) if p.strip()]
    cleaned: List[str] = []
    for part in parts:
        part = re.sub(r"^(?:سعر|بكام|كام|بديل|بدائل|price|substitute)\s*(?:لـ?|ل)?\s*", "", part).strip()
        if len(part) >= 2:
            cleaned.append(part)
    return cleaned if len(cleaned) > 1 else []


def resolve_followup_from_history(query: str, history: list) -> Optional[str]:
    """Reconstruct the last product query when user sends an ambiguous follow-up."""
    if not is_ambiguous_followup(query):
        return None
    for msg in reversed(history or []):
        if msg.get("role") != "user":
            continue
        prev = (msg.get("content") or "").strip()
        if prev and not is_ambiguous_followup(prev):
            return prev
    return None


DRUG_EXTRACT_PATTERNS = [
    r"(?:بديل|بدائل|بدل|مكان)\s+(?:لـ?|ل)?\s*(.+)",
    r"(?:سعر|بكام|كام)\s+(?:لـ?|ل)?\s*(.+)",
    r"(?:alternative|substitute|generic)\s+(?:for|of)?\s*(.+)",
    r"(?:ما هو|ايه|إيه|what is)\s+(?:دواء\s+)?(.+)",
    r"(?:المادة الفعالة|active ingredient)\s+(?:لـ?|ل|of)?\s*(.+)",
    r"(?:تفاصيل|details|info)\s+(?:عن|about|of)?\s*(.+)",
]


def normalize_text(text: str) -> str:
    """Arabic-aware normalization for matching."""
    text = (text or "").translate(AR_NUMS)
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ة", "ه").replace("ى", "ي")
    text = re.sub(r"[^\w\s+/.-]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def franco_to_arabic(text: str) -> str:
    """Convert common Franco-Arabic (Arabizi) spellings to Arabic."""
    if not text:
        return ""
    out = text.lower()
    for latin, arabic in sorted(FRANCO_MAP.items(), key=lambda x: -len(x[0])):
        out = out.replace(latin, arabic)
    return out


def resolve_trade_alias(text: str) -> str:
    """Map known aliases to canonical English trade name."""
    norm = normalize_text(text)
    if norm in TRADE_NAME_ALIASES:
        return TRADE_NAME_ALIASES[norm]
    franco = normalize_text(franco_to_arabic(text))
    if franco in TRADE_NAME_ALIASES:
        return TRADE_NAME_ALIASES[franco]
    for alias, canonical in TRADE_NAME_ALIASES.items():
        if alias in norm or norm in alias:
            return canonical
    return norm


def strip_form_noise(text: str) -> str:
    """Remove dosage/form tokens for a cleaner name match."""
    cleaned = STRIP_TOKENS_RE.sub(" ", text or "")
    return re.sub(r"\s+", " ", cleaned).strip()


def generate_search_variants(query: str) -> List[str]:
    """Produce search variants: raw, normalized, alias, Franco, stripped."""
    if not query or len(query.strip()) < 2:
        return []

    raw = query.strip()
    norm = normalize_text(raw)
    alias = resolve_trade_alias(raw)
    franco = normalize_text(franco_to_arabic(raw))
    stripped = strip_form_noise(norm)
    stripped_alias = strip_form_noise(alias)

    variants: List[str] = []
    seen: Set[str] = set()
    for v in (raw, norm, alias, franco, stripped, stripped_alias):
        key = v.lower().strip()
        if key and len(key) >= 2 and key not in seen:
            seen.add(key)
            variants.append(v)
    return variants


def extract_drug_name_from_query(query: str) -> Optional[str]:
    """Extract the target drug name from product/substitute questions."""
    q = (query or "").strip()
    if not q:
        return None

    norm = normalize_text(q)
    for pattern in DRUG_EXTRACT_PATTERNS:
        m = re.search(pattern, norm, re.IGNORECASE)
        if m:
            candidate = m.group(1).strip()
            candidate = re.sub(r"[؟?!.،,]+$", "", candidate).strip()
            if len(candidate) >= 2:
                return candidate

    for marker in SUBSTITUTE_MARKERS + ["سعر", "بكام", "price"]:
        if marker in norm:
            parts = re.split(rf"{re.escape(marker)}\s*(?:لـ?|ل)?", norm, maxsplit=1)
            if len(parts) > 1 and parts[1].strip():
                return parts[1].strip()

    # Short queries that look like a drug name only
    if len(norm) <= 40 and not any(m in norm for m in SYMPTOM_TREATMENT_MARKERS):
        if any(m in norm for m in PRODUCT_INFO_MARKERS):
            return None
        words = norm.split()
        if 1 <= len(words) <= 5:
            return norm

    return None


def classify_query(query: str) -> str:
    """
    Classify user intent for routing.

    Returns: product_info | substitute | symptom_treatment | general
    """
    norm = normalize_text(query)
    if any(m in norm for m in SUBSTITUTE_MARKERS):
        return "substitute"
    if any(m in norm for m in PRODUCT_INFO_MARKERS):
        return "product_info"
    if any(m in norm for m in SYMPTOM_TREATMENT_MARKERS):
        return "symptom_treatment"
    drug = extract_drug_name_from_query(query)
    if drug and len(drug) >= 3:
        return "product_info"
    return "general"
