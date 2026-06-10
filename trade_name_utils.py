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
