"""
response_grounding.py — Safe, dataset-grounded medical response assembly.

Ensures visible text does not hallucinate drug rows/prices and strips unsafe phrasing.
"""

from __future__ import annotations

import re
from typing import List, Optional

# Phrases that imply diagnosis or dismiss severity
DIAGNOSIS_PATTERNS = [
    (r"عندك\s+(?:انفلونزا|التهاب\s+رئوي|كورونا|covid)", "الأعراض محتاجة تقييم طبي"),
    (r"ده\s+(?:انفلونزا|التهاب|عدوى\s+بكتير)", "الأعراض محتاجة تقييم طبي"),
    (r"تشخيص(?:ك|ي)?\s*(?:هو|إنه|انه)", "التشخيص مش من اختصاص الصيدلي"),
    (r"you have (?:flu|pneumonia|infection)", "symptoms need medical evaluation"),
]

OVERCONFIDENT_PHRASES = [
    "متقلقش خالص", "متقلقش", "بسيطة", "حاجة بسيطة", "إن شاء الله حاجة بسيطة",
    "مفيش حاجة تقلق", "عادي خالص", "اكيد حاجة بسيطة", "مفيش خطر",
]

# Strip LLM-invented drug blocks and row citations from visible text
HALLUCINATED_DRUG_BLOCK = re.compile(
    r"(?:^|\n)\s*💊\s*\*\*.+?\*\*.*?(?:\(صف\s*\d+\)|\(row\s*\d+\)).*?(?=\n\s*💊|\n\s*---|\Z)",
    re.DOTALL | re.IGNORECASE,
)
HALLUCINATED_ROW_REF = re.compile(r"\(صف\s*\d+\)|\(row\s*\d+\)", re.IGNORECASE)
DATASET_SECTION_HEADER = re.compile(
    r"\n*---\n📋\s*\*\*من قاعدة الأدوية المصرية:\*\*.*",
    re.DOTALL,
)


def sanitize_medical_text(text: str) -> str:
    out = (text or "").strip()
    for phrase in OVERCONFIDENT_PHRASES:
        out = out.replace(phrase, "محتاجين نجمع تفاصيل أكتر قبل الحكم")
    for pattern, replacement in DIAGNOSIS_PATTERNS:
        out = re.sub(pattern, replacement, out, flags=re.IGNORECASE)
    return out.strip()


def strip_hallucinated_drug_content(text: str) -> str:
    """Remove LLM-generated drug blocks; structured retrieval re-adds them."""
    out = text or ""
    out = HALLUCINATED_DRUG_BLOCK.sub("", out)
    out = DATASET_SECTION_HEADER.sub("", out)
    out = HALLUCINATED_ROW_REF.sub("", out)
    return re.sub(r"\n{3,}", "\n\n", out).strip()


def assemble_grounded_response(
    visible_text: str,
    drug_section: str,
    medications: List[dict],
    safety_suffix: Optional[List[str]] = None,
) -> str:
    """
    Build final user-facing text from sanitized LLM prose + verified drug blocks only.
    """
    base = strip_hallucinated_drug_content(sanitize_medical_text(visible_text))
    parts = [base] if base else []

    if drug_section and medications:
        parts.append(drug_section.strip())

    if safety_suffix:
        parts.append("\n".join(safety_suffix))

    return "\n\n".join(p for p in parts if p).strip()


def validate_row_citations(text: str, valid_rows: List[int]) -> bool:
    """True when every (صف N) in text refers to a retrieved row."""
    cited = {int(m.group(1)) for m in re.finditer(r"\(صف\s*(\d+)\)", text or "", re.IGNORECASE)}
    if not cited:
        return True
    valid = set(valid_rows)
    return cited.issubset(valid)
