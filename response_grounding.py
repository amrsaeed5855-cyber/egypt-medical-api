"""
response_grounding.py — Safe, dataset-grounded medical response assembly.

When structured medication cards are available, drug details appear only in cards —
not duplicated in the assistant text.
"""

from __future__ import annotations

import re
from typing import List, Optional

OVERCONFIDENT_PHRASES = [
    "متقلقش خالص", "متقلقش", "بسيطة", "حاجة بسيطة", "إن شاء الله حاجة بسيطة",
    "مفيش حاجة تقلق", "عادي خالص", "اكيد حاجة بسيطة", "مفيش خطر",
]

HALLUCINATED_DRUG_BLOCK = re.compile(
    r"(?:^|\n)\s*💊\s*\*\*.+?\*\*.*?(?:\(صف\s*\d+\)|\(row\s*\d+\)).*?(?=\n\s*💊|\n\s*---|\Z)",
    re.DOTALL | re.IGNORECASE,
)
HALLUCINATED_ROW_REF = re.compile(r"\(صف\s*\d+\)|\(row\s*\d+\)", re.IGNORECASE)
DATASET_SECTION_HEADER = re.compile(
    r"\n*---\n📋\s*\*\*من قاعدة الأدوية المصرية:\*\*.*",
    re.DOTALL,
)
INLINE_DRUG_DETAILS = re.compile(
    r"(?:^|\n)\s*(?:💊|🔹)\s*\*\*.+?\*\*.*?(?=\n\s*(?:💊|🔹|💡|---)|\Z)",
    re.DOTALL,
)


def sanitize_medical_text(text: str) -> str:
    out = (text or "").strip()
    for phrase in OVERCONFIDENT_PHRASES:
        out = out.replace(phrase, "محتاجين نجمع تفاصيل أكتر قبل الحكم")
    return out.strip()


def strip_hallucinated_drug_content(text: str) -> str:
    """Remove LLM-generated drug blocks and row citations from visible text."""
    out = text or ""
    out = HALLUCINATED_DRUG_BLOCK.sub("", out)
    out = INLINE_DRUG_DETAILS.sub("", out)
    out = DATASET_SECTION_HEADER.sub("", out)
    out = HALLUCINATED_ROW_REF.sub("", out)
    return re.sub(r"\n{3,}", "\n\n", out).strip()


def assemble_grounded_response(
    visible_text: str,
    drug_section: str,
    medications: List[dict],
    safety_suffix: Optional[List[str]] = None,
    cards_only: bool = True,
) -> str:
    """
    Build final user-facing text.

    When medications[] is populated and cards_only=True (default), drug details
    are shown only in structured cards — the text contains guidance only.
    """
    base = strip_hallucinated_drug_content(sanitize_medical_text(visible_text))
    parts = [base] if base else []

    if not cards_only and drug_section and medications:
        parts.append(drug_section.strip())

    if safety_suffix:
        parts.append("\n".join(safety_suffix))

    return "\n\n".join(p for p in parts if p).strip()
