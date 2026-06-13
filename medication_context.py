# medication_context.py — Conversation-level medication search context and refinement resolution.

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from retrieval import extract_form_key, extract_strengths
from trade_name_utils import (
    classify_query,
    extract_drug_name_from_query,
    normalize_text,
    resolve_trade_alias,
    strip_form_noise,
)

REFINEMENT_RE = re.compile(
    r"^\s*(?:"
    r"\d+(?:\.\d+)?\s*(?:mg|mcg|g|gm|ml|iu|iu/ml|%|tab|tabs|cap|caps)?"
    r"|\d+(?:\.\d+)?"
    r")\s*$",
    re.IGNORECASE,
)

VOLUME_RE = re.compile(r"(\d+(?:\.\d+)?)\s*ml\b", re.IGNORECASE)

FORM_QUERY_RULES = (
    (("حقن", "امبول", "امبولة", "injection", "inject", "ampoule", "ampule", "vial"), "injection"),
    (("اقراص", "قرص", "tablet", "tab", "tabs"), "oral_solid"),
    (("كبسول", "capsule", "cap", "caps"), "oral_solid"),
    (("شراب", "syrup", "suspension", "susp", "معلق"), "liquid"),
    (("كريم", "cream", "gel", "مرهم", "ointment"), "topical"),
    (("قطرة", "drop", "drops", "spray", "بخاخ"), "drops_spray"),
    (("اكياس", "sachet", "sachets"), "sachet"),
)

DRUG_NOISE_PREFIXES = ("دواء", "دوا", "medicine", "drug", "medication")

NO_APPROPRIATE_RESULT_MSG = "No medically appropriate result found in the database."
FORM_NOT_FOUND_MSG = (
    "Requested dosage form not found. "
    "Would you like alternatives in another form?"
)


@dataclass
class MedicationSearchContext:
    """Conversation-level medication context preserved across follow-up turns."""

    drug_name: str = ""
    active_ingredient: str = ""
    form_key: str = ""
    strengths: Set[str] = field(default_factory=set)
    volume_ml: Optional[str] = None
    query_intent: str = "product_info"
    source_query: str = ""

    def is_active(self) -> bool:
        return bool(self.drug_name)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "drug_name": self.drug_name,
            "active_ingredient": self.active_ingredient,
            "form_key": self.form_key,
            "strengths": sorted(self.strengths),
            "volume_ml": self.volume_ml,
            "query_intent": self.query_intent,
            "source_query": self.source_query,
        }


def strip_drug_noise(name: str) -> str:
    norm = normalize_text(name or "")
    for prefix in DRUG_NOISE_PREFIXES:
        if norm.startswith(prefix + " "):
            norm = norm[len(prefix) + 1 :].strip()
    return norm


def extract_requested_form(query: str) -> tuple[str, str]:
    """
    Detect an explicit dosage form in the query.

    Returns (form_key, query_with_form_tokens_removed).
    """
    norm = normalize_text(query or "")
    remaining = norm
    for keywords, form_key in FORM_QUERY_RULES:
        for kw in keywords:
            if re.search(rf"\b{re.escape(kw)}\b", remaining):
                remaining = re.sub(rf"\b{re.escape(kw)}\b", " ", remaining)
                remaining = re.sub(r"\s+", " ", remaining).strip()
                return form_key, remaining
    return "", norm


def extract_volume_ml(query: str) -> Optional[str]:
    m = VOLUME_RE.search(query or "")
    if not m:
        return None
    return f"{m.group(1)}ml".replace(" ", "").lower()


def is_refinement_followup(query: str, prior_context: Optional[MedicationSearchContext] = None) -> bool:
    """Short numeric/unit replies that refine an existing medication context."""
    if not query or not prior_context or not prior_context.is_active():
        return False
    norm = normalize_text(query)
    if REFINEMENT_RE.match(norm):
        return True
    if len(norm.split()) <= 2 and extract_strengths(norm):
        return True
    if len(norm) <= 6 and norm.isdigit():
        return True
    return False


def build_context_from_query(query: str, query_type: str = "") -> MedicationSearchContext:
    form_key, stripped = extract_requested_form(query)
    raw_drug = strip_drug_noise(extract_drug_name_from_query(stripped) or extract_drug_name_from_query(query) or "")
    volume = extract_volume_ml(query) or extract_volume_ml(raw_drug)
    drug = strip_form_noise(raw_drug)
    if volume:
        drug = re.sub(rf"\b{re.escape(volume.replace('ml', ''))}\s*ml\b", "", drug, flags=re.IGNORECASE).strip()
        drug = strip_form_noise(drug)
    strengths = extract_strengths(query)
    intent = query_type or classify_query(query)
    if volume and not form_key:
        form_key = "liquid"
    ingredient = ingredient_hint_for_trade(drug) if drug else ""
    return MedicationSearchContext(
        drug_name=drug,
        active_ingredient=ingredient,
        form_key=form_key,
        strengths=strengths,
        volume_ml=volume,
        query_intent=intent,
        source_query=query.strip(),
    )


def build_context_from_row(row: dict, source_query: str, query_type: str, ingredient_col: str = "active_ingredient") -> MedicationSearchContext:
    ing = str(row.get(ingredient_col, "") or row.get("active_ingredient", "") or "")
    text = " ".join(
        str(row.get(k, "") or "")
        for k in (ingredient_col, "active_ingredient", "dosage_clean", "name_en", "name_ar")
    )
    return MedicationSearchContext(
        drug_name=strip_drug_noise(extract_drug_name_from_query(source_query) or source_query),
        active_ingredient=ing,
        form_key=extract_form_key(row),
        strengths=extract_strengths(text),
        volume_ml=extract_volume_ml(text),
        query_intent=query_type,
        source_query=source_query.strip(),
    )


def extract_medication_context_from_history(history: list) -> MedicationSearchContext:
    """Reconstruct medication context from prior user turns (stateless session)."""
    if not history:
        return MedicationSearchContext()

    last_ctx = MedicationSearchContext()
    for msg in history:
        if msg.get("role") != "user":
            continue
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        if is_refinement_followup(content, last_ctx):
            continue
        last_ctx = build_context_from_query(content, classify_query(content))
    return last_ctx


def merge_refinement_with_context(context: MedicationSearchContext, refinement: str) -> str:
    """Build an effective product query from preserved context + follow-up token."""
    ref = normalize_text(refinement)
    base = context.drug_name or strip_form_noise(context.source_query)
    intent_prefix = "بديل" if context.query_intent == "substitute" else ""

    ref_strengths = extract_strengths(ref)
    ref_volume = extract_volume_ml(ref)

    parts: List[str] = []
    if intent_prefix:
        parts.append(intent_prefix)
    if base:
        parts.append(base)

    if ref_volume:
        parts.append(ref_volume)
    elif ref_strengths:
        parts.extend(sorted(ref_strengths))
    elif ref.isdigit():
        num = ref
        if context.form_key == "liquid" or (context.query_intent == "substitute" and float(num) <= 250):
            parts.append(f"{num}ml")
        elif context.strengths:
            parts.append(num)
        else:
            parts.append(num)

    merged = " ".join(p for p in parts if p).strip()
    if context.form_key and context.form_key not in merged:
        form_tokens = {
            "injection": "حقن",
            "liquid": "شراب",
            "oral_solid": "",
            "topical": "كريم",
            "drops_spray": "قطرة",
            "sachet": "اكياس",
        }
        token = form_tokens.get(context.form_key, "")
        if token:
            merged = f"{token} {merged}".strip()
    return merged or context.source_query


def resolve_conversation_query(
    query: str,
    history: list,
) -> tuple[str, MedicationSearchContext, bool, str]:
    """
    Resolve effective search query using conversation medication context.

    Returns (effective_query, context, is_refinement, requested_form_key).
    """
    prior = extract_medication_context_from_history(history)
    form_key, stripped = extract_requested_form(query)
    requested_form = form_key or prior.form_key

    if is_refinement_followup(query, prior):
        effective = merge_refinement_with_context(prior, query)
        ctx = MedicationSearchContext(
            drug_name=prior.drug_name,
            active_ingredient=prior.active_ingredient or ingredient_hint_for_trade(prior.drug_name),
            form_key=prior.form_key or requested_form or ("liquid" if extract_volume_ml(effective) else ""),
            strengths=extract_strengths(effective) or prior.strengths,
            volume_ml=extract_volume_ml(effective) or prior.volume_ml,
            query_intent=prior.query_intent,
            source_query=prior.source_query,
        )
        return effective, ctx, True, requested_form

    ctx = build_context_from_query(query)
    if not ctx.form_key and prior.form_key:
        ctx.form_key = prior.form_key
    if requested_form:
        ctx.form_key = requested_form
    effective = stripped if form_key else query
    if form_key:
        drug = strip_drug_noise(extract_drug_name_from_query(effective) or effective)
        ctx.drug_name = drug or ctx.drug_name
    return effective, ctx, False, requested_form


def row_matches_requested_form(row: dict, form_key: str) -> bool:
    if not form_key:
        return True
    return extract_form_key(row) == form_key


TRADE_INGREDIENT_HINTS: dict[str, str] = {
    "dexa": "dexamethasone",
    "ator": "atorvastatin",
    "cetal": "paracetamol",
    "augmentin": "amoxicillin",
    "panadol": "paracetamol",
}


def score_row_for_query(row: dict, query: str, med_context: Optional[MedicationSearchContext] = None) -> float:
    """Rank how well a row matches explicit query constraints (form, strength, volume)."""
    from trade_name_utils import normalize_text, resolve_trade_alias, strip_form_noise

    score = float(row.get("retrieval_score") or 0)
    query_norm = normalize_text(query)
    drug = strip_drug_noise(extract_drug_name_from_query(query) or query)
    alias = resolve_trade_alias(drug or query)
    name_en = normalize_text(row.get("name_en", ""))
    name_ar = normalize_text(row.get("name_ar", ""))
    combined = f"{name_ar} {name_en}"

    if alias and len(alias) >= 3:
        if name_en.startswith(alias) or f" {alias}" in f" {name_en}":
            score += 0.20
        elif alias in combined:
            score += 0.10

    variant_penalty = ("extra", "cold", "flu", "sinus", "joint", "migraine", "baby", "infant", "advance", "actifast")
    for mod in variant_penalty:
        if mod in name_en and mod not in query_norm:
            score -= 0.10
    if "panadol" in (alias, query_norm):
        if "extra" not in query_norm:
            if "panadol 500" in name_en or ("500" in name_en and "extra" not in name_en):
                score += 0.12

    strengths = extract_strengths(query)
    if med_context and med_context.strengths:
        strengths = strengths | med_context.strengths
    row_strengths = extract_strengths(
        " ".join(str(row.get(k, "") or "") for k in ("active_ingredient", "ingredient_clean", "dosage_clean", "name_en"))
    )
    if strengths:
        score += 0.35 if strengths & row_strengths else -0.30

    form_key = (med_context.form_key if med_context else "") or extract_requested_form(query)[0]
    volume = (med_context.volume_ml if med_context else None) or extract_volume_ml(query)
    row_form = extract_form_key(row)

    if form_key:
        score += 0.25 if row_form == form_key else -0.40
    if volume:
        row_text = f"{row.get('name_en', '')} {row.get('name_ar', '')}".lower()
        if volume in row_text.replace(" ", ""):
            score += 0.30
        elif row_form == "liquid":
            score += 0.05
        else:
            score -= 0.35
    elif re.search(r"\b120\s*ml\b", query_norm):
        if "120ml" in combined.replace(" ", ""):
            score += 0.30
        elif row_form == "liquid":
            score += 0.08
        elif row_form == "oral_solid":
            score -= 0.40

    return score


def ingredient_hint_for_trade(name: str) -> str:
    norm = normalize_text(name or "")
    alias = resolve_trade_alias(norm)
    return TRADE_INGREDIENT_HINTS.get(alias, TRADE_INGREDIENT_HINTS.get(norm, ""))


def row_matches_context_refinement(row: dict, context: MedicationSearchContext) -> bool:
    """Filter rows when user sent a numeric refinement against an active context."""
    if not context.is_active():
        return True

    text = " ".join(str(row.get(k, "") or "") for k in ("name_en", "name_ar", "dosage_clean", "active_ingredient", "ingredient_clean"))
    row_strengths = extract_strengths(text)
    row_volume = extract_volume_ml(text)

    if context.volume_ml and row_volume and context.volume_ml != row_volume:
        return False

    if context.strengths and row_strengths and not (context.strengths & row_strengths):
        if context.volume_ml:
            return False
        if len(context.strengths) == 1:
            return False

    if context.form_key and not row_matches_requested_form(row, context.form_key):
        return False

    if context.drug_name:
        drug_norm = strip_form_noise(normalize_text(context.drug_name))
        combined = normalize_text(f"{row.get('name_ar', '')} {row.get('name_en', '')} {row.get('active_ingredient', '')}")
        if drug_norm and len(drug_norm) >= 3:
            if drug_norm not in combined and drug_norm.split()[0] not in combined:
                ing = normalize_text(str(row.get("active_ingredient", "") or row.get("ingredient_clean", "")))
                if context.active_ingredient:
                    if normalize_text(context.active_ingredient) not in ing:
                        return False
                elif context.query_intent != "substitute":
                    return False
    return True
