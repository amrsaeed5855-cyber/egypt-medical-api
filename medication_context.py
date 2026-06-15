# medication_context.py — Conversation-level medication search context and refinement resolution.

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from retrieval import (
    extract_concentration,
    extract_form_key,
    extract_pack_size,
    extract_strengths,
    extract_volume_ml,
    strengths_compatible,
)
from trade_name_utils import (
    classify_query,
    extract_drug_name_from_query,
    normalize_text,
    resolve_trade_alias,
    strip_drug_noise,
    strip_form_noise,
)

REFINEMENT_RE = re.compile(
    r"^\s*(?:"
    r"\d+(?:\.\d+)?\s*(?:mg|mcg|g|gm|ml|iu|iu/ml|%|tab|tabs|cap|caps)?"
    r"|\d+(?:\.\d+)?"
    r")\s*$",
    re.IGNORECASE,
)

FORM_QUERY_RULES = (
    (("حقن", "امبول", "امبولة", "injection", "inject", "ampoule", "ampule", "vial"), "injection"),
    (("اقراص", "قرص", "tablet", "tab", "tabs"), "oral_solid"),
    (("كبسول", "capsule", "cap", "caps"), "oral_solid"),
    (("شراب", "syrup", "suspension", "susp", "معلق"), "liquid"),
    (("كريم", "cream", "gel", "مرهم", "ointment"), "topical"),
    (("قطرة", "drop", "drops", "spray", "بخاخ"), "drops_spray"),
    (("اكياس", "sachet", "sachets"), "sachet"),
)

NO_APPROPRIATE_RESULT_MSG = "No medically appropriate result found in the database."
FORM_NOT_FOUND_MSG = (
    "Requested dosage form not found. "
    "Would you like alternatives in another form?"
)
NO_SUBSTITUTE_MSG = (
    "No medically appropriate substitute found with the same strength and dosage form."
)


@dataclass
class MedicationSearchContext:
    """Conversation-level medication context preserved across follow-up turns."""

    trade_name: str = ""
    drug_name: str = ""
    active_ingredient: str = ""
    strength: str = ""
    concentration: str = ""
    dosage_form: str = ""
    form_key: str = ""
    volume: str = ""
    volume_ml: Optional[str] = None
    pack_size: str = ""
    strengths: Set[str] = field(default_factory=set)
    query_intent: str = "product_info"
    source_query: str = ""

    def is_active(self) -> bool:
        return bool(self.trade_name or self.drug_name or self.active_ingredient)

    def sync_fields(self) -> None:
        if self.trade_name and not self.drug_name:
            self.drug_name = self.trade_name
        elif self.drug_name and not self.trade_name:
            self.trade_name = self.drug_name
        if self.form_key and not self.dosage_form:
            self.dosage_form = self.form_key
        elif self.dosage_form and not self.form_key:
            self.form_key = self.dosage_form
        if self.volume_ml and not self.volume:
            self.volume = self.volume_ml
        elif self.volume and not self.volume_ml:
            self.volume_ml = self.volume
        if self.strengths and not self.strength:
            self.strength = sorted(self.strengths)[0]

    def to_dict(self) -> Dict[str, Any]:
        self.sync_fields()
        return {
            "trade_name": self.trade_name,
            "drug_name": self.drug_name,
            "active_ingredient": self.active_ingredient,
            "strength": self.strength,
            "concentration": self.concentration,
            "dosage_form": self.dosage_form,
            "form_key": self.form_key,
            "volume": self.volume,
            "volume_ml": self.volume_ml,
            "pack_size": self.pack_size,
            "strengths": sorted(self.strengths),
            "query_intent": self.query_intent,
            "source_query": self.source_query,
        }


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


def extract_constraints_from_text(text: str) -> Dict[str, Any]:
    norm = text or ""
    strengths = extract_strengths(norm)
    concentration = extract_concentration(norm) or ""
    volume = extract_volume_ml(norm) or ""
    pack_size = extract_pack_size(norm) or ""
    form_key, _ = extract_requested_form(norm)
    strength = sorted(strengths)[0] if strengths else ""
    return {
        "strengths": strengths,
        "strength": strength,
        "concentration": concentration,
        "volume_ml": volume or None,
        "volume": volume,
        "pack_size": pack_size,
        "form_key": form_key,
        "dosage_form": form_key,
    }


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
    if len(norm.split()) <= 3 and extract_volume_ml(norm):
        return True
    return False


def merge_context(
    prior: MedicationSearchContext,
    new: MedicationSearchContext,
) -> MedicationSearchContext:
    """Merge new query fields into prior conversation state without resetting."""
    merged = MedicationSearchContext(
        trade_name=new.trade_name or prior.trade_name,
        drug_name=new.drug_name or prior.drug_name,
        active_ingredient=new.active_ingredient or prior.active_ingredient,
        strength=new.strength or prior.strength,
        concentration=new.concentration or prior.concentration,
        dosage_form=new.dosage_form or prior.dosage_form,
        form_key=new.form_key or prior.form_key,
        volume=new.volume or prior.volume,
        volume_ml=new.volume_ml or prior.volume_ml,
        pack_size=new.pack_size or prior.pack_size,
        strengths=new.strengths or set(prior.strengths),
        query_intent=new.query_intent if new.query_intent != "product_info" else prior.query_intent,
        source_query=new.source_query or prior.source_query,
    )
    if new.strengths:
        merged.strengths = set(prior.strengths) | set(new.strengths)
    if new.strength:
        merged.strength = new.strength
    merged.sync_fields()
    return merged


def apply_refinement_to_context(context: MedicationSearchContext, refinement: str) -> MedicationSearchContext:
    ref = normalize_text(refinement)
    constraints = extract_constraints_from_text(ref)
    updated = MedicationSearchContext(
        trade_name=context.trade_name,
        drug_name=context.drug_name,
        active_ingredient=context.active_ingredient,
        strength=constraints["strength"] or context.strength,
        concentration=constraints["concentration"] or context.concentration,
        dosage_form=context.dosage_form,
        form_key=context.form_key,
        volume=constraints["volume"] or context.volume,
        volume_ml=constraints["volume_ml"] or context.volume_ml,
        pack_size=constraints["pack_size"] or context.pack_size,
        strengths=set(context.strengths) | set(constraints["strengths"]),
        query_intent=context.query_intent,
        source_query=context.source_query,
    )
    if ref.isdigit():
        num = ref
        if context.form_key == "liquid" or (context.query_intent == "substitute" and float(num) <= 250):
            updated.volume_ml = f"{num}ml"
            updated.volume = updated.volume_ml
        elif not updated.strengths:
            updated.strength = num
            updated.strengths = extract_strengths(f"{num}mg") or {num}
    if constraints["form_key"]:
        updated.form_key = constraints["form_key"]
        updated.dosage_form = constraints["form_key"]
    updated.sync_fields()
    return updated


def build_context_from_query(query: str, query_type: str = "") -> MedicationSearchContext:
    form_key, stripped = extract_requested_form(query)
    raw_drug = strip_drug_noise(extract_drug_name_from_query(stripped) or extract_drug_name_from_query(query) or "")
    constraints = extract_constraints_from_text(query)
    volume = constraints["volume_ml"]
    drug = strip_form_noise(raw_drug)
    if volume:
        drug = re.sub(rf"\b{re.escape(volume.replace('ml', ''))}\s*ml\b", "", drug, flags=re.IGNORECASE).strip()
        drug = strip_form_noise(drug)
    intent = query_type or classify_query(query)
    if volume and not form_key:
        form_key = "liquid"
    ingredient = ingredient_hint_for_trade(drug) if drug else ""
    ctx = MedicationSearchContext(
        trade_name=drug,
        drug_name=drug,
        active_ingredient=ingredient,
        strength=constraints["strength"],
        concentration=constraints["concentration"],
        dosage_form=form_key,
        form_key=form_key,
        volume=constraints["volume"],
        volume_ml=volume,
        pack_size=constraints["pack_size"],
        strengths=set(constraints["strengths"]),
        query_intent=intent,
        source_query=query.strip(),
    )
    ctx.sync_fields()
    return ctx


def extract_medication_context_from_history(history: list) -> MedicationSearchContext:
    """Reconstruct medication context from prior user turns (stateless session)."""
    if not history:
        return MedicationSearchContext()

    ctx = MedicationSearchContext()
    for msg in history:
        if msg.get("role") != "user":
            continue
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        if is_refinement_followup(content, ctx):
            ctx = apply_refinement_to_context(ctx, content)
        else:
            ctx = merge_context(ctx, build_context_from_query(content, classify_query(content)))
    return ctx


def merge_refinement_with_context(context: MedicationSearchContext, refinement: str) -> str:
    """Build an effective product query from preserved context + follow-up token."""
    updated = apply_refinement_to_context(context, refinement)
    ref = normalize_text(refinement)
    base = updated.trade_name or updated.drug_name or strip_form_noise(updated.source_query)
    intent_prefix = "بديل" if updated.query_intent == "substitute" else ""

    parts: List[str] = []
    if intent_prefix:
        parts.append(intent_prefix)
    if base:
        parts.append(base)
    if updated.volume_ml:
        parts.append(updated.volume_ml)
    elif updated.strengths:
        parts.extend(sorted(updated.strengths))
    elif updated.strength:
        parts.append(updated.strength)
    elif ref.isdigit():
        parts.append(ref)

    merged = " ".join(p for p in parts if p).strip()
    if updated.form_key and updated.form_key not in merged:
        form_tokens = {
            "injection": "حقن",
            "liquid": "شراب",
            "oral_solid": "",
            "topical": "كريم",
            "drops_spray": "قطرة",
            "sachet": "اكياس",
        }
        token = form_tokens.get(updated.form_key, "")
        if token:
            merged = f"{token} {merged}".strip()
    return merged or updated.source_query


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
    requested_form = form_key or prior.form_key or prior.dosage_form

    if is_refinement_followup(query, prior):
        effective = merge_refinement_with_context(prior, query)
        ctx = apply_refinement_to_context(prior, query)
        ctx = merge_context(prior, ctx)
        ctx.sync_fields()
        return effective, ctx, True, requested_form

    ctx = merge_context(prior, build_context_from_query(query))
    if requested_form:
        ctx.form_key = requested_form
        ctx.dosage_form = requested_form
    effective = stripped if form_key else query
    if form_key:
        drug = strip_drug_noise(extract_drug_name_from_query(effective) or effective)
        if drug:
            ctx.trade_name = drug
            ctx.drug_name = drug
    ctx.sync_fields()
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
        score += 0.35 if strengths_compatible(strengths, row_strengths) else -0.30

    form_key = (med_context.form_key if med_context else "") or extract_requested_form(query)[0]
    volume = (med_context.volume_ml if med_context else None) or extract_volume_ml(query)
    pack_size = (med_context.pack_size if med_context else "") or extract_pack_size(query) or ""
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
    if pack_size:
        row_pack = extract_pack_size(f"{row.get('name_en', '')} {row.get('name_ar', '')}") or ""
        score += 0.20 if row_pack == pack_size else -0.25
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
    row_pack = extract_pack_size(text) or ""

    if context.volume_ml and row_volume and context.volume_ml != row_volume:
        return False

    if context.pack_size and row_pack and context.pack_size != row_pack:
        return False

    if context.strengths and row_strengths and not strengths_compatible(context.strengths, row_strengths):
        if context.volume_ml:
            return False
        if len(context.strengths) == 1:
            return False

    if context.form_key and not row_matches_requested_form(row, context.form_key):
        return False

    if context.drug_name or context.trade_name:
        drug_norm = strip_form_noise(normalize_text(context.drug_name or context.trade_name))
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
