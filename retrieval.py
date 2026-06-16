# retrieval.py — Hybrid drug retrieval, reranking, and relevance filtering.
# Changed: stricter trade-name gates (tuned threshold), substitute alias exclusion,
# generic-ingredient row skip, and known-brand vs unknown-query matching rules.

"""
retrieval.py — Hybrid drug retrieval, reranking, and relevance filtering.

Architecture
------------
1. Candidate generation: exact → normalized → alias → fuzzy (multi-scorer)
2. Optional FAISS semantic boost
3. Reranking: ingredient overlap, form preference, substitute compatibility
4. Medication matching: INN ingredient, trade-name, and substitute lookup
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import faiss
from rapidfuzz import fuzz, process

from data_cleaning import (
    INGREDIENT_ALIAS_MAP,
    SUBSTITUTE_ELIGIBLE,
    TUNED_MIN_TRADE_RERANK,
    is_generic_ingredient,
    is_known_brand_query,
    normalize_ingredient_canonical,
)
from trade_name_utils import (
    collect_name_variants,
    generate_search_variants,
    normalize_text as trade_normalize,
    resolve_trade_alias,
    strip_form_noise,
)

# ── Tuning constants ─────────────────────────────────────────────────────────
MIN_SEMANTIC_SCORE = float(__import__("os").getenv("MIN_SEMANTIC_SCORE", "0.32"))
MIN_LEXICAL_SCORE = float(__import__("os").getenv("MIN_LEXICAL_SCORE", "70"))
MIN_RERANK_SCORE = float(__import__("os").getenv("MIN_RERANK_SCORE", "0.40"))
MIN_TRADE_NAME_SCORE = float(__import__("os").getenv("MIN_TRADE_NAME_SCORE", "72"))
HYBRID_CANDIDATE_POOL = int(__import__("os").getenv("HYBRID_CANDIDATE_POOL", "80"))

INGREDIENT_SYNONYMS: Dict[str, str] = INGREDIENT_ALIAS_MAP

CONFUSABLE_TRADE_PREFIXES: Dict[str, Tuple[str, ...]] = {
    "cetal": ("citalo", "citalopram", "cital"),
    "citalo": ("cetal",),
}

PREFERRED_FORM_KEYWORDS = (
    "tablet", "tab", "cap", "capsule", "syrup", "suspension", "sachet",
    "قرص", "كبسول", "شراب", "اكياس",
)

SKIP_INGREDIENT_TERMS = {
    "useful for cough", "useful for pain", "علاج", "دواء", "unknown", "",
}

# Ingredients that change therapeutic role — must match for substitutes
MUSCLE_RELAXANT_MARKERS = {
    "chlorzoxazone", "orphenadrine", "methocarbamol", "tolperisone", "thiocolchicoside",
}
DECONGESTANT_MARKERS = {"pseudoephedrine", "phenylephrine", "phenylepherine"}
CAFFEINE_MARKERS = {"caffeine", "coffeine"}

STRENGTH_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:mg|g|gm|ml|%|mcg|iu)", re.IGNORECASE)
CONCENTRATION_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*mg\s*/\s*(\d+(?:\.\d+)?)\s*ml",
    re.IGNORECASE,
)
VOLUME_RE = re.compile(r"(\d+(?:\.\d+)?)\s*ml\b", re.IGNORECASE)
PACK_SIZE_RE = re.compile(
    r"\b(\d+)\s*(?:"
    r"tab|tabs|cap|caps|softgel|softgels|vial|vials|amp|amps|ampoule|ampoules|"
    r"sachet|sachets|strip|strips|puff|puffs|dose|doses"
    r")\b|"
    r"\b(\d+)(?:tab|tabs|cap|caps|softgel|softgels|vial|vials|amp|amps|ampoule|ampoules|"
    r"sachet|sachets|strip|strips|puff|puffs|dose|doses)\b",
    re.IGNORECASE,
)
_UNIT_TO_MG = {"mg": 1.0, "g": 1000.0, "gm": 1000.0, "mcg": 0.001}


@dataclass
class ScoredCandidate:
    row_index: int
    row: dict
    semantic: float = 0.0
    lexical: float = 0.0
    name_lexical: float = 0.0
    combined_lexical: float = 0.0
    exact_bonus: float = 0.0
    rrf_boost: float = 0.0
    rerank: float = 0.0
    match_sources: List[str] = field(default_factory=list)


RowFilter = Callable[[dict, int, str], Optional[str]]


def normalize_ingredient_query(raw: str) -> str:
    q = (raw or "").strip().lower()
    q = re.sub(r"\s+", " ", q)
    mapped = INGREDIENT_SYNONYMS.get(q, q)
    return normalize_ingredient_canonical(mapped) if mapped else mapped


def ingredient_tokens(ingredient: str) -> Set[str]:
    norm = normalize_ingredient_query(ingredient)
    parts = re.split(r"[+/\s,\-]+", norm)
    return {p.strip() for p in parts if len(p.strip()) >= 3}


def parse_ingredient_profile(ingredient: str) -> Set[str]:
    """Normalized set of active-ingredient tokens from a row."""
    ai = (ingredient or "").lower()
    ai = ai.replace("acetaminophen", "paracetamol")
    tokens: Set[str] = set()
    for part in re.split(r"[+/\s,\-()]+", ai):
        t = part.strip()
        if len(t) >= 3 and t not in ("acid", "tablet", "capsule", "cream", "syrup"):
            tokens.add(t)
    return tokens


def extract_strengths(text: str) -> Set[str]:
    return {m.group(0).lower().replace(" ", "") for m in STRENGTH_RE.finditer(text or "")}


def extract_dose_strengths(text: str) -> Set[str]:
    """Strength tokens excluding standalone volume (e.g. 120ml) confused with dose."""
    strengths = extract_strengths(text)
    return {s for s in strengths if not re.fullmatch(r"\d+(?:\.\d+)?ml", s.replace(" ", ""))}


def extract_concentration(text: str) -> Optional[str]:
    match = CONCENTRATION_RE.search(text or "")
    if not match:
        return None
    mg = float(match.group(1))
    ml = float(match.group(2))
    return f"{mg:g}mg/{ml:g}ml".lower().replace(" ", "")


def extract_volume_ml(text: str) -> Optional[str]:
    match = VOLUME_RE.search(text or "")
    if not match:
        return None
    return f"{float(match.group(1)):g}ml".lower().replace(" ", "")


def extract_pack_size(text: str) -> Optional[str]:
    match = PACK_SIZE_RE.search(text or "")
    if not match:
        return None
    return match.group(1) or match.group(2)


def _strength_to_mg(token: str) -> Optional[float]:
    match = re.match(r"(\d+(?:\.\d+)?)(mg|g|gm|mcg|iu|%)", (token or "").lower().replace(" ", ""))
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2)
    factor = _UNIT_TO_MG.get(unit)
    if factor is None:
        return None
    return round(value * factor, 4)


def strength_mg_values(strengths: Set[str]) -> Set[float]:
    return {mg for s in strengths if (mg := _strength_to_mg(s)) is not None}


def strengths_compatible(source: Set[str], candidate: Set[str]) -> bool:
    if not source:
        return True
    if not candidate:
        return False
    if source & candidate:
        return True
    src_mg = strength_mg_values(source)
    cand_mg = strength_mg_values(candidate)
    if src_mg and cand_mg:
        return bool(src_mg & cand_mg)
    return False


def row_constraint_text(row: dict, ingredient_col: str) -> str:
    return " ".join(
        str(row.get(k, "") or "")
        for k in (ingredient_col, "active_ingredient", "dosage", "dosage_clean", "name_en", "name_ar")
    )


def row_numeric_constraints(row: dict, ingredient_col: str) -> Dict[str, Any]:
    text = row_constraint_text(row, ingredient_col)
    return {
        "strengths": extract_dose_strengths(text),
        "concentration": extract_concentration(text),
        "volume_ml": extract_volume_ml(text),
        "pack_size": extract_pack_size(text),
    }


def extract_form_key(row: dict) -> str:
    form = " ".join(
        str(row.get(k, "") or "") for k in ("form", "form_clean", "name_en", "name_ar", "dosage_clean")
    ).lower()
    if re.search(r"ampoule|ampule|\bamp\b|\bamps\b|injection|حقن|امبول|vial|\bi\.?v\b", form):
        return "injection"
    if any(k in form for k in ("cream", "gel", "كريم", "مرهم")):
        return "topical"
    if any(k in form for k in ("syrup", "suspension", "شراب", "معلق")):
        return "liquid"
    if any(k in form for k in ("sachet", "اكياس", "sachets")):
        return "sachet"
    if any(k in form for k in ("tablet", "tab", "cap", "قرص", "كبسول")):
        return "oral_solid"
    if any(k in form for k in ("drop", "قطرة", "spray", "بخاخ")):
        return "drops_spray"
    return "other"


def therapeutic_compatible(source_profile: Set[str], candidate_profile: Set[str]) -> bool:
    """Reject substitutes that change therapeutic role (e.g. muscle relaxant combo)."""
    src_relax = bool(source_profile & MUSCLE_RELAXANT_MARKERS)
    cand_relax = bool(candidate_profile & MUSCLE_RELAXANT_MARKERS)
    if src_relax != cand_relax:
        return False

    src_decong = bool(source_profile & DECONGESTANT_MARKERS)
    cand_decong = bool(candidate_profile & DECONGESTANT_MARKERS)
    if src_decong != cand_decong:
        return False

    src_caf = bool(source_profile & CAFFEINE_MARKERS)
    cand_caf = bool(candidate_profile & CAFFEINE_MARKERS)
    if src_caf != cand_caf:
        return False

    return True


def ingredient_profiles_match(source: Set[str], candidate: Set[str]) -> bool:
    """Substitutes must share the same active-ingredient profile."""
    if not source or not candidate:
        return False
    src_norm = {normalize_ingredient_query(t) for t in source}
    cand_norm = {normalize_ingredient_query(t) for t in candidate}
    return src_norm == cand_norm


def ingredient_matches_query(active_ingredient: str, query: str) -> bool:
    ai = (active_ingredient or "").lower()
    tokens = ingredient_tokens(query)
    if not tokens:
        return False
    return all(t in ai for t in tokens)


def display_row_id(row_index: int, row: dict) -> int:
    for key in ("row_id", "id", "drug_id", "source_row"):
        val = row.get(key)
        if val not in (None, "", "nan"):
            try:
                return int(float(val))
            except (ValueError, TypeError):
                pass
    return row_index + 1


def attach_row_metadata(row: dict, row_index: int) -> dict:
    out = dict(row)
    out["row_index"] = row_index
    out["row_id"] = display_row_id(row_index, row)
    return out


def _multi_scorer(query: str, choice: str, score_cutoff: float = 0, **_) -> float:
    """Best fuzzy score across multiple algorithms (rapidfuzz-compatible signature)."""
    if not query or not choice:
        return 0.0
    q, c = query.lower(), choice.lower()
    if q == c:
        score = 100.0
    elif q in c or c in q:
        score = max(92.0, fuzz.partial_ratio(q, c))
    else:
        score = max(
            fuzz.ratio(q, c),
            fuzz.partial_ratio(q, c),
            fuzz.token_sort_ratio(q, c),
            fuzz.token_set_ratio(q, c),
            fuzz.WRatio(q, c),
        )
    return score if score >= score_cutoff else 0.0


class DrugRetrievalEngine:
    """Hybrid retrieval with reranking for the Egyptian drugs dataset."""

    def __init__(
        self,
        df,
        index: Optional[faiss.Index],
        ingredient_col: str,
        get_embed_model: Callable[[], Any],
        enable_semantic: bool,
    ):
        self.df = df
        self.index = index
        self.ingredient_col = ingredient_col
        self.get_embed_model = get_embed_model
        self.enable_semantic = enable_semantic
        self._ingredient_texts: Optional[List[str]] = None
        self._combined_texts: Optional[List[str]] = None
        self._name_ar_texts: Optional[List[str]] = None
        self._name_en_texts: Optional[List[str]] = None
        self._name_ar_norm: Optional[List[str]] = None
        self._name_en_norm: Optional[List[str]] = None

    @property
    def empty(self) -> bool:
        return self.df is None or self.df.empty

    def _ensure_caches(self) -> None:
        if self.empty:
            return
        if self._ingredient_texts is None:
            self._ingredient_texts = self.df[self.ingredient_col].fillna("").astype(str).tolist()
        if self._combined_texts is None and "combined" in self.df.columns:
            self._combined_texts = self.df["combined"].fillna("").astype(str).tolist()
        if self._name_ar_texts is None and "name_ar" in self.df.columns:
            self._name_ar_texts = self.df["name_ar"].fillna("").astype(str).tolist()
            self._name_ar_norm = [trade_normalize(t) for t in self._name_ar_texts]
        if self._name_en_texts is None and "name_en" in self.df.columns:
            self._name_en_texts = self.df["name_en"].fillna("").astype(str).tolist()
            self._name_en_norm = [trade_normalize(t) for t in self._name_en_texts]

    def _vector_candidates(self, query: str, top_k: int) -> Dict[int, ScoredCandidate]:
        out: Dict[int, ScoredCandidate] = {}
        if not self.enable_semantic or self.index is None:
            return out
        model = self.get_embed_model()
        if model is None:
            return out
        try:
            q = model.encode([query]).astype("float32")
            faiss.normalize_L2(q)
            scores, ids = self.index.search(q, min(top_k, self.index.ntotal))
            for score, idx in zip(scores[0], ids[0]):
                if idx < 0 or float(score) < MIN_SEMANTIC_SCORE:
                    continue
                row = self.df.iloc[int(idx)].to_dict()
                out[int(idx)] = ScoredCandidate(
                    row_index=int(idx),
                    row=row,
                    semantic=float(score),
                    match_sources=["semantic"],
                )
        except Exception:
            pass
        return out

    def _exact_name_candidates(self, variants: List[str]) -> Dict[int, ScoredCandidate]:
        """Exact and normalized whole-name matches (highest priority)."""
        out: Dict[int, ScoredCandidate] = {}
        if self.empty:
            return out
        self._ensure_caches()
        norm_variants = {trade_normalize(v) for v in variants}
        norm_variants |= {strip_form_noise(v) for v in norm_variants}
        norm_variants |= {resolve_trade_alias(v) for v in variants}

        for idx in range(len(self.df)):
            ar = (self._name_ar_norm or [""])[idx] if self._name_ar_norm else ""
            en = (self._name_en_norm or [""])[idx] if self._name_en_norm else ""
            ar_stripped = strip_form_noise(ar)
            en_stripped = strip_form_noise(en)

            matched = False
            for nv in norm_variants:
                if not nv or len(nv) < 2:
                    continue
                if nv in (ar, en, ar_stripped, en_stripped):
                    matched = True
                    break
                if ar.startswith(nv) or en.startswith(nv):
                    if len(nv) >= 4:
                        matched = True
                        break

            if matched:
                row = self.df.iloc[idx].to_dict()
                out[idx] = ScoredCandidate(
                    row_index=idx,
                    row=row,
                    name_lexical=1.0,
                    exact_bonus=0.25,
                    match_sources=["exact_name"],
                )
        return out

    def _lexical_field_candidates(
        self,
        query: str,
        texts: List[str],
        source: str,
        top_k: int,
        min_score: float,
    ) -> Dict[int, ScoredCandidate]:
        out: Dict[int, ScoredCandidate] = {}
        if not texts:
            return out
        hits = process.extract(query, texts, scorer=_multi_scorer, limit=top_k)
        for _, score, idx in hits:
            if score < min_score:
                continue
            row = self.df.iloc[idx].to_dict()
            cand = out.get(idx)
            if cand is None:
                cand = ScoredCandidate(row_index=idx, row=row, match_sources=[source])
                out[idx] = cand
            else:
                if source not in cand.match_sources:
                    cand.match_sources.append(source)
            score_norm = float(score) / 100.0
            if source == "ingredient_lexical":
                cand.lexical = max(cand.lexical, score_norm)
            elif source == "combined_lexical":
                cand.combined_lexical = max(cand.combined_lexical, score_norm)
            elif source in ("name_lexical", "name_ar", "name_en"):
                cand.name_lexical = max(cand.name_lexical, score_norm)
        return out

    def hybrid_candidates(
        self,
        query: str,
        mode: str = "ingredient",
        top_k: int = HYBRID_CANDIDATE_POOL,
        variants: Optional[List[str]] = None,
    ) -> List[ScoredCandidate]:
        if self.empty or not query or len(query.strip()) < 2:
            return []

        self._ensure_caches()
        search_variants = variants or ([query] if mode != "trade_name" else generate_search_variants(query))
        merged: Dict[int, ScoredCandidate] = {}

        if mode == "trade_name":
            for idx, cand in self._exact_name_candidates(search_variants).items():
                merged[idx] = cand

        for variant in search_variants[:6]:
            norm_query = normalize_ingredient_query(variant) if mode == "ingredient" else variant.strip()
            if not norm_query:
                continue

            for idx, cand in self._vector_candidates(norm_query, top_k).items():
                if idx in merged:
                    merged[idx].semantic = max(merged[idx].semantic, cand.semantic)
                    merged[idx].match_sources.extend(c for c in cand.match_sources if c not in merged[idx].match_sources)
                else:
                    merged[idx] = cand

            if self._ingredient_texts is not None and mode == "ingredient":
                for idx, cand in self._lexical_field_candidates(
                    norm_query, self._ingredient_texts, "ingredient_lexical", top_k, MIN_LEXICAL_SCORE
                ).items():
                    if idx in merged:
                        merged[idx].lexical = max(merged[idx].lexical, cand.lexical)
                        merged[idx].match_sources.extend(c for c in cand.match_sources if c not in merged[idx].match_sources)
                    else:
                        merged[idx] = cand

            if self._combined_texts is not None:
                min_combined = MIN_LEXICAL_SCORE - 5 if mode == "ingredient" else MIN_TRADE_NAME_SCORE - 8
                for idx, cand in self._lexical_field_candidates(
                    norm_query, self._combined_texts, "combined_lexical", top_k, min_combined
                ).items():
                    if idx in merged:
                        merged[idx].combined_lexical = max(merged[idx].combined_lexical, cand.combined_lexical)
                        merged[idx].match_sources.extend(c for c in cand.match_sources if c not in merged[idx].match_sources)
                    else:
                        merged[idx] = cand

            if mode == "trade_name":
                for texts, source in (
                    (self._name_ar_texts, "name_ar"),
                    (self._name_en_texts, "name_en"),
                ):
                    if texts:
                        for idx, cand in self._lexical_field_candidates(
                            norm_query, texts, source, top_k, MIN_TRADE_NAME_SCORE - 6
                        ).items():
                            if idx in merged:
                                merged[idx].name_lexical = max(merged[idx].name_lexical, cand.name_lexical)
                                merged[idx].match_sources.extend(c for c in cand.match_sources if c not in merged[idx].match_sources)
                            else:
                                merged[idx] = cand

        for cand in merged.values():
            unique_sources = len(set(cand.match_sources))
            if unique_sources >= 2:
                cand.rrf_boost = 0.08 * (unique_sources - 1)

        return list(merged.values())

    @staticmethod
    def _form_bonus(row: dict) -> float:
        form = " ".join(
            str(row.get(k, "") or "") for k in ("form", "form_clean", "name_en", "name_ar")
        ).lower()
        if any(k in form for k in PREFERRED_FORM_KEYWORDS):
            return 0.06
        return 0.0

    @staticmethod
    def _price_bonus(row: dict) -> float:
        price = row.get("price_egp", "")
        if price and str(price).strip() not in ("", "nan", "0"):
            return 0.04
        return 0.0

    def _ingredient_overlap_score(self, active_ingredient: str, query: str) -> float:
        tokens = ingredient_tokens(query)
        if not tokens:
            return 0.0
        ai = (active_ingredient or "").lower()
        hits = sum(1 for t in tokens if t in ai)
        return hits / len(tokens)

    @staticmethod
    def _row_strengths(row: dict, ingredient_col: str) -> Set[str]:
        text = " ".join(
            str(row.get(k, "") or "")
            for k in (ingredient_col, "active_ingredient", "dosage_clean", "dosage", "name_en", "name_ar")
        )
        return extract_strengths(text)

    @staticmethod
    def _strength_match_score(query_strengths: Set[str], row_strengths: Set[str]) -> float:
        if not query_strengths:
            return 0.0
        if strengths_compatible(query_strengths, row_strengths):
            if query_strengths & row_strengths:
                return 0.55
            return 0.45
        query_nums = strength_mg_values(query_strengths)
        row_nums = strength_mg_values(row_strengths)
        if query_nums & row_nums:
            return 0.25
        return -0.35

    @staticmethod
    def _form_match_score(requested_form: str, row: dict) -> float:
        if not requested_form:
            return 0.0
        row_form = extract_form_key(row)
        if row_form == requested_form:
            return 0.18
        return -0.50

    def rerank_candidates(
        self,
        candidates: List[ScoredCandidate],
        query: str,
        mode: str = "ingredient",
        requested_form: str = "",
        query_strengths: Optional[Set[str]] = None,
    ) -> List[ScoredCandidate]:
        norm_query = normalize_ingredient_query(query) if mode == "ingredient" else query
        strengths = query_strengths if query_strengths is not None else extract_strengths(query)
        reranked: List[ScoredCandidate] = []

        for cand in candidates:
            ai = cand.row.get(self.ingredient_col, "")
            overlap = self._ingredient_overlap_score(ai, norm_query) if mode == "ingredient" else 0.0
            exact_bonus = 0.12 if mode == "ingredient" and ingredient_matches_query(ai, norm_query) else 0.0
            row_strengths = self._row_strengths(cand.row, self.ingredient_col)
            strength_score = self._strength_match_score(strengths, row_strengths)
            form_score = self._form_match_score(requested_form, cand.row)

            if mode == "trade_name":
                cand.rerank = (
                    0.30 * cand.name_lexical
                    + 0.12 * cand.combined_lexical
                    + 0.08 * cand.semantic
                    + 0.06 * cand.lexical
                    + cand.exact_bonus
                    + cand.rrf_boost
                    + strength_score
                    + form_score
                    + self._form_bonus(cand.row)
                    + self._price_bonus(cand.row)
                )
            else:
                cand.rerank = (
                    0.38 * cand.semantic
                    + 0.28 * cand.lexical
                    + 0.12 * cand.combined_lexical
                    + 0.14 * cand.name_lexical
                    + 0.10 * overlap
                    + exact_bonus
                    + strength_score
                    + form_score
                    + cand.rrf_boost
                    + self._form_bonus(cand.row)
                    + self._price_bonus(cand.row)
                )
            reranked.append(cand)

        reranked.sort(key=lambda c: c.rerank, reverse=True)
        return reranked

    def _name_matches_query(self, row: dict, variants: List[str]) -> bool:
        combined = trade_normalize(
            f"{row.get('name_ar', '')} {row.get('name_en', '')}"
        )
        for variant in variants:
            nv = strip_form_noise(trade_normalize(variant))
            if len(nv) < 3:
                continue
            if nv in combined:
                return True
            brand = nv.split()[0]
            if len(brand) >= 4 and brand in combined:
                return True
        return False

    def _is_confusable_trade_match(self, row: dict, query: str) -> bool:
        canonical = resolve_trade_alias(query)
        blocked = CONFUSABLE_TRADE_PREFIXES.get(canonical, ())
        if not blocked:
            return False
        first = trade_normalize(row.get("name_en", "")).split()[0]
        return first in blocked

    def passes_relevance_gate(
        self,
        cand: ScoredCandidate,
        mode: str = "ingredient",
        query: str = "",
        variants: Optional[List[str]] = None,
        strict_unknown: bool = False,
    ) -> bool:
        if "exact_name" in cand.match_sources:
            return True
        if cand.rerank < MIN_RERANK_SCORE * 0.75:
            return False
        if mode == "ingredient":
            ai = cand.row.get(self.ingredient_col, "")
            if not ingredient_matches_query(ai, query) and cand.lexical < 0.82 and cand.semantic < 0.45:
                return False
        if mode == "trade_name":
            min_rerank = TUNED_MIN_TRADE_RERANK if TUNED_MIN_TRADE_RERANK else MIN_RERANK_SCORE
            if strict_unknown or not is_known_brand_query(query):
                # Unknown drug names must not pass on partial fuzzy alone
                if cand.rerank < min_rerank:
                    return False
                if cand.name_lexical < 0.88 and cand.exact_bonus == 0:
                    return False
            else:
                if cand.name_lexical < (MIN_TRADE_NAME_SCORE / 100.0) and cand.exact_bonus == 0:
                    return False
                if cand.rerank < min_rerank * 0.72:
                    return False
            if variants:
                if not self._name_matches_query(cand.row, variants):
                    return False
        return True

    def match_by_ingredient(
        self,
        ingredient: str,
        excluded: Set[str],
        row_filter: RowFilter,
        max_results: int = 2,
        caution_fn: Optional[Callable[[str, Any], List[str]]] = None,
        ctx: Any = None,
    ) -> List[dict]:
        norm = normalize_ingredient_query(ingredient)
        if norm in SKIP_INGREDIENT_TERMS or len(norm) < 3:
            return []

        candidates = self.hybrid_candidates(norm, mode="ingredient")
        candidates = self.rerank_candidates(candidates, norm, mode="ingredient")

        results: List[dict] = []
        seen_bases: Set[str] = set()

        for cand in candidates:
            if not self.passes_relevance_gate(cand, mode="ingredient", query=norm, strict_unknown=True):
                continue
            ai = cand.row.get(self.ingredient_col, "").strip().lower()
            reject = row_filter(cand.row, cand.row_index, norm)
            if reject:
                continue
            if any(excl in ai for excl in excluded):
                continue

            base = re.split(r"[+/\s\-]", ai)[0].strip()
            if base in seen_bases and len(results) >= max_results:
                continue

            row_dict = attach_row_metadata(cand.row, cand.row_index)
            row_dict["retrieval_score"] = round(cand.rerank, 3)
            row_dict["match_sources"] = list(set(cand.match_sources))
            if caution_fn and ctx is not None:
                row_dict["safety_cautions"] = caution_fn(ai, ctx)
            results.append(row_dict)
            seen_bases.add(base)
            if len(results) >= max_results:
                break

        return results

    def match_by_trade_name(
        self,
        name: str,
        row_filter: RowFilter,
        max_results: int = 5,
        caution_fn: Optional[Callable[[str, Any], List[str]]] = None,
        ctx: Any = None,
        normalize_fn: Optional[Callable[[str], str]] = None,
        relaxed_filter: bool = False,
        requested_form: str = "",
        query_strengths: Optional[Set[str]] = None,
        require_form: bool = False,
    ) -> List[dict]:
        if self.empty or not name or len(name.strip()) < 2:
            return []

        variants = generate_search_variants(name)
        if normalize_fn:
            variants = list(dict.fromkeys([normalize_fn(v) for v in variants] + variants))

        strengths = query_strengths if query_strengths is not None else extract_strengths(name)
        candidates = self.hybrid_candidates(name, mode="trade_name", variants=variants)
        if strengths or requested_form:
            filtered: List[ScoredCandidate] = []
            for cand in candidates:
                if requested_form and extract_form_key(cand.row) != requested_form:
                    continue
                row_strengths = self._row_strengths(cand.row, self.ingredient_col)
                if strengths and not strengths_compatible(strengths, row_strengths):
                    continue
                filtered.append(cand)
            if filtered:
                candidates = filtered
        candidates = self.rerank_candidates(
            candidates,
            name,
            mode="trade_name",
            requested_form=requested_form,
            query_strengths=strengths,
        )

        results: List[dict] = []
        seen: Set[int] = set()

        strict = not is_known_brand_query(name)
        for cand in candidates:
            if cand.row_index in seen:
                continue
            if require_form and requested_form and extract_form_key(cand.row) != requested_form:
                continue
            if not self.passes_relevance_gate(
                cand, mode="trade_name", query=name, variants=variants, strict_unknown=strict
            ):
                continue
            if strengths and self._strength_match_score(strengths, self._row_strengths(cand.row, self.ingredient_col)) < 0:
                continue
            if not relaxed_filter:
                reject = row_filter(cand.row, cand.row_index, name)
                if reject:
                    continue
            if self._is_confusable_trade_match(cand.row, name):
                continue

            ai = cand.row.get(self.ingredient_col, "").strip().lower()
            row_dict = attach_row_metadata(cand.row, cand.row_index)
            row_dict["retrieval_score"] = round(cand.rerank, 3)
            row_dict["match_sources"] = list(set(cand.match_sources))
            if caution_fn and ctx is not None:
                row_dict["safety_cautions"] = caution_fn(ai, ctx)
            results.append(row_dict)
            seen.add(cand.row_index)
            if len(results) >= max_results:
                break

        if require_form and requested_form and not results:
            return []

        return results

    def find_substitutes(
        self,
        source_row: dict,
        source_index: int,
        row_filter: RowFilter,
        max_results: int = 5,
        caution_fn: Optional[Callable[[str, Any], List[str]]] = None,
        ctx: Any = None,
    ) -> List[dict]:
        """
        Medically sound substitutes: same ingredient profile, form, strength, therapeutic role.
        Excludes the source product itself.
        """
        if self.empty or not source_row:
            return []

        src_ing = source_row.get(self.ingredient_col, "")
        src_profile = parse_ingredient_profile(src_ing)
        if not src_profile:
            return []

        src_form = extract_form_key(source_row)
        src_constraints = row_numeric_constraints(source_row, self.ingredient_col)
        src_strengths = src_constraints["strengths"]
        src_concentration = src_constraints["concentration"]
        src_volume = src_constraints["volume_ml"]
        src_pack = src_constraints["pack_size"]
        if is_generic_ingredient(src_ing):
            return []

        src_name_variants = collect_name_variants(source_row)
        src_name_en = trade_normalize(source_row.get("name_en", ""))
        src_brand_token = src_name_en.split()[0] if src_name_en else ""

        scored: List[Tuple[float, int, dict]] = []

        for idx in range(len(self.df)):
            if idx == source_index:
                continue
            if idx < len(SUBSTITUTE_ELIGIBLE) and not SUBSTITUTE_ELIGIBLE[idx]:
                continue
            row = self.df.iloc[idx].to_dict()
            cand_ing = row.get(self.ingredient_col, "")
            if is_generic_ingredient(cand_ing):
                continue
            cand_profile = parse_ingredient_profile(cand_ing)

            if not ingredient_profiles_match(src_profile, cand_profile):
                continue
            if not therapeutic_compatible(src_profile, cand_profile):
                continue

            cand_variants = collect_name_variants(row)
            if src_name_variants & cand_variants:
                continue

            reject = row_filter(row, idx, src_ing)
            if reject:
                continue

            cand_form = extract_form_key(row)
            if src_form != "other" and cand_form != src_form:
                continue

            cand_constraints = row_numeric_constraints(row, self.ingredient_col)
            cand_strengths = cand_constraints["strengths"]
            if src_strengths and not strengths_compatible(src_strengths, cand_strengths):
                continue

            if src_concentration and cand_constraints["concentration"] and src_concentration != cand_constraints["concentration"]:
                continue

            if src_volume and cand_constraints["volume_ml"] and src_volume != cand_constraints["volume_ml"]:
                continue

            if src_pack and cand_constraints["pack_size"] and src_pack != cand_constraints["pack_size"]:
                continue

            form_score = 1.0
            strength_score = 1.0 if src_strengths and src_strengths == cand_strengths else (
                0.85 if strengths_compatible(src_strengths, cand_strengths) else 0.0
            )
            if strength_score <= 0:
                continue

            cand_name = trade_normalize(row.get("name_en", ""))
            brand_token = cand_name.split()[0] if cand_name else ""
            brand_bonus = 0.0 if brand_token == src_brand_token else 0.05

            volume_bonus = 0.0
            if src_volume and cand_constraints["volume_ml"] == src_volume:
                volume_bonus = 0.04

            pack_bonus = 0.0
            if src_pack and cand_constraints["pack_size"] == src_pack:
                pack_bonus = 0.03

            price_bonus = self._price_bonus(row)
            total = (
                0.45 * strength_score
                + 0.30 * form_score
                + 0.10 * brand_bonus
                + volume_bonus
                + pack_bonus
                + price_bonus
            )

            scored.append((total, idx, row))

        scored.sort(key=lambda x: x[0], reverse=True)

        if src_pack or src_volume:
            preferred: List[Tuple[float, int, dict]] = []
            fallback: List[Tuple[float, int, dict]] = []
            for item in scored:
                cand = row_numeric_constraints(item[2], self.ingredient_col)
                pack_ok = not src_pack or cand["pack_size"] == src_pack
                vol_ok = not src_volume or cand["volume_ml"] == src_volume
                if pack_ok and vol_ok:
                    preferred.append(item)
                elif (src_pack and not cand["pack_size"]) or (src_volume and not cand["volume_ml"]):
                    fallback.append(item)
            scored = preferred + fallback if preferred else scored

        results: List[dict] = []
        for total, idx, row in scored[:max_results]:
            ai = row.get(self.ingredient_col, "").strip().lower()
            row_dict = attach_row_metadata(row, idx)
            row_dict["retrieval_score"] = round(total, 3)
            row_dict["match_sources"] = ["substitute"]
            if caution_fn and ctx is not None:
                row_dict["safety_cautions"] = caution_fn(ai, ctx)
            results.append(row_dict)

        return results
