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

from trade_name_utils import (
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

INGREDIENT_SYNONYMS: Dict[str, str] = {
    "acetaminophen": "paracetamol",
    "apap": "paracetamol",
    "panadol": "paracetamol",
    "ibuprofen": "ibuprofen",
    "brufen": "ibuprofen",
    "advil": "ibuprofen",
    "cetirizine": "cetirizine",
    "zyrtec": "cetirizine",
    "loratadine": "loratadine",
    "claritin": "loratadine",
    "omeprazole": "omeprazole",
    "losec": "omeprazole",
    "amoxicillin": "amoxicillin",
    "augmentin": "amoxicillin clavulanic",
    "pseudo ephedrine": "pseudoephedrine",
    "phenyl ephedrine": "phenylephrine",
    "vitamin c": "ascorbic acid",
    "ascorbic": "ascorbic acid",
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
    return INGREDIENT_SYNONYMS.get(q, q)


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


def extract_form_key(row: dict) -> str:
    form = " ".join(
        str(row.get(k, "") or "") for k in ("form", "form_clean", "name_en", "name_ar", "dosage_clean")
    ).lower()
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
    if any(k in form for k in ("injection", "ampoule", "حقن", "امبول")):
        return "injection"
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

    def rerank_candidates(
        self,
        candidates: List[ScoredCandidate],
        query: str,
        mode: str = "ingredient",
    ) -> List[ScoredCandidate]:
        norm_query = normalize_ingredient_query(query) if mode == "ingredient" else query
        reranked: List[ScoredCandidate] = []

        for cand in candidates:
            ai = cand.row.get(self.ingredient_col, "")
            overlap = self._ingredient_overlap_score(ai, norm_query) if mode == "ingredient" else 0.0
            exact_bonus = 0.12 if mode == "ingredient" and ingredient_matches_query(ai, norm_query) else 0.0

            if mode == "trade_name":
                cand.rerank = (
                    0.45 * cand.name_lexical
                    + 0.18 * cand.combined_lexical
                    + 0.12 * cand.semantic
                    + 0.08 * cand.lexical
                    + cand.exact_bonus
                    + cand.rrf_boost
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

    def passes_relevance_gate(
        self,
        cand: ScoredCandidate,
        mode: str = "ingredient",
        query: str = "",
        variants: Optional[List[str]] = None,
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
            if cand.name_lexical < (MIN_TRADE_NAME_SCORE / 100.0) and cand.exact_bonus == 0:
                return False
            if variants:
                brand_tokens = []
                for v in variants:
                    token = strip_form_noise(resolve_trade_alias(v)).split()[0]
                    if len(token) >= 4:
                        brand_tokens.append(token)
                if brand_tokens and not self._name_matches_query(cand.row, variants):
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
            if not self.passes_relevance_gate(cand, mode="ingredient", query=norm):
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
    ) -> List[dict]:
        if self.empty or not name or len(name.strip()) < 2:
            return []

        variants = generate_search_variants(name)
        if normalize_fn:
            variants = list(dict.fromkeys([normalize_fn(v) for v in variants] + variants))

        candidates = self.hybrid_candidates(name, mode="trade_name", variants=variants)
        candidates = self.rerank_candidates(candidates, name, mode="trade_name")

        results: List[dict] = []
        seen: Set[int] = set()

        for cand in candidates:
            if cand.row_index in seen:
                continue
            if not self.passes_relevance_gate(cand, mode="trade_name", query=name, variants=variants):
                continue
            if not relaxed_filter:
                reject = row_filter(cand.row, cand.row_index, name)
                if reject:
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
        src_strengths = extract_strengths(
            " ".join(str(source_row.get(k, "") or "") for k in (self.ingredient_col, "dosage", "dosage_clean", "name_en"))
        )
        src_name_en = trade_normalize(source_row.get("name_en", ""))
        src_brand_token = src_name_en.split()[0] if src_name_en else ""

        scored: List[Tuple[float, int, dict]] = []

        for idx in range(len(self.df)):
            if idx == source_index:
                continue
            row = self.df.iloc[idx].to_dict()
            cand_ing = row.get(self.ingredient_col, "")
            cand_profile = parse_ingredient_profile(cand_ing)

            if not ingredient_profiles_match(src_profile, cand_profile):
                continue
            if not therapeutic_compatible(src_profile, cand_profile):
                continue

            reject = row_filter(row, idx, src_ing)
            if reject:
                continue

            cand_form = extract_form_key(row)
            form_score = 1.0 if cand_form == src_form else (0.5 if cand_form != "other" and src_form != "other" else 0.2)

            cand_strengths = extract_strengths(
                " ".join(str(row.get(k, "") or "") for k in (self.ingredient_col, "dosage", "dosage_clean", "name_en"))
            )
            strength_score = 1.0 if src_strengths and src_strengths == cand_strengths else (
                0.7 if src_strengths & cand_strengths else 0.3
            )

            cand_name = trade_normalize(row.get("name_en", ""))
            brand_token = cand_name.split()[0] if cand_name else ""
            brand_bonus = 0.0 if brand_token == src_brand_token else 0.1

            price_bonus = self._price_bonus(row)
            total = 0.45 * form_score + 0.40 * strength_score + 0.10 * brand_bonus + price_bonus

            scored.append((total, idx, row))

        scored.sort(key=lambda x: x[0], reverse=True)

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
