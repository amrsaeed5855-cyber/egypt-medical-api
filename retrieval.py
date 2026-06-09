"""
retrieval.py — Hybrid drug retrieval, reranking, and relevance filtering.

Architecture
------------
1. Candidate generation (hybrid): FAISS semantic + RapidFuzz lexical on multiple fields
2. Score fusion: weighted combination with reciprocal-rank boost for dual hits
3. Reranking: ingredient overlap, form preference, price signal, safety-aware context
4. Relevance gate: minimum fused score before a row is returned
5. Medication matching: unified paths for INN ingredient and trade-name lookup
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import faiss
from rapidfuzz import fuzz, process

# ── Tuning constants ─────────────────────────────────────────────────────────
MIN_SEMANTIC_SCORE = float(__import__("os").getenv("MIN_SEMANTIC_SCORE", "0.32"))
MIN_LEXICAL_SCORE = float(__import__("os").getenv("MIN_LEXICAL_SCORE", "70"))
MIN_RERANK_SCORE = float(__import__("os").getenv("MIN_RERANK_SCORE", "0.40"))
MIN_TRADE_NAME_SCORE = float(__import__("os").getenv("MIN_TRADE_NAME_SCORE", "76"))
HYBRID_CANDIDATE_POOL = int(__import__("os").getenv("HYBRID_CANDIDATE_POOL", "80"))

# Canonical INN aliases → preferred search term
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


@dataclass
class ScoredCandidate:
    row_index: int
    row: dict
    semantic: float = 0.0
    lexical: float = 0.0
    name_lexical: float = 0.0
    combined_lexical: float = 0.0
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


def ingredient_matches_query(active_ingredient: str, query: str) -> bool:
    """True when every query token appears in the row's active ingredient."""
    ai = (active_ingredient or "").lower()
    tokens = ingredient_tokens(query)
    if not tokens:
        return False
    return all(t in ai for t in tokens)


def display_row_id(row_index: int, row: dict) -> int:
    """Stable 1-based row id for UI; prefer explicit dataset id columns."""
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
        if self._name_en_texts is None and "name_en" in self.df.columns:
            self._name_en_texts = self.df["name_en"].fillna("").astype(str).tolist()

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
        hits = process.extract(query, texts, scorer=fuzz.partial_ratio, limit=top_k)
        for _, score, idx in hits:
            if score < min_score:
                continue
            row = self.df.iloc[idx].to_dict()
            cand = out.get(idx)
            if cand is None:
                cand = ScoredCandidate(row_index=idx, row=row, match_sources=[source])
                out[idx] = cand
            else:
                cand.match_sources.append(source)
            if source == "ingredient_lexical":
                cand.lexical = max(cand.lexical, float(score) / 100.0)
            elif source == "combined_lexical":
                cand.combined_lexical = max(cand.combined_lexical, float(score) / 100.0)
            elif source in ("name_lexical", "name_ar", "name_en"):
                cand.name_lexical = max(cand.name_lexical, float(score) / 100.0)
        return out

    def hybrid_candidates(
        self,
        query: str,
        mode: str = "ingredient",
        top_k: int = HYBRID_CANDIDATE_POOL,
    ) -> List[ScoredCandidate]:
        if self.empty or not query or len(query.strip()) < 2:
            return []

        self._ensure_caches()
        norm_query = normalize_ingredient_query(query) if mode == "ingredient" else query.strip()
        merged: Dict[int, ScoredCandidate] = {}

        for idx, cand in self._vector_candidates(norm_query, top_k).items():
            merged[idx] = cand

        if self._ingredient_texts is not None and mode == "ingredient":
            for idx, cand in self._lexical_field_candidates(
                norm_query, self._ingredient_texts, "ingredient_lexical", top_k, MIN_LEXICAL_SCORE
            ).items():
                if idx in merged:
                    merged[idx].lexical = max(merged[idx].lexical, cand.lexical)
                    merged[idx].match_sources.extend(cand.match_sources)
                else:
                    merged[idx] = cand

        if self._combined_texts is not None:
            for idx, cand in self._lexical_field_candidates(
                norm_query, self._combined_texts, "combined_lexical", top_k, MIN_LEXICAL_SCORE - 5
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
                        norm_query, texts, source, top_k, MIN_TRADE_NAME_SCORE
                    ).items():
                        if idx in merged:
                            merged[idx].name_lexical = max(merged[idx].name_lexical, cand.name_lexical)
                            merged[idx].match_sources.extend(c for c in cand.match_sources if c not in merged[idx].match_sources)
                        else:
                            merged[idx] = cand

        # Reciprocal-rank style boost when multiple channels agree
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
                    0.55 * cand.name_lexical
                    + 0.20 * cand.combined_lexical
                    + 0.15 * cand.semantic
                    + 0.10 * cand.lexical
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

    def passes_relevance_gate(self, cand: ScoredCandidate, mode: str = "ingredient", query: str = "") -> bool:
        if cand.rerank < MIN_RERANK_SCORE:
            return False
        if mode == "ingredient":
            ai = cand.row.get(self.ingredient_col, "")
            if not ingredient_matches_query(ai, query) and cand.lexical < 0.82 and cand.semantic < 0.45:
                return False
        if mode == "trade_name" and cand.name_lexical < (MIN_TRADE_NAME_SCORE / 100.0):
            return False
        if mode == "trade_name":
            return cand.rerank >= MIN_RERANK_SCORE * 0.85
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
    ) -> List[dict]:
        if self.empty or not name or len(name.strip()) < 3:
            return []

        norm_name = normalize_fn(name) if normalize_fn else name.strip()
        candidates = self.hybrid_candidates(norm_name, mode="trade_name")
        candidates = self.rerank_candidates(candidates, norm_name, mode="trade_name")

        results: List[dict] = []
        seen: Set[int] = set()

        for cand in candidates:
            if cand.row_index in seen:
                continue
            if not self.passes_relevance_gate(cand, mode="trade_name", query=norm_name):
                continue
            reject = row_filter(cand.row, cand.row_index, norm_name)
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
