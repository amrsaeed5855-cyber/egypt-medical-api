# substitute_eval.py — Automated substitute-quality evaluation against the full dataset.
# Generates test cases from row metadata, validates retrieval output, and reports failures.

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd

from data_cleaning import SUBSTITUTE_ELIGIBLE, is_generic_ingredient
from medication_context import extract_volume_ml
from retrieval import (
    MIN_RERANK_SCORE,
    RowFilter,
    attach_row_metadata,
    display_row_id,
    extract_form_key,
    extract_pack_size,
    extract_strengths,
    ingredient_profiles_match,
    parse_ingredient_profile,
    therapeutic_compatible,
)
from trade_name_utils import collect_name_variants

DEFAULT_K = int(os.getenv("SUBSTITUTE_EVAL_K", "3"))
MIN_SUBSTITUTE_CONFIDENCE = float(os.getenv("MIN_SUBSTITUTE_CONFIDENCE", str(MIN_RERANK_SCORE * 0.75)))
DEFAULT_REPORT_DIR = os.getenv("SUBSTITUTE_EVAL_REPORT_DIR", "eval_reports")
REGRESSION_CASES_PATH = os.getenv("SUBSTITUTE_REGRESSION_CASES", "eval_regression_cases.json")

STRENGTH_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:mg|g|gm|ml|%|mcg|iu)", re.IGNORECASE)

FAILURE_WRONG_INGREDIENT = "wrong_active_ingredient"
FAILURE_WRONG_FORM = "wrong_dosage_form"
FAILURE_WRONG_STRENGTH = "wrong_strength"
FAILURE_IGNORED_NUMERIC = "ignored_numeric_constraints"
FAILURE_LOW_CONFIDENCE = "low_confidence_retrieval"
FAILURE_EMPTY_COVERAGE = "empty_result_despite_valid_alternatives"

FAILURE_TYPES = (
    FAILURE_WRONG_INGREDIENT,
    FAILURE_WRONG_FORM,
    FAILURE_WRONG_STRENGTH,
    FAILURE_IGNORED_NUMERIC,
    FAILURE_LOW_CONFIDENCE,
    FAILURE_EMPTY_COVERAGE,
)


def row_text(row: dict, ingredient_col: str) -> str:
    return " ".join(
        str(row.get(k, "") or "")
        for k in (ingredient_col, "dosage", "dosage_clean", "name_en", "name_ar", "form", "form_clean")
    )


def row_strengths(row: dict, ingredient_col: str) -> Set[str]:
    return extract_strengths(row_text(row, ingredient_col))


def row_volume_ml(row: dict) -> Optional[str]:
    text = f"{row.get('name_en', '')} {row.get('name_ar', '')}"
    return extract_volume_ml(text)


def extract_pack_size_from_row(row: dict) -> Optional[str]:
    text = f"{row.get('name_en', '')} {row.get('name_ar', '')}"
    return extract_pack_size(text)


def source_has_test_metadata(row: dict, ingredient_col: str) -> bool:
    ing = str(row.get(ingredient_col, "") or "").strip()
    if not ing or is_generic_ingredient(ing):
        return False
    if not parse_ingredient_profile(ing):
        return False
    form = extract_form_key(row)
    if form == "other" and not str(row.get("form_clean") or row.get("form") or "").strip():
        return False
    return bool(row_strengths(row, ingredient_col))


def build_substitute_query(row: dict) -> str:
    name = str(row.get("name_en") or row.get("name_ar") or "").strip()
    if not name:
        return "substitute"
    return f"substitute for {name}"


def ingredient_profile_tokens(ingredient: str) -> List[str]:
    tokens = parse_ingredient_profile(ingredient)
    return sorted(
        token
        for token in tokens
        if not STRENGTH_RE.fullmatch(token)
        and not token.replace(".", "").isdigit()
    )


def source_fingerprint(row: dict, ingredient_col: str) -> Dict[str, Any]:
    ing = str(row.get(ingredient_col, "") or "")
    return {
        "row_id": display_row_id(int(row.get("row_index", 0)), row),
        "ingredient_tokens": ingredient_profile_tokens(ing),
        "form_key": extract_form_key(row),
        "strengths": sorted(row_strengths(row, ingredient_col)),
        "volume_ml": row_volume_ml(row),
        "pack_size": extract_pack_size_from_row(row),
    }


@dataclass
class SubstituteTestCase:
    source_index: int
    query: str
    fingerprint: Dict[str, Any]
    oracle_count: int = 0


@dataclass
class SubstituteFailure:
    source_index: int
    source_fingerprint: Dict[str, Any]
    query: str
    failure_types: List[str]
    substitute_row_id: Optional[int] = None
    substitute_fingerprint: Optional[Dict[str, Any]] = None
    retrieval_score: Optional[float] = None
    rank: Optional[int] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalMetrics:
    total_cases: int = 0
    cases_with_oracle: int = 0
    retrieved_rows: int = 0
    valid_retrieved_rows: int = 0
    precision_at_k: float = 0.0
    active_ingredient_preservation_rate: float = 0.0
    dosage_form_preservation_rate: float = 0.0
    constraint_preservation_rate: float = 0.0
    substitute_coverage: float = 0.0
    failure_distribution: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvalReport:
    generated_at: str
    dataset_rows: int
    test_cases: int
    k: int
    min_confidence: float
    metrics: EvalMetrics
    failures: List[SubstituteFailure]
    improvement_hints: Dict[str, List[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "dataset_rows": self.dataset_rows,
            "test_cases": self.test_cases,
            "k": self.k,
            "min_confidence": self.min_confidence,
            "metrics": self.metrics.to_dict(),
            "failures": [asdict(f) for f in self.failures],
            "improvement_hints": self.improvement_hints,
        }


def validate_substitute_candidate(
    source_row: dict,
    candidate_row: dict,
    ingredient_col: str,
    *,
    min_confidence: float = MIN_SUBSTITUTE_CONFIDENCE,
    retrieval_score: Optional[float] = None,
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """Return (is_valid, failure_types, details) for one retrieved substitute."""
    failures: List[str] = []
    details: Dict[str, Any] = {}

    src_profile = parse_ingredient_profile(source_row.get(ingredient_col, ""))
    cand_profile = parse_ingredient_profile(candidate_row.get(ingredient_col, ""))
    if not ingredient_profiles_match(src_profile, cand_profile):
        failures.append(FAILURE_WRONG_INGREDIENT)
        details["source_profile"] = sorted(src_profile)
        details["candidate_profile"] = sorted(cand_profile)

    if not therapeutic_compatible(src_profile, cand_profile):
        failures.append(FAILURE_WRONG_INGREDIENT)
        details["therapeutic_mismatch"] = True

    src_form = extract_form_key(source_row)
    cand_form = extract_form_key(candidate_row)
    if src_form != "other" and cand_form != src_form:
        failures.append(FAILURE_WRONG_FORM)
        details["source_form"] = src_form
        details["candidate_form"] = cand_form

    src_strengths = row_strengths(source_row, ingredient_col)
    cand_strengths = row_strengths(candidate_row, ingredient_col)
    if src_strengths:
        if not cand_strengths or not (src_strengths & cand_strengths):
            failures.append(FAILURE_WRONG_STRENGTH)
            details["source_strengths"] = sorted(src_strengths)
            details["candidate_strengths"] = sorted(cand_strengths)

    numeric_failures: List[str] = []
    src_volume = row_volume_ml(source_row)
    cand_volume = row_volume_ml(candidate_row)
    if src_volume and cand_volume and src_volume != cand_volume:
        numeric_failures.append("volume_ml")
        details["source_volume_ml"] = src_volume
        details["candidate_volume_ml"] = cand_volume

    src_pack = extract_pack_size_from_row(source_row)
    cand_pack = extract_pack_size_from_row(candidate_row)
    if src_pack and cand_pack and src_pack != cand_pack:
        numeric_failures.append("pack_size")
        details["source_pack_size"] = src_pack
        details["candidate_pack_size"] = cand_pack

    if numeric_failures:
        failures.append(FAILURE_IGNORED_NUMERIC)
        details["ignored_numeric_fields"] = numeric_failures

    score = retrieval_score
    if score is None:
        score = float(candidate_row.get("retrieval_score") or 0.0)
    if score < min_confidence:
        failures.append(FAILURE_LOW_CONFIDENCE)
        details["retrieval_score"] = score
        details["min_confidence"] = min_confidence

    return (not failures, failures, details)


def oracle_substitute_indices(
    df: pd.DataFrame,
    source_index: int,
    ingredient_col: str,
    substitute_eligible: Sequence[bool],
    row_filter: RowFilter,
) -> Set[int]:
    """Ground-truth substitute set using the same medical rules as retrieval."""
    if source_index < 0 or source_index >= len(df):
        return set()

    source_row = attach_row_metadata(df.iloc[source_index].to_dict(), source_index)
    src_ing = source_row.get(ingredient_col, "")
    src_profile = parse_ingredient_profile(src_ing)
    if not src_profile or is_generic_ingredient(src_ing):
        return set()

    src_form = extract_form_key(source_row)
    src_strengths = row_strengths(source_row, ingredient_col)
    src_name_variants = collect_name_variants(source_row)
    valid: Set[int] = set()

    for idx in range(len(df)):
        if idx == source_index:
            continue
        if idx < len(substitute_eligible) and not substitute_eligible[idx]:
            continue

        row = df.iloc[idx].to_dict()
        cand_ing = row.get(ingredient_col, "")
        if is_generic_ingredient(cand_ing):
            continue

        cand_profile = parse_ingredient_profile(cand_ing)
        if not ingredient_profiles_match(src_profile, cand_profile):
            continue
        if not therapeutic_compatible(src_profile, cand_profile):
            continue
        if src_name_variants & collect_name_variants(row):
            continue
        if row_filter(row, idx, src_ing):
            continue

        cand_form = extract_form_key(row)
        if src_form != "other" and cand_form != src_form:
            continue

        cand_strengths = row_strengths(row, ingredient_col)
        if src_strengths:
            if not cand_strengths or not (src_strengths & cand_strengths):
                continue

        valid.add(idx)

    return valid


def generate_test_cases(
    df: pd.DataFrame,
    ingredient_col: str,
    substitute_eligible: Sequence[bool],
    row_filter: RowFilter,
    *,
    max_cases: Optional[int] = None,
    require_oracle: bool = True,
) -> List[SubstituteTestCase]:
    cases: List[SubstituteTestCase] = []
    for idx in range(len(df)):
        row = attach_row_metadata(df.iloc[idx].to_dict(), idx)
        if idx < len(substitute_eligible) and not substitute_eligible[idx]:
            continue
        if not source_has_test_metadata(row, ingredient_col):
            continue

        oracle = oracle_substitute_indices(df, idx, ingredient_col, substitute_eligible, row_filter)
        if require_oracle and not oracle:
            continue

        cases.append(
            SubstituteTestCase(
                source_index=idx,
                query=build_substitute_query(row),
                fingerprint=source_fingerprint(row, ingredient_col),
                oracle_count=len(oracle),
            )
        )
        if max_cases and len(cases) >= max_cases:
            break
    return cases


def evaluate_case(
    source_row: dict,
    substitutes: Sequence[dict],
    oracle_indices: Set[int],
    ingredient_col: str,
    *,
    k: int = DEFAULT_K,
    min_confidence: float = MIN_SUBSTITUTE_CONFIDENCE,
) -> Tuple[List[SubstituteFailure], Dict[str, Any]]:
    """Evaluate one source case. Returns failures and per-case counters."""
    failures: List[SubstituteFailure] = []
    top_k = list(substitutes[:k])

    ingredient_hits = 0
    form_hits = 0
    constraint_checks = 0
    constraint_hits = 0
    valid_in_top_k = 0

    src_form = extract_form_key(source_row)
    src_strengths = row_strengths(source_row, ingredient_col)
    src_volume = row_volume_ml(source_row)
    src_pack = extract_pack_size_from_row(source_row)

    for rank, sub in enumerate(top_k, start=1):
        score = float(sub.get("retrieval_score") or 0.0)
        is_valid, fail_types, details = validate_substitute_candidate(
            source_row,
            sub,
            ingredient_col,
            min_confidence=min_confidence,
            retrieval_score=score,
        )

        cand_profile = parse_ingredient_profile(sub.get(ingredient_col, ""))
        src_profile = parse_ingredient_profile(source_row.get(ingredient_col, ""))
        if ingredient_profiles_match(src_profile, cand_profile) and therapeutic_compatible(src_profile, cand_profile):
            ingredient_hits += 1

        if src_form == "other" or extract_form_key(sub) == src_form:
            form_hits += 1

        if src_strengths:
            constraint_checks += 1
            if row_strengths(sub, ingredient_col) & src_strengths:
                constraint_hits += 1
        if src_volume:
            constraint_checks += 1
            if row_volume_ml(sub) == src_volume:
                constraint_hits += 1
        if src_pack:
            constraint_checks += 1
            if extract_pack_size_from_row(sub) == src_pack:
                constraint_hits += 1

        if is_valid:
            valid_in_top_k += 1
        else:
            failures.append(
                SubstituteFailure(
                    source_index=int(source_row.get("row_index", 0)),
                    source_fingerprint=source_fingerprint(source_row, ingredient_col),
                    query=build_substitute_query(source_row),
                    failure_types=fail_types,
                    substitute_row_id=display_row_id(int(sub.get("row_index", 0)), sub),
                    substitute_fingerprint=source_fingerprint(sub, ingredient_col),
                    retrieval_score=score,
                    rank=rank,
                    details=details,
                )
            )

    if oracle_indices and not top_k:
        failures.append(
            SubstituteFailure(
                source_index=int(source_row.get("row_index", 0)),
                source_fingerprint=source_fingerprint(source_row, ingredient_col),
                query=build_substitute_query(source_row),
                failure_types=[FAILURE_EMPTY_COVERAGE],
                details={"oracle_alternatives": len(oracle_indices)},
            )
        )

    counters = {
        "retrieved": len(top_k),
        "valid_in_top_k": valid_in_top_k,
        "ingredient_hits": ingredient_hits,
        "form_hits": form_hits,
        "constraint_checks": constraint_checks,
        "constraint_hits": constraint_hits,
        "has_oracle": bool(oracle_indices),
        "covered": bool(top_k) if oracle_indices else True,
    }
    return failures, counters


def run_evaluation(
    df: pd.DataFrame,
    retrieve_fn: Callable[[SubstituteTestCase], List[dict]],
    ingredient_col: str,
    substitute_eligible: Sequence[bool],
    row_filter: RowFilter,
    *,
    k: int = DEFAULT_K,
    min_confidence: float = MIN_SUBSTITUTE_CONFIDENCE,
    max_cases: Optional[int] = None,
) -> EvalReport:
    cases = generate_test_cases(
        df,
        ingredient_col,
        substitute_eligible,
        row_filter,
        max_cases=max_cases,
        require_oracle=True,
    )

    all_failures: List[SubstituteFailure] = []
    failure_counter: Counter = Counter()

    total_retrieved = 0
    total_valid = 0
    total_ingredient_hits = 0
    total_form_hits = 0
    total_constraint_checks = 0
    total_constraint_hits = 0
    cases_with_oracle = 0
    covered_cases = 0

    for case in cases:
        source_row = attach_row_metadata(df.iloc[case.source_index].to_dict(), case.source_index)
        oracle = oracle_substitute_indices(
            df, case.source_index, ingredient_col, substitute_eligible, row_filter
        )
        if oracle:
            cases_with_oracle += 1

        substitutes = retrieve_fn(case)
        case_failures, counters = evaluate_case(
            source_row,
            substitutes,
            oracle,
            ingredient_col,
            k=k,
            min_confidence=min_confidence,
        )
        all_failures.extend(case_failures)
        for failure in case_failures:
            for ft in failure.failure_types:
                failure_counter[ft] += 1

        total_retrieved += counters["retrieved"]
        total_valid += counters["valid_in_top_k"]
        total_ingredient_hits += counters["ingredient_hits"]
        total_form_hits += counters["form_hits"]
        total_constraint_checks += counters["constraint_checks"]
        total_constraint_hits += counters["constraint_hits"]
        if counters["has_oracle"] and counters["covered"]:
            covered_cases += 1

    precision_at_k = (total_valid / total_retrieved) if total_retrieved else 1.0
    metrics = EvalMetrics(
        total_cases=len(cases),
        cases_with_oracle=cases_with_oracle,
        retrieved_rows=total_retrieved,
        valid_retrieved_rows=total_valid,
        precision_at_k=round(precision_at_k, 4),
        active_ingredient_preservation_rate=round(
            (total_ingredient_hits / total_retrieved) if total_retrieved else 1.0, 4
        ),
        dosage_form_preservation_rate=round(
            (total_form_hits / total_retrieved) if total_retrieved else 1.0, 4
        ),
        constraint_preservation_rate=round(
            (total_constraint_hits / total_constraint_checks) if total_constraint_checks else 1.0, 4
        ),
        substitute_coverage=round(
            (covered_cases / cases_with_oracle) if cases_with_oracle else 1.0, 4
        ),
        failure_distribution=dict(failure_counter),
    )

    return EvalReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        dataset_rows=len(df),
        test_cases=len(cases),
        k=k,
        min_confidence=min_confidence,
        metrics=metrics,
        failures=all_failures,
        improvement_hints=build_improvement_hints(all_failures),
    )


def build_improvement_hints(failures: Iterable[SubstituteFailure]) -> Dict[str, List[str]]:
    hints: Dict[str, Set[str]] = defaultdict(set)
    for failure in failures:
        for ft in failure.failure_types:
            if ft == FAILURE_WRONG_INGREDIENT:
                hints[ft].add("Tighten ingredient profile matching and therapeutic role gates in retrieval filters.")
            elif ft == FAILURE_WRONG_FORM:
                hints[ft].add("Enforce dosage-form compatibility before ranking; verify form_key extraction from names.")
            elif ft == FAILURE_WRONG_STRENGTH:
                hints[ft].add("Require strength overlap in candidate filtering and boost exact strength matches in reranking.")
            elif ft == FAILURE_IGNORED_NUMERIC:
                fields = failure.details.get("ignored_numeric_fields") or []
                if "volume_ml" in fields:
                    hints[ft].add("Apply volume constraints during substitute filtering, not only as a ranking bonus.")
                if "pack_size" in fields:
                    hints[ft].add("Parse and preserve pack-size constraints from product names when specified.")
            elif ft == FAILURE_LOW_CONFIDENCE:
                hints[ft].add("Raise minimum substitute score threshold or improve ranking weights for constraint matches.")
            elif ft == FAILURE_EMPTY_COVERAGE:
                hints[ft].add(
                    "Investigate false-negative filtering (name-variant overlap, strength parsing, substitute eligibility)."
                )
    return {key: sorted(values) for key, values in hints.items()}


def save_failure_report(report: EvalReport, output_path: str) -> str:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report.to_dict(), handle, ensure_ascii=False, indent=2)
    return output_path


def failures_to_regression_cases(failures: Sequence[SubstituteFailure]) -> List[Dict[str, Any]]:
    """Convert failures into structural regression cases (no product-specific rules)."""
    cases: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    for failure in failures:
        fp = failure.source_fingerprint
        key = json.dumps(fp, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)

        assertions: Dict[str, Any] = {
            "max_failures": 0,
            "forbidden_failure_types": sorted(set(failure.failure_types)),
        }
        if FAILURE_EMPTY_COVERAGE in failure.failure_types:
            assertions["min_substitutes"] = 1

        forbidden_tokens: Set[str] = set()
        if failure.substitute_fingerprint:
            src_tokens = set(fp.get("ingredient_tokens") or [])
            cand_tokens = set(failure.substitute_fingerprint.get("ingredient_tokens") or [])
            forbidden_tokens = cand_tokens - src_tokens
        if forbidden_tokens:
            assertions["forbidden_ingredient_tokens"] = sorted(forbidden_tokens)

        if fp.get("form_key"):
            assertions["required_form_key"] = fp["form_key"]
        if fp.get("strengths"):
            assertions["required_strength_overlap"] = fp["strengths"]

        cases.append(
            {
                "source": fp,
                "assertions": assertions,
                "origin_failure_types": sorted(set(failure.failure_types)),
            }
        )
    return cases


def load_regression_cases(path: str = REGRESSION_CASES_PATH) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload.get("cases", [])


def save_regression_cases(
    cases: Sequence[Dict[str, Any]],
    path: str = REGRESSION_CASES_PATH,
    *,
    merge: bool = True,
) -> str:
    existing = load_regression_cases(path) if merge and os.path.exists(path) else []
    merged: Dict[str, Dict[str, Any]] = {}
    for case in list(existing) + list(cases):
        merged[json.dumps(case.get("source", {}), sort_keys=True)] = case

    payload = {"cases": list(merged.values())}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return path


def fingerprint_matches_row(row: dict, fingerprint: Dict[str, Any], ingredient_col: str) -> bool:
    actual = source_fingerprint(row, ingredient_col)
    for key in ("ingredient_tokens", "form_key", "strengths", "volume_ml", "pack_size"):
        expected = fingerprint.get(key)
        if expected in (None, "", []):
            continue
        if actual.get(key) != expected:
            return False
    return True


def run_regression_case(
    df: pd.DataFrame,
    engine: Any,
    case: Dict[str, Any],
    ingredient_col: str,
    substitute_eligible: Sequence[bool],
    row_filter: RowFilter,
    *,
    k: int = DEFAULT_K,
    min_confidence: float = MIN_SUBSTITUTE_CONFIDENCE,
) -> Tuple[bool, List[str]]:
    """Execute one regression case. Returns (passed, error_messages)."""
    fingerprint = case.get("source") or {}
    assertions = case.get("assertions") or {}
    errors: List[str] = []

    source_index = None
    for idx in range(len(df)):
        row = attach_row_metadata(df.iloc[idx].to_dict(), idx)
        if fingerprint_matches_row(row, fingerprint, ingredient_col):
            source_index = idx
            break
    if source_index is None:
        return False, ["No source row matches regression fingerprint"]

    source_row = attach_row_metadata(df.iloc[source_index].to_dict(), source_index)
    substitutes = engine.find_substitutes(
        source_row=source_row,
        source_index=source_index,
        row_filter=row_filter,
        max_results=k,
    )

    min_required = int(assertions.get("min_substitutes", 0))
    if min_required and len(substitutes) < min_required:
        errors.append(f"Expected at least {min_required} substitutes, got {len(substitutes)}")

    forbidden_types = set(assertions.get("forbidden_failure_types") or [])
    forbidden_tokens = set(assertions.get("forbidden_ingredient_tokens") or [])
    required_form = assertions.get("required_form_key")
    required_strengths = set(assertions.get("required_strength_overlap") or [])

    for sub in substitutes:
        cand_tokens = set(parse_ingredient_profile(sub.get(ingredient_col, "")))
        if forbidden_tokens & cand_tokens:
            errors.append(f"Forbidden ingredient tokens returned: {sorted(forbidden_tokens & cand_tokens)}")

        if required_form and extract_form_key(sub) != required_form:
            errors.append(f"Returned form {extract_form_key(sub)!r}, expected {required_form!r}")

        if required_strengths:
            sub_strengths = row_strengths(sub, ingredient_col)
            if not (sub_strengths & required_strengths):
                errors.append("Returned substitute missing required strength overlap")

        _, fail_types, _ = validate_substitute_candidate(
            source_row,
            sub,
            ingredient_col,
            min_confidence=min_confidence,
        )
        unexpected = set(fail_types) & forbidden_types
        if unexpected:
            errors.append(f"Regression failure types reappeared: {sorted(unexpected)}")

    return (not errors, errors)
