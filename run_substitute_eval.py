#!/usr/bin/env python3
"""Run automated substitute-quality evaluation across the full medication dataset."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone

import pandas as pd

from data_cleaning import SUBSTITUTE_ELIGIBLE, clean_dataframe
from retrieval import DrugRetrievalEngine
from substitute_eval import (
    DEFAULT_K,
    DEFAULT_REPORT_DIR,
    MIN_SUBSTITUTE_CONFIDENCE,
    REGRESSION_CASES_PATH,
    SubstituteTestCase,
    failures_to_regression_cases,
    run_evaluation,
    save_failure_report,
    save_regression_cases,
)


def _noop_row_filter(_row, _idx, _query):
    return None


def load_dataset(csv_path: str) -> tuple[pd.DataFrame, str]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    raw = pd.read_csv(csv_path, encoding="utf-8").fillna("")
    cleaned, meta = clean_dataframe(raw)
    return cleaned, meta["ingredient_col"]


def build_engine(df: pd.DataFrame, ingredient_col: str, enable_semantic: bool) -> DrugRetrievalEngine:
    index = None
    get_embed_model = lambda: None
    if enable_semantic:
        import faiss

        index_path = os.getenv("FAISS_INDEX_PATH", "faiss.index")
        if os.path.exists(index_path):
            index = faiss.read_index(index_path)
            from rag_logic import get_embed_model as _get_embed_model

            get_embed_model = _get_embed_model
    return DrugRetrievalEngine(
        df=df,
        index=index,
        ingredient_col=ingredient_col,
        get_embed_model=get_embed_model,
        enable_semantic=enable_semantic and index is not None,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate substitute retrieval quality across the dataset.")
    parser.add_argument(
        "--csv",
        default=os.getenv("EGYPT_DRUGS_CSV", "egypt_drugs_cleaned_utf8.csv"),
        help="Path to medication CSV",
    )
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Top-k substitutes to evaluate")
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=MIN_SUBSTITUTE_CONFIDENCE,
        help="Minimum acceptable substitute retrieval_score",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="Limit generated test cases (0 = all eligible cases)",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Failure report JSON path (default: eval_reports/substitute_failures_<timestamp>.json)",
    )
    parser.add_argument(
        "--update-regression",
        action="store_true",
        help="Append failed cases to eval_regression_cases.json for pytest regression coverage",
    )
    parser.add_argument(
        "--regression-path",
        default=REGRESSION_CASES_PATH,
        help="Regression cases JSON path",
    )
    parser.add_argument(
        "--semantic",
        action="store_true",
        help="Enable semantic search for engine initialization (not used by substitute scan)",
    )
    parser.add_argument(
        "--fail-on-errors",
        action="store_true",
        help="Exit with code 1 when any evaluation failures are found",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    max_cases = args.max_cases if args.max_cases > 0 else None

    try:
        df, ingredient_col = load_dataset(args.csv)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2

    engine = build_engine(df, ingredient_col, enable_semantic=args.semantic)
    substitute_eligible = SUBSTITUTE_ELIGIBLE if len(SUBSTITUTE_ELIGIBLE) == len(df) else [
        True
    ] * len(df)

    def retrieve(case: SubstituteTestCase):
        source_row = df.iloc[case.source_index].to_dict()
        source_row = dict(source_row)
        source_row["row_index"] = case.source_index
        return engine.find_substitutes(
            source_row=source_row,
            source_index=case.source_index,
            row_filter=_noop_row_filter,
            max_results=args.k,
        )

    report = run_evaluation(
        df,
        retrieve,
        ingredient_col,
        substitute_eligible,
        _noop_row_filter,
        k=args.k,
        min_confidence=args.min_confidence,
        max_cases=max_cases,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = args.output or os.path.join(DEFAULT_REPORT_DIR, f"substitute_failures_{timestamp}.json")
    save_failure_report(report, output_path)

    metrics = report.metrics
    print(f"Dataset rows: {report.dataset_rows}")
    print(f"Generated test cases: {report.test_cases}")
    print(f"Precision@{args.k}: {metrics.precision_at_k}")
    print(f"Active ingredient preservation: {metrics.active_ingredient_preservation_rate}")
    print(f"Dosage form preservation: {metrics.dosage_form_preservation_rate}")
    print(f"Constraint preservation: {metrics.constraint_preservation_rate}")
    print(f"Substitute coverage: {metrics.substitute_coverage}")
    print(f"Failure distribution: {metrics.failure_distribution}")
    print(f"Failures logged: {len(report.failures)}")
    print(f"Report saved to: {output_path}")

    if report.improvement_hints:
        print("\nImprovement hints:")
        for failure_type, hints in report.improvement_hints.items():
            print(f"  [{failure_type}]")
            for hint in hints:
                print(f"    - {hint}")

    if args.update_regression and report.failures:
        regression_cases = failures_to_regression_cases(report.failures)
        regression_path = save_regression_cases(regression_cases, args.regression_path, merge=True)
        print(f"Updated regression cases: {regression_path} (+{len(regression_cases)} cases)")

    if args.fail_on_errors and report.failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
