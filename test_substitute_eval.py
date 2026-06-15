"""Tests for substitute evaluation framework and structural regression cases."""

import os

import pandas as pd
import pytest

from retrieval import DrugRetrievalEngine, attach_row_metadata
from substitute_eval import (
    FAILURE_EMPTY_COVERAGE,
    FAILURE_WRONG_INGREDIENT,
    REGRESSION_CASES_PATH,
    SubstituteTestCase,
    build_substitute_query,
    evaluate_case,
    generate_test_cases,
    load_regression_cases,
    oracle_substitute_indices,
    run_evaluation,
    run_regression_case,
    source_has_test_metadata,
    validate_substitute_candidate,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "name_ar": "بانادول",
            "name_en": "Panadol 500mg 24 Tabs",
            "active_ingredient": "paracetamol 500mg tablet",
            "combined": "بانادول Panadol paracetamol 500mg tablet",
            "price_egp": "25",
            "form": "tablet",
            "form_clean": "tablet",
            "dosage_clean": "500mg",
        },
        {
            "name_ar": "فارما دول",
            "name_en": "Pharmadol 500mg 24 Tabs",
            "active_ingredient": "paracetamol 500mg tablet",
            "combined": "Pharmadol paracetamol 500mg tablet",
            "price_egp": "20",
            "form": "tablet",
            "form_clean": "tablet",
            "dosage_clean": "500mg",
        },
        {
            "name_ar": "مايولاكس",
            "name_en": "Myolax 20 Caps",
            "active_ingredient": "chlorzoxazone+paracetamol(acetaminophen)",
            "combined": "Myolax chlorzoxazone paracetamol",
            "price_egp": "40",
            "form": "capsule",
            "form_clean": "capsule",
            "dosage_clean": "500mg",
        },
        {
            "name_ar": "بروفين",
            "name_en": "Brufen 400mg 20 Tabs",
            "active_ingredient": "ibuprofen 400mg tablet",
            "combined": "Brufen ibuprofen 400mg tablet",
            "price_egp": "30",
            "form": "tablet",
            "form_clean": "tablet",
            "dosage_clean": "400mg",
        },
    ])


def _engine(df: pd.DataFrame) -> DrugRetrievalEngine:
    return DrugRetrievalEngine(
        df=df,
        index=None,
        ingredient_col="active_ingredient",
        get_embed_model=lambda: None,
        enable_semantic=False,
    )


def _row_filter(_row, _idx, _query):
    return None


def test_source_metadata_gate():
    df = _sample_df()
    row = attach_row_metadata(df.iloc[0].to_dict(), 0)
    assert source_has_test_metadata(row, "active_ingredient")
    assert build_substitute_query(row).startswith("substitute for")


def test_oracle_excludes_same_product_variants():
    df = _sample_df()
    eligible = [True] * len(df)
    oracle = oracle_substitute_indices(df, 0, "active_ingredient", eligible, _row_filter)
    assert 1 in oracle
    assert 2 not in oracle


def test_validate_detects_wrong_ingredient():
    df = _sample_df()
    source = attach_row_metadata(df.iloc[0].to_dict(), 0)
    bad = attach_row_metadata(df.iloc[2].to_dict(), 2)
    valid, failures, _ = validate_substitute_candidate(source, bad, "active_ingredient", min_confidence=0.0)
    assert not valid
    assert FAILURE_WRONG_INGREDIENT in failures


def test_generate_test_cases_from_sample_df():
    df = _sample_df()
    eligible = [True] * len(df)
    cases = generate_test_cases(df, "active_ingredient", eligible, _row_filter, require_oracle=True)
    assert cases
    assert all(case.oracle_count > 0 for case in cases)


def test_evaluate_case_empty_coverage_failure():
    df = _sample_df()
    source = attach_row_metadata(df.iloc[0].to_dict(), 0)
    eligible = [True] * len(df)
    oracle = oracle_substitute_indices(df, 0, "active_ingredient", eligible, _row_filter)
    failures, counters = evaluate_case(source, [], oracle, "active_ingredient", k=3, min_confidence=0.0)
    assert counters["covered"] is False
    assert any(FAILURE_EMPTY_COVERAGE in f.failure_types for f in failures)


def test_engine_substitutes_pass_validation_on_sample():
    df = _sample_df()
    engine = _engine(df)
    source = attach_row_metadata(df.iloc[0].to_dict(), 0)
    subs = engine.find_substitutes(
        source_row=source,
        source_index=0,
        row_filter=_row_filter,
        max_results=3,
    )
    for sub in subs:
        valid, failures, _ = validate_substitute_candidate(source, sub, "active_ingredient", min_confidence=0.0)
        assert valid, failures


@pytest.mark.parametrize("case", load_regression_cases(REGRESSION_CASES_PATH))
def test_structural_regression_cases(case):
    df = _sample_df()
    engine = _engine(df)
    passed, errors = run_regression_case(
        df,
        engine,
        case,
        "active_ingredient",
        [True] * len(df),
        _row_filter,
        k=3,
        min_confidence=0.0,
    )
    assert passed, errors


@pytest.mark.skipif(not os.path.exists("egypt_drugs_cleaned_utf8.csv"), reason="full dataset required")
def test_full_dataset_eval_smoke():
    from data_cleaning import clean_dataframe, SUBSTITUTE_ELIGIBLE

    raw = pd.read_csv("egypt_drugs_cleaned_utf8.csv", encoding="utf-8").fillna("")
    df, meta = clean_dataframe(raw)
    ingredient_col = meta["ingredient_col"]
    engine = _engine(df)
    eligible = SUBSTITUTE_ELIGIBLE

    def retrieve(case: SubstituteTestCase):
        source_row = attach_row_metadata(df.iloc[case.source_index].to_dict(), case.source_index)
        return engine.find_substitutes(
            source_row=source_row,
            source_index=case.source_index,
            row_filter=_row_filter,
            max_results=3,
        )

    report = run_evaluation(
        df,
        retrieve,
        ingredient_col,
        eligible,
        _row_filter,
        k=3,
        max_cases=50,
    )
    assert report.test_cases > 0
    assert 0.0 <= report.metrics.precision_at_k <= 1.0
