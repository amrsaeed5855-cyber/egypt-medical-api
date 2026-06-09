"""Unit tests for hybrid retrieval (no CSV/FAISS required)."""

import pandas as pd

from retrieval import DrugRetrievalEngine, display_row_id, ingredient_matches_query, normalize_ingredient_query


def _sample_df():
    return pd.DataFrame([
        {
            "name_ar": "بانادول",
            "name_en": "Panadol",
            "active_ingredient": "paracetamol 500mg tablet",
            "combined": "بانادول Panadol paracetamol 500mg tablet",
            "price_egp": "25",
            "form": "tablet",
        },
        {
            "name_ar": "بروفين",
            "name_en": "Brufen",
            "active_ingredient": "ibuprofen 400mg tablet",
            "combined": "بروفين Brufen ibuprofen 400mg tablet",
            "price_egp": "30",
            "form": "tablet",
        },
        {
            "name_ar": "قطرة عين",
            "name_en": "Eye Drops",
            "active_ingredient": "chloramphenicol eye drop",
            "combined": "قطرة عين Eye Drops chloramphenicol eye drop",
            "price_egp": "15",
            "form": "eye drop",
        },
    ])


def test_ingredient_normalization():
    assert normalize_ingredient_query("acetaminophen") == "paracetamol"


def test_ingredient_token_match():
    assert ingredient_matches_query("paracetamol 500mg", "paracetamol")
    assert not ingredient_matches_query("ibuprofen 400mg", "paracetamol")


def test_display_row_id_prefers_explicit():
    assert display_row_id(5, {"id": "100"}) == 100
    assert display_row_id(5, {}) == 6


def test_lexical_ingredient_match():
    engine = DrugRetrievalEngine(
        df=_sample_df(),
        index=None,
        ingredient_col="active_ingredient",
        get_embed_model=lambda: None,
        enable_semantic=False,
    )
    rows = engine.match_by_ingredient(
        "paracetamol",
        excluded=set(),
        row_filter=lambda _r, _i, _q: None,
        max_results=2,
    )
    assert len(rows) >= 1
    assert "paracetamol" in rows[0]["active_ingredient"].lower()


def test_trade_name_match():
    engine = DrugRetrievalEngine(
        df=_sample_df(),
        index=None,
        ingredient_col="active_ingredient",
        get_embed_model=lambda: None,
        enable_semantic=False,
    )
    rows = engine.match_by_trade_name(
        "panadol",
        row_filter=lambda _r, _i, _q: None,
        max_results=1,
    )
    assert rows and "panadol" in rows[0]["name_en"].lower()
