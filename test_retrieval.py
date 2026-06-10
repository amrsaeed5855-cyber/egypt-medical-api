"""Unit tests for hybrid retrieval (no CSV/FAISS required)."""

import pandas as pd

from retrieval import (
    DrugRetrievalEngine,
    display_row_id,
    ingredient_matches_query,
    ingredient_profiles_match,
    normalize_ingredient_query,
    parse_ingredient_profile,
    therapeutic_compatible,
)
from trade_name_utils import classify_query, extract_drug_name_from_query, generate_search_variants


def _sample_df():
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
            "name_ar": "بروفين",
            "name_en": "Brufen",
            "active_ingredient": "ibuprofen 400mg tablet",
            "combined": "بروفين Brufen ibuprofen 400mg tablet",
            "price_egp": "30",
            "form": "tablet",
        },
        {
            "name_ar": "مايولاكس",
            "name_en": "Myolax 20 Caps",
            "active_ingredient": "chlorzoxazone+paracetamol(acetaminophen)",
            "combined": "Myolax chlorzoxazone paracetamol",
            "price_egp": "40",
            "form": "capsule",
            "form_clean": "capsule",
        },
        {
            "name_ar": "بانادول اكسترا",
            "name_en": "Panadol Extra 24 Tabs",
            "active_ingredient": "caffeine+paracetamol(acetaminophen)",
            "combined": "Panadol Extra caffeine paracetamol",
            "price_egp": "54",
            "form": "tablet",
            "form_clean": "tablet",
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


def test_arabic_trade_name():
    engine = DrugRetrievalEngine(
        df=_sample_df(),
        index=None,
        ingredient_col="active_ingredient",
        get_embed_model=lambda: None,
        enable_semantic=False,
    )
    rows = engine.match_by_trade_name(
        "بانادول",
        row_filter=lambda _r, _i, _q: None,
        max_results=1,
    )
    assert rows


def test_substitute_excludes_muscle_relaxant_combo():
    engine = DrugRetrievalEngine(
        df=_sample_df(),
        index=None,
        ingredient_col="active_ingredient",
        get_embed_model=lambda: None,
        enable_semantic=False,
    )
    source = engine.match_by_trade_name("panadol", row_filter=lambda _r, _i, _q: None, max_results=1)[0]
    subs = engine.find_substitutes(
        source_row=source,
        source_index=source["row_index"],
        row_filter=lambda _r, _i, _q: None,
        max_results=5,
    )
    for s in subs:
        assert "chlorzoxazone" not in s["active_ingredient"].lower()


def test_therapeutic_compatible():
    paracetamol_only = parse_ingredient_profile("paracetamol 500mg")
    combo = parse_ingredient_profile("chlorzoxazone+paracetamol")
    assert not therapeutic_compatible(paracetamol_only, combo)


def test_query_classification():
    assert classify_query("بديل البنادول") == "substitute"
    assert classify_query("سعر بانادول") == "product_info"
    assert classify_query("عاوز دواء للصداع") == "symptom_treatment"


def test_extract_drug_name():
    assert "بنادول" in (extract_drug_name_from_query("بديل البنادول") or "")


def test_unknown_trade_name_blocked():
    engine = DrugRetrievalEngine(
        df=_sample_df(),
        index=None,
        ingredient_col="active_ingredient",
        get_embed_model=lambda: None,
        enable_semantic=False,
    )
    rows = engine.match_by_trade_name(
        "toblaxiel",
        row_filter=lambda _r, _i, _q: None,
        max_results=3,
    )
    assert len(rows) == 0


def test_split_multi_drug():
    from trade_name_utils import split_multi_drug_names
    parts = split_multi_drug_names("سعر ديفارول وفيدروب")
    assert len(parts) >= 2
