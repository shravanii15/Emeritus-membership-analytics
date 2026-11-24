import pandas as pd
import pytest

from src.data_cleaning import (
    normalize_asu_id,
    normalize_text,
    normalize_zip,
    linkage_key,
    build_clean_dataset,
)


class TestNormalizeText:
    def test_lowercases_and_strips_punctuation(self):
        assert normalize_text("O'Brien-Smith") == "obriensmith"

    def test_handles_none(self):
        assert normalize_text(None) == ""

    def test_handles_nan(self):
        assert normalize_text(float("nan")) == ""

    def test_strips_accents(self):
        assert normalize_text("José") == "jose"


class TestNormalizeZip:
    def test_plain_five_digit(self):
        assert normalize_zip("85287") == "85287"

    def test_zip_plus_four(self):
        assert normalize_zip("85287-1234") == "85287"

    def test_float_zip_from_excel(self):
        assert normalize_zip(85287.0) == "85287"

    def test_short_zip_zero_padded(self):
        assert normalize_zip("1234") == "01234"

    def test_handles_none(self):
        assert normalize_zip(None) == ""


class TestNormalizeAsuId:
    def test_valid_id(self):
        assert normalize_asu_id(1000423727) == 1000423727

    def test_string_unknown_becomes_none(self):
        assert normalize_asu_id("Unknown") is None

    def test_none_stays_none(self):
        assert normalize_asu_id(None) is None

    def test_numeric_string(self):
        assert normalize_asu_id("1000423727") == 1000423727


class TestLinkageKey:
    def test_same_person_different_case_matches(self):
        k1 = linkage_key("John", "Smith", "85287")
        k2 = linkage_key("JOHN", "smith", "85287-1234")
        assert k1 == k2

    def test_different_people_differ(self):
        k1 = linkage_key("John", "Smith", "85287")
        k2 = linkage_key("Jane", "Smith", "85287")
        assert k1 != k2


class TestBuildCleanDataset:
    @pytest.fixture(scope="class")
    def clean_df(self):
        yield build_clean_dataset()

    def test_returns_dataframe(self, clean_df):
        assert isinstance(clean_df, pd.DataFrame)

    def test_has_expected_row_count(self, clean_df):
        # 523 active + 231 deceased + 440 special (no linkage-key collisions
        # expected across these three independent identity pools)
        assert len(clean_df) == 523 + 231 + 440

    def test_member_id_is_unique(self, clean_df):
        assert clean_df["member_id"].is_unique

    def test_source_status_values(self, clean_df):
        assert set(clean_df["source_status"].unique()) == {"active", "deceased", "special_case"}

    def test_most_asu_ids_resolved(self, clean_df):
        resolved_fraction = clean_df["asu_id"].notna().mean()
        assert resolved_fraction > 0.9

    def test_ev_subscription_flag_is_boolean(self, clean_df):
        assert set(clean_df["has_current_ev_subscription"].unique()).issubset({True, False})
