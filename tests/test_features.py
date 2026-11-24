import pandas as pd
import pytest

from src.data_cleaning import build_clean_dataset
from src.features import engineer_features


@pytest.fixture(scope="module")
def features_df():
    clean = build_clean_dataset()
    return engineer_features(clean)


def test_only_active_members_included(features_df):
    # engineer_features filters to source_status == "active" internally;
    # row count should match the active-roster size.
    assert len(features_df) == 523


def test_renewed_target_is_binary(features_df):
    assert set(features_df["renewed"].unique()).issubset({0, 1})


def test_renewal_rate_is_plausible(features_df):
    rate = features_df["renewed"].mean()
    # Sanity bound: real-world membership renewal rates rarely sit below 20%
    # or above 90% for an engaged alumni-style organization.
    assert 0.2 < rate < 0.9


def test_engagement_score_bounded(features_df):
    assert features_df["engagement_score"].between(0, 4).all()


def test_years_since_retirement_non_negative(features_df):
    assert (features_df["years_since_retirement"].dropna() >= 0).all()


def test_no_missing_member_id(features_df):
    assert features_df["member_id"].notna().all()


def test_discipline_bucketed_has_other_category(features_df):
    assert "Other" in features_df["discipline_bucketed"].unique()
