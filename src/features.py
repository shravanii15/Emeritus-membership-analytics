"""
features.py
============
Feature engineering layer: turns the cleaned member table into a
model-ready feature matrix for renewal/churn prediction and segmentation.

Target definition (renewal / churn):
  Since this is a membership organization, "churn" = an active member who
  has NOT paid current-cycle dues and shows no current EV subscription
  (i.e., shows no sign of renewed engagement this cycle). This mirrors a
  realistic nonprofit/association renewal-prediction problem.

Run directly to produce data/processed/member_features.csv:
    python -m src.features
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
REFERENCE_DATE = pd.Timestamp("2026-11-24")  # "today" for recency calculations


def load_clean() -> pd.DataFrame:
    path = PROCESSED_DIR / "members_clean.csv"
    df = pd.read_csv(path, parse_dates=[
        "joined_ec", "retirement_date", "deceased_date",
        "dues_date_paid", "ev_subscription_date_paid",
    ])
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Only active, living members are in scope for a renewal model —
    # deceased members and special-case rosters aren't being "renewed".
    model_pop = df[df["source_status"] == "active"].copy()

    # --- Tenure & recency ---
    model_pop["years_since_retirement"] = (
        (REFERENCE_DATE - model_pop["retirement_date"]).dt.days / 365.25
    )
    model_pop["years_since_joined_ec"] = (
        (REFERENCE_DATE - model_pop["joined_ec"]).dt.days / 365.25
    )
    model_pop["years_since_retirement"] = model_pop["years_since_retirement"].clip(lower=0)
    model_pop["years_since_joined_ec"] = model_pop["years_since_joined_ec"].clip(lower=0)

    # Gap between retiring and joining Emeritus College — a behavioral signal:
    # people who join quickly after retiring tend to be more engaged.
    model_pop["years_retirement_to_join_gap"] = (
        model_pop["years_since_retirement"] - model_pop["years_since_joined_ec"]
    ).clip(lower=0)

    model_pop["dues_paid_recency_days"] = (
        REFERENCE_DATE - model_pop["dues_date_paid"]
    ).dt.days
    model_pop["has_paid_dues_this_cycle"] = model_pop["dues_date_paid"].notna().astype(int)
    model_pop["has_ev_subscription"] = (model_pop["ev_subscription"] == "Yes").astype(int)
    model_pop["has_current_ev_subscription"] = model_pop["has_current_ev_subscription"].astype(int)
    model_pop["opted_out_of_email"] = model_pop["opted_out_of_email"].astype(int)
    model_pop["has_phone_on_file"] = model_pop["phone"].notna().astype(int)
    model_pop["has_email_on_file"] = model_pop["email"].notna().astype(int)
    model_pop["is_in_state"] = (model_pop["state"].astype(str).str.strip() == "AZ").astype(int)

    # Simple composite engagement score (0-4): sums observable engagement signals.
    model_pop["engagement_score"] = (
        model_pop["has_paid_dues_this_cycle"]
        + model_pop["has_ev_subscription"]
        + model_pop["has_current_ev_subscription"]
        + (1 - model_pop["opted_out_of_email"])
    )

    # --- Target: renewed vs. churn-risk this cycle ---
    # "Renewed" = paid dues this cycle OR holds a current EV subscription.
    model_pop["renewed"] = (
        (model_pop["has_paid_dues_this_cycle"] == 1)
        | (model_pop["has_current_ev_subscription"] == 1)
    ).astype(int)

    # discipline: keep top-N categories, bucket the rest as "Other" to avoid
    # a huge sparse one-hot matrix on a modest dataset.
    top_disciplines = model_pop["discipline"].value_counts().nlargest(10).index
    model_pop["discipline_bucketed"] = np.where(
        model_pop["discipline"].isin(top_disciplines), model_pop["discipline"], "Other"
    )

    feature_cols = [
        "member_id", "years_since_retirement", "years_since_joined_ec",
        "years_retirement_to_join_gap", "dues_paid_recency_days",
        "has_paid_dues_this_cycle", "has_ev_subscription",
        "has_current_ev_subscription", "opted_out_of_email",
        "has_phone_on_file", "has_email_on_file", "is_in_state",
        "engagement_score", "discipline_bucketed", "membership_fee",
        "renewed",
    ]
    return model_pop[feature_cols]


def main():
    df = load_clean()
    features = engineer_features(df)
    out_path = PROCESSED_DIR / "member_features.csv"
    features.to_csv(out_path, index=False)
    print(f"Wrote {len(features)} feature rows to {out_path}")
    print(f"Renewal rate (target=1): {features['renewed'].mean():.1%}")
    print(features.describe(include='all').T[["count", "mean"]].head(20))


if __name__ == "__main__":
    main()
