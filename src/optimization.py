"""
optimization.py
=================
Decision layer: turns the renewal-risk predictions into an actionable,
budget-constrained outreach plan. This is the piece that connects the ML
model to a real business decision rather than leaving it as an abstract
score.

Scenario: the Emeritus College volunteer phone-campaign has a limited
number of caller-hours each cycle. We cannot call every at-risk member, so
we must choose WHOM to call to maximize expected renewals recovered,
subject to a call-capacity constraint — a classic knapsack-style resource
allocation problem.

Two allocation strategies are computed and compared:
  1. Greedy / highest-risk-first — call the members most likely to lapse.
  2. Expected-value-maximizing (marginal uplift) — call members where a
     call has the highest *expected marginal benefit*: members who are
     genuinely on the fence (mid-range renewal probability) are more
     persuadable than members who were never going to renew regardless of
     being called, or members who were always going to renew anyway. This
     is implemented via a simple assumed "lift" function peaking in the
     mid-probability range, and solved as a 0/1 knapsack with
     dynamic programming (capacity = number of available caller-hours).

Run directly to produce an outreach priority list:
    python -m src.optimization
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

MINUTES_PER_CALL = 12          # avg. volunteer phone-call length incl. notes
DEFAULT_CALLER_HOURS = 40      # a typical campaign's total volunteer hours available


def load_scored_population() -> pd.DataFrame:
    scored = pd.read_csv(PROCESSED_DIR / "scored_members.csv")
    features = pd.read_csv(PROCESSED_DIR / "member_features.csv")
    return scored.merge(features, on="member_id", suffixes=("", "_feat"))


def assumed_call_lift(p_renew: np.ndarray) -> np.ndarray:
    """Assumed probability a call *changes the outcome* (from lapse to renew),
    peaking for members who are genuinely on the fence (~0.35-0.55 baseline
    renewal probability) and near-zero for members who were always going to
    renew or were always going to lapse regardless of contact. This reflects
    a common finding in nonprofit renewal/membership research: outreach has
    the most marginal impact on the "movable middle," not the extremes."""
    return np.exp(-((p_renew - 0.42) ** 2) / (2 * 0.18 ** 2)) * 0.35


def knapsack_select(values: np.ndarray, costs: np.ndarray, capacity: int) -> np.ndarray:
    """0/1 knapsack via dynamic programming (values/costs scaled to ints).
    Returns a boolean selection mask. Capacity and costs are in whole call-slots."""
    n = len(values)
    scaled_values = (values * 1000).round().astype(int)
    dp = np.zeros((n + 1, capacity + 1), dtype=np.int64)
    keep = np.zeros((n + 1, capacity + 1), dtype=bool)

    for i in range(1, n + 1):
        cost = costs[i - 1]
        val = scaled_values[i - 1]
        # Vectorized over capacity dimension
        dp[i, :] = dp[i - 1, :]
        if cost <= capacity:
            affordable = np.arange(cost, capacity + 1)
            candidate = dp[i - 1, affordable - cost] + val
            better = candidate > dp[i, affordable]
            dp[i, affordable[better]] = candidate[better]
            keep[i, affordable[better]] = True

    selected = np.zeros(n, dtype=bool)
    c = capacity
    for i in range(n, 0, -1):
        if keep[i, c]:
            selected[i - 1] = True
            c -= costs[i - 1]
    return selected


def build_outreach_plan(df: pd.DataFrame, caller_hours: int = DEFAULT_CALLER_HOURS) -> dict:
    capacity_calls = int(caller_hours * 60 / MINUTES_PER_CALL)
    df = df.copy()
    df["call_lift"] = assumed_call_lift(df["predicted_renewal_probability"].values)
    df["expected_marginal_renewals"] = df["call_lift"]  # 1 call slot = cost 1, so value == lift
    costs = np.ones(len(df), dtype=int)

    # Strategy 1: greedy highest-risk-first (lowest predicted probability first)
    greedy_order = df.sort_values("predicted_renewal_probability").head(capacity_calls)
    greedy_expected_gain = greedy_order["call_lift"].sum()

    # Strategy 2: expected-value-maximizing via knapsack on call_lift
    capacity_calls_capped = min(capacity_calls, len(df))
    selected_mask = knapsack_select(df["expected_marginal_renewals"].values, costs, capacity_calls_capped)
    optimized = df[selected_mask].sort_values("call_lift", ascending=False)
    optimized_expected_gain = optimized["call_lift"].sum()

    improvement_pct = (
        (optimized_expected_gain - greedy_expected_gain) / greedy_expected_gain * 100
        if greedy_expected_gain > 0 else 0.0
    )

    summary = {
        "caller_hours": caller_hours,
        "capacity_calls": capacity_calls,
        "greedy_expected_renewals_recovered": float(greedy_expected_gain),
        "optimized_expected_renewals_recovered": float(optimized_expected_gain),
        "improvement_pct": float(improvement_pct),
    }
    return {
        "summary": summary,
        "greedy_plan": greedy_order,
        "optimized_plan": optimized,
        "full_scored": df,
    }


def main():
    df = load_scored_population()
    result = build_outreach_plan(df)

    print("Budget-constrained outreach optimization")
    print(json.dumps(result["summary"], indent=2))

    out_cols = ["member_id", "predicted_renewal_probability", "call_lift", "expected_marginal_renewals"]
    result["optimized_plan"][out_cols].to_csv(PROCESSED_DIR / "outreach_plan_optimized.csv", index=False)
    result["greedy_plan"][out_cols].to_csv(PROCESSED_DIR / "outreach_plan_greedy.csv", index=False)
    result["full_scored"][out_cols].to_csv(PROCESSED_DIR / "outreach_scored_all.csv", index=False)

    with open(PROCESSED_DIR / "optimization_summary.json", "w") as f:
        json.dump(result["summary"], f, indent=2)

    print(f"\nWrote outreach plans to {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
