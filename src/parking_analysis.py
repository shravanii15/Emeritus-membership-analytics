"""
parking_analysis.py
====================
Budget-impact analysis: estimating the savings available by not issuing
on-campus parking passes to out-of-state ("snowbird"/remote) members, who
are geographically unlikely to use campus parking.

This mirrors a real internal analysis built for the Emeritus College
membership office — rebuilt here on 100% synthetic data, with the key
assumption (not every out-of-state member necessarily holds/uses a pass,
and the per-pass cost is a planning estimate) made explicit and
stress-tested rather than presented as a single confident number.

Run directly to produce data/processed/parking_savings_summary.json and
data/processed/parking_by_state.csv:
    python -m src.parking_analysis
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

DEFAULT_PASS_COST = 280.0          # $ / year, a typical campus parking-permit price
DEFAULT_HOLDER_RATE = 1.0          # assumed share of out-of-state members who hold a pass


def load_active_members() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / "members_clean.csv")
    return df[df["source_status"] == "active"].copy()


def state_breakdown(active: pd.DataFrame) -> pd.DataFrame:
    """Out-of-state member counts by state, sorted descending."""
    out_of_state = active[active["state"].astype(str).str.strip() != "AZ"]
    counts = (
        out_of_state["state"]
        .value_counts()
        .rename_axis("state")
        .reset_index(name="member_count")
    )
    return counts


def estimate_savings(
    active: pd.DataFrame,
    pass_cost: float = DEFAULT_PASS_COST,
    holder_rate: float = DEFAULT_HOLDER_RATE,
) -> dict:
    """
    Core estimate: if `holder_rate` of out-of-state members currently hold a
    parking pass at `pass_cost`/year, and those passes were not renewed for
    members who live out of state, how much would that save per year?
    """
    total_active = len(active)
    is_out_of_state = active["state"].astype(str).str.strip() != "AZ"
    n_out_of_state = int(is_out_of_state.sum())
    pct_out_of_state = n_out_of_state / total_active if total_active else 0.0

    estimated_pass_holders = n_out_of_state * holder_rate
    estimated_annual_savings = estimated_pass_holders * pass_cost

    return {
        "total_active_members": total_active,
        "n_out_of_state": n_out_of_state,
        "pct_out_of_state": pct_out_of_state,
        "pass_cost": pass_cost,
        "holder_rate": holder_rate,
        "estimated_pass_holders": estimated_pass_holders,
        "estimated_annual_savings": estimated_annual_savings,
    }


def sensitivity_table(
    active: pd.DataFrame,
    cost_scenarios: list[float] | None = None,
    holder_rate_scenarios: list[float] | None = None,
) -> pd.DataFrame:
    """
    Cross a few plausible per-pass costs against a few plausible
    "actually holds/uses a pass" rates, so the recommendation is shown as a
    defensible range rather than a single number that looks more precise
    than the underlying assumption deserves.
    """
    if cost_scenarios is None:
        cost_scenarios = [180.0, 280.0, 380.0]
    if holder_rate_scenarios is None:
        holder_rate_scenarios = [0.5, 0.75, 1.0]

    rows = []
    for cost in cost_scenarios:
        for rate in holder_rate_scenarios:
            result = estimate_savings(active, pass_cost=cost, holder_rate=rate)
            rows.append({
                "pass_cost": cost,
                "holder_rate": rate,
                "estimated_annual_savings": result["estimated_annual_savings"],
            })
    return pd.DataFrame(rows)


def main():
    active = load_active_members()
    base = estimate_savings(active)
    by_state = state_breakdown(active)
    sens = sensitivity_table(active)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROCESSED_DIR / "parking_savings_summary.json", "w") as f:
        json.dump(base, f, indent=2)
    by_state.to_csv(PROCESSED_DIR / "parking_by_state.csv", index=False)
    sens.to_csv(PROCESSED_DIR / "parking_sensitivity.csv", index=False)

    print(f"Active members: {base['total_active_members']}")
    print(f"Out-of-state: {base['n_out_of_state']} ({base['pct_out_of_state']:.1%})")
    print(f"Base-case estimated annual savings: ${base['estimated_annual_savings']:,.0f}")
    print(f"Sensitivity range: ${sens['estimated_annual_savings'].min():,.0f} - "
          f"${sens['estimated_annual_savings'].max():,.0f}")


if __name__ == "__main__":
    main()
