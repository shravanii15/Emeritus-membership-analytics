import pandas as pd
import pytest

from src.parking_analysis import estimate_savings, sensitivity_table, state_breakdown


@pytest.fixture
def toy_active():
    return pd.DataFrame({
        "member_id": [1, 2, 3, 4],
        "state": ["AZ", "AZ", "CA", "OR"],
        "source_status": ["active"] * 4,
    })


def test_estimate_savings_counts_out_of_state(toy_active):
    result = estimate_savings(toy_active, pass_cost=100.0, holder_rate=1.0)
    assert result["total_active_members"] == 4
    assert result["n_out_of_state"] == 2
    assert result["pct_out_of_state"] == pytest.approx(0.5)
    assert result["estimated_annual_savings"] == pytest.approx(200.0)


def test_estimate_savings_scales_with_holder_rate(toy_active):
    full = estimate_savings(toy_active, pass_cost=100.0, holder_rate=1.0)
    half = estimate_savings(toy_active, pass_cost=100.0, holder_rate=0.5)
    assert half["estimated_annual_savings"] == pytest.approx(full["estimated_annual_savings"] / 2)


def test_estimate_savings_handles_no_out_of_state_members():
    all_az = pd.DataFrame({"member_id": [1, 2], "state": ["AZ", "AZ"], "source_status": ["active"] * 2})
    result = estimate_savings(all_az)
    assert result["n_out_of_state"] == 0
    assert result["estimated_annual_savings"] == 0.0


def test_state_breakdown_excludes_az_and_sorts_desc(toy_active):
    breakdown = state_breakdown(toy_active)
    assert "AZ" not in breakdown["state"].values
    assert set(breakdown["state"]) == {"CA", "OR"}
    assert (breakdown["member_count"] == 1).all()


def test_sensitivity_table_covers_all_combinations(toy_active):
    sens = sensitivity_table(toy_active, cost_scenarios=[100.0, 200.0], holder_rate_scenarios=[0.5, 1.0])
    assert len(sens) == 4
    assert set(sens["pass_cost"]) == {100.0, 200.0}
    assert set(sens["holder_rate"]) == {0.5, 1.0}
    assert (sens["estimated_annual_savings"] >= 0).all()
