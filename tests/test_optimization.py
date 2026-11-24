import numpy as np
import pandas as pd

from src.optimization import assumed_call_lift, knapsack_select, build_outreach_plan


def test_call_lift_peaks_in_mid_range():
    probs = np.array([0.0, 0.42, 1.0])
    lifts = assumed_call_lift(probs)
    assert lifts[1] > lifts[0]
    assert lifts[1] > lifts[2]


def test_call_lift_bounded_nonnegative():
    probs = np.linspace(0, 1, 50)
    lifts = assumed_call_lift(probs)
    assert (lifts >= 0).all()
    assert (lifts <= 0.35).all()


def test_knapsack_respects_capacity():
    values = np.array([5, 3, 8, 2, 9])
    costs = np.array([1, 1, 1, 1, 1])
    selected = knapsack_select(values, costs, capacity=2)
    assert selected.sum() == 2
    # Should pick the two highest-value items (indices 2 and 4)
    assert selected[2] and selected[4]


def test_knapsack_handles_zero_capacity():
    values = np.array([5, 3, 8])
    costs = np.array([1, 1, 1])
    selected = knapsack_select(values, costs, capacity=0)
    assert selected.sum() == 0


def test_build_outreach_plan_never_exceeds_capacity():
    df = pd.DataFrame({
        "member_id": range(100),
        "predicted_renewal_probability": np.random.uniform(0, 1, 100),
    })
    result = build_outreach_plan(df, caller_hours=5)
    capacity = result["summary"]["capacity_calls"]
    assert len(result["optimized_plan"]) <= capacity
    assert len(result["greedy_plan"]) <= capacity


def test_optimized_plan_is_at_least_as_good_as_greedy():
    df = pd.DataFrame({
        "member_id": range(200),
        "predicted_renewal_probability": np.random.uniform(0, 1, 200),
    })
    result = build_outreach_plan(df, caller_hours=20)
    assert (
        result["summary"]["optimized_expected_renewals_recovered"]
        >= result["summary"]["greedy_expected_renewals_recovered"] - 1e-6
    )
