from src.linkage_robustness import corrupt_string, corrupt_zip, run_experiment


def test_zero_corruption_is_identity():
    assert corrupt_string("smith", 0.0) == "smith"
    assert corrupt_zip("85287", 0.0) == "85287"


def test_corruption_changes_string_with_high_probability():
    # With p=1.0 every character position is touched, so the result should
    # (almost certainly) differ from the input for a non-trivial string.
    original = "worthington"
    corrupted = corrupt_string(original, 1.0)
    assert corrupted != original


def test_zip_corruption_preserves_length_when_no_dup_or_del():
    # corrupt_zip only substitutes digits in place, so length is always preserved.
    assert len(corrupt_zip("85287", 0.5)) == 5


def test_experiment_runs_and_recall_decreases_with_corruption():
    df = run_experiment()
    strategy_a = df[df["strategy"].str.startswith("A")].groupby("corruption_level")["recall"].mean()
    levels = sorted(strategy_a.index)
    # Recall at zero corruption should be perfect; recall at the highest
    # corruption level should be far lower.
    assert strategy_a[levels[0]] == 1.0
    assert strategy_a[levels[-1]] < strategy_a[levels[0]]


def test_false_match_rate_stays_low_due_to_zip_gate():
    df = run_experiment()
    # Both strategies require some form of zip agreement, so false matches
    # (linking to the WRONG identity) should remain rare even under noise.
    assert df["false_match_rate"].mean() < 0.05
