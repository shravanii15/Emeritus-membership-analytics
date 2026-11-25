import pandas as pd
import plotly.express as px
import streamlit as st

from src.optimization import build_outreach_plan, sensitivity_analysis, MINUTES_PER_CALL
from src.ui_theme import inject_theme, page_nav, hero, read_more

st.set_page_config(page_title="Outreach Budget Optimizer", page_icon="📞", layout="wide")
inject_theme()
page_nav()
hero("📞", "Budget-Constrained Outreach Optimizer",
     "Given limited volunteer caller-hours, whom should the phone campaign call to maximize renewals recovered?")

st.warning(
    "⚠️ **The 'call lift' curve below is an assumption, not a measured effect** — there's no "
    "randomized experiment behind it. Use the sliders to stress-test whether 'optimization beats "
    "greedy' survives disagreement about that assumption (see the **Sensitivity analysis** tab)."
)

read_more(
    "ℹ️ The two strategies being compared (click to expand)",
    """
1. **Greedy (highest-risk-first)** — call whoever is least likely to renew.
2. **Expected-value-maximizing (0/1 knapsack)** — call members where a conversation has the
   highest *marginal* persuasive value — the "movable middle" who are genuinely undecided, rather
   than members who were always going to lapse (a call won't help) or always going to renew anyway
   (a call is wasted effort).
""",
)

try:
    scored = pd.read_csv("data/processed/scored_members.csv")
    features = pd.read_csv("data/processed/member_features.csv")
except FileNotFoundError:
    st.warning("Run `python -m src.pipeline` first.")
    st.stop()

merged = scored.merge(features, on="member_id", suffixes=("", "_feat"))

tab1, tab2 = st.tabs(["🎯 Interactive planner", "🧪 Sensitivity analysis"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        caller_hours = st.slider("Volunteer caller-hours available this cycle", 5, 100, 40, step=5)
    with c2:
        st.markdown("**Call-lift assumption** (adjust to test your own belief)")
    lc1, lc2, lc3 = st.columns(3)
    with lc1:
        peak = st.slider("Peak persuadability (predicted renewal probability)", 0.1, 0.9, 0.42, 0.01)
    with lc2:
        width = st.slider("Width of the 'movable middle'", 0.05, 0.4, 0.18, 0.01)
    with lc3:
        max_lift = st.slider("Maximum assumed lift from a call", 0.0, 0.6, 0.35, 0.01)

    capacity_calls = int(caller_hours * 60 / MINUTES_PER_CALL)
    st.caption(f"≈ {MINUTES_PER_CALL} minutes/call → **{capacity_calls} calls** possible with {caller_hours} caller-hours.")

    lift_params = {"peak": peak, "width": width, "max_lift": max_lift}
    result = build_outreach_plan(merged, caller_hours=caller_hours, lift_params=lift_params)
    summary = result["summary"]

    m1, m2, m3 = st.columns(3)
    m1.metric("Greedy strategy: expected renewals recovered", f"{summary['greedy_expected_renewals_recovered']:.1f}")
    m2.metric("Optimized strategy: expected renewals recovered", f"{summary['optimized_expected_renewals_recovered']:.1f}")
    m3.metric("Improvement from optimization", f"+{summary['improvement_pct']:.1f}%")

    plot_df = result["full_scored"][["predicted_renewal_probability", "call_lift"]].sort_values("predicted_renewal_probability")
    fig = px.line(plot_df, x="predicted_renewal_probability", y="call_lift",
                  title="Assumed persuasive impact of a call, by baseline renewal probability (your current assumption)",
                  labels={"predicted_renewal_probability": "Predicted renewal probability (no call)",
                          "call_lift": "Assumed probability a call changes the outcome"},
                  color_discrete_sequence=["#8C1D40"])
    fig.add_vrect(x0=max(peak - width, 0), x1=min(peak + width, 1), fillcolor="#FFC627", opacity=0.15, line_width=0,
                  annotation_text="The 'movable middle'", annotation_position="top left")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Optimized call list (top priority)")
    display_cols = ["member_id", "predicted_renewal_probability", "call_lift"]
    st.dataframe(
        result["optimized_plan"][display_cols].rename(columns={
            "predicted_renewal_probability": "Predicted renewal probability",
            "call_lift": "Expected marginal impact of a call",
        }).round(3),
        use_container_width=True,
        height=400,
    )

    csv = result["optimized_plan"][display_cols].to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download optimized call list (CSV)", data=csv, file_name="outreach_call_list.csv", mime="text/csv")

with tab2:
    st.markdown(
        """
        Instead of trusting one guessed lift curve, this re-runs the optimizer
        under four scenarios — a pessimistic case, the base case, an optimistic
        case, and a **null case where calls are assumed to have no effect at
        all** (a sanity check: optimization should show zero advantage here,
        since there's nothing to optimize for).
        """
    )
    sens = sensitivity_analysis(merged, caller_hours=40)
    display = sens[["scenario", "lift_peak", "lift_width", "lift_max",
                     "greedy_expected_renewals_recovered", "optimized_expected_renewals_recovered",
                     "improvement_pct"]].rename(columns={
        "lift_peak": "Peak", "lift_width": "Width", "lift_max": "Max lift",
        "greedy_expected_renewals_recovered": "Greedy (expected)",
        "optimized_expected_renewals_recovered": "Optimized (expected)",
        "improvement_pct": "Improvement %",
    })
    st.dataframe(display.round(2), use_container_width=True, hide_index=True)

    fig = px.bar(sens, x="scenario", y="improvement_pct",
                 title="Does optimization beat greedy under every assumption?",
                 labels={"scenario": "", "improvement_pct": "Improvement over greedy (%)"},
                 color_discrete_sequence=["#8C1D40"])
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f"""
**Finding:** the optimized (knapsack) strategy beats greedy highest-risk-first
targeting by **{sens[sens.scenario == 'Pessimistic (narrow, low lift)']['improvement_pct'].values[0]:.0f}–
{sens[sens.scenario == 'Optimistic (wide, high lift)']['improvement_pct'].values[0]:.0f}%**
across every non-null scenario tested, and correctly collapses to exactly
0% improvement under the null (no-effect) assumption — a reassuring sanity
check that the optimization logic isn't manufacturing an advantage out of
nothing. The conclusion "optimize, don't just call highest-risk-first" is
reasonably robust to disagreement about the exact shape of the persuasion
curve — but the *size* of the benefit is clearly sensitive to that
assumption, which is exactly why it's shown as a range here rather than a
single confident number.
"""
    )
