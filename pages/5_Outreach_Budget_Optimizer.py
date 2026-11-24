import pandas as pd
import plotly.express as px
import streamlit as st

from src.optimization import build_outreach_plan, MINUTES_PER_CALL

st.set_page_config(page_title="Outreach Budget Optimizer", page_icon="📞", layout="wide")
st.title("📞 Budget-Constrained Outreach Optimizer")

st.markdown(
    """
This page turns the renewal-risk model into an **actionable decision**: given
a limited number of volunteer caller-hours this cycle, whom should the phone
campaign call to maximize expected renewals recovered?

Two strategies are compared:
1. **Greedy (highest-risk-first)** — call whoever is least likely to renew.
2. **Expected-value-maximizing (0/1 knapsack)** — call members where a
   conversation has the highest *marginal* persuasive value — the
   "movable middle" who are genuinely undecided, rather than members who
   were always going to lapse (a call won't help) or always going to renew
   anyway (a call is wasted effort).
"""
)

try:
    scored = pd.read_csv("data/processed/scored_members.csv")
    features = pd.read_csv("data/processed/member_features.csv")
except FileNotFoundError:
    st.warning("Run `python -m src.pipeline` first.")
    st.stop()

merged = scored.merge(features, on="member_id", suffixes=("", "_feat"))

caller_hours = st.slider("Volunteer caller-hours available this cycle", 5, 100, 40, step=5)
capacity_calls = int(caller_hours * 60 / MINUTES_PER_CALL)
st.caption(f"≈ {MINUTES_PER_CALL} minutes/call → **{capacity_calls} calls** possible with {caller_hours} caller-hours.")

result = build_outreach_plan(merged, caller_hours=caller_hours)
summary = result["summary"]

c1, c2, c3 = st.columns(3)
c1.metric("Greedy strategy: expected renewals recovered", f"{summary['greedy_expected_renewals_recovered']:.1f}")
c2.metric("Optimized strategy: expected renewals recovered", f"{summary['optimized_expected_renewals_recovered']:.1f}")
c3.metric("Improvement from optimization", f"+{summary['improvement_pct']:.1f}%")

st.subheader("Expected call-lift by predicted renewal probability")
plot_df = result["full_scored"][["predicted_renewal_probability", "call_lift"]].sort_values("predicted_renewal_probability")
fig = px.line(plot_df, x="predicted_renewal_probability", y="call_lift",
              title="Assumed persuasive impact of a call, by baseline renewal probability",
              labels={"predicted_renewal_probability": "Predicted renewal probability (no call)",
                      "call_lift": "Assumed probability a call changes the outcome"},
              color_discrete_sequence=["#8C1D40"])
fig.add_vrect(x0=0.3, x1=0.55, fillcolor="#FFC627", opacity=0.15, line_width=0,
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
