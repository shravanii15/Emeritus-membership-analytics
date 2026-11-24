import json

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Member Segmentation", page_icon="🧩", layout="wide")
st.title("🧩 Member Segmentation (Unsupervised Learning)")

try:
    segments = pd.read_csv("data/processed/member_segments.csv")
    with open("data/processed/segmentation_summary.json") as f:
        summary = json.load(f)
except FileNotFoundError:
    st.warning("Run `python -m src.pipeline` first.")
    st.stop()

st.markdown(
    f"""
K-Means clustering groups active members into behaviorally distinct
segments for targeted outreach messaging, independent of (and complementary
to) the renewal-prediction model. The number of clusters (**k = {summary['chosen_k']}**)
was chosen by maximizing silhouette score across k = 2–5.
"""
)

c1, c2 = st.columns([2, 1])
with c1:
    fig = px.scatter(
        segments, x="pca_x", y="pca_y", color="cluster_name",
        title="Member segments (PCA-reduced to 2D)",
        labels={"pca_x": "Principal Component 1", "pca_y": "Principal Component 2"},
        color_discrete_sequence=px.colors.qualitative.Safe,
        hover_data=["member_id", "engagement_score", "years_since_retirement"],
    )
    st.plotly_chart(fig, use_container_width=True)
with c2:
    st.markdown("**Silhouette score by k**")
    sil_df = pd.DataFrame(list(summary["silhouette_scores"].items()), columns=["k", "silhouette"])
    fig2 = px.line(sil_df, x="k", y="silhouette", markers=True)
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(
        f"PCA explains {sum(summary['explained_variance_ratio']):.0%} of variance "
        f"in the first 2 components."
    )

st.subheader("Segment profiles")
profile_cols = ["engagement_score", "years_since_retirement", "is_in_state",
                 "opted_out_of_email", "has_paid_dues_this_cycle", "has_ev_subscription"]
profile = segments.groupby("cluster_name")[profile_cols].mean().round(2)
profile["size"] = segments["cluster_name"].value_counts()
st.dataframe(profile, use_container_width=True)

st.subheader("Suggested outreach message per segment")
descriptions = {
    "Engaged": "Thank-you / ambassador messaging — ask them to refer a colleague.",
    "Disengaged": "Win-back campaign — personal phone call, remind them of EC benefits.",
}
for name in segments["cluster_name"].unique():
    tag = "Engaged" if "Engaged" in name else "Disengaged"
    st.markdown(f"- **{name}**: {descriptions[tag]}")
