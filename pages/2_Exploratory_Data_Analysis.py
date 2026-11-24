import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Exploratory Data Analysis", page_icon="📊", layout="wide")
st.title("📊 Exploratory Data Analysis")

try:
    features = pd.read_csv("data/processed/member_features.csv")
    clean = pd.read_csv("data/processed/members_clean.csv")
except FileNotFoundError:
    st.warning("Run `python -m src.pipeline` first.")
    st.stop()

st.markdown("Exploring the **active member population** (n = {}) that the renewal model is trained on.".format(len(features)))

tab1, tab2, tab3, tab4 = st.tabs(["Renewal patterns", "Geography", "Tenure", "Discipline"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        renewal_by_state = features.copy()
        renewal_by_state["location"] = renewal_by_state["is_in_state"].map({1: "In-state (AZ)", 0: "Out-of-state"})
        grp = renewal_by_state.groupby("location")["renewed"].mean().reset_index()
        fig = px.bar(grp, x="location", y="renewed", title="Renewal rate: in-state vs. out-of-state",
                     color="location", color_discrete_sequence=px.colors.qualitative.Safe)
        fig.update_yaxes(tickformat=".0%", title="Renewal rate")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        grp2 = features.groupby("opted_out_of_email")["renewed"].mean().reset_index()
        grp2["opted_out_of_email"] = grp2["opted_out_of_email"].map({0: "Subscribed", 1: "Opted out"})
        fig2 = px.bar(grp2, x="opted_out_of_email", y="renewed", title="Renewal rate: email opt-out status",
                      color="opted_out_of_email", color_discrete_sequence=px.colors.qualitative.Safe)
        fig2.update_yaxes(tickformat=".0%", title="Renewal rate")
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

with tab2:
    city_counts = clean[clean["source_status"] == "active"]["city"].value_counts().nlargest(12).reset_index()
    city_counts.columns = ["city", "count"]
    fig = px.bar(city_counts, x="count", y="city", orientation="h",
                 title="Top cities among active members", color_discrete_sequence=["#8C1D40"])
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    fig = px.histogram(features, x="years_since_retirement", nbins=30,
                        title="Distribution: years since retirement", color_discrete_sequence=["#8C1D40"])
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.scatter(features, x="years_since_retirement", y="engagement_score",
                       color="renewed", title="Engagement score vs. years since retirement, by renewal outcome",
                       labels={"renewed": "Renewed"}, color_continuous_scale="RdYlGn",
                       opacity=0.6)
    st.plotly_chart(fig2, use_container_width=True)

with tab4:
    disc_counts = features["discipline_bucketed"].value_counts().reset_index()
    disc_counts.columns = ["discipline", "count"]
    fig = px.pie(disc_counts, names="discipline", values="count", title="Discipline mix (top 10 + Other)")
    st.plotly_chart(fig, use_container_width=True)

    disc_renewal = features.groupby("discipline_bucketed")["renewed"].mean().sort_values(ascending=False).reset_index()
    fig2 = px.bar(disc_renewal, x="renewed", y="discipline_bucketed", orientation="h",
                  title="Renewal rate by discipline", color_discrete_sequence=["#FFC627"])
    fig2.update_xaxes(tickformat=".0%")
    fig2.update_layout(yaxis={"categoryorder": "total ascending"}, yaxis_title="")
    st.plotly_chart(fig2, use_container_width=True)
