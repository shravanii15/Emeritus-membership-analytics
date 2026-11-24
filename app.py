"""
Emeritus College Membership Analytics — Streamlit entry point.

A multi-page portfolio dashboard built on fully synthetic data (see
data/raw/README.md). Run locally with:

    streamlit run app.py
"""
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Emeritus College Membership Analytics",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 Emeritus College Membership Analytics")
st.caption(
    "An end-to-end data science project: record linkage, EDA, renewal-prediction "
    "modeling with SHAP interpretability, member segmentation, and a budget-"
    "constrained outreach optimizer — built entirely on **synthetic** data."
)

st.info(
    "⚠️ **All data on this site is synthetically generated** (via the `Faker` "
    "library) and does not represent any real person. It was built to mirror "
    "the structure and statistical texture of a real university emeritus-"
    "association membership database, without exposing any private member "
    "information. See the **About & Methodology** page for details.",
    icon="🔒",
)

st.markdown("### 👈 Use the sidebar to explore each stage of the project")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        "#### 🧹 Data Engineering\n"
        "- Record linkage across 3 messy source spreadsheets\n"
        "- Fuzzy name + zip matching\n"
        "- Data-quality auditing"
    )
with col2:
    st.markdown(
        "#### 🤖 Machine Learning\n"
        "- Renewal/churn classification (LogReg, RF, XGBoost)\n"
        "- SHAP model interpretability\n"
        "- K-Means member segmentation"
    )
with col3:
    st.markdown(
        "#### 📊 Decision Science\n"
        "- Budget-constrained outreach optimizer (0/1 knapsack)\n"
        "- Expected-value vs. greedy targeting comparison\n"
        "- Interactive what-if budget slider"
    )

st.divider()

try:
    members = pd.read_csv("data/processed/members_clean.csv")
    features = pd.read_csv("data/processed/member_features.csv")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total records linked", f"{len(members):,}")
    m2.metric("Active members modeled", f"{len(features):,}")
    m3.metric("Renewal rate (this cycle)", f"{features['renewed'].mean():.0%}")
    m4.metric("Data sources merged", "3")
except FileNotFoundError:
    st.warning("Run `python -m src.pipeline` first to generate the processed data files.")

st.divider()
st.markdown(
    "**Source code:** this project's GitHub repository (see README for data "
    "dictionary, methodology, and reproduction steps)."
)
