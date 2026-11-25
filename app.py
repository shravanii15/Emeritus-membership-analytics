"""
Emeritus College Membership Analytics — Streamlit entry point.

Run locally with:
    streamlit run app.py
"""
import pandas as pd
import streamlit as st

from src.ui_theme import inject_theme, page_nav, hero, read_more

st.set_page_config(
    page_title="Emeritus College Membership Analytics",
    page_icon="🎓",
    layout="wide",
)
inject_theme()
page_nav()

hero("🎓", "Emeritus College Membership Analytics",
     "A retiree association's membership data lived in 3 messy spreadsheets with no shared ID. This rebuilds it into one clean dataset and a decision-support tool on top of it.")

st.markdown(
    """
> **In plain English:** a university's emeritus (retired-faculty) association kept its member
> list, dues payments, and subscription records in three separate spreadsheets that didn't line
> up — duplicate people, inconsistent IDs, typos. Nobody could reliably answer basic questions
> like *"who's at risk of not renewing this year?"* or *"are we spending money on benefits
> people don't use?"*
>
> This project (1) **cleans and merges** those three files into one trustworthy member record,
> then uses that clean data to (2) **predict who's likely to lapse**, (3) **group members into
> types** for more relevant outreach, (4) **decide who a limited volunteer phone team should
> call** to get the most renewals for the hours available, and (5) **check whether a specific
> cost — parking passes for out-of-state members — is being spent well**. Everything below is
> built on fake (synthetic) member data, so it can be shown publicly.
"""
)

try:
    members = pd.read_csv("data/processed/members_clean.csv")
    features = pd.read_csv("data/processed/member_features.csv")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Records linked", f"{len(members):,}", help="Rows reconciled across the 3 source spreadsheets")
    m2.metric("Active members modeled", f"{len(features):,}", help="Living, active members used for prediction/segmentation")
    m3.metric("Renewal rate", f"{features['renewed'].mean():.0%}", help="Share of active members who paid dues or hold a current subscription this cycle")
    m4.metric("Data sources merged", "3", help="Member roster, dues ledger, EV-subscription ledger")
except FileNotFoundError:
    st.warning("Run `python -m src.pipeline` first to generate the processed data files.")

st.markdown("### Explore each step (in the order the story above follows)")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.page_link("pages/1_Data_Quality_and_Linkage.py", label="🧹 **Data Quality & Linkage**", help="How 3 messy spreadsheets get joined")
    st.page_link("pages/2_Exploratory_Data_Analysis.py", label="📊 **Exploratory Data Analysis**", help="Renewal patterns by geography, tenure, discipline")
with c2:
    st.page_link("pages/3_Renewal_Prediction_Model.py", label="🤖 **Renewal Prediction Model**", help="LogReg/RF/XGBoost + SHAP, with honest confidence intervals")
    st.page_link("pages/4_Member_Segmentation.py", label="🧩 **Member Segmentation**", help="K-Means clusters for targeted outreach")
with c3:
    st.page_link("pages/5_Outreach_Budget_Optimizer.py", label="📞 **Outreach Budget Optimizer**", help="Interactive budget-constrained call-list optimizer")
    st.page_link("pages/8_Parking_Budget_Analysis.py", label="🅿️ **Parking Budget Analysis**", help="Out-of-state parking savings estimate + sensitivity analysis")
with c4:
    st.page_link("pages/7_Linkage_Robustness_Experiment.py", label="🔬 **Linkage Robustness Experiment**", help="The project's research question")
    st.page_link("pages/6_About_and_Methodology.py", label="📖 **About & Methodology**", help="Data privacy, skills map, limitations")

read_more(
    "ℹ️ About this project & data privacy (click to expand)",
    """
**All data on this site is synthetically generated** (via the `Faker` library) and does not
represent any real person. It mirrors the structure and statistical texture of a real university
emeritus-association membership database — inconsistent IDs, mixed zip formats, free-text notes
sitting in data fields — without exposing any private information. See the **About & Methodology**
page for the full writeup, and the **Linkage Robustness Experiment** page for the project's one
genuinely falsifiable research finding (independent of the data being synthetic).

**What each stage demonstrates:**
- 🧹 **Data engineering** — fuzzy record linkage across 3 inconsistent spreadsheets
- 🤖 **Machine learning** — renewal/churn classification with SHAP interpretability, honestly reported with confidence intervals
- 🧩 **Unsupervised learning** — K-Means member segmentation
- 📞 **Decision science** — a budget-constrained 0/1 knapsack outreach optimizer, stress-tested with a sensitivity analysis
- 🅿️ **Budget-impact analysis** — an out-of-state parking-pass savings estimate, framed as a sensitivity range rather than a single confident number
- 🔬 **Research methodology** — a controlled experiment on record-linkage robustness under data-entry noise
""",
)

st.caption("Source code: see the GitHub repository linked in the About page for the data dictionary and reproduction steps.")
