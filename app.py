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
     "Record linkage → ML renewal prediction → segmentation → budget optimization — on synthetic data.")

try:
    members = pd.read_csv("data/processed/members_clean.csv")
    features = pd.read_csv("data/processed/member_features.csv")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Records linked", f"{len(members):,}")
    m2.metric("Active members modeled", f"{len(features):,}")
    m3.metric("Renewal rate", f"{features['renewed'].mean():.0%}")
    m4.metric("Data sources merged", "3")
except FileNotFoundError:
    st.warning("Run `python -m src.pipeline` first to generate the processed data files.")

st.markdown("### Jump to a stage")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.page_link("pages/1_Data_Quality_and_Linkage.py", label="🧹 **Data Quality & Linkage**", help="How 3 messy spreadsheets get joined")
    st.page_link("pages/2_Exploratory_Data_Analysis.py", label="📊 **Exploratory Data Analysis**", help="Renewal patterns by geography, tenure, discipline")
with c2:
    st.page_link("pages/3_Renewal_Prediction_Model.py", label="🤖 **Renewal Prediction Model**", help="LogReg/RF/XGBoost + SHAP, with honest confidence intervals")
    st.page_link("pages/4_Member_Segmentation.py", label="🧩 **Member Segmentation**", help="K-Means clusters for targeted outreach")
with c3:
    st.page_link("pages/5_Outreach_Budget_Optimizer.py", label="📞 **Outreach Budget Optimizer**", help="Interactive budget-constrained call-list optimizer")
    st.page_link("pages/7_Linkage_Robustness_Experiment.py", label="🔬 **Linkage Robustness Experiment**", help="The project's research question")
with c4:
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
- 🔬 **Research methodology** — a controlled experiment on record-linkage robustness under data-entry noise
""",
)

st.caption("Source code: see the GitHub repository linked in the About page for the data dictionary and reproduction steps.")
