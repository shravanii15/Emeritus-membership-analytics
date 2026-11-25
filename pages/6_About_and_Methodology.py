import streamlit as st

from src.ui_theme import inject_theme, page_nav, hero

st.set_page_config(page_title="About & Methodology", page_icon="📖", layout="wide")
inject_theme()
page_nav()
hero("📖", "About & Methodology", "Data privacy, what this project demonstrates, how to reproduce it, and its limitations.")

tab1, tab2, tab3, tab4 = st.tabs(["🔒 Data & Privacy", "🎯 Skills Map", "⚙️ Reproduce", "⚠️ Limitations"])

with tab1:
    st.markdown(
        """
This project was built from the membership-management workflow of a real university
emeritus-faculty association. **All data used in this public project is 100% synthetic**,
generated with the Python `Faker` library and seeded random sampling — no real member names,
addresses, emails, or phone numbers appear anywhere in this repository or dashboard.

The synthetic generator (`src/make_synthetic_data.py`) was deliberately built to preserve the
*statistical structure* of the real data (file layout, row/column counts, realistic multi-source
data-quality issues, and genuine cross-file identity overlap so that record linkage is meaningful)
without exposing any private information. See `data/raw/README.md` for the full data dictionary.
"""
    )

with tab2:
    st.markdown("Built to demonstrate the breadth of skills a Master's-level Data Science role in the US typically expects:")
    st.table({
        "Skill area": ["Data engineering", "EDA & storytelling", "Supervised ML", "Model interpretability",
                       "Unsupervised ML", "Decision science / OR", "Research methodology",
                       "Software engineering", "Deployment & communication"],
        "Where it shows up": [
            "Multi-source record linkage with fuzzy matching, handling inconsistent keys/formats",
            "Interactive visual exploration across geography, tenure, discipline",
            "Binary classification with proper CV, multi-model comparison, bootstrap confidence intervals",
            "SHAP global + local explanations, with readable feature labels",
            "K-Means segmentation with silhouette-based k selection, PCA visualization",
            "Budget-constrained 0/1 knapsack optimization + a sensitivity analysis over its core assumption",
            "A controlled experiment on linkage-strategy robustness — a real, falsifiable finding",
            "Modular src/ package, pytest suite (39 tests), CI via GitHub Actions, reproducible pipeline",
            "Multi-page interactive Streamlit dashboard, deployed publicly",
        ],
    })

with tab3:
    st.code(
        """git clone https://github.com/shravanii15/Emeritus-membership-analytics.git
cd Emeritus-membership-analytics
pip install -r requirements.txt
python -m src.pipeline          # runs the full data -> model -> optimization pipeline
streamlit run app.py            # launches the dashboard locally""",
        language="bash",
    )

with tab4:
    st.markdown(
        """
- The renewal signal in the synthetic data is intentionally modest and noisy (ROC-AUC ≈ 0.5–0.7,
  with overlapping confidence intervals across models) to reflect how real membership-renewal
  behavior actually looks — a perfect classifier here would be a red flag for data leakage, not a
  success.
- The outreach "call lift" curve (used by the budget optimizer) is an **assumed** behavioral
  model, not fit from a real randomized experiment — stress-tested via sensitivity analysis on
  that page rather than presented as fact.
- Discipline and location categories are drawn from a realistic but simplified reference list, not
  the full real set of academic departments.
- The one claim in this project that *is* a real, reproducible finding independent of the
  synthetic data — see the **Linkage Robustness Experiment** page.
"""
    )

st.caption("Built as a portfolio project. See the repository README for more.")
