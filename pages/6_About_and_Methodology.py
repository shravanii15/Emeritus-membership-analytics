import streamlit as st

st.set_page_config(page_title="About & Methodology", page_icon="📖", layout="wide")
st.title("📖 About & Methodology")

st.markdown(
    """
## Project origin & data privacy

This project was built from the membership-management workflow of a real
university emeritus-faculty association. **All data used in this public
project is 100% synthetic**, generated with the Python `Faker` library and
seeded random sampling — no real member names, addresses, emails, or phone
numbers appear anywhere in this repository or dashboard.

The synthetic generator (`src/make_synthetic_data.py`) was deliberately
built to preserve the *statistical structure* of the real data (file
layout, row/column counts, realistic multi-source data-quality issues, and
genuine cross-file identity overlap so that record linkage is meaningful)
without exposing any private information. See `data/raw/README.md` for the
full data dictionary.

## Why this project

Built to demonstrate the breadth of skills a Master's-level Data Science
role in the US typically expects:

| Skill area | Where it shows up |
|---|---|
| **Data engineering** | Multi-source record linkage with fuzzy matching, handling inconsistent keys/formats |
| **EDA & data storytelling** | Interactive visual exploration across geography, tenure, discipline |
| **Supervised ML** | Binary classification (renewal/churn) with proper CV, multiple model comparison, threshold-aware evaluation |
| **Model interpretability** | SHAP global + local explanations |
| **Unsupervised ML** | K-Means segmentation with silhouette-based k selection, PCA visualization |
| **Decision science / OR** | Budget-constrained 0/1 knapsack optimization connecting the model to a real resource-allocation decision |
| **Software engineering** | Modular `src/` package, pytest test suite, CI via GitHub Actions, reproducible pipeline |
| **Deployment & communication** | Multi-page interactive Streamlit dashboard, deployed publicly, written for a non-technical stakeholder audience |

## Reproducing this project

```bash
git clone <this-repo-url>
cd emeritus-membership-analytics
pip install -r requirements.txt
python -m src.pipeline          # runs the full data → model → optimization pipeline
streamlit run app.py            # launches the dashboard locally
```

## Limitations & honest caveats

- The renewal signal in the synthetic data is intentionally modest and
  noisy (ROC-AUC ≈ 0.6–0.7) to reflect how real membership-renewal behavior
  actually looks — a perfect classifier here would be a red flag for data
  leakage, not a success.
- The outreach "call lift" curve (used by the budget optimizer) is an
  **assumed** behavioral model, not fit from a real randomized experiment —
  in a production setting this would ideally come from an A/B test of
  calling vs. not calling members at different risk levels.
- Discipline and location categories are drawn from a realistic but
  simplified reference list, not the full real set of academic departments.

## Contact

Built as a portfolio project. See the repository README for more.
"""
)
