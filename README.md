# 🎓 Emeritus College Membership Analytics

An end-to-end data science project — record linkage, exploratory analysis,
renewal/churn prediction with model interpretability, member segmentation,
and a budget-constrained outreach optimizer — built on fully **synthetic**
data and deployed as an interactive Streamlit dashboard.

**🔗 Live demo:** _add your Streamlit Community Cloud URL here after deploying_
**📂 Repo:** https://github.com/shravanii15/Emeritus-membership-analytics

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Tests](https://github.com/shravanii15/Emeritus-membership-analytics/actions/workflows/tests.yml/badge.svg)
![License](https://img.shields.io/badge/license-MIT-green)

## Research question

Most of this project is a methods *demonstration* (the data is synthetic,
so there's nothing real to "discover" about emeritus faculty). But one
page asks — and answers — a genuine, falsifiable question that doesn't
depend on the data being real:

> **How robust is fuzzy name+zip record linkage to data-entry noise, and
> is there a linkage strategy that degrades more gracefully?**

`src/linkage_robustness.py` runs a controlled experiment: synthetic
identities are corrupted with realistic typos/transpositions at increasing
rates, and two linkage strategies (the exact-match key actually used
elsewhere in this project vs. a zip-gated similarity-threshold alternative)
are compared on recall and false-match rate. The finding — exact-match
recall collapses to ~43% at just 5% character corruption while the
alternative holds up far better, with **no measurable increase in false
matches for either strategy** — is a real, reproducible, and actionable
result about the *method*, independent of what the underlying data is. See
the **Linkage Robustness Experiment** page in the dashboard.

---

## Why this project exists

Most "membership org" style datasets you'll find online are either toy
(Iris, Titanic) or came pre-cleaned. Real membership data — the kind an
actual association office keeps — looks nothing like that: it is scattered
across several spreadsheets with no shared ID, inconsistent formats, and
administrative notes typed directly into data fields. This project
simulates that reality end to end, from messy raw files to a deployed
decision-support tool, using **synthetic data generated to preserve the
real statistical structure of a university emeritus-faculty association's
membership records — without exposing any private information.**

> ⚠️ **Data privacy note:** every row in `data/raw/` is synthetically
> generated (see `data/raw/README.md`). No real member's name, address,
> email, or phone number appears anywhere in this repository.

## What's inside

| Stage | Module | What it demonstrates |
|---|---|---|
| 🧹 Data engineering | `src/data_cleaning.py` | Record linkage across 3 inconsistent spreadsheets via a fuzzy name+zip key; data-quality detection and repair |
| 🏗️ Feature engineering | `src/features.py` | Tenure/recency features, engagement scoring, renewal-target construction |
| 🤖 Supervised ML | `src/modeling.py` | Logistic Regression / Random Forest / XGBoost comparison, stratified CV, ROC-AUC & PR-AUC, SHAP interpretability |
| 🧩 Unsupervised ML | `src/segmentation.py` | K-Means segmentation with silhouette-based k selection, PCA visualization |
| 📞 Decision science | `src/optimization.py` | Budget-constrained 0/1 knapsack outreach optimizer, with a sensitivity analysis over the assumed behavioral response instead of treating it as fact |
| 🔬 Methods research | `src/linkage_robustness.py` | A controlled experiment on linkage-strategy robustness under data-entry noise — a real, falsifiable finding independent of the synthetic data |
| 📊 Deployment | `app.py` + `pages/` | 7-page interactive Streamlit dashboard |
| ✅ Quality | `tests/` + `.github/workflows/tests.yml` | 34 pytest unit tests, run automatically in CI on every push |

## Dashboard pages

1. **Home** — project overview & headline metrics
2. **Data Quality & Linkage** — the record-linkage problem and how it's solved
3. **Exploratory Data Analysis** — renewal patterns, geography, tenure, discipline
4. **Renewal Prediction Model** — model comparison, ROC/PR curves, confusion matrix, global + per-member SHAP explanations
5. **Member Segmentation** — interactive cluster map with suggested outreach messaging per segment
6. **Outreach Budget Optimizer** — interactive caller-hours and lift-assumption sliders; compares a greedy vs. expected-value-maximizing call list, plus a sensitivity analysis across four behavioral-response scenarios, with CSV export
7. **About & Methodology** — data privacy notes, skills map, limitations and honest caveats
8. **Linkage Robustness Experiment** — the project's research question: a controlled test of how two record-linkage strategies hold up under increasing data-entry noise

## Quickstart

```bash
git clone https://github.com/shravanii15/Emeritus-membership-analytics.git
cd Emeritus-membership-analytics
pip install -r requirements.txt

# Regenerate the processed data / model artifacts from the raw synthetic files
python -m src.pipeline

# Launch the dashboard
streamlit run app.py
```

Run the test suite:

```bash
pytest tests/ -v
```

## Project structure

```
.
├── app.py                      # Streamlit entry point
├── pages/                      # Streamlit multipage app
├── src/
│   ├── make_synthetic_data.py  # Generates the synthetic raw data
│   ├── data_cleaning.py        # Record linkage & data-quality repair
│   ├── features.py             # Feature engineering & target construction
│   ├── modeling.py             # Supervised renewal/churn model + SHAP
│   ├── segmentation.py         # K-Means member segmentation
│   ├── optimization.py         # Budget-constrained outreach optimizer
│   ├── linkage_robustness.py   # Research question: linkage strategy robustness experiment
│   └── pipeline.py             # Orchestrates the full pipeline end to end
├── tests/                      # pytest unit tests
├── data/
│   ├── raw/                    # Synthetic source spreadsheets + data dictionary
│   └── processed/              # Generated: cleaned data, model artifacts, plans
├── notebooks/                  # Executed EDA + modeling walkthrough notebook
├── .github/workflows/tests.yml # CI: runs the pipeline + full test suite on every push
└── requirements.txt
```

## Methodology notes & honest limitations

- The renewal signal is deliberately modest (ROC-AUC ≈ 0.6–0.7), mirroring
  how real membership-renewal behavior looks: useful but imperfect signal,
  not a deterministic outcome. See the **About & Methodology** page for the
  full discussion, including why a *higher* AUC here would actually be a
  red flag for data leakage.
- The outreach "call lift" curve is an assumed behavioral model for
  demonstration purposes, not fit from a real randomized experiment.

## License

MIT — see [LICENSE](LICENSE).
