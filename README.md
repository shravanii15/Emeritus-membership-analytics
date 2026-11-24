# 🎓 Emeritus College Membership Analytics

An end-to-end data science project — record linkage, exploratory analysis,
renewal/churn prediction with model interpretability, member segmentation,
and a budget-constrained outreach optimizer — built on fully **synthetic**
data and deployed as an interactive Streamlit dashboard.

**🔗 Live demo:** _add your Streamlit Community Cloud URL here after deploying_

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Tests](https://github.com/<your-username>/<your-repo>/actions/workflows/tests.yml/badge.svg)
![License](https://img.shields.io/badge/license-MIT-green)

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
| 📞 Decision science | `src/optimization.py` | Budget-constrained 0/1 knapsack outreach optimizer — connects the model to an actual resource-allocation decision |
| 📊 Deployment | `app.py` + `pages/` | 6-page interactive Streamlit dashboard |
| ✅ Quality | `tests/` + `.github/workflows/tests.yml` | 34 pytest unit tests, run automatically in CI on every push |

## Dashboard pages

1. **Home** — project overview & headline metrics
2. **Data Quality & Linkage** — the record-linkage problem and how it's solved
3. **Exploratory Data Analysis** — renewal patterns, geography, tenure, discipline
4. **Renewal Prediction Model** — model comparison, ROC/PR curves, confusion matrix, global + per-member SHAP explanations
5. **Member Segmentation** — interactive cluster map with suggested outreach messaging per segment
6. **Outreach Budget Optimizer** — interactive caller-hours slider; compares a greedy vs. expected-value-maximizing call list, with CSV export
7. **About & Methodology** — data privacy notes, skills map, limitations and honest caveats

## Quickstart

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
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
