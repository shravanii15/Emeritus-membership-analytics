# 🎓 Emeritus College Membership Analytics

An end-to-end data science project — record linkage, exploratory analysis,
renewal/churn prediction with model interpretability, member segmentation,
and a budget-constrained outreach optimizer — built on fully **synthetic**
data and deployed as an interactive Streamlit dashboard.

**📄 Case study (start here):** https://shravanii15.github.io/Emeritus-membership-analytics/ _(enable via repo Settings → Pages → source: `docs/` on `main` — see below)_
**🔗 Live technical dashboard:** _add your Streamlit Community Cloud URL here after deploying_
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
| 💰 Decision analysis | `src/parking_analysis.py` | Budget-impact estimate (out-of-state parking-pass cost) framed as an assumption-driven range with sensitivity analysis, not a single confident number |
| 📊 Deployment | `app.py` + `pages/` | 8-page interactive Streamlit dashboard |
| ✅ Quality | `tests/` + `.github/workflows/tests.yml` | 44 pytest unit tests, run automatically in CI on every push |

## Dashboard pages

1. **Home** — project overview & headline metrics
2. **Data Quality & Linkage** — the record-linkage problem and how it's solved
3. **Exploratory Data Analysis** — renewal patterns, geography, tenure, discipline
4. **Renewal Prediction Model** — model comparison, ROC/PR curves, confusion matrix, global + per-member SHAP explanations
5. **Member Segmentation** — interactive cluster map with suggested outreach messaging per segment
6. **Outreach Budget Optimizer** — interactive caller-hours and lift-assumption sliders; compares a greedy vs. expected-value-maximizing call list, plus a sensitivity analysis across four behavioral-response scenarios, with CSV export
7. **About & Methodology** — data privacy notes, skills map, limitations and honest caveats
8. **Linkage Robustness Experiment** — the project's research question: a controlled test of how two record-linkage strategies hold up under increasing data-entry noise
9. **Parking Budget Analysis** — estimates the savings available by not issuing campus parking passes to out-of-state members, with a state-level breakdown, a live sensitivity analysis over the two driving assumptions, and a worked case study (problem / constraints / decisions / limitations)

## Architecture

```
┌─────────────────────┐     ┌──────────────────────┐     ┌────────────────────┐
│   data/raw/*.xlsx    │     │  src/data_cleaning.py │     │  src/features.py   │
│  3 inconsistent      │ ──▶ │  fuzzy name+zip key   │ ──▶ │  tenure, recency,  │
│  synthetic sheets    │     │  linkage + repair     │     │  engagement, target│
└─────────────────────┘     └──────────────────────┘     └──────────┬─────────┘
                                                                      │
                     ┌────────────────────────────────────────────────┼─────────────────────────────┐
                     ▼                                                ▼                             ▼
          ┌─────────────────────┐                         ┌─────────────────────┐      ┌─────────────────────┐
          │  src/modeling.py    │                         │ src/segmentation.py │      │ src/optimization.py │
          │  LR/RF/XGB + SHAP + │                         │  K-Means + PCA      │      │  knapsack outreach + │
          │  bootstrap CIs      │                         │  (silhouette k)     │      │  sensitivity analysis│
          └──────────┬──────────┘                         └──────────┬──────────┘      └──────────┬──────────┘
                     │                                                │                            │
                     └────────────────────────┬───────────────────────┴──────────────┬─────────────┘
                                               ▼                                      ▼
                                 ┌─────────────────────────┐         ┌─────────────────────────────┐
                                 │  src/linkage_robustness  │         │  src/parking_analysis.py     │
                                 │  .py — controlled noise  │         │  — assumption-driven budget  │
                                 │  experiment (research Q)  │         │  impact estimate + sensitivity│
                                 └────────────┬─────────────┘         └───────────────┬─────────────┘
                                              └──────────────────┬───────────────────┘
                                                                  ▼
                                                   ┌───────────────────────────┐
                                                   │  app.py + pages/*.py      │
                                                   │  9-page Streamlit app     │
                                                   │  (deployed publicly)      │
                                                   └───────────────────────────┘
```

`src/pipeline.py` runs every stage above, in order, in one command.

## Case-study landing page

`docs/index.html` is a single, polished, non-technical case-study page — the thing to link from
your résumé or LinkedIn, instead of the Streamlit app directly. It leads with the plain-English
problem, headline numbers, and a short case study, with a button through to the full technical
dashboard for anyone who wants to dig in.

**To publish it for free on GitHub Pages:**
1. Push this repo (already done if you're reading this on GitHub).
2. Go to the repo's **Settings → Pages**.
3. Under "Build and deployment", set **Source: Deploy from a branch**, **Branch: `main`**,
   **Folder: `/docs`**. Save.
4. GitHub gives you a URL like `https://<username>.github.io/<repo-name>/` within a minute or two.
5. Open `docs/index.html`, find the two `share.streamlit.io` placeholder links, and replace them
   with your actual deployed Streamlit URL once you have it.

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
│   ├── parking_analysis.py     # Budget-impact estimate + sensitivity analysis
│   └── pipeline.py             # Orchestrates the full pipeline end to end
├── tests/                      # pytest unit tests
├── data/
│   ├── raw/                    # Synthetic source spreadsheets + data dictionary
│   └── processed/              # Generated: cleaned data, model artifacts, plans
├── notebooks/                  # Executed EDA + modeling walkthrough notebook
├── docs/index.html             # Polished case-study landing page (publish via GitHub Pages)
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

## Case study: Emeritus College Membership Analytics

**Problem.** Years of fragmented membership and dues spreadsheets meant the
association had no reliable, unified view of who was an active member, let
alone whether a cost like campus parking passes was being allocated to
people unlikely to use it.

**Constraints.** The underlying data is sensitive (names, addresses,
payment records), and the people who needed the answer — membership-office
staff — are not technical users. Any tool built on it had to protect
member privacy by design and be usable without a data background.

**Decisions.**
- Two-tier record linkage (exact-ID match first, normalized name+zip match
  second) to reconcile three separately maintained spreadsheets into one
  clean member record, instead of requiring a shared ID that didn't exist.
- This public version is rebuilt entirely on **synthetic data** — generated
  to preserve the real statistical structure (file layout, data-quality
  issues, cross-file identity overlap) without exposing any real member's
  information — so the method and code can be shared publicly.
- Dollar estimates (the parking-budget analysis) are presented as a
  **range under stated assumptions**, not a single number, because the
  source data can't actually confirm usage — an honest, audit-first
  recommendation is more credible than false precision.
- A model-interpretability layer (SHAP) and bootstrap confidence intervals
  on model comparison, so performance claims are never overstated.

**Results.** _Replace this paragraph with the specific numbers already on
your résumé from the real engagement — members reconciled, discrepancies
found, dollars identified, hours saved. This synthetic rebuild intentionally
doesn't invent those numbers._

**What I'd do differently.** Capture actual benefit-usage data (e.g. parking
pass swipes) at intake, so budget-impact analyses like the parking one are
measured rather than estimated — the single biggest limitation called out
on the **About & Methodology** and **Parking Budget Analysis** pages.

## License

MIT — see [LICENSE](LICENSE).
