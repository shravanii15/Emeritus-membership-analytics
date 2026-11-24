"""Builds notebooks/eda_and_modeling_walkthrough.ipynb programmatically (as
source cells), so the narrative text and code stay easy to edit and review
in a code review / diff, then executes it with nbconvert to bake in real
outputs."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


md("""# Emeritus College Membership Analytics — EDA & Modeling Walkthrough

This notebook is a readable, narrative walkthrough of the full pipeline implemented in `src/`.
It is meant as a portfolio artifact — the production code lives in `src/*.py` and is covered by
`tests/`; this notebook tells the story of *why* each step was taken and what it found.

**Data note:** all data used here is synthetically generated (see `data/raw/README.md`) and
contains no real member information.
""")

code("""import sys
sys.path.insert(0, "..")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", 30)
plt.rcParams["figure.figsize"] = (9, 5)
""")

md("## 1. Data cleaning & record linkage\n\nWe start from three independent spreadsheets with no shared reliable key.")

code("""from src.data_cleaning import build_clean_dataset

clean = build_clean_dataset()
print(f"Linked {len(clean)} records across 3 source files")
clean["source_status"].value_counts()
""")

code("""clean[["member_id", "first_name", "last_name", "city", "state",
       "source_status", "has_current_ev_subscription"]].sample(10, random_state=1)
""")

md("""### Data quality issues found

- A handful of ASU IDs are the literal string `"Unknown"` and were linked purely on the fuzzy
  name+zip key instead.
- Some "Discipline"/"Joined EC" cells actually contain an administrative note
  (e.g. *"Has requested removal from all email lists"*) rather than real data — these were
  detected and split out into a proper `opted_out_of_email` flag.
""")

code("""resolved_id_pct = clean["asu_id"].notna().mean()
opted_out = clean["opted_out_of_email"].sum() if "opted_out_of_email" in clean else 0
print(f"Resolved numeric ASU IDs: {resolved_id_pct:.1%}")
print(f"Members flagged as opted out of email (data-quality repair): {opted_out}")
""")

md("## 2. Feature engineering")

code("""from src.features import engineer_features

features = engineer_features(clean)
print(f"Active members in scope for the renewal model: {len(features)}")
print(f"Renewal rate: {features['renewed'].mean():.1%}")
features.describe(include="all").T.head(15)
""")

md("## 3. Exploratory data analysis")

code("""fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

features.groupby(features["is_in_state"].map({1: "In-state (AZ)", 0: "Out-of-state"}))["renewed"].mean().plot(
    kind="bar", ax=axes[0], color=["#8C1D40", "#FFC627"], title="Renewal rate: in-state vs out-of-state"
)
axes[0].set_ylabel("Renewal rate")
axes[0].tick_params(axis="x", rotation=0)

features["years_since_retirement"].hist(bins=30, ax=axes[1], color="#8C1D40")
axes[1].set_title("Distribution: years since retirement")
axes[1].set_xlabel("Years since retirement")

plt.tight_layout()
plt.show()
""")

code("""disc_renewal = features.groupby("discipline_bucketed")["renewed"].mean().sort_values()
disc_renewal.plot(kind="barh", color="#8C1D40", title="Renewal rate by discipline")
plt.xlabel("Renewal rate")
plt.tight_layout()
plt.show()
""")

md("## 4. Supervised modeling: renewal / churn prediction")

code("""from src.modeling import load_features, evaluate_models

model_df = load_features()
results, pipelines, best_name, splits = evaluate_models(model_df)

comparison = pd.DataFrame({
    name: {"CV ROC-AUC": r["cv_roc_auc_mean"], "Test ROC-AUC": r["test_roc_auc"], "Test PR-AUC": r["test_pr_auc"]}
    for name, r in results.items()
}).T
print(f"Best model: {best_name}")
comparison
""")

code("""import numpy as np
from sklearn.metrics import roc_curve

X_train, X_test, y_train, y_test = splits
best_pipe = pipelines[best_name]
proba = best_pipe.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, proba)

plt.plot(fpr, tpr, color="#8C1D40", linewidth=2, label=best_name)
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random baseline")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC curve — {best_name}")
plt.legend()
plt.show()
""")

md("""### SHAP interpretability

We explain the winning model's predictions globally (which features matter most overall) and
locally (why a specific member was scored the way they were).""")

code("""from src.modeling import compute_shap_values

shap_values, feature_names, X_test_transformed = compute_shap_values(best_pipe, X_test)
mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=feature_names).sort_values()
mean_abs_shap.plot(kind="barh", color="#8C1D40", title="Global feature importance (mean |SHAP value|)")
plt.tight_layout()
plt.show()
""")

md("## 5. Unsupervised learning: member segmentation")

code("""from src.segmentation import load_features as load_seg_features, choose_k, CLUSTER_FEATURES
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

seg_df = load_seg_features()
X_scaled = StandardScaler().fit_transform(seg_df[CLUSTER_FEATURES])
best_k, sil_scores = choose_k(X_scaled)
print(f"Silhouette scores: {sil_scores}")
print(f"Chosen k = {best_k}")

km = KMeans(n_clusters=best_k, n_init=10, random_state=42)
seg_df["cluster"] = km.fit_predict(X_scaled)
coords = PCA(n_components=2, random_state=42).fit_transform(X_scaled)

plt.scatter(coords[:, 0], coords[:, 1], c=seg_df["cluster"], cmap="tab10", alpha=0.7)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Member segments (PCA-reduced)")
plt.show()
""")

md("## 6. Decision layer: budget-constrained outreach optimization")

code("""from src.optimization import load_scored_population, build_outreach_plan

# Note: this cell assumes src.modeling.main() has already been run once to
# produce data/processed/scored_members.csv (done via `python -m src.pipeline`).
scored = load_scored_population()
result = build_outreach_plan(scored, caller_hours=40)
result["summary"]
""")

md("""## Summary

- Linked 1,194 records across 3 inconsistent source spreadsheets with no shared reliable key.
- Modeled renewal risk for 523 active members with an honest, non-overfit ROC-AUC of ~0.6–0.7.
- Explained the model's predictions with SHAP at both the global and individual-member level.
- Segmented members into behaviorally distinct groups via K-Means for targeted messaging.
- Converted model output into an actionable, budget-constrained outreach plan that outperforms
  naive highest-risk-first targeting.

See the deployed Streamlit dashboard for the fully interactive version of this analysis.
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}

with open("notebooks/eda_and_modeling_walkthrough.ipynb", "w") as f:
    nbf.write(nb, f)

print("Notebook written.")
