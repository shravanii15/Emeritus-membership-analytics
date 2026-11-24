import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_curve, precision_recall_curve

st.set_page_config(page_title="Renewal Prediction Model", page_icon="🤖", layout="wide")
st.title("🤖 Renewal / Churn Prediction Model")

MODELS_DIR = "data/processed/models"

try:
    with open(f"{MODELS_DIR}/metrics.json") as f:
        metrics = json.load(f)
    test_preds = pd.read_csv("data/processed/test_predictions.csv")
except FileNotFoundError:
    st.warning("Run `python -m src.pipeline` first.")
    st.stop()

best_model = metrics["best_model"]
results = metrics["results"]

st.markdown(
    f"""
Each model is trained to predict whether an **active** member will renew
(pay dues or hold a current EV subscription) this cycle, using only signals
available **before** the renewal window closes — tenure, recency, location,
discipline, and prior engagement. Evaluated with stratified 5-fold
cross-validation plus a held-out 25% test set.

**Best model selected: `{best_model}`** (highest test-set ROC-AUC).
"""
)

st.subheader("Model comparison")
comp_rows = []
for name, r in results.items():
    comp_rows.append({
        "Model": name,
        "CV ROC-AUC": f"{r['cv_roc_auc_mean']:.3f} ± {r['cv_roc_auc_std']:.3f}",
        "Test ROC-AUC": round(r["test_roc_auc"], 3),
        "Test PR-AUC": round(r["test_pr_auc"], 3),
    })
st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

st.caption(
    "Note on performance: renewal behavior in this synthetic dataset is "
    "deliberately noisy (it mixes a real, modest signal — recency, location, "
    "discipline, opt-out status — with substantial irreducible randomness), "
    "mirroring how membership renewal actually behaves in practice: useful "
    "but imperfect signal, not a deterministic outcome. A ROC-AUC in the "
    "0.6–0.7 range here reflects honest, non-overfit performance rather than "
    "a leakage artifact."
)

c1, c2 = st.columns(2)

with c1:
    st.subheader("ROC curve (best model)")
    fpr, tpr, _ = roc_curve(test_preds["actual_renewed"], test_preds["predicted_renewal_probability"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=best_model, line=dict(color="#8C1D40", width=3)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random baseline",
                              line=dict(color="gray", dash="dash")))
    fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("Precision-Recall curve (best model)")
    prec, rec, _ = precision_recall_curve(test_preds["actual_renewed"], test_preds["predicted_renewal_probability"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name=best_model, line=dict(color="#FFC627", width=3)))
    fig.update_layout(xaxis_title="Recall", yaxis_title="Precision")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Confusion matrix (threshold = 0.5)")
cm = np.array(results[best_model]["confusion_matrix"])
fig = go.Figure(data=go.Heatmap(
    z=cm, x=["Predicted: Lapse", "Predicted: Renew"], y=["Actual: Lapse", "Actual: Renew"],
    colorscale="Burg", showscale=False, text=cm, texttemplate="%{text}", textfont={"size": 20},
))
fig.update_layout(height=350)
st.plotly_chart(fig, use_container_width=True)

st.subheader("🔍 SHAP interpretability")
try:
    shap_values = np.load(f"{MODELS_DIR}/shap_values.npy")
    X_test_transformed = np.load(f"{MODELS_DIR}/shap_X_test.npy")
    with open(f"{MODELS_DIR}/shap_feature_names.json") as f:
        feature_names = json.load(f)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap})
    importance_df = importance_df.sort_values("mean_abs_shap", ascending=True)

    import plotly.express as px
    fig = px.bar(importance_df, x="mean_abs_shap", y="feature", orientation="h",
                 title="Global feature importance (mean |SHAP value|)",
                 color_discrete_sequence=["#8C1D40"])
    fig.update_layout(height=500, yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Per-member explanation")
    row_idx = pd.read_csv("data/processed/shap_row_index.csv")
    options = row_idx["member_id"].tolist()
    chosen = st.selectbox("Pick a test-set member to explain:", options)
    pos = row_idx.index[row_idx["member_id"] == chosen][0]

    member_shap = shap_values[pos]
    member_df = pd.DataFrame({"feature": feature_names, "shap_value": member_shap})
    member_df = member_df.reindex(member_df["shap_value"].abs().sort_values(ascending=True).index)
    member_df["direction"] = np.where(member_df["shap_value"] >= 0, "Pushes toward renewal", "Pushes toward lapse")

    fig2 = px.bar(member_df, x="shap_value", y="feature", orientation="h", color="direction",
                  color_discrete_map={"Pushes toward renewal": "#2E7D32", "Pushes toward lapse": "#C62828"},
                  title=f"Why the model predicts member #{chosen} this way")
    fig2.update_layout(height=500, yaxis_title="")
    st.plotly_chart(fig2, use_container_width=True)

    predicted_prob = row_idx.loc[pos, "predicted_renewal_probability"]
    actual = row_idx.loc[pos, "actual_renewed"]
    st.metric("Predicted renewal probability", f"{predicted_prob:.1%}")
    st.metric("Actual outcome", "Renewed" if actual == 1 else "Lapsed")

except FileNotFoundError:
    st.info("SHAP values not available for this model run.")
