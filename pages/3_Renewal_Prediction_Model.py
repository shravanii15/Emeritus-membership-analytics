import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_curve, precision_recall_curve

st.set_page_config(page_title="Renewal Prediction Model", page_icon="🤖", layout="wide")
st.title("🤖 Renewal / Churn Prediction Model")

MODELS_DIR = "data/processed/models"

FEATURE_LABELS = {
    "years_since_retirement": "Years since retirement",
    "years_since_joined_ec": "Years since joining Emeritus College",
    "years_retirement_to_join_gap": "Gap: retirement → joining EC",
    "has_phone_on_file": "Has phone on file",
    "has_email_on_file": "Has email on file",
    "is_in_state": "Lives in-state (AZ)",
    "opted_out_of_email": "Opted out of email",
}


MODEL_SHORT_NAMES = {
    "logistic_regression": "Logistic Reg.",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}


def readable(name: str) -> str:
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]
    if name.startswith("discipline_bucketed_"):
        return f"Discipline: {name.replace('discipline_bucketed_', '')}"
    return name.replace("_", " ").capitalize()


try:
    with open(f"{MODELS_DIR}/metrics.json") as f:
        metrics = json.load(f)
    test_preds = pd.read_csv("data/processed/test_predictions.csv")
except FileNotFoundError:
    st.warning("Run `python -m src.pipeline` first.")
    st.stop()

best_model = metrics["best_model"]
results = metrics["results"]
indistinguishable = metrics.get("indistinguishable_from_best", [])
best_ci = results[best_model]["test_roc_auc_ci"]

# --- KPI summary row --------------------------------------------------------
n_test = len(test_preds)
n_at_risk = (test_preds["predicted_renewal_probability"] < 0.5).sum()
k1, k2, k3, k4 = st.columns(4)
k1.metric("Test-set members", n_test)
k2.metric("Best model (by point estimate)", MODEL_SHORT_NAMES.get(best_model, best_model))
k3.metric("Test ROC-AUC (95% CI)", f"{best_ci['point_estimate']:.2f}",
          help=f"[{best_ci['ci_lower']:.3f}, {best_ci['ci_upper']:.3f}] — bootstrap, {best_ci['n_boot_valid']} resamples")
k4.metric("Flagged at-risk (p < 0.5)", f"{n_at_risk} / {n_test}")

st.markdown(
    """
Each model predicts whether an **active** member will renew (pay dues or
hold a current EV subscription) this cycle, using only signals available
**before** the renewal window closes — tenure, recency, location,
discipline, and prior engagement. Evaluated with stratified 5-fold
cross-validation plus a held-out 25% test set.
"""
)

if indistinguishable:
    st.warning(
        f"⚠️ **Honest caveat:** `{best_model}` has the highest point-estimate ROC-AUC, but its "
        f"95% bootstrap confidence interval overlaps **{', '.join(m.replace('_', ' ') for m in indistinguishable)}**. "
        f"On this test set (n={n_test}), these models are **not statistically distinguishable** — "
        f"the 'best model' label reflects a point estimate, not a significant difference. "
        f"`{best_model}` is used below because it's the simplest of the models that are tied for best "
        f"(fewer hyperparameters, directly interpretable coefficients)."
    )

tab1, tab2, tab3 = st.tabs(["📊 Model comparison", "🎯 Threshold explorer", "🔍 SHAP interpretability"])

with tab1:
    comp_rows = []
    for name, r in results.items():
        ci = r["test_roc_auc_ci"]
        comp_rows.append({
            "Model": name.replace("_", " "),
            "CV ROC-AUC": f"{r['cv_roc_auc_mean']:.3f} ± {r['cv_roc_auc_std']:.3f}",
            "Test ROC-AUC": f"{r['test_roc_auc']:.3f}",
            "95% CI": f"[{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}]",
            "Test PR-AUC": round(r["test_pr_auc"], 3),
        })
    st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

    st.caption(
        "Note on performance: renewal behavior in this synthetic dataset is deliberately noisy "
        "(it mixes a real, modest signal — recency, location, discipline, opt-out status — with "
        "substantial irreducible randomness), mirroring how membership renewal actually behaves in "
        "practice. A ROC-AUC in the 0.5–0.7 range with overlapping confidence intervals is honest, "
        "non-overfit, non-leaked performance — not a result to oversell."
    )

    c1, c2 = st.columns(2)
    with c1:
        fpr, tpr, _ = roc_curve(test_preds["actual_renewed"], test_preds["predicted_renewal_probability"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=best_model, line=dict(color="#8C1D40", width=3)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random baseline",
                                  line=dict(color="gray", dash="dash")))
        fig.update_layout(title="ROC curve (best model)", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        prec, rec, _ = precision_recall_curve(test_preds["actual_renewed"], test_preds["predicted_renewal_probability"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name=best_model, line=dict(color="#FFC627", width=3)))
        fig.update_layout(title="Precision-Recall curve (best model)", xaxis_title="Recall", yaxis_title="Precision")
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown(
        "A classifier outputs a probability, not a decision — **the threshold you pick determines "
        "the trade-off** between catching more at-risk members (recall) and not wasting outreach on "
        "members who would have renewed anyway (precision). Move the slider to see that trade-off live."
    )
    threshold = st.slider("Decision threshold (flag as 'at risk of lapsing' if predicted probability < threshold)",
                           0.0, 1.0, 0.5, 0.05)
    y_true = test_preds["actual_renewed"]
    y_proba = test_preds["predicted_renewal_probability"]
    # "At risk" = predicted NOT to renew: flag when proba < threshold
    y_pred_lapse = (y_proba < threshold).astype(int)
    y_true_lapse = (y_true == 0).astype(int)

    cm = confusion_matrix(y_true_lapse, y_pred_lapse, labels=[0, 1])
    prec = precision_score(y_true_lapse, y_pred_lapse, zero_division=0)
    rec = recall_score(y_true_lapse, y_pred_lapse, zero_division=0)

    m1, m2, m3 = st.columns(3)
    m1.metric("Members flagged at-risk", int(y_pred_lapse.sum()))
    m2.metric("Precision (of flagged, % truly at-risk)", f"{prec:.0%}")
    m3.metric("Recall (of truly at-risk, % caught)", f"{rec:.0%}")

    fig = go.Figure(data=go.Heatmap(
        z=cm, x=["Predicted: Will renew", "Predicted: At risk"], y=["Actual: Will renew", "Actual: At risk"],
        colorscale="Burg", showscale=False, text=cm, texttemplate="%{text}", textfont={"size": 20},
    ))
    fig.update_layout(height=350, title=f"Confusion matrix at threshold = {threshold:.2f}")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    try:
        shap_values = np.load(f"{MODELS_DIR}/shap_values.npy")
        with open(f"{MODELS_DIR}/shap_feature_names.json") as f:
            feature_names = json.load(f)
        readable_names = [readable(n) for n in feature_names]

        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        importance_df = pd.DataFrame({"feature": readable_names, "mean_abs_shap": mean_abs_shap})
        importance_df = importance_df.sort_values("mean_abs_shap", ascending=True)

        fig = px.bar(importance_df, x="mean_abs_shap", y="feature", orientation="h",
                     title="Global feature importance (mean |SHAP value|)",
                     labels={"mean_abs_shap": "Mean |SHAP value|", "feature": ""},
                     color_discrete_sequence=["#8C1D40"])
        fig.update_layout(height=500, yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Per-member explanation")
        row_idx = pd.read_csv("data/processed/shap_row_index.csv")
        options = row_idx["member_id"].tolist()
        chosen = st.selectbox("Pick a test-set member to explain:", options)
        pos = row_idx.index[row_idx["member_id"] == chosen][0]

        member_shap = shap_values[pos]
        member_df = pd.DataFrame({"feature": readable_names, "shap_value": member_shap})
        member_df = member_df.reindex(member_df["shap_value"].abs().sort_values(ascending=True).index)
        member_df["direction"] = np.where(member_df["shap_value"] >= 0, "Pushes toward renewal", "Pushes toward lapse")

        fig2 = px.bar(member_df, x="shap_value", y="feature", orientation="h", color="direction",
                      color_discrete_map={"Pushes toward renewal": "#2E7D32", "Pushes toward lapse": "#C62828"},
                      title=f"Why the model predicts member #{chosen} this way",
                      labels={"shap_value": "SHAP value", "feature": ""})
        fig2.update_layout(height=500, yaxis_title="")
        st.plotly_chart(fig2, use_container_width=True)

        predicted_prob = row_idx.loc[pos, "predicted_renewal_probability"]
        actual = row_idx.loc[pos, "actual_renewed"]
        c1, c2 = st.columns(2)
        c1.metric("Predicted renewal probability", f"{predicted_prob:.1%}")
        c2.metric("Actual outcome", "Renewed" if actual == 1 else "Lapsed")

    except FileNotFoundError:
        st.info("SHAP values not available for this model run.")
