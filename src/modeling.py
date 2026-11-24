"""
modeling.py
============
Supervised learning: predicts whether an active Emeritus College member will
renew (pay dues / hold an EV subscription) this cycle, so outreach staff can
prioritize at-risk members before they lapse.

Trains and compares:
  - Logistic Regression (interpretable baseline)
  - Random Forest (nonlinear baseline)
  - XGBoost (gradient boosting; typically the strongest tabular performer)

Evaluates with stratified 5-fold cross-validation and a held-out test set:
ROC-AUC, PR-AUC, and a confusion matrix at a chosen operating threshold.
Explains the winning model with SHAP values (both global importance and
per-member local explanations — used later by the Streamlit app).

Run directly to train, evaluate, and persist the model + metrics + SHAP values:
    python -m src.modeling
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, confusion_matrix, roc_auc_score,
    classification_report, roc_curve, precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "models"

NUMERIC_FEATURES = [
    "years_since_retirement", "years_since_joined_ec", "years_retirement_to_join_gap",
    "has_phone_on_file", "has_email_on_file", "is_in_state", "opted_out_of_email",
]
CATEGORICAL_FEATURES = ["discipline_bucketed"]
TARGET = "renewed"
RANDOM_STATE = 42


def load_features() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / "member_features.csv")
    # Impute: years_since_* can be NaN for a handful of rows with a bad/missing
    # retirement date in the source data — fill with the column median rather
    # than drop, so the model still gets to use every member's other signals.
    for col in ["years_since_retirement", "years_since_joined_ec", "years_retirement_to_join_gap"]:
        df[col] = df[col].fillna(df[col].median())
    return df


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])


def get_models() -> dict:
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=5, random_state=RANDOM_STATE
        ),
    }
    try:
        from xgboost import XGBClassifier
        models["xgboost"] = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.9,
            eval_metric="logloss", random_state=RANDOM_STATE,
        )
    except ImportError:
        pass
    return models


def evaluate_models(df: pd.DataFrame):
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    fitted_pipelines = {}

    for name, model in get_models().items():
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", model)])
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        preds = (proba >= 0.5).astype(int)

        results[name] = {
            "cv_roc_auc_mean": float(cv_scores.mean()),
            "cv_roc_auc_std": float(cv_scores.std()),
            "test_roc_auc": float(roc_auc_score(y_test, proba)),
            "test_pr_auc": float(average_precision_score(y_test, proba)),
            "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
            "classification_report": classification_report(y_test, preds, output_dict=True),
        }
        fitted_pipelines[name] = pipe

    best_name = max(results, key=lambda n: results[n]["test_roc_auc"])
    return results, fitted_pipelines, best_name, (X_train, X_test, y_train, y_test)


def compute_shap_values(pipe: Pipeline, X_test: pd.DataFrame):
    """SHAP explanation for the winning model. Falls back gracefully if the
    model type isn't tree-based (SHAP's TreeExplainer needs tree models;
    for linear models we use a linear explainer)."""
    import shap

    prep = pipe.named_steps["prep"]
    clf = pipe.named_steps["clf"]
    X_test_transformed = prep.transform(X_test)
    feature_names = (
        NUMERIC_FEATURES
        + list(prep.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES))
    )

    if hasattr(clf, "feature_importances_"):  # tree-based
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_test_transformed)
        if isinstance(shap_values, list):  # some versions return [class0, class1]
            shap_values = shap_values[1]
    else:  # linear
        explainer = shap.LinearExplainer(clf, X_test_transformed)
        shap_values = explainer.shap_values(X_test_transformed)

    return shap_values, feature_names, X_test_transformed


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_features()
    results, pipelines, best_name, splits = evaluate_models(df)
    X_train, X_test, y_train, y_test = splits

    print("Model comparison (test set ROC-AUC / PR-AUC):")
    for name, r in results.items():
        print(f"  {name:20s} ROC-AUC={r['test_roc_auc']:.3f}  PR-AUC={r['test_pr_auc']:.3f}  "
              f"CV ROC-AUC={r['cv_roc_auc_mean']:.3f}±{r['cv_roc_auc_std']:.3f}")
    print(f"\nBest model: {best_name}")

    best_pipe = pipelines[best_name]

    with open(MODELS_DIR / "best_model.pkl", "wb") as f:
        pickle.dump(best_pipe, f)

    metrics_out = {"best_model": best_name, "results": results}
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2, default=str)

    # Persist test-set predictions for the Streamlit app
    proba = best_pipe.predict_proba(X_test)[:, 1]
    test_out = X_test.copy()
    test_out["member_id"] = df.loc[X_test.index, "member_id"].values
    test_out["actual_renewed"] = y_test.values
    test_out["predicted_renewal_probability"] = proba
    test_out.to_csv(PROCESSED_DIR / "test_predictions.csv", index=False)

    # Full-population scored predictions (for the budget optimizer)
    X_full = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    full_proba = best_pipe.predict_proba(X_full)[:, 1]
    scored = df[["member_id", "renewed"]].copy()
    scored["predicted_renewal_probability"] = full_proba
    scored.to_csv(PROCESSED_DIR / "scored_members.csv", index=False)

    try:
        shap_values, feature_names, X_test_transformed = compute_shap_values(best_pipe, X_test)
        np.save(MODELS_DIR / "shap_values.npy", shap_values)
        np.save(MODELS_DIR / "shap_X_test.npy", X_test_transformed)
        with open(MODELS_DIR / "shap_feature_names.json", "w") as f:
            json.dump(feature_names, f)
        test_out.reset_index(drop=True).to_csv(PROCESSED_DIR / "shap_row_index.csv", index=False)
        print(f"\nSHAP values computed for {len(feature_names)} features on {len(X_test)} test rows.")
    except Exception as e:
        print(f"SHAP computation skipped ({e}).")

    print(f"\nSaved model to {MODELS_DIR / 'best_model.pkl'}")
    print(f"Saved metrics to {MODELS_DIR / 'metrics.json'}")
    print(f"Saved scored population to {PROCESSED_DIR / 'scored_members.csv'}")


if __name__ == "__main__":
    main()
