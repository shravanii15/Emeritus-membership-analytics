"""
segmentation.py
================
Unsupervised learning: K-Means clustering to segment Emeritus College
members into behaviorally distinct groups (e.g., "highly engaged in-state
loyalists" vs. "lapsed out-of-state alumni") for targeted outreach
messaging — a complement to the supervised renewal model.

Chooses k via silhouette score, then reduces to 2D with PCA for
visualization.

Run directly to persist cluster assignments + PCA coordinates:
    python -m src.segmentation
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

CLUSTER_FEATURES = [
    "years_since_retirement", "years_since_joined_ec", "years_retirement_to_join_gap",
    "is_in_state", "opted_out_of_email", "engagement_score",
    "has_paid_dues_this_cycle", "has_ev_subscription",
]


def load_features() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / "member_features.csv")
    for col in ["years_since_retirement", "years_since_joined_ec", "years_retirement_to_join_gap"]:
        df[col] = df[col].fillna(df[col].median())
    return df


def choose_k(X_scaled: np.ndarray, k_range=range(2, 6)) -> tuple[int, dict]:
    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X_scaled)
        scores[k] = float(silhouette_score(X_scaled, labels))
    best_k = max(scores, key=scores.get)
    return best_k, scores


def label_clusters(df: pd.DataFrame) -> dict:
    """Produce a short, human-readable name per cluster from its feature means,
    so the dashboard can show 'Engaged In-State Core' instead of 'Cluster 2'."""
    names = {}
    overall_engagement = df["engagement_score"].mean()
    overall_recency = df["years_since_retirement"].mean()
    for cluster_id, group in df.groupby("cluster"):
        tags = []
        tags.append("Engaged" if group["engagement_score"].mean() >= overall_engagement else "Disengaged")
        tags.append("In-State" if group["is_in_state"].mean() >= 0.5 else "Out-of-State")
        tags.append("Recent Retirees" if group["years_since_retirement"].mean() < overall_recency else "Long-Tenured")
        names[int(cluster_id)] = " ".join(tags)
    # Disambiguate clusters that land on the same label (common with higher k)
    seen = {}
    for cid, name in names.items():
        seen.setdefault(name, []).append(cid)
    for name, cids in seen.items():
        if len(cids) > 1:
            for cid in cids:
                names[cid] = f"{name} ({cid})"
    return names


def main():
    df = load_features()
    X = df[CLUSTER_FEATURES]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    best_k, silhouette_scores = choose_k(X_scaled)
    print(f"Silhouette scores by k: {silhouette_scores}")
    print(f"Chosen k = {best_k}")

    km = KMeans(n_clusters=best_k, n_init=10, random_state=42)
    df["cluster"] = km.fit_predict(X_scaled)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    df["pca_x"] = coords[:, 0]
    df["pca_y"] = coords[:, 1]

    cluster_names = label_clusters(df)
    df["cluster_name"] = df["cluster"].map(cluster_names)

    out_cols = ["member_id", "cluster", "cluster_name", "pca_x", "pca_y"] + CLUSTER_FEATURES
    df[out_cols].to_csv(PROCESSED_DIR / "member_segments.csv", index=False)

    summary = {
        "chosen_k": best_k,
        "silhouette_scores": silhouette_scores,
        "cluster_names": cluster_names,
        "cluster_sizes": df["cluster"].value_counts().to_dict(),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    }
    with open(PROCESSED_DIR / "segmentation_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nCluster sizes:\n{df['cluster'].value_counts()}")
    print(f"\nCluster names: {cluster_names}")
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"\nWrote segments to {PROCESSED_DIR / 'member_segments.csv'}")


if __name__ == "__main__":
    main()
