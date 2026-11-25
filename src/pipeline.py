"""
pipeline.py
============
Orchestrates the full data-to-decision pipeline end to end. Useful for a
one-shot rebuild (e.g. in CI, or after regenerating the synthetic data).

    python -m src.pipeline
"""
from src import (
    data_cleaning, features, modeling, segmentation, optimization,
    linkage_robustness, parking_analysis,
)


def run_all():
    print("=" * 70)
    print("STEP 1/7: Data cleaning & record linkage")
    print("=" * 70)
    data_cleaning.main()

    print("\n" + "=" * 70)
    print("STEP 2/7: Feature engineering")
    print("=" * 70)
    features.main()

    print("\n" + "=" * 70)
    print("STEP 3/7: Supervised modeling (renewal prediction)")
    print("=" * 70)
    modeling.main()

    print("\n" + "=" * 70)
    print("STEP 4/7: Unsupervised segmentation")
    print("=" * 70)
    segmentation.main()

    print("\n" + "=" * 70)
    print("STEP 5/7: Budget-constrained outreach optimization")
    print("=" * 70)
    optimization.main()

    print("\n" + "=" * 70)
    print("STEP 6/7: Linkage robustness experiment (research question)")
    print("=" * 70)
    linkage_robustness.main()

    print("\n" + "=" * 70)
    print("STEP 7/7: Parking budget impact analysis")
    print("=" * 70)
    parking_analysis.main()

    print("\nPipeline complete.")


if __name__ == "__main__":
    run_all()
