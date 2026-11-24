"""
pipeline.py
============
Orchestrates the full data-to-decision pipeline end to end. Useful for a
one-shot rebuild (e.g. in CI, or after regenerating the synthetic data).

    python -m src.pipeline
"""
from src import data_cleaning, features, modeling, segmentation, optimization


def run_all():
    print("=" * 70)
    print("STEP 1/5: Data cleaning & record linkage")
    print("=" * 70)
    data_cleaning.main()

    print("\n" + "=" * 70)
    print("STEP 2/5: Feature engineering")
    print("=" * 70)
    features.main()

    print("\n" + "=" * 70)
    print("STEP 3/5: Supervised modeling (renewal prediction)")
    print("=" * 70)
    modeling.main()

    print("\n" + "=" * 70)
    print("STEP 4/5: Unsupervised segmentation")
    print("=" * 70)
    segmentation.main()

    print("\n" + "=" * 70)
    print("STEP 5/5: Budget-constrained outreach optimization")
    print("=" * 70)
    optimization.main()

    print("\nPipeline complete.")


if __name__ == "__main__":
    run_all()
