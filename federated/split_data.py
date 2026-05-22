"""
split_data.py
=============
Creates three DEMOGRAPHICALLY DISTINCT hospital datasets — a realistic
non-IID federated learning scenario where each hospital has a different
patient population (different age skews, BMI ranges, pregnancy patterns).

This is closer to real-world federated learning than a simple random
split:  in practice, every hospital has its own demographic profile
(urban vs. rural, specialty vs. general, young vs. elderly).

Hospital identities
-------------------
  Hospital A — "Urban Wellness Clinic"
      Patient skew: younger adults  (age 21–45 weighted higher)
      Use case   : preventive care, screening

  Hospital B — "Senior Care Hospital"
      Patient skew: older adults    (age 45+ weighted higher)
      Use case   : geriatric medicine, late-onset diabetes

  Hospital C — "Metro General Hospital"
      Patient skew: balanced cohort (mixed age groups)
      Use case   : general practice, broad demographics

Privacy principle
-----------------
After this step, each hospital ONLY ever reads its own CSV file.
No cross-hospital data sharing occurs at any point in the pipeline.
"""

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Paths — resolved relative to this file so it works from any cwd
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH    = os.path.join(PROJECT_ROOT, "ml", "dataset", "diabetes.csv")
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "federated", "data")

# Reproducible seed
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Hospital metadata — drives split logic AND dashboard display names
# ---------------------------------------------------------------------------
HOSPITAL_METADATA = {
    "hospital_a": {
        "display_name": "Hospital A — Urban Wellness Clinic",
        "specialty":    "Preventive Care",
        "csv_name":     "hospital_a.csv",
        # Age preference: skew toward 21–45 (younger urban demographics)
        "age_min":       21,
        "age_max":       45,
        "age_weight":    3.0,    # 3× boost for patients in this age range
    },
    "hospital_b": {
        "display_name": "Hospital B — Senior Care Hospital",
        "specialty":    "Geriatric Medicine",
        "csv_name":     "hospital_b.csv",
        # Age preference: skew toward 45+ (older adults)
        "age_min":       45,
        "age_max":       100,
        "age_weight":    3.0,
    },
    "hospital_c": {
        "display_name": "Hospital C — Metro General Hospital",
        "specialty":    "General Medicine",
        "csv_name":     "hospital_c.csv",
        # Balanced — no age bias
        "age_min":       21,
        "age_max":       100,
        "age_weight":    1.0,
    },
}


# ---------------------------------------------------------------------------
# Non-IID partitioning logic
# ---------------------------------------------------------------------------

def _weighted_sample(df: pd.DataFrame, meta: dict, n: int, seed: int) -> pd.DataFrame:
    """
    Draw `n` samples from `df` using age-weighted probabilities so the
    resulting sample skews toward the hospital's preferred age range.

    Within the preferred range  → weight = meta["age_weight"]  (e.g., 3.0)
    Outside the preferred range → weight = 1.0  (still possible, just less)

    This produces non-IID data while keeping every patient possibly drawn
    by every hospital (no hard cut-offs).
    """
    weights = np.where(
        (df["Age"] >= meta["age_min"]) & (df["Age"] <= meta["age_max"]),
        meta["age_weight"],
        1.0,
    )
    # Normalise to a probability distribution
    probs = weights / weights.sum()

    rng = np.random.default_rng(seed)
    chosen_idx = rng.choice(df.index, size=n, replace=False, p=probs)
    return df.loc[chosen_idx].copy()


# ---------------------------------------------------------------------------
# Main split entrypoint
# ---------------------------------------------------------------------------

def split_and_save() -> dict:
    """
    Load the full diabetes dataset and produce three non-IID hospital
    partitions, each saved as its own CSV file.

    Returns
    -------
    dict mapping hospital_name → pandas DataFrame (for inspection / tests)
    """

    print("=" * 70)
    print("  MEDISHIELD — MULTI-HOSPITAL DATASET PARTITIONING")
    print("  Strategy: non-IID demographic split (age-weighted sampling)")
    print("=" * 70)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Source dataset missing: {DATA_PATH}\n"
            "Make sure ml/dataset/diabetes.csv exists before running."
        )

    df_full = pd.read_csv(DATA_PATH)
    print(f"\n[split] Source dataset : {len(df_full)} rows × {df_full.shape[1]} cols")
    print(f"[split] Class balance  : "
          f"{int(df_full['Outcome'].sum())} diabetic / "
          f"{len(df_full) - int(df_full['Outcome'].sum())} non-diabetic")

    # ------------------------------------------------------------------
    # Each hospital draws ~1/3 of the dataset.  With replace=False across
    # hospitals we avoid the same patient appearing in two hospitals
    # (medically realistic — no double-counting).
    #
    # We pop drawn rows from a working copy so subsequent hospitals draw
    # from the remaining pool.
    # ------------------------------------------------------------------
    samples_per_hospital = len(df_full) // 3
    working_pool = df_full.copy()
    splits: dict[str, pd.DataFrame] = {}

    print("\n[split] Drawing demographically-weighted samples per hospital:")
    print(f"  Samples per hospital  : ~{samples_per_hospital}")

    for i, (key, meta) in enumerate(HOSPITAL_METADATA.items()):
        sample = _weighted_sample(
            working_pool,
            meta,
            n=samples_per_hospital,
            seed=RANDOM_SEED + i,        # different seed per hospital
        )
        splits[key] = sample.reset_index(drop=True)
        # Remove drawn rows so the next hospital draws from remaining patients
        working_pool = working_pool.drop(sample.index)

    # ------------------------------------------------------------------
    # Save each hospital's CSV and print a demographic summary so the
    # demo audience can SEE that hospitals genuinely have different cohorts.
    # ------------------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n[split] Per-hospital demographic profile:")
    print(f"  {'Hospital':<14} {'Specialty':<22} {'Rows':>5} "
          f"{'AvgAge':>7} {'Diabetic':>9} {'NonDiab':>8}")
    print("  " + "-" * 70)

    for key, partition in splits.items():
        meta = HOSPITAL_METADATA[key]
        out_path = os.path.join(OUTPUT_DIR, meta["csv_name"])
        partition.to_csv(out_path, index=False)

        diabetic     = int(partition["Outcome"].sum())
        non_diabetic = len(partition) - diabetic
        avg_age      = float(partition["Age"].mean())

        print(f"  {key:<14} {meta['specialty']:<22} {len(partition):>5} "
              f"{avg_age:>7.1f} {diabetic:>9} {non_diabetic:>8}")

    print(f"\n[split] CSV files saved to: {OUTPUT_DIR}")
    print(f"[split] Each hospital will load ONLY its own CSV — no cross-access.")
    print("=" * 70)

    return splits


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    split_and_save()
    print("\nMulti-hospital dataset partitioning complete.")
