"""
preprocess.py
=============
Production-ready preprocessing pipeline for the Pima Indians Diabetes dataset.

Pipeline steps:
  1. Load the CSV dataset
  2. Inspect missing / invalid zero values
  3. Impute zeros with column-wise medians (Glucose, BloodPressure,
     SkinThickness, Insulin, BMI)
  4. Split features (X) and target (y)
  5. Normalise features with StandardScaler
  6. Perform an 80/20 train-test split
  7. Convert all splits to PyTorch tensors
  8. Print final dataset shapes
"""

import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Path to the CSV file, resolved relative to this script's location so the
# module can be imported from any working directory.
DATA_PATH = os.path.join(os.path.dirname(__file__), "dataset", "diabetes.csv")

# Columns where a value of 0 is physiologically impossible and should be
# treated as a missing value, then imputed with the column median.
ZERO_IMPUTE_COLS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]

# The column we want to predict.
TARGET_COL = "Outcome"

# Random seed for reproducibility across train-test splits.
RANDOM_SEED = 42

# Fraction of data reserved for testing.
TEST_SIZE = 0.20


# ---------------------------------------------------------------------------
# Step 1 – Load dataset
# ---------------------------------------------------------------------------

def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the diabetes CSV and return a pandas DataFrame."""
    df = pd.read_csv(path)
    print(f"[load_data] Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


# ---------------------------------------------------------------------------
# Step 2 – Inspect data quality
# ---------------------------------------------------------------------------

def inspect_data(df: pd.DataFrame) -> None:
    """
    Print a quick data-quality report:
      - Standard NaN missing values per column
      - Count of invalid zero values in the physiologically-constrained columns
    """
    print("\n" + "=" * 55)
    print("DATA QUALITY REPORT")
    print("=" * 55)

    # 2a. Standard NaN counts
    nan_counts = df.isnull().sum()
    print("\n[inspect_data] NaN counts per column:")
    print(nan_counts.to_string())

    # 2b. Invalid zeros in biometric columns
    print(f"\n[inspect_data] Invalid-zero counts in constrained columns:")
    for col in ZERO_IMPUTE_COLS:
        n_zeros = (df[col] == 0).sum()
        pct = n_zeros / len(df) * 100
        print(f"  {col:<20s}: {n_zeros:>4d} zeros  ({pct:.1f} %)")

    print("=" * 55 + "\n")


# ---------------------------------------------------------------------------
# Step 3 – Impute invalid zeros with column median
# ---------------------------------------------------------------------------

def impute_zeros(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace physiologically impossible zero values with each column's
    *post-masking* median (i.e. the median calculated only over non-zero rows),
    so that the zeros themselves do not bias the imputed value.

    A copy of the DataFrame is returned; the original is not modified.
    """
    df = df.copy()

    for col in ZERO_IMPUTE_COLS:
        # Compute the median ignoring the invalid zeros
        valid_median = df.loc[df[col] != 0, col].median()
        n_replaced = (df[col] == 0).sum()

        # Replace zeros with the computed median
        df[col] = df[col].replace(0, valid_median)

        print(
            f"[impute_zeros] '{col}': replaced {n_replaced} zeros "
            f"with median = {valid_median:.4f}"
        )

    return df


# ---------------------------------------------------------------------------
# Step 4 – Split features and target
# ---------------------------------------------------------------------------

def split_features_target(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate the feature matrix X from the target vector y.

    Returns
    -------
    X : pd.DataFrame  – all columns except the target
    y : pd.Series     – the binary target (0 = no diabetes, 1 = diabetes)
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    print(f"\n[split_features_target] Features: {list(X.columns)}")
    print(f"[split_features_target] Target  : '{target_col}'  (classes: {sorted(y.unique())})")
    return X, y


# ---------------------------------------------------------------------------
# Step 5 – Normalise features
# ---------------------------------------------------------------------------

def normalise_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Fit a StandardScaler on X_train only (to avoid data leakage), then
    transform both X_train and X_test.

    Returns
    -------
    X_train_scaled : np.ndarray
    X_test_scaled  : np.ndarray
    scaler         : fitted StandardScaler (save this for inference)
    """
    scaler = StandardScaler()

    # Fit ONLY on training data – never on test data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n[normalise_features] StandardScaler fitted on training set.")
    print(f"[normalise_features] Feature means  (train): {scaler.mean_.round(4)}")
    print(f"[normalise_features] Feature stdevs (train): {scaler.scale_.round(4)}")

    return X_train_scaled, X_test_scaled, scaler


# ---------------------------------------------------------------------------
# Step 6 – Train-test split
# ---------------------------------------------------------------------------

def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_SEED,
) -> tuple:
    """
    Stratified 80/20 train-test split.

    Stratification ensures that the class distribution (diabetic vs.
    non-diabetic) is preserved in both splits.

    Returns
    -------
    X_train, X_test, y_train, y_test  (all pd.DataFrame / pd.Series)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,          # preserve class balance
    )
    print(
        f"\n[split_train_test] Train: {len(X_train)} samples  "
        f"| Test: {len(X_test)} samples  "
        f"| Split: {int((1 - test_size) * 100)}/{int(test_size * 100)}"
    )
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Step 7 – Convert to PyTorch tensors
# ---------------------------------------------------------------------------

def to_tensors(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert numpy arrays / pandas Series to PyTorch tensors.

    Features  → float32  (required by most PyTorch layers)
    Labels    → float32  (required by BCELoss / BCEWithLogitsLoss)

    Returns
    -------
    X_train_t, X_test_t, y_train_t, y_test_t
    """
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32)
    y_test_t  = torch.tensor(y_test.values,  dtype=torch.float32)

    print("\n[to_tensors] Converted splits to PyTorch tensors.")
    return X_train_t, X_test_t, y_train_t, y_test_t


# ---------------------------------------------------------------------------
# Step 8 – Print final shapes
# ---------------------------------------------------------------------------

def print_shapes(
    X_train_t: torch.Tensor,
    X_test_t:  torch.Tensor,
    y_train_t: torch.Tensor,
    y_test_t:  torch.Tensor,
) -> None:
    """Pretty-print the shape of every tensor in the final dataset."""
    print("\n" + "=" * 55)
    print("FINAL DATASET SHAPES")
    print("=" * 55)
    print(f"  X_train : {tuple(X_train_t.shape)}")
    print(f"  X_test  : {tuple(X_test_t.shape)}")
    print(f"  y_train : {tuple(y_train_t.shape)}")
    print(f"  y_test  : {tuple(y_test_t.shape)}")
    print("=" * 55 + "\n")


# ---------------------------------------------------------------------------
# Main pipeline – orchestrates all steps in order
# ---------------------------------------------------------------------------

def preprocess(
    data_path: str = DATA_PATH,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_SEED,
) -> dict:
    """
    Run the full preprocessing pipeline end-to-end.

    Parameters
    ----------
    data_path    : path to the raw CSV file
    test_size    : fraction of samples reserved for the test set
    random_state : reproducibility seed

    Returns
    -------
    A dictionary with keys:
        'X_train', 'X_test', 'y_train', 'y_test'  → torch.Tensor
        'scaler'                                    → fitted StandardScaler
        'feature_names'                             → list[str]
    """
    # 1. Load
    df = load_data(data_path)

    # 2. Inspect
    inspect_data(df)

    # 3. Impute invalid zeros
    df = impute_zeros(df)

    # 4. Feature / target split
    X, y = split_features_target(df)
    feature_names = list(X.columns)

    # 5. Train-test split  (done before scaling to avoid leakage)
    X_train, X_test, y_train, y_test = split_train_test(
        X, y, test_size=test_size, random_state=random_state
    )

    # 6. Normalise features
    X_train_scaled, X_test_scaled, scaler = normalise_features(X_train, X_test)

    # 7. Convert to PyTorch tensors
    X_train_t, X_test_t, y_train_t, y_test_t = to_tensors(
        X_train_scaled, X_test_scaled, y_train, y_test
    )

    # 8. Print shapes
    print_shapes(X_train_t, X_test_t, y_train_t, y_test_t)

    return {
        "X_train": X_train_t,
        "X_test":  X_test_t,
        "y_train": y_train_t,
        "y_test":  y_test_t,
        "scaler":  scaler,
        "feature_names": feature_names,
    }


# ---------------------------------------------------------------------------
# Entry point – run as a standalone script for a quick sanity-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    data = preprocess()
    print("Preprocessing complete. Ready for model training.")
