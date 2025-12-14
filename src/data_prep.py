import os
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# -----------------------------
# CONFIG
# -----------------------------
HF_DATASET_REPO = "VJBharathkumar/tourism-wellness"

# IMPORTANT: put the EXACT csv filename present in your HF dataset repo
# Example: "tourism.csv"
HF_DATA_FILE = "tourism.csv"

DATA_DIR = "data"

TARGET = "ProdTaken"
DROP_COLS = ["Unnamed: 0", "CustomerID"]  # index/identifier columns commonly safe to drop


# -----------------------------
# HELPERS
# -----------------------------
def load_from_hf(repo_id: str, filename: str) -> pd.DataFrame:
    """
    Loads ONLY the specified CSV file from the HF dataset repo.
    This avoids HF trying to merge multiple csv files with different columns.
    """
    hf_path = f"hf://datasets/{repo_id}/{filename}"
    ds = load_dataset("csv", data_files=hf_path)
    return ds["train"].to_pandas()


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Drop unwanted columns (if present)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

    # Ensure target exists
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found. Columns: {list(df.columns)}")

    # Fill missing values
    for col in df.columns:
        if col == TARGET:
            continue

        if df[col].dtype == "object":
            if df[col].isna().any():
                mode_vals = df[col].mode(dropna=True)
                df[col] = df[col].fillna(mode_vals.iloc[0] if not mode_vals.empty else "Unknown")
        else:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())

    # Make sure target is int (0/1)
    df[TARGET] = df[TARGET].astype(int)

    return df


def split_and_save(df: pd.DataFrame, out_dir: str = DATA_DIR):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, "train.csv")
    test_path = os.path.join(out_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    return train_df, test_df, train_path, test_path


# -----------------------------
# MAIN
# -----------------------------
def main():
    print(f"Loading data from HF dataset: {HF_DATASET_REPO}")
    print(f"Using file: {HF_DATA_FILE}")

    df = load_from_hf(HF_DATASET_REPO, HF_DATA_FILE)
    print("Raw shape:", df.shape)

    df = clean_dataframe(df)
    print("After cleaning shape:", df.shape)

    train_df, test_df, train_path, test_path = split_and_save(df)

    print("\nData prep complete")
    print("Train shape:", train_df.shape)
    print("Test shape :", test_df.shape)
    print("Saved files:")
    print(" -", train_path)
    print(" -", test_path)


if __name__ == "__main__":
    main()