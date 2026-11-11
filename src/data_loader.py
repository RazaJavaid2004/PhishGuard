"""
PhishGuard: Multi-Source Email Threat Classification
Data Loader and Alignment Pipeline (Phase 2)
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple
import pandas as pd


# Candidate names prioritized for dynamic column resolution
TEXT_CANDIDATE_EXACT = [
    "text_combined",
    "body",
    "text",
    "message",
    "email",
    "content",
    "mail",
    "email_text",
    "body_text",
    "clean_text",
    "message_body",
]

LABEL_CANDIDATE_EXACT = [
    "label",
    "class",
    "target",
    "spam",
    "is_spam",
    "category",
    "status",
    "classification",
    "threat",
]

# Exclusion keywords when matching fallback substrings
EXCLUDE_FOR_TEXT = ["sender", "receiver", "to", "from", "date", "url", "id", "subject"]
EXCLUDE_FOR_LABEL = ["date", "subject", "body", "text", "sender", "receiver", "url"]


def find_column(
    columns: List[str],
    exact_candidates: List[str],
    fallback_keywords: List[str],
    exclude_keywords: List[str],
) -> Optional[str]:
    """
    Dynamically find matching column name using exact candidates first,
    then falling back to keyword heuristics.
    """
    lower_col_map = {col.strip().lower(): col for col in columns}

    # 1. Exact match check against candidates in order of priority
    for candidate in exact_candidates:
        if candidate in lower_col_map:
            return lower_col_map[candidate]

    # 2. Heuristic substring check
    for lower_name, original_name in lower_col_map.items():
        if any(ex in lower_name for ex in exclude_keywords):
            continue
        if any(kw in lower_name for kw in fallback_keywords):
            return original_name

    return None


def standardize_labels(series: pd.Series) -> pd.Series:
    """
    Standardize target labels into integer 0 (benign/ham) and 1 (phishing/spam).
    Handles string values, floats, and numeric types.
    """
    string_map = {
        "ham": 0,
        "legitimate": 0,
        "benign": 0,
        "normal": 0,
        "clean": 0,
        "safe": 0,
        "0": 0,
        "0.0": 0,
        "spam": 1,
        "phishing": 1,
        "fraud": 1,
        "malicious": 1,
        "threat": 1,
        "1": 1,
        "1.0": 1,
    }

    def convert_val(val):
        if pd.isna(val):
            return None
        # Handle string representation
        val_str = str(val).strip().lower()
        if val_str in string_map:
            return string_map[val_str]
        # Attempt direct numeric cast
        try:
            num = float(val)
            if num in (0.0, 1.0):
                return int(num)
        except (ValueError, TypeError):
            pass
        return None

    return series.apply(convert_val)


def load_and_standardize_csv(filepath: Path) -> Optional[pd.DataFrame]:
    """
    Load an individual CSV file, dynamically map text and label columns,
    and return a standardized DataFrame with ['text', 'label'].
    """
    print(f"-> Reading: {filepath.name}")

    # Try standard utf-8 first, fallback to latin1 for dirty email headers
    try:
        df = pd.read_csv(filepath, encoding="utf-8", low_memory=False)
    except (UnicodeDecodeError, Exception):
        df = pd.read_csv(
            filepath, encoding="latin1", encoding_errors="replace", low_memory=False
        )

    text_col = find_column(
        df.columns.tolist(),
        exact_candidates=TEXT_CANDIDATE_EXACT,
        fallback_keywords=["body", "text", "message", "content"],
        exclude_keywords=EXCLUDE_FOR_TEXT,
    )

    label_col = find_column(
        df.columns.tolist(),
        exact_candidates=LABEL_CANDIDATE_EXACT,
        fallback_keywords=["label", "class", "target", "spam"],
        exclude_keywords=EXCLUDE_FOR_LABEL,
    )

    if not text_col or not label_col:
        print(
            f"   [WARNING] Could not map columns for {filepath.name}. "
            f"Found text: {text_col}, label: {label_col}. Columns: {list(df.columns)}"
        )
        return None

    print(f"   Mapped: '{text_col}' -> 'text', '{label_col}' -> 'label' (rows: {len(df):,})")

    standardized_df = pd.DataFrame({
        "text": df[text_col],
        "label": df[label_col],
    })

    return standardized_df


def merge_and_clean_datasets(
    raw_data_dir: Path, output_file: Path
) -> pd.DataFrame:
    """
    Iterate over raw CSV files, standardize, concatenate, clean,
    and save the final master DataFrame.
    """
    csv_files = sorted(list(raw_data_dir.glob("*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_data_dir}")

    dfs = []
    for file_path in csv_files:
        df = load_and_standardize_csv(file_path)
        if df is not None and not df.empty:
            dfs.append(df)

    if not dfs:
        raise ValueError("No valid dataframes could be loaded.")

    print("\nConcatenating all datasets...")
    master_df = pd.concat(dfs, ignore_index=True)
    raw_count = len(master_df)
    print(f"Total raw concatenated rows: {raw_count:,}")

    # Standardize labels
    master_df["label"] = standardize_labels(master_df["label"])

    # Basic cleaning
    # 1. Drop rows with missing values in text or label
    master_df = master_df.dropna(subset=["text", "label"])

    # Ensure text is clean string and not empty whitespace
    master_df["text"] = master_df["text"].astype(str).str.strip()
    master_df = master_df[master_df["text"] != ""]

    # 2. Convert label to integer
    master_df["label"] = master_df["label"].astype(int)

    # 3. Remove duplicate email entries based on content
    before_dedup = len(master_df)
    master_df = master_df.drop_duplicates(subset=["text"], keep="first")
    dedup_dropped = before_dedup - len(master_df)

    master_df = master_df.reset_index(drop=True)

    # Ensure output directory exists and save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    master_df.to_csv(output_file, index=False)

    print("\n================ DATA CLEANING SUMMARY ================")
    print(f"Raw total rows:              {raw_count:,}")
    print(f"Rows after dropping nulls:   {before_dedup:,}")
    print(f"Duplicate email rows removed: {dedup_dropped:,}")
    print(f"Final cleaned dataset shape: {master_df.shape}")
    print("Class distribution:")
    label_counts = master_df["label"].value_counts().to_dict()
    for lbl, count in label_counts.items():
        name = "Phishing / Threat (1)" if lbl == 1 else "Legitimate / Benign (0)"
        pct = (count / len(master_df)) * 100
        print(f"  {name}: {count:,} ({pct:.2f}%)")
    print(f"Saved cleaned data to: {output_file}")
    print("=======================================================")

    return master_df


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    raw_dir = project_root / "data" / "raw"
    processed_file = project_root / "data" / "processed" / "cleaned_phishing_data.csv"

    print(f"Starting Phase 2 data loading and merging pipeline...")
    print(f"Raw data directory: {raw_dir}")
    print(f"Output file: {processed_file}\n")

    cleaned_df = merge_and_clean_datasets(raw_dir, processed_file)
    print(f"\nFinal dataset shape: {cleaned_df.shape}")
