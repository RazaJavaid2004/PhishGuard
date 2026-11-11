"""
PhishGuard: Multi-Source Email Threat Classification
Feature Preprocessing Pipeline (Phase 4)

Loads the cleaned dataset, applies TF-IDF vectorization,
and performs a reproducible train-test split.
"""

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import spmatrix


# ── Configuration ────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.20
MAX_FEATURES = 10_000
NGRAM_RANGE = (1, 2)          # unigrams + bigrams for richer signal
SUBLINEAR_TF = True           # apply log-scaling to raw term frequencies
MIN_DF = 3                    # ignore ultra-rare tokens (< 3 docs)
MAX_DF = 0.95                 # ignore tokens appearing in > 95 % of docs


def load_dataset(path: Path) -> pd.DataFrame:
    """Load the cleaned CSV and validate required columns."""
    df = pd.read_csv(path)
    assert {"text", "label"}.issubset(df.columns), \
        f"Expected columns 'text' and 'label', got {list(df.columns)}"
    df["text"] = df["text"].astype(str).fillna("")
    df["label"] = df["label"].astype(int)
    return df


def build_tfidf_vectorizer() -> TfidfVectorizer:
    """Return a configured TF-IDF vectorizer."""
    return TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        sublinear_tf=SUBLINEAR_TF,
        min_df=MIN_DF,
        max_df=MAX_DF,
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",
        strip_accents="unicode",
    )


def preprocess_pipeline(
    data_path: Path,
) -> Tuple[spmatrix, spmatrix, pd.Series, pd.Series, TfidfVectorizer]:
    """
    End-to-end preprocessing: load → vectorize → split.

    Returns
    -------
    X_train, X_test : sparse TF-IDF matrices
    y_train, y_test : label Series
    vectorizer      : fitted TfidfVectorizer (for persistence)
    """
    print(f"Loading dataset from {data_path} ...")
    df = load_dataset(data_path)
    print(f"  Rows: {len(df):,}  |  Class 0: {(df['label']==0).sum():,}  |  Class 1: {(df['label']==1).sum():,}")

    # Train-test split (stratified to preserve class ratio)
    X_text_train, X_text_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )
    print(f"  Train size: {len(X_text_train):,}  |  Test size: {len(X_text_test):,}")

    # Fit TF-IDF on training data only, then transform both splits
    vectorizer = build_tfidf_vectorizer()
    print(f"  Fitting TF-IDF (max_features={MAX_FEATURES}, ngram_range={NGRAM_RANGE}) ...")
    X_train = vectorizer.fit_transform(X_text_train)
    X_test = vectorizer.transform(X_text_test)
    print(f"  TF-IDF matrix shape  →  train: {X_train.shape}  |  test: {X_test.shape}")

    return X_train, X_test, y_train, y_test, vectorizer


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "processed" / "cleaned_phishing_data.csv"
    X_train, X_test, y_train, y_test, vec = preprocess_pipeline(data_path)
    print("\nPreprocessing complete.  Ready for model training.")
