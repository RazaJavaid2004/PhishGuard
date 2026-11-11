"""
PhishGuard: Multi-Source Email Threat Classification
Model Training & Evaluation Pipeline (Phase 4)

Trains a Logistic Regression classifier on TF-IDF features,
evaluates on the held-out test set, saves the confusion matrix
plot and persists the trained model + vectorizer for reuse.
"""

from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from preprocess import preprocess_pipeline


# ── Configuration ────────────────────────────────────────────────────
RANDOM_STATE = 42
MODEL_DIR_NAME = "models"
ASSETS_DIR_NAME = "docs/assets"


def train_model(X_train, y_train):
    """Train a Logistic Regression classifier."""
    print("\n════════════════ MODEL TRAINING ════════════════")
    print("Algorithm : Logistic Regression (saga solver, L2 penalty)")
    print(f"Train set : {X_train.shape[0]:,} samples  ×  {X_train.shape[1]:,} features")

    model = LogisticRegression(
        solver="saga",        # scales well to large sparse datasets
        penalty="l2",
        C=1.0,
        max_iter=1000,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )

    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"Training completed in {elapsed:.1f}s")

    return model


def evaluate_model(model, X_test, y_test, assets_dir: Path):
    """Evaluate the model and save the confusion matrix plot."""
    print("\n════════════════ MODEL EVALUATION ════════════════")
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")

    print("\n── Detailed Classification Report ──")
    target_names = ["Legitimate (0)", "Phishing (1)"]
    print(classification_report(y_test, y_pred, target_names=target_names))

    # ── Confusion Matrix Plot ────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=",d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
        linewidths=1,
        linecolor="white",
        cbar_kws={"label": "Count"},
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12, labelpad=10)
    ax.set_ylabel("True Label", fontsize=12, labelpad=10)
    ax.set_title(
        f"PhishGuard Confusion Matrix  —  Accuracy: {acc*100:.2f}%",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    plt.tight_layout()

    cm_path = assets_dir / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=300, bbox_inches="tight")
    print(f"\nConfusion matrix saved → {cm_path.resolve()}")
    plt.close(fig)

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def save_artifacts(model, vectorizer, model_dir: Path):
    """Persist the trained model and fitted vectorizer."""
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "logistic_regression_model.joblib"
    vec_path = model_dir / "tfidf_vectorizer.joblib"

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vec_path)

    print("\n════════════════ ARTIFACTS SAVED ════════════════")
    print(f"  Model      → {model_path.resolve()}")
    print(f"  Vectorizer → {vec_path.resolve()}")


# ── Main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "processed" / "cleaned_phishing_data.csv"
    model_dir = project_root / MODEL_DIR_NAME
    assets_dir = project_root / ASSETS_DIR_NAME

    assets_dir.mkdir(parents=True, exist_ok=True)

    # 1. Preprocess
    X_train, X_test, y_train, y_test, vectorizer = preprocess_pipeline(data_path)

    # 2. Train
    model = train_model(X_train, y_train)

    # 3. Evaluate
    metrics = evaluate_model(model, X_test, y_test, assets_dir)

    # 4. Save
    save_artifacts(model, vectorizer, model_dir)

    print("\n✔  Phase 4 complete. Model trained, evaluated, and persisted.")
