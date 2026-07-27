"""
Part A: Classic ML baselines for IMDB sentiment analysis.

Loads the pre-split data from data/processed/, applies TF-IDF vectorization,
and trains two classifiers: Logistic Regression and LinearSVC.  Results
(metrics, confusion-matrix figures, and the fitted vectorizer) are saved
to outputs/ for later comparison with Part B.

Usage:
    python src/part_a_ml.py
"""

import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")                     # non-interactive backend, safe for headless runs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.svm import LinearSVC

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent          # sentiment-analysis/
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
OUTPUTS_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
RESULTS_DIR = OUTPUTS_DIR / "results"

MODEL_PATHS = {
    "logistic_regression": RESULTS_DIR / "logistic_regression.joblib",
    "linear_svc": RESULTS_DIR / "linear_svc.joblib",
}
VECTORIZER_PATH = RESULTS_DIR / "tfidf_vectorizer.joblib"
METRICS_CSV_PATH = RESULTS_DIR / "part_a_metrics.csv"

# Ensure output directories exist
for _dir in (FIGURES_DIR, RESULTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Load the saved train/test split
# ---------------------------------------------------------------------------
def load_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read train.csv and test.csv from data/processed/."""
    train_path = PROCESSED_DIR / "train.csv"
    test_path = PROCESSED_DIR / "test.csv"

    for p in (train_path, test_path):
        if not p.exists():
            print(f"[ERROR] Expected file not found: {p}")
            print("Run `python src/data_prep.py` first to create the split.")
            sys.exit(1)

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    print(f"  Loaded train set: {train.shape[0]:,} rows")
    print(f"  Loaded test set:  {test.shape[0]:,} rows")
    return train, test


# ---------------------------------------------------------------------------
# 2. TF-IDF vectorization
# ---------------------------------------------------------------------------
def vectorize(
    X_train_text: pd.Series, X_test_text: pd.Series
) -> tuple[np.ndarray, np.ndarray, TfidfVectorizer]:
    """Fit TF-IDF on the training set, transform both train and test.

    Returns (X_train_tfidf, X_test_tfidf, fitted_vectorizer).
    """
    print("\n[TF-IDF] Fitting vectorizer (max_features=10 000, ngram_range=(1,2))...")
    vectorizer = TfidfVectorizer(
        max_features=10_000,
        ngram_range=(1, 2),
        sublinear_tf=True,            # use 1 + log(tf) for better scaling
    )
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)
    print(f"  Train shape: {X_train.shape}")
    print(f"  Test shape:  {X_test.shape}")
    return X_train, X_test, vectorizer


# ---------------------------------------------------------------------------
# 3 & 4. Train models
# ---------------------------------------------------------------------------
def train_logistic_regression(X_train: np.ndarray, y_train: pd.Series):
    """Train a Logistic Regression classifier."""
    print("\n[Logistic Regression] Training (max_iter=1000)...")
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    print("  [OK] Logistic Regression trained.")
    return clf


def train_linear_svc(X_train: np.ndarray, y_train: pd.Series):
    """Train a LinearSVC classifier — fast on sparse TF-IDF features."""
    print("\n[LinearSVC] Training...")
    clf = LinearSVC(random_state=42, max_iter=2000, dual="auto")
    clf.fit(X_train, y_train)
    print("  [OK] LinearSVC trained.")
    return clf


# ---------------------------------------------------------------------------
# 5. Compute and print metrics
# ---------------------------------------------------------------------------
def evaluate_model(
    clf, X_test: np.ndarray, y_test: pd.Series, model_name: str
) -> dict:
    """Compute accuracy, precision, recall, F1 and print them."""
    y_pred = clf.predict(X_test)

    metrics = {
        "model": model_name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, average="binary"), 4),
        "recall": round(recall_score(y_test, y_pred, average="binary"), 4),
        "f1_score": round(f1_score(y_test, y_pred, average="binary"), 4),
    }

    print(f"\n  {model_name}")
    print(f"  {'-' * (len(model_name) + 2)}")
    print(f"    Accuracy :  {metrics['accuracy']:.4f}")
    print(f"    Precision:  {metrics['precision']:.4f}")
    print(f"    Recall   :  {metrics['recall']:.4f}")
    print(f"    F1-score :  {metrics['f1_score']:.4f}")

    # Full classification report for richer diagnostics
    print(f"\n  Classification report:\n")
    print(classification_report(y_test, y_pred, target_names=["negative", "positive"]))

    return metrics, y_pred


# ---------------------------------------------------------------------------
# 6. Generate and save confusion matrices
# ---------------------------------------------------------------------------
def save_confusion_matrix(
    y_test: pd.Series, y_pred: np.ndarray, model_name: str, filename: str
) -> None:
    """Plot a labelled confusion-matrix heatmap and save to outputs/figures/."""
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)  # row-wise %

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_norm,
        annot=cm,                      # show raw counts inside cells
        fmt="d",
        cmap="Blues",
        xticklabels=["negative", "positive"],
        yticklabels=["negative", "positive"],
        cbar_kws={"label": "Proportion"},
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Confusion Matrix — {model_name}")

    fig.tight_layout()
    save_path = FIGURES_DIR / filename
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  [Figure] Saved confusion matrix -> {save_path}")


# ---------------------------------------------------------------------------
# 7. Save results summary CSV
# ---------------------------------------------------------------------------
def save_metrics_csv(metrics_list: list[dict], path: Path) -> None:
    """Write a CSV with one row per model."""
    df = pd.DataFrame(metrics_list)
    df.to_csv(path, index=False)
    print(f"  [Metrics] Saved -> {path}")


# ---------------------------------------------------------------------------
# 8. Save the fitted vectorizer (joblib)
# ---------------------------------------------------------------------------
def save_vectorizer(vectorizer: TfidfVectorizer, path: Path) -> None:
    joblib.dump(vectorizer, path)
    print(f"  [Vectorizer] Saved -> {path}")


# ---------------------------------------------------------------------------
# 9. Print comparison
# ---------------------------------------------------------------------------
def print_comparison(metrics_list: list[dict]) -> None:
    """Print a side-by-side comparison of Logistic Regression vs SVM."""
    if len(metrics_list) != 2:
        return

    lr = {m["model"]: m for m in metrics_list}.get("Logistic Regression", {})
    svm = {m["model"]: m for m in metrics_list}.get("LinearSVC", {})

    header = f"{'Metric':<15} {'Logistic Regression':<22} {'LinearSVC':<15} {'Winner':<10}"
    sep = "-" * len(header)

    print("\n" + "=" * 65)
    print("  MODEL COMPARISON  —  Logistic Regression  vs  LinearSVC")
    print("=" * 65)
    print(header)
    print(sep)

    for metric in ("accuracy", "precision", "recall", "f1_score"):
        lr_val = lr.get(metric, 0)
        svm_val = svm.get(metric, 0)
        winner = "LR" if lr_val >= svm_val else "SVM"
        print(
            f"{metric.capitalize():<15} {lr_val:<22.4f} {svm_val:<15.4f} {winner:<10}"
        )

    print(sep)
    print()


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("=" * 60)
    print("  Part A — Classic ML Baselines")
    print("=" * 60)

    # 1. Load data
    print("\n[1/6] Loading pre-split data...")
    train, test = load_split()

    X_train_text = train["cleaned_review"]
    y_train = train["sentiment"]
    X_test_text = test["cleaned_review"]
    y_test = test["sentiment"]

    # 2. TF-IDF
    print("\n[2/6] Vectorizing with TF-IDF...")
    X_train_tfidf, X_test_tfidf, vectorizer = vectorize(X_train_text, X_test_text)

    # 3 + 4. Train both models
    print("\n[3/6] Training Logistic Regression...")
    lr_clf = train_logistic_regression(X_train_tfidf, y_train)

    print("\n[4/6] Training LinearSVC...")
    svm_clf = train_linear_svc(X_train_tfidf, y_train)

    # 5. Evaluate both
    print("\n[5/6] Evaluating models...")
    lr_metrics, lr_pred = evaluate_model(lr_clf, X_test_tfidf, y_test, "Logistic Regression")
    svm_metrics, svm_pred = evaluate_model(svm_clf, X_test_tfidf, y_test, "LinearSVC")

    # 6. Confusion matrices
    print("\n[6/6] Generating outputs...")
    save_confusion_matrix(
        y_test, lr_pred, "Logistic Regression", "part_a_lr_confusion_matrix.png"
    )
    save_confusion_matrix(
        y_test, svm_pred, "LinearSVC", "part_a_svm_confusion_matrix.png"
    )

    # 7. Metrics CSV
    save_metrics_csv([lr_metrics, svm_metrics], METRICS_CSV_PATH)

    # 8. Save vectorizer
    save_vectorizer(vectorizer, VECTORIZER_PATH)

    # 9. Comparison
    print_comparison([lr_metrics, svm_metrics])

    print("Done. Part A complete.")


if __name__ == "__main__":
    main()
