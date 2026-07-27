"""
Part B: LSTM deep-learning baseline for IMDB sentiment analysis.

Loads the same pre-split data from data/processed/ used by Part A,
tokenizes with Keras' Tokenizer, and trains a small LSTM suitable
for a CPU-only machine with 8 GB RAM.

Usage:
    python src/part_b_lstm.py
"""

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                     # non-interactive backend, safe for headless runs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent          # sentiment-analysis/
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
OUTPUTS_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
RESULTS_DIR = OUTPUTS_DIR / "results"

METRICS_CSV_PATH = RESULTS_DIR / "part_b_metrics.csv"

# Ensure directories exist
for _dir in (FIGURES_DIR, RESULTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_NUM_WORDS = 10_000          # vocabulary size for the Tokenizer
MAX_SEQUENCE_LENGTH = 200       # pad / truncate to this length
EMBEDDING_DIM = 128
LSTM_UNITS = 64
DROPOUT_RATE = 0.5
BATCH_SIZE = 64
EPOCHS = 10
VALIDATION_SPLIT = 0.10         # carve 10 % from training set
EARLY_STOPPING_PATIENCE = 2


# ---------------------------------------------------------------------------
# 1. Load the saved train/test split (identical to part_a_ml.py)
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
# 2. Tokenize with Keras Tokenizer -> sequences -> padded sequences
# ---------------------------------------------------------------------------
def tokenize_texts(
    X_train_text: pd.Series, X_test_text: pd.Series
) -> tuple[np.ndarray, np.ndarray, Tokenizer]:
    """Fit a Keras Tokenizer on the training texts, then transform both splits.

    Returns (X_train_pad, X_test_pad, fitted_tokenizer).
    """
    print("\n[Tokenizer] Fitting on training texts (num_words=10,000)...")
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train_text)
    print(f"  Vocabulary size: {min(len(tokenizer.word_index) + 1, MAX_NUM_WORDS):,}")

    print("[Tokenizer] Converting texts to sequences...")
    X_train_seq = tokenizer.texts_to_sequences(X_train_text)
    X_test_seq = tokenizer.texts_to_sequences(X_test_text)

    print(f"[Tokenizer] Padding sequences to maxlen={MAX_SEQUENCE_LENGTH}...")
    X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post")
    X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post")

    print(f"  Train shape: {X_train_pad.shape}")
    print(f"  Test shape:  {X_test_pad.shape}")
    return X_train_pad, X_test_pad, tokenizer


# ---------------------------------------------------------------------------
# 3. Build the LSTM model
# ---------------------------------------------------------------------------
def build_lstm_model() -> models.Sequential:
    """Build and compile a small LSTM suitable for CPU training.

    Architecture:
        Embedding(10k, 128) -> LSTM(64) -> Dropout(0.5) -> Dense(1, sigmoid)
    """
    model = models.Sequential(name="lstm_sentiment")
    model.add(layers.Input(shape=(MAX_SEQUENCE_LENGTH,)))
    # input_dim MUST be at least max_index + 1.  The Keras Tokenizer
    # with num_words=10k + oov_token assigns indices 1..10k (padding=0),
    # so the vocabulary size is MAX_NUM_WORDS + 1.
    model.add(
        layers.Embedding(
            input_dim=MAX_NUM_WORDS + 1,
            output_dim=EMBEDDING_DIM,
            mask_zero=True,                      # skip padding tokens efficiently
        )
    )
    model.add(layers.LSTM(LSTM_UNITS))
    model.add(layers.Dropout(DROPOUT_RATE))
    model.add(layers.Dense(1, activation="sigmoid"))

    # 4. Compile
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    return model


# ---------------------------------------------------------------------------
# 5 + 6. Train with EarlyStopping and validation split
# ---------------------------------------------------------------------------
def train_model(
    model: models.Sequential,
    X_train: np.ndarray,
    y_train: pd.Series,
) -> tuple[models.Sequential, dict, float]:
    """Train the LSTM with EarlyStopping and a validation split.

    Returns (trained_model, history.history dict, training_time_seconds).
    """
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )

    print(f"\n[Training] Starting (batch_size={BATCH_SIZE}, max_epochs={EPOCHS}, "
          f"val_split={VALIDATION_SPLIT})...")
    start = time.perf_counter()
    history = model.fit(
        X_train,
        y_train,
        validation_split=VALIDATION_SPLIT,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[early_stop],
        verbose=1,
    )
    elapsed = time.perf_counter() - start

    epochs_done = len(history.history["loss"])
    print(f"\n  Training finished after {epochs_done}/{EPOCHS} epochs")
    print(f"  Training time: {elapsed:.1f} seconds (CPU)")

    return model, history.history, elapsed


# ---------------------------------------------------------------------------
# 7. Plot training vs validation curves
# ---------------------------------------------------------------------------
def save_training_curves(history: dict, filename: str) -> None:
    """Save a two-panel figure: accuracy and loss over epochs."""
    epochs_range = range(1, len(history["accuracy"]) + 1)

    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy
    ax_acc.plot(epochs_range, history["accuracy"], "b-o", label="Training accuracy")
    ax_acc.plot(epochs_range, history["val_accuracy"], "r-o", label="Validation accuracy")
    ax_acc.set_title("Training and Validation Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.legend()
    ax_acc.grid(True, linestyle="--", alpha=0.5)

    # Loss
    ax_loss.plot(epochs_range, history["loss"], "b-o", label="Training loss")
    ax_loss.plot(epochs_range, history["val_loss"], "r-o", label="Validation loss")
    ax_loss.set_title("Training and Validation Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()
    ax_loss.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    save_path = FIGURES_DIR / filename
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  [Figure] Saved training curves -> {save_path}")


# ---------------------------------------------------------------------------
# 8. Evaluate on test set
# ---------------------------------------------------------------------------
def evaluate_model(
    model: models.Sequential,
    X_test: np.ndarray,
    y_test: pd.Series,
) -> tuple[dict, np.ndarray]:
    """Compute Accuracy, Precision, Recall, F1 on the test set.

    Returns (metrics_dict, y_pred_classes).
    """
    print("\n[Evaluation] Predicting on test set...")
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).ravel()

    metrics = {
        "model": "LSTM",
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, average="binary")), 4),
        "recall": round(float(recall_score(y_test, y_pred, average="binary")), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, average="binary")), 4),
    }

    print(f"\n  LSTM")
    print(f"  {'-' * 6}")
    print(f"    Accuracy :  {metrics['accuracy']:.4f}")
    print(f"    Precision:  {metrics['precision']:.4f}")
    print(f"    Recall   :  {metrics['recall']:.4f}")
    print(f"    F1-score :  {metrics['f1_score']:.4f}")

    print(f"\n  Classification report:\n")
    print(classification_report(y_test, y_pred, target_names=["negative", "positive"]))

    return metrics, y_pred


# ---------------------------------------------------------------------------
# 9. Confusion matrix
# ---------------------------------------------------------------------------
def save_confusion_matrix(
    y_test: pd.Series, y_pred: np.ndarray, filename: str
) -> None:
    """Plot a labelled confusion-matrix heatmap and save to outputs/figures/."""
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)  # row-wise %

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_norm,
        annot=cm,
        fmt="d",
        cmap="Blues",
        xticklabels=["negative", "positive"],
        yticklabels=["negative", "positive"],
        cbar_kws={"label": "Proportion"},
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix — LSTM")

    fig.tight_layout()
    save_path = FIGURES_DIR / filename
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  [Figure] Saved confusion matrix -> {save_path}")


# ---------------------------------------------------------------------------
# 10. Save results CSV
# ---------------------------------------------------------------------------
def save_metrics_csv(metrics: dict, path: Path) -> None:
    """Write a CSV with one row for the LSTM model."""
    df = pd.DataFrame([metrics])
    df.to_csv(path, index=False)
    print(f"  [Metrics] Saved -> {path}")


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("=" * 60)
    print("  Part B — LSTM Sentiment Classifier")
    print("=" * 60)

    # 1. Load data
    print("\n[1/6] Loading pre-split data...")
    train, test = load_split()

    X_train_text = train["cleaned_review"]
    y_train = train["sentiment"]
    X_test_text = test["cleaned_review"]
    y_test = test["sentiment"]

    # 2. Tokenize + pad
    print("\n[2/6] Tokenizing and padding sequences...")
    X_train_pad, X_test_pad, tokenizer = tokenize_texts(X_train_text, X_test_text)

    # 3 + 4. Build and compile the LSTM model
    print("\n[3/6] Building LSTM model...")
    model = build_lstm_model()

    # 5 + 6. Train with EarlyStopping
    print("\n[4/6] Training LSTM...")
    model, history, training_time = train_model(model, X_train_pad, y_train)

    # 7. Plot curves
    print("\n[5/6] Generating outputs...")
    save_training_curves(history, "part_b_lstm_curves.png")

    # 8. Evaluate
    print("\n[6/6] Evaluating on test set...")
    metrics, y_pred = evaluate_model(model, X_test_pad, y_test)

    # 9. Confusion matrix
    save_confusion_matrix(y_test, y_pred, "part_b_lstm_confusion_matrix.png")

    # 10. Save metrics
    save_metrics_csv(metrics, METRICS_CSV_PATH)

    # 11. Print total training time
    print("\n" + "=" * 60)
    print("  TRAINING SUMMARY")
    print("=" * 60)
    print(f"  Total training time:  {training_time:.1f} seconds")
    print(f"  Platform:             CPU")
    print(f"  Epochs completed:     {len(history['loss'])} / {EPOCHS}")
    print(f"  Test accuracy:        {metrics['accuracy']:.4f}")
    print(f"  Test F1-score:        {metrics['f1_score']:.4f}")
    print("=" * 60)
    print("Done. Part B complete.")


if __name__ == "__main__":
    main()
