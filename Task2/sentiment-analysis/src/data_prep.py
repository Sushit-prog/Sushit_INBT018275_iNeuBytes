"""
Data preparation pipeline for IMDB sentiment analysis.

Loads the raw IMDB Dataset.csv, cleans review text, splits into
train/test (80/20), and saves the processed data to data/processed/
so that Part A and Part B both load the same canonical split.

Usage:
    python src/data_prep.py
"""

import re
import string
import sys
from pathlib import Path

import nltk
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# NLTK resource downloads — wrapped in try/except so they only download once
# ---------------------------------------------------------------------------
# NLTK 3.9+ moved tokenizer models to ``punkt_tab``; try both so the
# script works across NLTK versions.
for _resource in ("stopwords", "punkt", "punkt_tab"):
    try:
        if _resource in ("punkt", "punkt_tab"):
            nltk.data.find(f"tokenizers/{_resource}")
        else:
            nltk.data.find(f"corpora/{_resource}")
    except LookupError:
        nltk.download(_resource, quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Script is at  sentiment-analysis/src/data_prep.py
# Project root is sentiment-analysis/
ROOT_DIR = Path(__file__).resolve().parent.parent

# The user-specified path is ``data/IMDB Dataset.csv`` (relative to the project root).
# If the file is not there, fall back to ``../IMDB Dataset.csv`` (Task2/ level)
# so the script works immediately without manual copying.
_DEFAULT_RAW_PATH = ROOT_DIR / "data" / "IMDB Dataset.csv"
_FALLBACK_RAW_PATH = ROOT_DIR.parent / "IMDB Dataset.csv"

def _resolve_raw_data_path() -> Path:
    if _DEFAULT_RAW_PATH.exists():
        return _DEFAULT_RAW_PATH
    if _FALLBACK_RAW_PATH.exists():
        print(f"  [INFO] Using fallback path: {_FALLBACK_RAW_PATH}")
        return _FALLBACK_RAW_PATH
    # Neither exists — return the user-specified path so the error message is clear
    return _DEFAULT_RAW_PATH

RAW_DATA_PATH = _resolve_raw_data_path()

PROCESSED_DIR = ROOT_DIR / "data" / "processed"

# ---------------------------------------------------------------------------
# Stopwords set (cached globally so every call to clean_text reuses it)
# ---------------------------------------------------------------------------
_STOPWORDS = set(stopwords.words("english"))

# Regex that matches any HTML tag (e.g. <br />, <p>, </div>)
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def load_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the IMDB CSV into a DataFrame.

    The file is expected to have columns ``review`` (str) and
    ``sentiment`` (str with values ``positive`` / ``negative``).
    """
    if not path.exists():
        print(f"[ERROR] Raw data file not found at: {path.resolve()}")
        print("Make sure IMDB Dataset.csv is in the expected location.")
        sys.exit(1)
    df = pd.read_csv(path)
    # Strip any accidental whitespace from column names
    df.columns = df.columns.str.strip()
    required = {"review", "sentiment"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        print(f"[ERROR] Missing required columns: {missing}")
        sys.exit(1)
    return df


def map_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Map ``sentiment`` column: ``positive`` → 1, ``negative`` → 0."""
    mapping = {"positive": 1, "negative": 0}
    df = df.copy()
    df["sentiment"] = df["sentiment"].str.strip().str.lower().map(mapping)
    unmapped = df["sentiment"].isna().sum()
    if unmapped:
        print(f"[WARNING] {unmapped} row(s) had unrecognised sentiment values and will be dropped.")
        df = df.dropna(subset=["sentiment"])
    df["sentiment"] = df["sentiment"].astype(int)
    return df


def print_class_distribution(df: pd.DataFrame, label: str = "Full dataset") -> None:
    """Print count and percentage for each sentiment class."""
    print(f"\n--- Class distribution: {label} ---")
    counts = df["sentiment"].value_counts().sort_index()
    total = len(df)
    for sentiment_val in (0, 1):
        cnt = counts.get(sentiment_val, 0)
        pct = 100.0 * cnt / total
        label_name = "negative" if sentiment_val == 0 else "positive"
        print(f"  {label_name:>10} ({sentiment_val}): {cnt:>6,}  ({pct:5.2f}%)")

    # Check for imbalance
    ratio = counts.get(0, 0) / max(counts.get(1, 1), 1)
    if ratio < 0.8 or ratio > 1.25:
        print(
            "  [WARNING] Dataset is imbalanced. F1-score (macro) will be the "
            "primary metric for judging performance."
        )
    else:
        print("  [OK] Dataset is reasonably balanced.")


def clean_text(text: str) -> str:
    """Apply the full cleaning pipeline to a single review string.

    Steps
    -----
    1. Remove HTML tags (e.g. ``<br />``)
    2. Remove punctuation
    3. Lowercase
    4. Remove extra whitespace
    5. Remove stopwords
    """
    # 1. Remove HTML tags
    text = _HTML_TAG_RE.sub(" ", text)
    # 2. Remove punctuation (replace with space so words don't merge)
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    # 3. Lowercase
    text = text.lower()
    # 4. Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # 5. Remove stopwords (requires tokenizing)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in _STOPWORDS and len(t) > 0]
    return " ".join(tokens)


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------
    print("Loading data...")
    df = load_data()
    print(f"  Loaded {len(df):,} rows from {RAW_DATA_PATH.resolve()}")

    # ------------------------------------------------------------------
    # 2. Map sentiment to binary
    # ------------------------------------------------------------------
    df = map_sentiment(df)
    print(f"  After sentiment mapping: {len(df):,} rows")

    # ------------------------------------------------------------------
    # 3. Class distribution (before cleaning — raw distribution)
    # ------------------------------------------------------------------
    print_class_distribution(df, "Raw dataset")

    # ------------------------------------------------------------------
    # 4. Clean text (includes tokenisation & stopword removal)
    # ------------------------------------------------------------------
    print("\nCleaning review text...")
    df["cleaned_review"] = df["review"].apply(clean_text)
    print(f"  Cleaning complete. Sample cleaned review:")
    print(f"    {df['cleaned_review'].iloc[0][:200]}...")

    # ------------------------------------------------------------------
    # 5. Tokenisation note — the cleaning step above already tokenises
    #    and removes stopwords; we save the rejoined cleaned text.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # 6. Train/test split (80/20) — deterministic seed so Part A & B
    #    can reproduce the exact same split by calling the same seed.
    # ------------------------------------------------------------------
    print("\nSplitting into train/test (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        df["cleaned_review"],
        df["sentiment"],
        test_size=0.20,
        random_state=42,
        stratify=df["sentiment"],  # preserve class proportions
    )
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # ------------------------------------------------------------------
    # 7. Save to data/processed/
    # ------------------------------------------------------------------
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.DataFrame({"cleaned_review": X_train, "sentiment": y_train})
    test_df = pd.DataFrame({"cleaned_review": X_test, "sentiment": y_test})

    train_path = PROCESSED_DIR / "train.csv"
    test_path = PROCESSED_DIR / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\n  Saved train set  -> {train_path}  ({train_df.shape})")
    print(f"  Saved test set   -> {test_path}  ({test_df.shape})")

    # ------------------------------------------------------------------
    # 8. Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("  PREPROCESSING SUMMARY")
    print("=" * 55)
    print(f"  Total samples:              {len(df):>7,}")
    print(f"  Train samples:              {len(X_train):>7,}")
    print(f"  Test samples:               {len(X_test):>7,}")
    print(f"  Raw features:               cleaned review text")
    print(f"  Target:                     sentiment (0=negative, 1=positive)")
    print()
    print_class_distribution(train_df, "Train set")
    print_class_distribution(test_df, "Test set")
    print("=" * 55)


if __name__ == "__main__":
    main()
