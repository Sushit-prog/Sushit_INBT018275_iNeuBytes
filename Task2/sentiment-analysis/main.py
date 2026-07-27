"""
Pipeline orchestrator for IMDB sentiment analysis.

Runs the full pipeline end-to-end:
    1. data_prep.py   — load raw CSV, clean text, split, save
    2. part_a_ml.py   — TF-IDF + Logistic Regression + LinearSVC
    3. part_b_lstm.py — Tokenizer + LSTM classifier
    4. utils.py       — build & save final comparison table

Usage:
    python main.py
"""

import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT_DIR / "src"

SCRIPTS = [
    ("data_prep.py",     "DATA PREPARATION"),
    ("part_a_ml.py",     "PART A  —  ML Baselines"),
    ("part_b_lstm.py",   "PART B  —  LSTM Classifier"),
]

FINAL_SCRIPT = ("utils.py", "FINAL COMPARISON TABLE")


def run_script(script_name: str, stage_label: str) -> None:
    """Run a script via ``python src/<script_name>`` and time it."""
    script_path = SCRIPTS_DIR / script_name
    if not script_path.exists():
        print(f"\n[FATAL] Script not found: {script_path}")
        sys.exit(1)

    header = f"  {'=' * 60}"
    print(f"\n{header}")
    print(f"  >>> {stage_label}")
    print(f"{header}\n")

    start = time.perf_counter()
    # Use exec to keep the same Python process, or subprocess for isolation.
    # subprocess is cleaner for long-running scripts with different imports.
    import subprocess
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=False,  # let stdout/stderr flow through to the console
    )
    elapsed = time.perf_counter() - start

    if result.returncode != 0:
        print(f"\n[FAIL] {script_name} exited with code {result.returncode}")
        sys.exit(result.returncode)

    print(f"\n  [OK] {script_name} completed in {elapsed:.1f} seconds.\n")


def main():
    print("=" * 70)
    print("  IMDB SENTIMENT ANALYSIS  —  Full Pipeline")
    print("  CPU-only | 8 GB RAM")
    print("=" * 70)

    pipeline_start = time.perf_counter()

    # Stages 1–3
    for script_name, label in SCRIPTS:
        run_script(script_name, label)

    # Stage 4: comparison (import function rather than subprocess)
    print(f"\n  {'=' * 60}")
    print("  >>> FINAL COMPARISON TABLE")
    print(f"  {'=' * 60}\n")

    from src.utils import build_comparison_table, load_metrics, print_table, save_csv, save_table_png
    df = load_metrics()
    table = build_comparison_table(df)
    note = print_table(table)
    save_csv(table)
    save_table_png(table)

    total_elapsed = time.perf_counter() - pipeline_start

    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Total pipeline time:  {total_elapsed:.1f} seconds")
    print(f"  Note:                  {note}")
    print("=" * 70)


if __name__ == "__main__":
    main()
