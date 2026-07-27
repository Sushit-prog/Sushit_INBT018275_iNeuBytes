"""
Utility functions for the sentiment analysis pipeline.

The main function ``build_comparison_table()`` reads the saved metrics
from Part A and Part B, produces a three-model comparison (Logistic
Regression, SVM, LSTM), and saves the results as both CSV and a
formatted PNG table suitable for inclusion in reports.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "outputs" / "results"
FIGURES_DIR = ROOT_DIR / "outputs" / "figures"

PART_A_CSV = RESULTS_DIR / "part_a_metrics.csv"
PART_B_CSV = RESULTS_DIR / "part_b_metrics.csv"
FINAL_CSV = RESULTS_DIR / "final_comparison.csv"
FINAL_PNG = FIGURES_DIR / "final_comparison_table.png"


def load_metrics() -> pd.DataFrame:
    """Load Part A and Part B metrics CSVs and return a single DataFrame.

    Returns
    -------
    pd.DataFrame
        Columns: model, accuracy, precision, recall, f1_score
        Rows: one per model (Logistic Regression, LinearSVC, LSTM).
    """
    for csv_path, label in [(PART_A_CSV, "Part A"), (PART_B_CSV, "Part B")]:
        if not csv_path.exists():
            print(f"[ERROR] {label} metrics not found at: {csv_path}")
            print("Make sure both part_a_ml.py and part_b_lstm.py have been run.")
            sys.exit(1)

    df_a = pd.read_csv(PART_A_CSV)
    df_b = pd.read_csv(PART_B_CSV)
    combined = pd.concat([df_a, df_b], ignore_index=True)
    return combined


def build_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build a clean comparison table with columns renamed for readability."""
    table = df.copy()
    table = table.rename(
        columns={
            "model": "Model",
            "accuracy": "Accuracy",
            "precision": "Precision",
            "recall": "Recall",
            "f1_score": "F1-score",
        }
    )
    return table


def print_table(table: pd.DataFrame) -> str:
    """Print the comparison table to console in a clean format.

    Returns the note about LSTM vs best classical F1 as a string.
    """
    print("\n" + "=" * 75)
    print("  FINAL MODEL COMPARISON  —  Logistic Regression  vs  LinearSVC  vs  LSTM")
    print("=" * 75)
    print(table.to_string(index=False))
    print("=" * 75)

    # Identify the best classical (non-LSTM) model by F1
    classical = table[table["Model"] != "LSTM"]
    best_classical = classical.loc[classical["F1-score"].idxmax()]
    lstm_row = table[table["Model"] == "LSTM"].iloc[0]

    best_f1 = best_classical["F1-score"]
    lstm_f1 = lstm_row["F1-score"]
    best_name = best_classical["Model"]

    if lstm_f1 >= best_f1:
        note = (
            f"  The LSTM (F1={lstm_f1:.4f}) matched or exceeded the best "
            f"classical model ({best_name}, F1={best_f1:.4f})."
        )
    else:
        diff = best_f1 - lstm_f1
        note = (
            f"  The best classical model ({best_name}, F1={best_f1:.4f}) "
            f"outperforms the LSTM (F1={lstm_f1:.4f}) by {diff:.4f}."
        )

    print(note)
    print("=" * 75)
    return note


def save_csv(table: pd.DataFrame, path: Path = FINAL_CSV) -> None:
    """Save the comparison table as CSV."""
    table.to_csv(path, index=False)
    print(f"  [CSV] Saved comparison -> {path}")


def save_table_png(table: pd.DataFrame, path: Path = FINAL_PNG) -> None:
    """Render the comparison table as a styled PNG image for reports.

    Uses matplotlib to draw a clean table with colour-coded headers
    and alternating row colours.
    """
    col_labels = table.columns.tolist()
    row_labels = table["Model"].tolist()
    cell_text = [
        [f"{v:.4f}" if isinstance(v, float) else str(v) for v in row]
        for row in table.values
    ]

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")

    # Colour palette
    header_color = "#2c3e50"       # dark blue-grey
    header_text_color = "white"
    row_colors = ["#ecf0f1", "#ffffff"]  # light grey / white
    edge_color = "#bdc3c7"

    the_table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )

    # Style header row
    for j in range(len(col_labels)):
        cell = the_table[0, j]
        cell.set_facecolor(header_color)
        cell.set_text_props(color=header_text_color, fontweight="bold", fontsize=11)
        cell.set_edgecolor(edge_color)

    # Style data rows
    for i in range(len(row_labels)):
        bg = row_colors[i % 2]
        for j in range(len(col_labels)):
            cell = the_table[i + 1, j]
            cell.set_facecolor(bg)
            cell.set_edgecolor(edge_color)
            cell.set_text_props(fontsize=10)
            # Bold the model name in the first column
            if j == 0:
                cell.set_text_props(fontweight="bold", fontsize=10)

    # Scale font in cells
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)

    # Adjust column widths
    the_table.auto_set_column_width(col=list(range(len(col_labels))))

    # Title
    ax.set_title(
        "IMDB Sentiment Analysis — Model Performance Comparison",
        fontsize=13,
        fontweight="bold",
        pad=20,
        color="#2c3e50",
    )

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [PNG] Saved comparison table image -> {path}")


# ===========================================================================
# Convenience: run as standalone
# ===========================================================================
def main():
    """Load metrics, build the comparison table, print it, and save outputs."""
    print("=" * 75)
    print("  Building final comparison table...")
    print("=" * 75)

    df = load_metrics()
    table = build_comparison_table(df)
    note = print_table(table)
    save_csv(table)
    save_table_png(table)
    print("\nDone. All comparison outputs saved.")


if __name__ == "__main__":
    main()
