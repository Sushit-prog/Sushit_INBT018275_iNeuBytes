"""
Architecture diagrams for the IMDB sentiment analysis report.

Generates two presentation-quality PNG figures:
    1. ml_pipeline_diagram.png  —  data-flow diagram of the ML pipeline
       (Raw Text -> Cleaning -> TF-IDF -> [Logistic Regression, SVM])
    2. lstm_architecture_diagram.png  —  layer diagram of the LSTM model
       (Input -> Tokenizer -> Embedding -> LSTM -> Dropout -> Dense -> Output)

Usage:
    python src/generate_diagrams.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = ROOT_DIR / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
COLORS = {
    "input":      "#3498db",   # blue
    "process":    "#2ecc71",   # green
    "ml_model":   "#e67e22",   # orange
    "output":     "#9b59b6",   # purple
    "text":       "#2c3e50",   # dark grey
    "arrow":      "#7f8c8d",   # grey
    "bg":         "#fafafa",   # off-white
}


def _rounded_box(ax, xy, width, height, color, label, fontsize=10, text_color="white"):
    """Draw a rounded rectangle with centred label."""
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.15",
        facecolor=color,
        edgecolor=color,
        linewidth=1.5,
        zorder=3,
    )
    ax.add_patch(box)
    cx = xy[0] + width / 2
    cy = xy[1] + height / 2
    ax.text(
        cx, cy, label,
        ha="center", va="center",
        fontsize=fontsize, fontweight="bold",
        color=text_color, zorder=4,
    )
    return box


def _arrow(ax, x1, y1, x2, y2, color=None):
    """Draw a curved arrow from (x1,y1) to (x2,y2)."""
    if color is None:
        color = COLORS["arrow"]
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        color=color,
        linewidth=2,
        mutation_scale=20,
        zorder=2,
    )
    ax.add_patch(arrow)


def _title(ax, text, y=0.94):
    """Add a bold title at the top of the figure."""
    ax.text(
        0.5, y, text,
        ha="center", va="center",
        fontsize=14, fontweight="bold",
        color=COLORS["text"],
        transform=ax.transAxes,
    )


# ===========================================================================
# Diagram 1: ML Pipeline Flow
# ===========================================================================
def draw_ml_pipeline_diagram():
    """Flow diagram showing the classical ML pipeline."""
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.5)
    ax.axis("off")
    ax.set_facecolor(COLORS["bg"])
    fig.patch.set_facecolor(COLORS["bg"])

    _title(ax, "Traditional ML Pipeline — Sentiment Classification", y=0.96)

    # ---- Nodes ----
    bw, bh = 2.2, 0.65  # box width, box height
    gap = 0.5

    # Row 1: preprocessing
    y1 = 3.8
    x_positions_pipe = [0.4, 2.8, 5.2, 7.6]
    pipe_nodes = [
        ("Raw Text", COLORS["input"]),
        ("Cleaning\n(lowercase, remove\npunctuation, stopwords)", COLORS["process"]),
        ("Tokenization\n& Normalization", COLORS["process"]),
        ("TF-IDF\nVectorization\n(max_features=10k)", COLORS["process"]),
    ]

    for i, (label, color) in enumerate(pipe_nodes):
        x = x_positions_pipe[i]
        _rounded_box(ax, (x, y1), bw, bh, color, label, fontsize=9)

    # Arrows between preprocessing nodes
    for i in range(len(x_positions_pipe) - 1):
        x1 = x_positions_pipe[i] + bw
        x2 = x_positions_pipe[i + 1]
        _arrow(ax, x1, y1 + bh / 2, x2, y1 + bh / 2)

    # Row 2: branching to models
    y2 = 1.8
    bw2 = 2.8
    bh2 = 0.75

    # Down arrow from TF-IDF to models
    tfidf_x = x_positions_pipe[3] + bw / 2
    _arrow(ax, tfidf_x, y1, tfidf_x, y2 + bh2)

    # Two models side by side
    model_x_left = 1.5
    model_x_right = 4.5
    _rounded_box(ax, (model_x_left, y2), bw2, bh2, COLORS["ml_model"],
                 "Logistic\nRegression\n(max_iter=1000)", fontsize=9)
    _rounded_box(ax, (model_x_right, y2), bw2, bh2, COLORS["ml_model"],
                 "LinearSVC\n(SVM classifier)", fontsize=9)

    # Arrow splits from TF-IDF to both models
    split_y = y2 + bh2 + 0.3
    _arrow(ax, tfidf_x, split_y, model_x_left + bw2 / 2, split_y)
    _arrow(ax, tfidf_x, split_y, model_x_right + bw2 / 2, split_y)

    # Row 3: predictions
    y3 = 0.4
    pred_x_left = model_x_left + bw2 / 2 - bw / 2
    pred_x_right = model_x_right + bw2 / 2 - bw / 2

    _arrow(ax, model_x_left + bw2 / 2, y2, model_x_left + bw2 / 2, y3 + bh)
    _arrow(ax, model_x_right + bw2 / 2, y2, model_x_right + bw2 / 2, y3 + bh)

    _rounded_box(ax, (pred_x_left, y3), bw, bh, COLORS["output"],
                 "Predictions\n(pos/neg)", fontsize=9)
    _rounded_box(ax, (pred_x_right, y3), bw, bh, COLORS["output"],
                 "Predictions\n(pos/neg)", fontsize=9)

    # Legend
    legend_y = 0.15
    legend_items = [
        ("Input / Output", COLORS["input"]),
        ("Preprocessing", COLORS["process"]),
        ("ML Model", COLORS["ml_model"]),
        ("Prediction", COLORS["output"]),
    ]
    for i, (lbl, clr) in enumerate(legend_items):
        lx = 7.8 + i * 1.05
        ax.add_patch(plt.Rectangle((lx, legend_y), 0.2, 0.2, facecolor=clr, edgecolor=clr, zorder=3))
        ax.text(lx + 0.28, legend_y + 0.1, lbl, fontsize=8, va="center", color=COLORS["text"])

    fig.tight_layout()
    path = FIGURES_DIR / "ml_pipeline_diagram.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    print(f"  [Diagram] Saved ML pipeline diagram -> {path}")


# ===========================================================================
# Diagram 2: LSTM Architecture
# ===========================================================================
def draw_lstm_architecture_diagram():
    """Vertical layer diagram of the LSTM model."""
    fig, ax = plt.subplots(figsize=(8, 9))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.set_facecolor(COLORS["bg"])
    fig.patch.set_facecolor(COLORS["bg"])

    _title(ax, "LSTM Architecture — Sentiment Classification", y=0.97)

    # Layer definitions: (y_center, label, color, width, height, fontsize)
    bw = 4.5    # standard box width
    bh = 0.6    # standard box height
    cx = 4.0    # centre x

    layers = [
        (7.8,  "Input Text (cleaned review)", COLORS["input"], bw, bh, 9),
        (6.8,  "Tokenizer\n(num_words=10,000)", COLORS["process"], bw, bh, 9),
        (5.8,  "Padded Sequences\n(maxlen=200)", COLORS["process"], bw, bh, 9),
        (4.65, "Embedding Layer\n(input_dim=10001, output_dim=128)", "#2980b9", bw, bh, 9),
        (3.45, "LSTM(64)\n(64 hidden units)", "#8e44ad", bw, bh, 9),
        (2.45, "Dropout(0.5)\n(50% neurons dropped)", "#d35400", bw, bh, 9),
        (1.45, "Dense(1, sigmoid)\n(binary classification)", "#c0392b", bw, bh, 9),
        (0.5,  "Output\n(Positive / Negative)", COLORS["output"], bw, bh, 10),
    ]

    for y_center, label, color, w, h, fs in layers:
        x = cx - w / 2
        _rounded_box(ax, (x, y_center - h / 2), w, h, color, label, fontsize=fs)

    # Arrows connecting layers (vertical, centre to centre)
    for i in range(len(layers) - 1):
        y1 = layers[i][0] - layers[i][4] / 2  # bottom of current layer
        y2 = layers[i + 1][0] + layers[i + 1][4] / 2  # top of next layer
        _arrow(ax, cx, y1, cx, y2)

    # Side annotation: data shape info
    annotations = [
        ("Shape: (batch,)", 7.3),
        ("Shape: (batch,)", 6.3),
        ("Shape: (batch, 200)", 5.3),
        ("Shape: (batch, 200, 128)", 4.15),
        ("Shape: (batch, 64)", 2.95),
        ("Shape: (batch, 64)", 1.95),
        ("Shape: (batch, 1)", 0.95),
    ]
    for text, y in annotations:
        ax.text(
            7.2, y, text,
            fontsize=7.5, fontstyle="italic",
            color="#7f8c8d", ha="right", va="center",
        )

    fig.tight_layout()
    path = FIGURES_DIR / "lstm_architecture_diagram.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    print(f"  [Diagram] Saved LSTM architecture diagram -> {path}")


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("=" * 60)
    print("  Generating architecture diagrams...")
    print("=" * 60)
    draw_ml_pipeline_diagram()
    draw_lstm_architecture_diagram()
    print("\nDone. Both diagrams saved to outputs/figures/.")


if __name__ == "__main__":
    main()
