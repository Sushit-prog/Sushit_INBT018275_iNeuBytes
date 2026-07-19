"""Figure generation shared by both parts. Architecture diagrams need
pydot + graphviz; if either is missing (common on a fresh Windows
machine), we fall back to a text summary rather than crashing the run."""

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from src import config


def save_architecture_diagram(model: tf.keras.Model, filename: str) -> None:
    png_path = config.FIGURES_DIR / filename
    try:
        tf.keras.utils.plot_model(
            model, to_file=png_path, show_shapes=True, show_layer_names=True
        )
    except ImportError:
        txt_path = png_path.with_suffix(".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            model.summary(print_fn=lambda line: f.write(line + "\n"))
        print(
            f"[visualize] pydot/graphviz not installed — wrote text summary "
            f"to {txt_path} instead of a diagram."
        )


def save_training_curves(history: tf.keras.callbacks.History, filename: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / filename, dpi=150)
    plt.close(fig)


def save_confusion_matrix(cm: np.ndarray, filename: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(config.NUM_CLASSES))
    ax.set_yticks(range(config.NUM_CLASSES))
    ax.set_xticklabels(config.CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(config.CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)

    threshold = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            ax.text(j, i, cm[i, j], ha="center", va="center", color=color, fontsize=7)

    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / filename, dpi=150)
    plt.close(fig)