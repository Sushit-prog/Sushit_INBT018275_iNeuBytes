"""
Loads CIFAR-10, normalizes pixel values, and produces a FIXED
train/val/test split (seeded) that both Part A and Part B reuse.
"""

from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from src import config


@dataclass
class Dataset:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def load_cifar10_split() -> Dataset:
    """Loads CIFAR-10, normalizes to [0, 1], and carves a fixed val split.

    The split is deterministic given config.SEED, so calling this from
    both Part A and Part B scripts yields byte-identical train/val/test
    sets, satisfying the "same split reused in Part B" requirement.
    """
    (x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    x_train_full = _normalize(x_train_full)
    x_test = _normalize(x_test)
    y_train_full = y_train_full.squeeze(axis=1)
    y_test = y_test.squeeze(axis=1)

    x_train, y_train, x_val, y_val = _fixed_val_split(x_train_full, y_train_full)

    return Dataset(x_train, y_train, x_val, y_val, x_test, y_test)


def _normalize(images: np.ndarray) -> np.ndarray:
    return images.astype("float32") / 255.0


def _fixed_val_split(x: np.ndarray, y: np.ndarray):
    rng = np.random.default_rng(config.SEED)
    indices = rng.permutation(len(x))

    val_size = int(len(x) * config.VAL_FRACTION)
    val_idx, train_idx = indices[:val_size], indices[val_size:]

    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]