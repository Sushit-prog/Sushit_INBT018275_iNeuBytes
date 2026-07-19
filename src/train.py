"""Generic training runner used by both Part A and Part B, so timing,
history capture, and seeding logic is written once."""

import time
from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from src import config
from src.data import Dataset


@dataclass
class TrainResult:
    model: tf.keras.Model
    history: tf.keras.callbacks.History
    training_time_seconds: float
    epochs_trained: int


def set_global_seed(seed: int = config.SEED) -> None:
    tf.keras.utils.set_random_seed(seed)


def run_training(
    model: tf.keras.Model,
    data: Dataset,
    epochs: int,
    callbacks: list | None = None,
) -> TrainResult:
    set_global_seed()  # re-seed immediately before training for determinism

    start = time.perf_counter()
    history = model.fit(
        data.x_train,
        data.y_train,
        validation_data=(data.x_val, data.y_val),
        epochs=epochs,
        batch_size=config.BATCH_SIZE,
        callbacks=callbacks or [],
        verbose=2,
    )
    elapsed = time.perf_counter() - start

    epochs_trained = len(history.history["loss"])  # may be < epochs if early stopping is added later
    return TrainResult(model, history, elapsed, epochs_trained)