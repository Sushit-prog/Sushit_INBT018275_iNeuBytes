"""Evaluation metrics shared by both parts, so Part A and Part B are
scored with identical logic and are directly comparable."""

import json
from dataclasses import asdict, dataclass

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src import config
from src.data import Dataset


@dataclass
class EvalReport:
    model_name: str
    total_params: int
    trainable_params: int
    epochs_trained: int
    training_time_seconds: float
    train_accuracy: float
    val_accuracy: float
    test_accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_model(
    model: tf.keras.Model,
    data: Dataset,
    model_name: str,
    epochs_trained: int,
    training_time_seconds: float,
    history: tf.keras.callbacks.History,
) -> tuple[EvalReport, np.ndarray]:
    """Returns (metrics report, confusion matrix) for the test set."""
    y_pred = np.argmax(model.predict(data.x_test, verbose=0), axis=1)

    report = EvalReport(
        model_name=model_name,
        total_params=model.count_params(),
        trainable_params=_count_trainable_params(model),
        epochs_trained=epochs_trained,
        training_time_seconds=round(training_time_seconds, 1),
        train_accuracy=round(float(history.history["accuracy"][-1]), 4),
        val_accuracy=round(float(history.history["val_accuracy"][-1]), 4),
        test_accuracy=round(float(np.mean(y_pred == data.y_test)), 4),
        precision_macro=round(float(precision_score(data.y_test, y_pred, average="macro")), 4),
        recall_macro=round(float(recall_score(data.y_test, y_pred, average="macro")), 4),
        f1_macro=round(float(f1_score(data.y_test, y_pred, average="macro")), 4),
    )
    cm = confusion_matrix(data.y_test, y_pred)
    return report, cm


def _count_trainable_params(model: tf.keras.Model) -> int:
    return int(sum(np.prod(v.shape) for v in model.trainable_weights))


def save_report(report: EvalReport, path) -> None:
    with open(path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)


def compare_reports(traditional: EvalReport, custom: EvalReport) -> dict:
    """Checks the Part B success threshold: custom must beat traditional
    by >= config.CUSTOM_MIN_IMPROVEMENT_PP percentage points."""
    improvement_pp = (custom.test_accuracy - traditional.test_accuracy) * 100
    return {
        "traditional_test_accuracy": traditional.test_accuracy,
        "custom_test_accuracy": custom.test_accuracy,
        "improvement_percentage_points": round(improvement_pp, 2),
        "threshold_met": improvement_pp >= config.CUSTOM_MIN_IMPROVEMENT_PP,
        "traditional_params": traditional.total_params,
        "custom_params": custom.total_params,
        "traditional_training_time_seconds": traditional.training_time_seconds,
        "custom_training_time_seconds": custom.training_time_seconds,
    }