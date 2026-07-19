"""
CLI entry point for Task 1 (CIFAR-10 CNNs).

Usage:
    python main.py --part a       for Part A only (traditional CNN)
    python main.py --part b       for Part B only (custom CNN)
    python main.py --part both    for both + comparison table (default)
"""

import argparse
import json

from src import config, evaluate, models, train, visualize
from src.data import load_cifar10_split


def run_part_a(data):
    print("\n=== Part A: Traditional CNN ===")
    train.set_global_seed()
    model = models.build_traditional_cnn()
    model.summary()

    result = train.run_training(model, data, epochs=config.EPOCHS)
    result.model.save(config.OUTPUTS_DIR / "traditional_cnn.keras")  # save immediately, before eval/plots can fail

    report, cm = evaluate.evaluate_model(
        result.model, data, "traditional_cnn",
        result.epochs_trained, result.training_time_seconds, result.history,
    )

    evaluate.save_report(report, config.METRICS_DIR / "traditional_cnn_metrics.json")
    visualize.save_architecture_diagram(model, "traditional_cnn_architecture.png")
    visualize.save_training_curves(result.history, "traditional_cnn_curves.png")
    visualize.save_confusion_matrix(cm, "traditional_cnn_confusion_matrix.png", "Traditional CNN")

    _print_report(report)
    if report.test_accuracy < config.TRADITIONAL_MIN_TEST_ACC:
        print(
            f"[WARNING] Test accuracy {report.test_accuracy:.2%} is below the "
            f"{config.TRADITIONAL_MIN_TEST_ACC:.0%} success threshold."
        )
    return report


def run_part_b(data):
    print("\n=== Part B: Custom CNN ===")
    train.set_global_seed()
    model = models.build_custom_cnn()
    model.summary()

    epochs = config.resolve_custom_epochs()
    result = train.run_training(
        model, data, epochs=epochs, callbacks=models.custom_training_callbacks()
    )
    result.model.save(config.OUTPUTS_DIR / "custom_cnn.keras")  # save immediately, before eval/plots can fail

    report, cm = evaluate.evaluate_model(
        result.model, data, "custom_cnn",
        result.epochs_trained, result.training_time_seconds, result.history,
    )

    evaluate.save_report(report, config.METRICS_DIR / "custom_cnn_metrics.json")
    visualize.save_architecture_diagram(model, "custom_cnn_architecture.png")
    visualize.save_training_curves(result.history, "custom_cnn_curves.png")
    visualize.save_confusion_matrix(cm, "custom_cnn_confusion_matrix.png", "Custom CNN")

    _print_report(report)
    return report


def _print_report(report) -> None:
    for key, value in report.to_dict().items():
        print(f"  {key}: {value}")


def main():
    parser = argparse.ArgumentParser(description="Task 1: CIFAR-10 CNNs")
    parser.add_argument("--part", choices=["a", "b", "both"], default="both")
    args = parser.parse_args()

    data = load_cifar10_split()

    traditional_report = custom_report = None
    if args.part in ("a", "both"):
        traditional_report = run_part_a(data)
    if args.part in ("b", "both"):
        custom_report = run_part_b(data)

    if traditional_report and custom_report:
        comparison = evaluate.compare_reports(traditional_report, custom_report)
        print("\n=== Comparison ===")
        print(json.dumps(comparison, indent=2))
        with open(config.METRICS_DIR / "comparison.json", "w") as f:
            json.dump(comparison, f, indent=2)


if __name__ == "__main__":
    main()
