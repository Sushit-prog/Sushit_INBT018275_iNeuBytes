"""
Standalone comparison script — use this if Part A and Part B were run
as separate `python main.py --part a` / `--part b` invocations, so
main.py never had both reports in memory at once to compare them.

Usage:
    python compare_results.py
"""

import json

from src import config


def main():
    traditional = _load(config.METRICS_DIR / "traditional_cnn_metrics.json")
    custom = _load(config.METRICS_DIR / "custom_cnn_metrics.json")

    improvement_pp = (custom["test_accuracy"] - traditional["test_accuracy"]) * 100
    comparison = {
        "traditional_test_accuracy": traditional["test_accuracy"],
        "custom_test_accuracy": custom["test_accuracy"],
        "improvement_percentage_points": round(improvement_pp, 2),
        "threshold_met": improvement_pp >= config.CUSTOM_MIN_IMPROVEMENT_PP,
        "traditional_params": traditional["total_params"],
        "custom_params": custom["total_params"],
        "traditional_training_time_seconds": traditional["training_time_seconds"],
        "custom_training_time_seconds": custom["training_time_seconds"],
    }

    out_path = config.METRICS_DIR / "comparison.json"
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2)

    print(json.dumps(comparison, indent=2))
    print(f"\nWritten to {out_path}")


def _load(path):
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    main()