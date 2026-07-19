# AI Engineering Internship — iNeuBytes

**Intern:** Sushit &nbsp;|&nbsp; **Course ID:** AIINB10626 &nbsp;|&nbsp; **Organization:** iNeuBytes

This repository contains the code, results, and documentation for all
tasks completed during the internship. Each task lives in its own
folder with a self-contained README, so it can be run independently.

## Repository structure

```
.
├── Task1/
│   └── cifar10-cnn/        # CIFAR-10 image classification with CNNs
├── Task2/                  # (in progress)
└── MajorProject/           # (in progress)
```

## Task 1 — CIFAR-10 CNN Classification

**Folder:** [`Task1/cifar10-cnn`](Task1/cifar10-cnn)

Trained and compared two convolutional neural networks on CIFAR-10:

| Model | Test accuracy | Parameters | Notes |
|---|---|---|---|
| Traditional CNN (AlexNet-style baseline) | 74.57% | 3,376,970 | No regularization — clean baseline |
| Custom CNN | 85.57% | 4,030,218 | BatchNorm, dropout, augmentation, LR scheduling |

**Result:** the custom model improved test accuracy by **+11.0 percentage
points** over the baseline — well past the required +3pp threshold —
while shrinking the train/validation accuracy gap from ~22 points down
to ~1.6 points, indicating the added regularization meaningfully reduced
overfitting rather than just adding capacity.

Full write-up, architecture details, and metrics are in
[`Task1/cifar10-cnn/README.md`](Task1/cifar10-cnn/README.md) and the
detailed report submitted alongside this repository.

**Stack:** TensorFlow / Keras, scikit-learn, NumPy, Matplotlib

## Task 2

To be added.

## Major Project

To be added.

## Notes

All models in this repository were trained locally on CPU-only hardware
(no discrete GPU), which is reflected in the training-time figures
reported in each task's documentation.
