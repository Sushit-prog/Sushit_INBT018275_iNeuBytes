"""
Two model builders:

- build_traditional_cnn: AlexNet-style CNN adapted for 32x32 CIFAR-10
  images. AlexNet's original 227x227 input, 11x11/5x5 kernels, and
  96/256/384 filters would immediately over-downsample a 32x32 image
  (dimension mismatch), so this keeps AlexNet's core pattern —
  stacked conv blocks with growing filter counts, ReLU, pooling, then
  dense classification head — while using 3x3 kernels and the
  64/128/256 filter progression requested in the spec.

- build_custom_cnn: the same backbone with additions, each included
  for a specific, statable reason (see REPORT_TEMPLATE.md):
    * BatchNorm      -> stabilizes/accelerates training, allows a
                         slightly higher learning rate
    * Dropout        -> reduces overfitting from the deeper backbone
    * extra conv/blk -> more representational capacity per stage
    * augmentation   -> synthetic data diversity, reduces overfitting
                         on CIFAR-10's small 32x32 images
"""

import tensorflow as tf
from tensorflow.keras import layers, models

from src import config


def build_traditional_cnn() -> tf.keras.Model:
    model = models.Sequential(name="traditional_cnn_alexnet_style")
    model.add(layers.Input(shape=config.INPUT_SHAPE))

    for filters in (64, 128, 256):
        model.add(layers.Conv2D(filters, 3, activation="relu", padding="same"))
        model.add(layers.Conv2D(filters, 3, activation="relu", padding="same"))
        model.add(layers.MaxPooling2D(2))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dense(config.NUM_CLASSES, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_custom_cnn(learning_rate: float = 1e-3) -> tf.keras.Model:
    inputs = layers.Input(shape=config.INPUT_SHAPE)
    x = _augmentation_pipeline()(inputs)

    for filters in (64, 128, 256):
        x = _conv_bn_relu(x, filters)
        x = _conv_bn_relu(x, filters)
        x = _conv_bn_relu(x, filters)  # extra conv layer vs. traditional CNN
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(config.NUM_CLASSES, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="custom_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _conv_bn_relu(x, filters: int):
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    return layers.Activation("relu")(x)


def _augmentation_pipeline() -> tf.keras.Sequential:
    """Applied only during training (Keras preprocessing layers are
    no-ops at inference), so val/test accuracy stays a clean measure."""
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.06),
            layers.RandomTranslation(0.1, 0.1),
            layers.RandomZoom(0.1),
        ],
        name="augmentation",
    )


def custom_training_callbacks() -> list:
    """Learning-rate tuning strategy for the custom model (Part B
    requirement: 'optimizer tuning' / 'learning rate changes')."""
    return [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
        )
    ]