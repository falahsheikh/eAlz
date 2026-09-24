"""Frozen ImageNet backbones with a small classification head."""

from tensorflow import keras

from .config import CLASSES, HEAD_UNITS, IMG_SIZE, LEARNING_RATE

_APPS = keras.applications

BACKBONES = {
    "mobilenetv2": (_APPS.MobileNetV2, _APPS.mobilenet_v2.preprocess_input),
    "efficientnetv2b0": (_APPS.EfficientNetV2B0, _APPS.efficientnet_v2.preprocess_input),
    "densenet121": (_APPS.DenseNet121, _APPS.densenet.preprocess_input),
}


def preprocess_fn(backbone):
    """Architecture-specific input preprocessing (expects RGB pixels in [0, 255])."""
    return BACKBONES[backbone][1]


def build_model(backbone, weights="imagenet"):
    """Frozen backbone -> global average pooling -> Dense(128, ReLU) -> Dense(3, softmax)."""
    constructor, _ = BACKBONES[backbone]
    base = constructor(weights=weights, include_top=False, input_shape=(*IMG_SIZE, 3))
    base.trainable = False

    x = keras.layers.GlobalAveragePooling2D()(base.output)
    x = keras.layers.Dense(HEAD_UNITS, activation="relu")(x)
    outputs = keras.layers.Dense(len(CLASSES), activation="softmax")(x)

    model = keras.Model(base.input, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
