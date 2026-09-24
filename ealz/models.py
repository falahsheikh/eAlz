"""Frozen backbones with a small classification head."""

from tensorflow import keras

from .config import CLASSES, HEAD_UNITS, IMG_SIZE, LEARNING_RATE

_APPS = keras.applications

BACKBONES = {
    "mobilenetv2": (_APPS.MobileNetV2, _APPS.mobilenet_v2.preprocess_input),
    "efficientnetv2b0": (_APPS.EfficientNetV2B0, _APPS.efficientnet_v2.preprocess_input),
    "densenet121": (_APPS.DenseNet121, _APPS.densenet.preprocess_input),
}

# Layer names that identify each backbone in a saved model.
_SIGNATURES = {
    "mobilenetv2": {"Conv_1", "out_relu"},
    "efficientnetv2b0": {"stem_conv", "top_conv"},
    "densenet121": {"conv5_block16_concat"},
}


def preprocess_fn(backbone: str):
    """Architecture-specific input preprocessing (expects RGB pixels in [0, 255])."""
    return BACKBONES[backbone][1]


def build_model(backbone: str, weights: str | None = "imagenet") -> keras.Model:
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


def infer_backbone(model: keras.Model) -> str:
    """Name of the backbone in a model built by build_model, found from its layer names."""
    names = {layer.name for layer in model.layers}
    matches = [name for name, signature in _SIGNATURES.items() if signature <= names]
    if "densenet121" in matches and "conv5_block17_concat" in names:  # a deeper DenseNet
        matches.remove("densenet121")
    if len(matches) != 1:
        raise ValueError("cannot identify the backbone of this model; pass it explicitly")
    return matches[0]
