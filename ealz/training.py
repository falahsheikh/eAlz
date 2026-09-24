"""Training and inference helpers shared by train.py and cross_validate.py."""

import platform

import numpy as np
import tensorflow as tf
from tensorflow import keras

from .config import EARLY_STOPPING_PATIENCE, MAX_EPOCHS
from .data import make_generator
from .models import build_model, preprocess_fn


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and TensorFlow."""
    keras.utils.set_random_seed(seed)


def environment() -> dict:
    """Library versions, saved with every run."""
    return {
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
        "numpy": np.__version__,
    }


def fit(backbone, train_df, val_df, augment, epochs=MAX_EPOCHS, seed=0, weights="imagenet", verbose=1):
    """Train with early stopping on validation loss; the weights of the best epoch are restored.

    Returns (model, history, best_epoch), with best_epoch counted from 1.
    """
    preprocess = preprocess_fn(backbone)
    train_gen = make_generator(train_df, preprocess, augment=augment, shuffle=True, seed=seed)
    val_gen = make_generator(val_df, preprocess)
    model = build_model(backbone, weights=weights)
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True, verbose=verbose
    )
    history = model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=[early_stopping], verbose=verbose)
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    return model, history.history, best_epoch


def predict(model, backbone, df):
    """Return (filepaths, integer labels, class probabilities) for every image in df, in order."""
    gen = make_generator(df, preprocess_fn(backbone))
    probs = model.predict(gen, verbose=0)
    return gen.filenames, gen.classes, probs
