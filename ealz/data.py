"""Split files and Keras data generators."""

import os

import pandas as pd
from tensorflow import keras

from .config import AUGMENTATION, BATCH_SIZE, CLASSES, IMG_SIZE

SPLITS = ("train", "val", "test")


def read_split(csv_path, data_root):
    """Read a split CSV (columns: filepath, label) and resolve filepaths against data_root."""
    df = pd.read_csv(csv_path)
    df["filepath"] = [os.path.join(data_root, p) for p in df["filepath"]]
    return df


def read_splits(split_dir, data_root):
    """Return the train, val and test DataFrames from split_dir."""
    return [read_split(os.path.join(split_dir, f"{s}_split.csv"), data_root) for s in SPLITS]


def make_generator(df, preprocess, augment=False, shuffle=False, seed=None, batch_size=BATCH_SIZE):
    """Stream images from df. Augmentation is only ever applied to training data."""
    datagen = keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocess, **(AUGMENTATION if augment else {})
    )
    return datagen.flow_from_dataframe(
        df,
        x_col="filepath",
        y_col="label",
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        classes=list(CLASSES),
        shuffle=shuffle,
        seed=seed,
    )
