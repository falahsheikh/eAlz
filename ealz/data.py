"""Split files, cross-validation folds and Keras data generators."""

import os

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from tensorflow import keras

from .config import AUGMENTATION, BATCH_SIZE, CLASSES, IMG_SIZE

SPLITS = ("train", "val", "test")


def read_split(csv_path: str, data_root: str) -> pd.DataFrame:
    """Read a split CSV (columns: filepath, label) and resolve filepaths against data_root.

    Raises FileNotFoundError if any image is missing, so that no slice is silently left out.
    """
    df = pd.read_csv(csv_path)
    df["filepath"] = [os.path.join(data_root, p) for p in df["filepath"]]
    missing = [p for p in df["filepath"] if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of {len(df)} images listed in {csv_path} are missing, for example {missing[0]}"
        )
    return df


def read_splits(split_dir: str, data_root: str) -> list[pd.DataFrame]:
    """Return the train, val and test DataFrames from split_dir."""
    return [read_split(os.path.join(split_dir, f"{s}_split.csv"), data_root) for s in SPLITS]


def make_folds(slices: pd.DataFrame, n_folds: int = 5, seed: int = 0, val_fraction: float = 0.2):
    """Yield (fold, train, val, test) for stratified k-fold cross-validation.

    Each slice is in the test set of exactly one fold. The other slices of a fold are divided,
    stratified by label, into training and validation data; validation data is used only for early stopping.
    """
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (train_val_idx, test_idx) in enumerate(folds.split(slices, slices["label"]), start=1):
        train_val = slices.iloc[train_val_idx]
        train, val = train_test_split(train_val, test_size=val_fraction, stratify=train_val["label"], random_state=seed)
        yield fold, train, val, slices.iloc[test_idx]


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
