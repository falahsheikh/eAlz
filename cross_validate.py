"""Stratified k-fold cross-validation over all slices in the split files.

Each fold holds out 1/k of the slices for testing. The remaining slices are divided, stratified by
class, into 80% training and 20% validation data. The validation data is used only for early stopping.
With the default k = 5 and 3,000 slices, every fold has 1,920 training, 480 validation and 600 test slices.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from tensorflow import keras

from ealz.config import CLASSES, MAX_EPOCHS
from ealz.data import read_splits
from ealz.metrics import evaluate
from ealz.models import BACKBONES
from ealz.training import fit, predict, set_seed

SUMMARY_METRICS = ("accuracy", "balanced_accuracy", "macro_auc", "micro_auc", "brier")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbone", required=True, choices=sorted(BACKBONES))
    p.add_argument("--augment", action="store_true", help="apply on-the-fly augmentation to the training data")
    p.add_argument("--data-root", required=True, help="folder that contains the cn/, emci/ and lmci/ slice folders")
    p.add_argument("--splits", default="splits", help="folder with the split CSV files; all rows are pooled")
    p.add_argument("--out", required=True, help="output folder")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=MAX_EPOCHS, help="maximum number of epochs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-pretrained", action="store_true", help="use random initialization instead of ImageNet weights")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    slices = pd.concat(read_splits(args.splits, args.data_root), ignore_index=True)

    folds = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    results = []
    for fold, (train_val_idx, test_idx) in enumerate(folds.split(slices, slices["label"]), start=1):
        train_val, test = slices.iloc[train_val_idx], slices.iloc[test_idx]
        train, val = train_test_split(train_val, test_size=0.2, stratify=train_val["label"], random_state=args.seed)

        set_seed(args.seed + fold)
        model, history = fit(
            args.backbone,
            train,
            val,
            augment=args.augment,
            epochs=args.epochs,
            seed=args.seed + fold,
            weights=None if args.no_pretrained else "imagenet",
        )
        _, y_true, probs = predict(model, args.backbone, test)
        metrics = evaluate(y_true, probs, CLASSES, seed=args.seed)
        metrics.update(fold=fold, epochs_trained=len(history["loss"]), n_train=len(train), n_val=len(val))
        results.append(metrics)
        print(f"fold {fold}/{args.folds}: accuracy {metrics['accuracy']:.4f}, macro AUC {metrics['macro_auc']:.3f}")
        keras.backend.clear_session()

    summary = {
        m: {"mean": float(np.mean([r[m] for r in results])), "std": float(np.std([r[m] for r in results], ddof=1))}
        for m in SUMMARY_METRICS
    }
    with open(os.path.join(args.out, "cv_results.json"), "w") as f:
        json.dump({"summary": summary, "folds": results}, f, indent=2)

    acc = summary["accuracy"]
    print(f"{args.folds}-fold accuracy: {acc['mean']:.4f} +/- {acc['std']:.4f} (sample standard deviation)")
    return summary


if __name__ == "__main__":
    main()
