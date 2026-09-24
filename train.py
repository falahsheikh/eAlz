"""Train one configuration on the fixed train/validation/test split and evaluate it on the test set."""

import argparse
import json
import os

import pandas as pd

from ealz.config import CLASSES, MAX_EPOCHS
from ealz.data import read_splits
from ealz.metrics import evaluate
from ealz.models import BACKBONES
from ealz.training import environment, fit, predict, set_seed


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backbone", required=True, choices=sorted(BACKBONES))
    p.add_argument("--augment", action="store_true", help="apply on-the-fly augmentation to the training set")
    p.add_argument("--data-root", required=True, help="folder that contains the cn/, emci/ and lmci/ slice folders")
    p.add_argument("--splits", default="splits", help="folder with train_split.csv, val_split.csv and test_split.csv")
    p.add_argument("--out", required=True, help="output folder")
    p.add_argument("--epochs", type=int, default=MAX_EPOCHS, help="maximum number of epochs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-pretrained", action="store_true", help="use random initialization instead of ImageNet weights")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "run_config.json"), "w") as f:
        json.dump({"args": vars(args), "environment": environment()}, f, indent=2)

    train_df, val_df, test_df = read_splits(args.splits, args.data_root)
    model, history, best_epoch = fit(
        args.backbone,
        train_df,
        val_df,
        augment=args.augment,
        epochs=args.epochs,
        seed=args.seed,
        weights=None if args.no_pretrained else "imagenet",
    )

    filepaths, y_true, probs = predict(model, args.backbone, test_df)
    metrics = evaluate(y_true, probs, CLASSES, seed=args.seed)
    metrics.update(epochs_trained=len(history["loss"]), best_epoch=best_epoch)

    model.save(os.path.join(args.out, "model.keras"))
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(args.out, "history.json"), "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.items()}, f, indent=2)
    predictions = pd.DataFrame(probs, columns=[f"prob_{c}" for c in CLASSES])
    predictions.insert(0, "label", [CLASSES[i] for i in y_true])
    predictions.insert(0, "filepath", filepaths)
    predictions.to_csv(os.path.join(args.out, "predictions.csv"), index=False)

    print(
        f"{args.backbone} ({'augmented' if args.augment else 'original'}): "
        f"accuracy {metrics['accuracy']:.4f}, macro AUC {metrics['macro_auc']:.3f}, "
        f"Brier {metrics['brier']:.4f}, best epoch {best_epoch} of {metrics['epochs_trained']}"
    )
    return metrics


if __name__ == "__main__":
    main()
