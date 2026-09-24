"""End-to-end run of train.py, cross_validate.py and explain.py on a small synthetic dataset."""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import cross_validate
import explain
import train

CLASSES = ("cn", "emci", "lmci")


def make_dataset(root, per_class=12):
    rng = np.random.default_rng(0)
    rows = []
    for k, name in enumerate(CLASSES):
        (root / "slices" / name).mkdir(parents=True)
        for i in range(per_class):
            image = rng.uniform(0, 60, size=(224, 224))
            image[40 + 40 * k : 100 + 40 * k, 60:160] += 150  # class-dependent bright band
            relative = f"{name}/{name}_synthetic_{i:02d}_s100.png"
            plt.imsave(root / "slices" / relative, image, cmap="gray")
            rows.append((relative, name, i))
    df = pd.DataFrame(rows, columns=["filepath", "label", "i"])
    (root / "splits").mkdir()
    for split, keep in [("train", lambda i: i < 6), ("val", lambda i: 6 <= i < 9), ("test", lambda i: i >= 9)]:
        df[df["i"].map(keep)][["filepath", "label"]].to_csv(root / "splits" / f"{split}_split.csv", index=False)


def test_train_cross_validate_explain(tmp_path):
    make_dataset(tmp_path)
    common = ["--backbone", "mobilenetv2", "--data-root", str(tmp_path / "slices"), "--splits", str(tmp_path / "splits")]
    common += ["--epochs", "1", "--no-pretrained"]

    metrics = train.main(common + ["--augment", "--out", str(tmp_path / "run")])
    assert metrics["n"] == 9 and 0.0 <= metrics["accuracy"] <= 1.0
    assert json.loads((tmp_path / "run" / "metrics.json").read_text())["epochs_trained"] == 1
    predictions = pd.read_csv(tmp_path / "run" / "predictions.csv")
    assert len(predictions) == 9
    np.testing.assert_allclose(predictions[[f"prob_{c}" for c in CLASSES]].sum(axis=1), 1.0, rtol=1e-5)

    summary = cross_validate.main(common + ["--folds", "2", "--out", str(tmp_path / "cv")])
    folds = json.loads((tmp_path / "cv" / "cv_results.json").read_text())["folds"]
    assert [f["n"] for f in folds] == [18, 18] and set(summary) >= {"accuracy", "macro_auc", "brier"}

    image = tmp_path / "slices" / "cn" / "cn_synthetic_00_s100.png"
    explain.main(["--model", str(tmp_path / "run" / "model.keras"), "--backbone", "mobilenetv2"]
                 + ["--images", str(image), "--out", str(tmp_path / "xai")])
    for suffix in ("gradcampp", "guided_gradcampp"):
        assert (tmp_path / "xai" / f"cn_synthetic_00_s100_{suffix}.png").exists()
