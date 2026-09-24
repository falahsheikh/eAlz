"""Write Grad-CAM++ and Guided Grad-CAM++ maps for one or more MRI slices."""

import argparse
import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

from ealz.config import CLASSES, IMG_SIZE
from ealz.models import BACKBONES, preprocess_fn
from ealz.xai import feature_layer_name, gradcam_plus_plus, guided_gradcam_plus_plus, guided_model, overlay

# Models saved by Keras record their input as a list; calling them with a single tensor is correct.
warnings.filterwarnings("ignore", message="The structure of `inputs` doesn't match")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="trained .keras model from train.py")
    p.add_argument("--backbone", required=True, choices=sorted(BACKBONES), help="backbone of the model")
    p.add_argument("--images", nargs="+", required=True, help="PNG slices made by extract_slices.py")
    p.add_argument("--out", required=True, help="output folder")
    p.add_argument("--layer", help="layer to explain (default: the final feature map)")
    p.add_argument("--class-index", type=int, help="class to explain (default: the predicted class)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    model = keras.models.load_model(args.model)
    guided = guided_model(model)
    layer = args.layer or feature_layer_name(model)

    for path in args.images:
        rgb = keras.utils.img_to_array(keras.utils.load_img(path, color_mode="rgb", target_size=IMG_SIZE))
        batch = preprocess_fn(args.backbone)(rgb[np.newaxis].copy())
        probs = model(batch, training=False).numpy()[0]
        target = int(np.argmax(probs)) if args.class_index is None else args.class_index

        cam = gradcam_plus_plus(model, batch, target, layer)
        guided_cam = guided_gradcam_plus_plus(model, batch, target, layer, guided)

        stem = os.path.join(args.out, os.path.splitext(os.path.basename(path))[0])
        plt.imsave(f"{stem}_gradcampp.png", overlay(cam, rgb))
        plt.imsave(f"{stem}_guided_gradcampp.png", guided_cam, cmap="gray")
        scores = ", ".join(f"{c} {p:.3f}" for c, p in zip(CLASSES, probs))
        print(f"{path}: explained class {CLASSES[target]} (layer {layer}); probabilities: {scores}")


if __name__ == "__main__":
    main()
