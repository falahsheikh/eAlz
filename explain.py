"""Write Grad-CAM++ and Guided Grad-CAM++ maps for one or more MRI slices."""

import argparse
import os

import numpy as np
from matplotlib import image as mpimg
from tensorflow import keras

from ealz.config import CLASSES, IMG_SIZE
from ealz.models import BACKBONES, infer_backbone, preprocess_fn
from ealz.xai import feature_layer_name, gradcam_plus_plus, guided_gradcam_plus_plus, guided_model, overlay


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="trained .keras model from train.py")
    p.add_argument("--backbone", choices=sorted(BACKBONES), help="backbone of the model (default: read from the model)")
    p.add_argument("--images", nargs="+", required=True, help="PNG slices made by extract_slices.py")
    p.add_argument("--out", required=True, help="output folder")
    p.add_argument("--layer", help="layer to explain (default: the final feature map)")
    p.add_argument("--class-index", type=int, help="class to explain (default: the predicted class)")
    return p.parse_args(argv)


def resolve_backbone(model, requested=None):
    """Backbone of the model; a requested backbone must agree with the model's layers."""
    try:
        found = infer_backbone(model)
    except ValueError:
        if requested is None:
            raise
        return requested
    if requested is not None and requested != found:
        raise ValueError(f"--backbone {requested} does not match the model, which uses {found}")
    return found


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    model = keras.models.load_model(args.model)
    backbone = resolve_backbone(model, args.backbone)
    guided = guided_model(model)  # built once and used for all images
    layer = args.layer or feature_layer_name(model)

    for path in args.images:
        rgb = keras.utils.img_to_array(keras.utils.load_img(path, color_mode="rgb", target_size=IMG_SIZE))
        batch = preprocess_fn(backbone)(rgb[np.newaxis].copy())
        probs = model.predict(batch, verbose=0)[0]
        target = int(np.argmax(probs)) if args.class_index is None else args.class_index

        cam = gradcam_plus_plus(model, batch, target, layer)
        guided_cam = guided_gradcam_plus_plus(model, batch, target, layer, guided)

        stem = os.path.join(args.out, os.path.splitext(os.path.basename(path))[0])
        mpimg.imsave(f"{stem}_gradcampp.png", overlay(cam, rgb))
        mpimg.imsave(f"{stem}_guided_gradcampp.png", guided_cam, cmap="gray")
        scores = ", ".join(f"{c} {p:.3f}" for c, p in zip(CLASSES, probs, strict=True))
        print(f"{path}: explained class {CLASSES[target]} (layer {layer}); probabilities: {scores}")


if __name__ == "__main__":
    main()
