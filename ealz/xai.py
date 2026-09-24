"""Grad-CAM++ and guided backpropagation for the trained classifiers.

Both methods differentiate the class score before the softmax (the logit), as defined in
Grad-CAM (Selvaraju et al., 2017) and Grad-CAM++ (Chattopadhay et al., 2018).
"""

import matplotlib
import numpy as np
import tensorflow as tf
from tensorflow import keras

_RECTIFIERS = {"relu", "relu6", "silu", "swish"}


def feature_layer_name(model):
    """Name of the layer whose output feeds the classifier's global average pooling (the final feature map)."""
    # Search from the end: EfficientNet's squeeze-and-excitation blocks contain their own pooling layers.
    pooling = next((x for x in reversed(model.layers) if isinstance(x, keras.layers.GlobalAveragePooling2D)), None)
    if pooling is None:
        raise ValueError("expected a GlobalAveragePooling2D layer after the backbone")
    for layer in model.layers:
        if layer.output is pooling.input:
            return layer.name
    raise ValueError("could not find the layer feeding global average pooling")


def _logit_model(model, layer_name=None):
    """Model returning (feature map, head input); the logits are computed from the head input."""
    head = model.layers[-1]
    if not isinstance(head, keras.layers.Dense):
        raise ValueError("expected the model to end in a Dense layer")
    outputs = head.input if layer_name is None else [model.get_layer(layer_name).output, head.input]
    return keras.Model(model.input, outputs), head


def _logits(features, head):
    logits = tf.matmul(features, tf.convert_to_tensor(head.kernel))
    return logits if head.bias is None else logits + tf.convert_to_tensor(head.bias)


def _normalize(x):
    x = np.asarray(x, dtype=np.float32)
    peak = x.max()
    return x / peak if peak > 0 else np.zeros_like(x)


def gradcam_plus_plus(model, image, class_index, layer_name=None):
    """Grad-CAM++ heatmap in [0, 1] at input resolution for one preprocessed image of shape (1, H, W, 3)."""
    layer_name = layer_name or feature_layer_name(model)
    grad_model, head = _logit_model(model, layer_name)
    image = tf.convert_to_tensor(image, dtype=tf.float32)

    with tf.GradientTape() as tape:
        activations, features = grad_model(image, training=False)
        score = _logits(features, head)[:, class_index]
    grads = tape.gradient(score, activations)[0]
    activations = activations[0]

    # alpha_ij^k = g^2 / (2 g^2 + sum_ab(A_ab^k) g^3), with g the gradient at (i, j) of channel k.
    grads_2, grads_3 = tf.square(grads), tf.pow(grads, 3)
    denominator = 2.0 * grads_2 + tf.reduce_sum(activations, axis=(0, 1), keepdims=True) * grads_3
    denominator = tf.where(denominator != 0.0, denominator, tf.ones_like(denominator))
    weights = tf.reduce_sum(grads_2 / denominator * tf.nn.relu(grads), axis=(0, 1))

    cam = tf.nn.relu(tf.reduce_sum(activations * weights, axis=-1))
    cam = tf.image.resize(cam[..., tf.newaxis], image.shape[1:3], method="bilinear")[..., 0]
    return _normalize(cam)


def _guided(fn):
    """Wrap an activation so its gradient only passes where the input and the incoming gradient are positive."""

    @tf.custom_gradient
    def guided(x):
        def grad(upstream):
            with tf.GradientTape() as tape:
                tape.watch(x)
                y = fn(x)
            dx = tape.gradient(y, x, output_gradients=upstream)
            return dx * tf.cast(x > 0, dx.dtype) * tf.cast(upstream > 0, dx.dtype)

        return fn(x), grad

    return guided


def _clone_layer(layer):
    if isinstance(layer, keras.layers.ReLU):  # e.g. MobileNetV2's ReLU6 layers, which have no `activation` attribute

        def relu(x, config=layer):
            return keras.activations.relu(
                x, negative_slope=config.negative_slope, max_value=config.max_value, threshold=config.threshold
            )

        return keras.layers.Activation(_guided(relu), name=layer.name)
    return layer.__class__.from_config(layer.get_config())


def guided_model(model):
    """Copy of model whose rectifying activations use the guided backpropagation rule (Springenberg et al., 2015)."""
    clone = keras.models.clone_model(model, clone_function=_clone_layer)
    clone.set_weights(model.get_weights())
    for layer in clone.layers:
        if getattr(getattr(layer, "activation", None), "__name__", None) in _RECTIFIERS:
            layer.activation = _guided(layer.activation)
    return clone


def guided_backprop(model, image, class_index, guided=None):
    """Guided-backpropagation gradients of the class logit with respect to the input, shape (H, W, 3).

    Pass a precomputed guided_model(model) as `guided` to avoid rebuilding it for every image.
    """
    guided = guided or guided_model(model)
    feature_model, head = _logit_model(guided)
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(image)
        score = _logits(feature_model(image, training=False), head)[:, class_index]
    return tape.gradient(score, image)[0].numpy()


def guided_gradcam_plus_plus(model, image, class_index, layer_name=None, guided=None):
    """Guided Grad-CAM++: guided-backprop saliency multiplied by the upsampled Grad-CAM++ map, in [0, 1]."""
    cam = gradcam_plus_plus(model, image, class_index, layer_name)
    saliency = np.abs(guided_backprop(model, image, class_index, guided)).max(axis=-1)
    return _normalize(saliency * cam)


def overlay(heatmap, image_rgb, alpha=0.5):
    """Blend a [0, 1] heatmap (jet colormap) onto an RGB uint8 image."""
    colored = matplotlib.colormaps["jet"](heatmap)[..., :3] * 255.0
    return ((1 - alpha) * np.asarray(image_rgb, dtype=np.float32) + alpha * colored).clip(0, 255).astype(np.uint8)
