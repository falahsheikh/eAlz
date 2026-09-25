import numpy as np
import pytest
import tensorflow as tf
from tensorflow import keras

from ealz import xai


def small_cnn():
    keras.utils.set_random_seed(0)
    inputs = keras.Input((8, 8, 3))
    x = keras.layers.Conv2D(4, 3, padding="same", activation="relu", name="conv")(inputs)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(5, activation="relu")(x)
    return keras.Model(inputs, keras.layers.Dense(3, activation="softmax")(x))


def test_gradcam_plus_plus_matches_reference_equation():
    model = small_cnn()
    image = np.random.default_rng(1).normal(size=(1, 8, 8, 3)).astype("float32")
    target = 1

    # Reference: Chattopadhay et al. (2018), Eq. 19 with the exponential score, written in NumPy.
    head = model.layers[-1]
    features = keras.Model(model.input, [model.get_layer("conv").output, head.input])
    with tf.GradientTape() as tape:
        activations, pooled = features(image)
        logit = (tf.matmul(pooled, head.kernel) + head.bias)[:, target]
    grads = tape.gradient(logit, activations)[0].numpy()
    activations = activations[0].numpy()
    denominator = 2 * grads**2 + activations.sum(axis=(0, 1)) * grads**3
    denominator[denominator == 0] = 1
    weights = (grads**2 / denominator * np.maximum(grads, 0)).sum(axis=(0, 1))
    expected = np.maximum((activations * weights).sum(axis=-1), 0)
    expected = tf.image.resize(expected[..., None], (8, 8), method="bilinear")[..., 0].numpy()
    expected /= expected.max()

    assert xai.feature_layer_name(model) == "conv"
    np.testing.assert_allclose(xai.gradcam_plus_plus(model, image, target), expected, rtol=1e-5, atol=1e-6)


def test_logits_are_the_pre_softmax_scores():
    model = small_cnn()
    image = np.random.default_rng(2).normal(size=(1, 8, 8, 3)).astype("float32")
    feature_model, head = xai._logit_model(model)
    logits = xai._logits(feature_model([image]), head)
    np.testing.assert_allclose(tf.nn.softmax(logits).numpy(), model(image).numpy(), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("relu_as_layer", [False, True])
def test_guided_backprop_matches_hand_derivation(relu_as_layer):
    # x (4 values) -> z = x W1 -> ReLU -> logits = h W2. Guided backprop keeps a hidden unit only
    # where z > 0 and the gradient arriving from W2 is positive.
    inputs = keras.Input((2, 2, 1))
    x = keras.layers.Flatten()(inputs)
    if relu_as_layer:
        x = keras.layers.ReLU(6.0)(keras.layers.Dense(3, use_bias=False, name="hidden")(x))
    else:
        x = keras.layers.Dense(3, activation="relu", use_bias=False, name="hidden")(x)
    model = keras.Model(inputs, keras.layers.Dense(2, activation="softmax", use_bias=False, name="out")(x))

    w1 = np.array([[1.0, -1.0, 0.5], [0.5, 1.0, -1.0], [-0.5, 0.5, 1.0], [1.0, 1.0, 1.0]], dtype="float32")
    w2 = np.array([[1.0, -1.0], [-2.0, 1.0], [0.5, 0.5]], dtype="float32")
    model.get_layer("hidden").set_weights([w1])
    model.get_layer("out").set_weights([w2])
    image = np.ones((1, 2, 2, 1), dtype="float32")

    z = image.reshape(-1) @ w1  # [2.0, 1.5, 1.5]: every hidden unit is active
    upstream = w2[:, 0]  # [1.0, -2.0, 0.5]: guided backprop drops the second unit, plain gradients keep it
    guided = w1 @ (upstream * (z > 0) * (upstream > 0))
    vanilla = w1 @ (upstream * (z > 0))

    result = xai.guided_backprop(model, image, class_index=0).reshape(-1)
    np.testing.assert_allclose(result, guided, rtol=1e-6)
    assert not np.allclose(result, vanilla)
