import numpy as np
import pytest

from ealz.models import build_model
from ealz.xai import feature_layer_name

# Total parameters reported in Table 7 of the paper (2.42 M, 6.08 M and 7.17 M) and the trainable head size.
CASES = [
    ("mobilenetv2", 2_422_339, 1280 * 128 + 128 + 128 * 3 + 3, "out_relu"),
    ("efficientnetv2b0", 6_083_667, 1280 * 128 + 128 + 128 * 3 + 3, "top_activation"),
    ("densenet121", 7_169_091, 1024 * 128 + 128 + 128 * 3 + 3, "relu"),
]


@pytest.mark.parametrize("backbone, total, trainable, feature_layer", CASES)
def test_architecture_matches_paper(backbone, total, trainable, feature_layer):
    model = build_model(backbone, weights=None)
    assert model.count_params() == total
    assert sum(int(np.prod(w.shape)) for w in model.trainable_weights) == trainable
    assert model.output_shape == (None, 3)
    assert feature_layer_name(model) == feature_layer
