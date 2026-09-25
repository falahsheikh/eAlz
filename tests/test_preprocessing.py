import numpy as np

from ealz.preprocessing import crop_and_resize_slice, find_brain_bbox


def ellipse_slice(shape=(256, 200), center=(120, 90), radii=(90, 60), seed=0):
    rows, cols = np.indices(shape)
    inside = ((rows - center[0]) / radii[0]) ** 2 + ((cols - center[1]) / radii[1]) ** 2 < 1
    return inside * np.random.default_rng(seed).uniform(40, 200, size=shape)


def test_bbox_contains_the_brain_with_padding():
    # The brain covers rows 31-209 and columns 31-149; the padding is 10 pixels.
    assert find_brain_bbox(ellipse_slice()) == (21, 220, 21, 160)
    assert find_brain_bbox(np.zeros((10, 20))) == (0, 10, 0, 20)


def test_crop_and_resize_keeps_the_aspect_ratio():
    canvas = crop_and_resize_slice(ellipse_slice())
    assert canvas.shape == (224, 224)
    rows = np.flatnonzero(canvas.any(axis=1))
    cols = np.flatnonzero(canvas.any(axis=0))
    height, width = np.ptp(rows) + 1, np.ptp(cols) + 1
    assert height > 190  # the taller axis fills the canvas, except for the padding
    np.testing.assert_allclose(width / height, 119 / 179, rtol=0.05)  # the brain is 179 x 119 pixels
