"""Coronal slice extraction from skull-stripped MRI volumes (the data of the paper).

Each volume is reoriented to RAS. The 30 contiguous coronal slices centered on the middle coronal
index are cropped to the brain bounding box, scaled into a zero-padded 224x224 canvas without
distortion, rotated so that superior is at the top, scaled to 0-255 and saved as grayscale PNG files.
"""

import re
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
from matplotlib import image as mpimg
from scipy.ndimage import zoom
from skimage import measure, morphology

from .config import IMG_SIZE


def volume_name(path):
    """Name of a volume file without the .nii or .nii.gz extension and without a trailing "_stripped"."""
    name = re.sub(r"\.nii(\.gz)?$", "", Path(path).name)
    return re.sub(r"_stripped$", "", name)


def find_brain_bbox(slice_data, threshold_percentile=5, padding=10):
    """Bounding box (row_min, row_max, col_min, col_max) of the largest foreground region, with padding."""
    rows, cols = slice_data.shape
    if np.max(slice_data) == 0:
        return 0, rows, 0, cols
    threshold = np.percentile(slice_data[slice_data > 0], threshold_percentile)
    mask = morphology.remove_small_objects(slice_data > threshold, max_size=255)  # objects of < 256 pixels
    mask = morphology.closing(mask, morphology.disk(5))
    labels = measure.label(mask)
    if labels.max() == 0:
        return 0, rows, 0, cols
    min_row, min_col, max_row, max_col = max(measure.regionprops(labels), key=lambda r: r.area).bbox
    return (
        max(0, min_row - padding),
        min(rows, max_row + padding),
        max(0, min_col - padding),
        min(cols, max_col + padding),
    )


def crop_and_resize_slice(slice_data, target_size=IMG_SIZE):
    """Crop to the brain and scale it into a zero-padded target_size canvas, keeping the aspect ratio."""
    min_row, max_row, min_col, max_col = find_brain_bbox(slice_data)
    cropped = slice_data[min_row:max_row, min_col:max_col]
    if cropped.size == 0:
        return np.zeros(target_size, dtype=slice_data.dtype)
    scale = min(target_size[0] / cropped.shape[0], target_size[1] / cropped.shape[1])
    new_shape = (int(cropped.shape[0] * scale), int(cropped.shape[1] * scale))
    resized = zoom(cropped, (new_shape[0] / cropped.shape[0], new_shape[1] / cropped.shape[1]), order=1)
    canvas = np.zeros(target_size, dtype=resized.dtype)
    top, left = (target_size[0] - new_shape[0]) // 2, (target_size[1] - new_shape[1]) // 2
    canvas[top : top + new_shape[0], left : left + new_shape[1]] = resized
    return canvas


def save_slice_png(slice_data, png_path):
    """Rotate a (left-right, inferior-superior) slice for display, scale it to 0-255 and save it as a PNG."""
    rotated = np.rot90(slice_data, k=1)
    mpimg.imsave(png_path, cv2.normalize(rotated, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U), cmap="gray")


def extract_class(input_dir, output_dir, class_name, num_slices=30, per_class_cap=1000):
    """Write the slices of one class and return (number of PNG files, number of volumes used).

    Volumes are read in sorted order, and extraction stops at per_class_cap slices.
    """
    out = Path(output_dir) / class_name
    out.mkdir(parents=True, exist_ok=True)
    class_dir = Path(input_dir) / class_name
    volumes = sorted([*class_dir.rglob("*.nii.gz"), *class_dir.rglob("*.nii")])
    saved, used = 0, set()
    for path in volumes:
        if saved >= per_class_cap:
            break
        try:
            data = nib.as_closest_canonical(nib.load(path)).get_fdata()
        except Exception as e:  # nibabel raises many types of errors; skip unreadable volumes, as in the original run
            print(f"  skipped {path}: {e}")
            continue
        mid = data.shape[1] // 2
        for index in range(max(0, mid - num_slices // 2), min(data.shape[1], mid + num_slices // 2)):
            if saved >= per_class_cap:
                break
            coronal = crop_and_resize_slice(data[:, index, :])
            if np.sum(coronal) > 0:
                save_slice_png(coronal, out / f"{class_name}_{volume_name(path)}_s{index:03d}.png")
                saved += 1
                used.add(path)
    return saved, len(used)
