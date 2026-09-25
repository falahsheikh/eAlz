"""Extract 224x224 coronal PNG slices from skull-stripped T1-weighted MRI volumes.

Input:  <input-dir>/<class>/**/*.nii.gz (or .nii), with <class> in cn, emci, lmci.
Output: <output-dir>/<class>/<class>_<volume>_s<index>.png, the file names used in splits/*.csv.

Each volume is reoriented to RAS. The script takes 30 contiguous coronal slices centered on the
middle coronal index, crops each slice to the brain bounding box, and pads it to 224x224 without
distortion. A trailing "_stripped" in a volume name (added by SynthStrip pipelines) is removed.
See ealz/preprocessing.py for the steps.
"""

import argparse

from ealz.config import CLASSES
from ealz.preprocessing import extract_class


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", required=True, help="folder with cn/, emci/ and lmci/ subfolders of volumes")
    p.add_argument("--output-dir", required=True, help="folder for the PNG slices")
    p.add_argument("--num-slices", type=int, default=30, help="coronal slices per volume")
    p.add_argument("--per-class-cap", type=int, default=1000, help="maximum number of slices per class")
    args = p.parse_args(argv)
    for class_name in CLASSES:
        saved, n_volumes = extract_class(
            args.input_dir, args.output_dir, class_name, args.num_slices, args.per_class_cap
        )
        print(f"{class_name}: {saved} slices from {n_volumes} volumes")


if __name__ == "__main__":
    main()
