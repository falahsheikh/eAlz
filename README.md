# eAlz: lightweight explainable models for early Alzheimer's detection

[![tests](https://github.com/falahsheikh/eAlz/actions/workflows/tests.yml/badge.svg)](https://github.com/falahsheikh/eAlz/actions/workflows/tests.yml)
[![DOI](https://img.shields.io/badge/DOI-10.3390%2Fdiagnostics15212709-blue)](https://doi.org/10.3390/diagnostics15212709)

This repository contains the code for the paper
[Lightweight Deep Learning Models with Explainable AI for Early Alzheimer's Detection from Standard MRI Scans](https://doi.org/10.3390/diagnostics15212709)
(*Diagnostics*, 2025).

The models classify one coronal slice of a T1-weighted MRI scan into three classes:

- cognitively normal (CN)
- early mild cognitive impairment (EMCI)
- late mild cognitive impairment (LMCI).

Each model is a frozen ImageNet backbone with a small classification head.
On a CPU, one prediction takes 175 ms with MobileNetV2 and 346 ms with EfficientNetV2B0.
Grad-CAM++ and Guided Grad-CAM++ show the image regions that have an effect on each prediction.

## Results

These values come from the paper.
All metrics are at the slice level, on the fixed test split (600 slices, 200 for each class).

| Backbone | Training data | Parameters | Accuracy | Macro AUC [95% CI] | Brier score |
| --- | --- | ---: | ---: | --- | ---: |
| EfficientNetV2B0 | Augmented | 6.08 M | **0.880** | **0.973** [0.963, 0.982] | **0.0588** |
| EfficientNetV2B0 | Original | 6.08 M | 0.875 | 0.971 [0.961, 0.980] | 0.0643 |
| MobileNetV2 | Augmented | 2.42 M | 0.865 | 0.970 [0.961, 0.979] | 0.0628 |
| MobileNetV2 | Original | 2.42 M | 0.860 | 0.962 [0.951, 0.972] | 0.0706 |
| DenseNet121 (baseline) | Augmented | 7.17 M | 0.817 | 0.947 [0.933, 0.960] | 0.0891 |

With 5-fold stratified cross-validation, EfficientNetV2B0 (augmented) has a mean accuracy of 0.880 (SD 0.010).

## Repository contents

```
eAlz/
├── ealz/                      Python package
│   ├── config.py              hyperparameters from the paper
│   ├── data.py                split files, cross-validation folds and data generators
│   ├── models.py              backbones and classification head
│   ├── training.py            training with early stopping, and inference
│   ├── metrics.py             accuracy, AUC, Brier score, sensitivity, specificity, PPV, NPV
│   └── xai.py                 Grad-CAM++, guided backpropagation, Guided Grad-CAM++
├── extract_slices.py          MRI volumes -> 224x224 PNG slices
├── train.py                   training and test evaluation on the fixed split
├── cross_validate.py          stratified k-fold cross-validation
├── explain.py                 explanation maps for single slices
├── splits/                    train, validation and test split files
├── notebooks/training_code.ipynb   original notebook for the paper
└── tests/                     tests with synthetic data
```

## Installation

1. Install Python 3.11.
2. Make a virtual environment and activate it:

   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the tests. The tests use synthetic images and do not need ADNI data.

   ```bash
   pytest
   ```

GitHub Actions runs `ruff check`, `ruff format --check` and `pytest` for each push.

## Data

The data comes from the [Alzheimer's Disease Neuroimaging Initiative (ADNI)](https://adni.loni.usc.edu).
The ADNI data use agreement does not permit redistribution.
Thus, this repository contains no images.
To use the data, ask ADNI for access.

The files in `splits/` identify each slice by its file name.
Each file name contains the ADNI subject ID, the image ID and the slice index.
For example: `cn/cn_ADNI_013_S_1035_MR_MPR__GradWarp_Br_20070426170809875_S23101_I51479_s120.png`.

Do these steps to make the slices:

1. Download the T1-weighted MPRAGE volumes that the split files identify.
2. Remove the skull from each volume with [SynthStrip](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/).
3. Keep the original ADNI name of each volume.
   The script removes the suffix `_stripped` if it is present.
4. Put the volumes in one folder for each class: `data/volumes/cn/`, `data/volumes/emci/` and `data/volumes/lmci/`.
5. Make the slices:

   ```bash
   python extract_slices.py --input-dir data/volumes --output-dir data/slices
   ```

The script turns each volume to RAS orientation.
It takes 30 adjacent coronal slices at the center of the volume.
It crops each slice to the brain and puts it on a 224x224 canvas with zero padding.
It keeps the aspect ratio.
The default limit of 1,000 slices for each class gives the 3,000 slices in the paper.

The training scripts stop with an error if an image in the split files is missing.

## Usage

Train and evaluate one configuration on the fixed split:

```bash
python train.py --backbone efficientnetv2b0 --augment --data-root data/slices --out runs/efficientnetv2b0_aug
```

- Use `--backbone mobilenetv2` or `--backbone densenet121` for the other backbones.
- Do not use `--augment` if you want to train on the original data.
- The output folder contains `model.keras`, `metrics.json`, `history.json`, `predictions.csv` and `run_config.json`.
- `metrics.json` contains all the metrics in Tables 8 to 10 of the paper.
  It also gives the epoch with the lowest validation loss and the number of epochs trained.
- `run_config.json` records the arguments and the library versions of the run.

Run 5-fold stratified cross-validation on all 3,000 slices:

```bash
python cross_validate.py --backbone efficientnetv2b0 --augment --data-root data/slices --out runs/cv_efficientnetv2b0_aug
```

Each fold uses 600 slices for the test.
The script divides the other 2,400 slices into 1,920 training slices and 480 validation slices.
It uses the validation slices only for early stopping.

Make Grad-CAM++ and Guided Grad-CAM++ maps for one or more slices:

```bash
python explain.py --model runs/efficientnetv2b0_aug/model.keras --images data/slices/cn/<slice>.png --out runs/xai
```

The script reads the backbone from the model, so the input scaling always agrees with training.

## Training configuration

| Setting | Value |
| --- | --- |
| Input | 224 x 224 x 3 |
| Backbone | ImageNet weights, frozen |
| Head | global average pooling, Dense(128, ReLU), Dense(3, softmax) |
| Optimizer | Adam, learning rate 0.001 |
| Loss | categorical cross-entropy |
| Batch size | 16 |
| Epochs | 100 maximum, early stopping on validation loss (patience 7), best weights restored |
| Augmentation | rotation 10 degrees, shift 10%, zoom 10%, shear 0.1 degrees, horizontal flip, brightness 0.9 to 1.1 |

## Notes on evaluation

- The metrics are at the slice level.
- The splits are at the slice level.
  Thus, slices from one subject can be in the training, validation and test sets.
  The paper discusses this limitation.
- The 95% confidence interval of the macro AUC is a percentile bootstrap over the test slices (1,000 resamples).
- The Brier score is the mean of the one-vs-rest Brier scores of the three classes.
- Grad-CAM++ and guided backpropagation use the class score before the softmax, as in the original methods.
- The notebook in `notebooks/` is the original Google Colab notebook for the paper.
  It trains the augmented configurations on the fixed split.
  The scripts use the same settings.

## Citation

```bibtex
@article{sheikh2025lightweight,
  title   = {Lightweight Deep Learning Models with Explainable {AI} for Early {Alzheimer's} Detection from Standard {MRI} Scans},
  author  = {Sheikh, Falah and Al Marouf, Ahmed and Rokne, Jon George and Alhajj, Reda},
  journal = {Diagnostics},
  volume  = {15},
  number  = {21},
  pages   = {2709},
  year    = {2025},
  doi     = {10.3390/diagnostics15212709}
}
```

## License

[CC BY-NC 4.0](LICENSE)

## Acknowledgments

The data for this work came from the ADNI database (adni.loni.usc.edu).
An Alberta Innovates Summer Research Studentship supported this work.
