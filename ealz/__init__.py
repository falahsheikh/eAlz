"""Lightweight, explainable CNNs for early Alzheimer's detection from 2D MRI slices.

- config: training configuration of the paper
- preprocessing: coronal slice extraction from MRI volumes
- data: split files, cross-validation folds and data generators
- models: backbones and classification head
- training: training with early stopping, and inference
- metrics: evaluation metrics of the paper
- xai: Grad-CAM++ and guided backpropagation

The scripts in the repository root (extract_slices.py, train.py, cross_validate.py and explain.py)
are the command-line interfaces.
"""

__version__ = "1.2.0"
