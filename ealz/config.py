"""Training configuration used for the paper (Tables 5 and 6), identical to notebooks/training_code.ipynb."""

IMG_SIZE = (224, 224)
CLASSES = ("cn", "emci", "lmci")

BATCH_SIZE = 16
LEARNING_RATE = 1e-3
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 7
HEAD_UNITS = 128

# On-the-fly augmentation, passed to Keras' ImageDataGenerator. Keras reads shear_range
# as an angle in degrees, so 0.1 here is the value that was actually used in training.
AUGMENTATION = dict(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    brightness_range=(0.9, 1.1),
    fill_mode="nearest",
)
