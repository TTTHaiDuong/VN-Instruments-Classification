import tensorflow as tf
import config.types as type
from scripts.f1_score import MacroF1Score


# Main model path
# MAIN_MODEL = r"bestmodel\model1_20250519.h5"
MAIN_MODEL = r"bestmodel\model1.h5"



# Dataset directories
# DATASET_PATH = "dataset"
# TRAIN_DATA = "dataset/train"
# VAL_DATA = "dataset/val"
# TEST_DATA = "dataset/test"
DATASET_PATH = "images_dataset"
TRAIN_DATA = "images_dataset/train"
VAL_DATA = "images_dataset/val"
TEST_DATA = "images_dataset/test"

# Checkpoint and model saving directories
CHECKPOINT_PATH = "checkpoint"
BEST_MODEL = "bestmodel"
TRAIN_HISTORY_PATH = "train_history"



# Class names and their corresponding directory labels
CLASS_LABELS = ["danbau", "dannhi", "dantranh", "dantrung", "sao"]

INSTRUMENTS = type.InstrumentRegistry([
    { "name": "Đàn bầu",    "label": "danbau",   "index": 0 },
    { "name": "Đàn nhị",    "label": "dannhi",   "index": 1 },
    { "name": "Đàn tranh",  "label": "dantranh", "index": 2 },
    { "name": "Đàn T'rưng", "label": "dantrung", "index": 3 },
    { "name": "Sáo",        "label": "sao",      "index": 4 },
])


MEL_CONFIG = type.MelConfig(
    target_dbfs     = -60.0,
    sr              = 22050,
    duration        = 5,
    hop_length      = 512,
    n_fft           = 2048,
    n_mels          = 128,
    fmin            = 20,
    fmax            = None,         # None -> sr/2
    top_db          = 80,
    pre_emphasis    = 0.97,
    normalize       = "minmax_0_1", # "minmax_0_1" | "zscore"
    pad_mode        = "constant"
)


TRAIN_CONFIG = type.TrainConfig(
    input_shape             = (128, 212, 3),
    batch_size              = 32,
    validation_batch_size   = 64,
    epochs                  = 50
)


CALLBACK_CONFIG = type.CallbackConfig(
    early_patience      = 10,
    reduce_lr_patience  = 7,
    reduce_lr_factor    = 0.5,
    min_lr              = 1e-6
)


# Class weights to address class imbalance during training
CLASS_WEIGHT_DICT1 = {
    0: 1.2,
    1: 2.0,
    2: 1.0,
    3: 1.5
}
CLASS_WEIGHT_DICT2 = {
    0: 1.2,
    1: 2.0,
    2: 1.2,
    3: 1.5
}



# Image data augmentation configuration. Apply to all sample
RESCALE = 1./255,                  # Chuẩn hoá giá trị pixel về [0, 1]. Tiền xử lý bắt buộc.
IMAGE_AUGMENTATION = {
    "width_shift_range": 0.02,          # 
    "brightness_range": (0.95, 1.05),   # 
    "zoom_range": 0.025
}


# Audio augmentation default parameters
AUDIO_AUGMENTATION = {
    "pitch_shift": (-1, 1),
    "time_stretch": (0.9, 1.1),
    "add_noise": (0, 0.005)
}
# Audio augmentation parameters for each class (used with save_augment_audios_to_mel_spectrogram)
AUDIO_CLASS_AUGMENTATION = {
    "danbau": {
        "time_stretch": (0.98, 1.02)
    },
    "dannhi": {
        "pitch_shift": (-0.5, 0.5),
        "add_noise": (0, 0.01),
        "time_stretch": (0.95, 1.05)
    },
    "dantranh": {
        "add_noise": (0, 0.005)
    },
    "sao": {
        "pitch_shift": (-0.5, 0.5),
        "add_noise": (0, 0.01),
        "time_stretch": (0.95, 1.05)
    }
}



# CNN input shape configuration
INPUT_SHAPE = (128, 212, 3) # Mel-spectrogram image dimensions (Mel bands, time frames, channels)


# Model training configuration
OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=0.0003, weight_decay=1e-4)
METRICS = [
    "accuracy", 
    MacroF1Score(name="macro_f1_score", num_classes=5)
]
# LOSS = "categorical_crossentropy"  # Loss function for multi-class classification
LOSS = "sparse_categorical_crossentropy"  # Loss function for multi-class classification
BATCH_SIZE = 128                   # Training batch size
EPOCHS = 50                        # Number of training epochs
VALIDATION_BATCH_SIZE = 128        # Validation batch size


# Model checkpoint configuration
CHECKPOINT_MONITOR = "val_accuracy"  # Metric to monitor for saving best model
CHECKPOINT_REGEX = r"model1_\d+_(\d+\.\d{4})\.weights\.h5"  # Regex pattern for checkpoint files

# Early stopping configuration
EARLY_MONITOR = "val_loss"  # Metric to monitor for early stopping
EARLY_PATIENCE = 10         # Number of epochs without improvement before stopping
VERBOSE = 1                 # Verbosity level for training logs

# Reduce learning rate on plateau configuration
REDUCE_LR_PATIENCE = 7     # Number of epochs with no improvement to wait before reducing LR
MIN_LR = 1e-6              # Minimum learning rate
LR_FACTOR = 0.5            # Factor to reduce learning rate by
REDUCE_LR_VERBOSE = 1      # Verbosity level for LR reduction