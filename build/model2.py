import os
import sys
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, GlobalMaxPooling2D, Dense
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.general import *
from scripts.checkpoint import DualCheckpoint



# Transfer from IRMAS, VGG16
# A Single Predominant Instrument Recognition of Polyphonic Music Using CNN-based Timbre Analysis
def get_model2(input_shape = (128, 128, 3), n_class = 4):
    model = tf.keras.models.Sequential([
        # First convolution
        Conv2D(64, kernel_size=(3, 3), padding='same', input_shape=input_shape),
        Activation(tf.keras.layers.LeakyReLU(alpha=0.33)),

        # Second convolution
        Conv2D(64, kernel_size=(3, 3), padding='same'),
        Activation(tf.keras.layers.LeakyReLU(alpha=0.33)),
        MaxPooling2D(pool_size=(3, 3)),
        Dropout(0.25),

        # Third convolution
        Conv2D(128, kernel_size=(3, 3), padding='same'),
        Activation(tf.keras.layers.LeakyReLU(alpha=0.33)),

        # Fourth convolution
        Conv2D(128, kernel_size=(3, 3), padding='same'),
        Activation(tf.keras.layers.LeakyReLU(alpha=0.33)),
        MaxPooling2D(pool_size=(3, 3)),
        Dropout(0.25),

        # Fifth convolution
        Conv2D(256, kernel_size=(3, 3), padding='same'),
        Activation(tf.keras.layers.LeakyReLU(alpha=0.33)),

        # Sixth convolution
        Conv2D(256, kernel_size=(3, 3), padding='same'),
        Activation(tf.keras.layers.LeakyReLU(alpha=0.33)),
        MaxPooling2D(pool_size=(3, 3)),
        Dropout(0.5),

        # Seventh convolution
        Conv2D(512, kernel_size=(3, 3), padding='same'),
        Activation(tf.keras.layers.LeakyReLU(alpha=0.33)),

        # Eighth convolution
        Conv2D(512, kernel_size=(3, 3), padding='same'),
        Activation(tf.keras.layers.LeakyReLU(alpha=0.33)),
        GlobalMaxPooling2D(),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        Dense(n_class, activation='softmax')
    ])

    if not os.path.exists(os.path.join(CHECKPOINT_PATH, 'model2')):
        os.makedirs(os.path.join(CHECKPOINT_PATH, 'model2'))

    checkpoint= DualCheckpoint(
        filepath1 = os.path.join(CHECKPOINT_PATH, 'model2', 'model2_epoch{epoch:02d}_{val_macro_f1_score:.4f}.weights.h5'),
        filepath2 = os.path.join(BEST_MODEL, 'model2.h5'),
        monitor = "val_macro_f1_score",
        mode="max",
        save_best_only = True,
        save_weights_only = True,
        verbose = 1
    )

    early = EarlyStopping(
        monitor="val_macro_f1_score",
        mode="max",
        patience = EARLY_PATIENCE,
        restore_best_weights= True,
        verbose = VERBOSE
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_macro_f1_score',
        patience=REDUCE_LR_PATIENCE,
        min_lr=MIN_LR,
        factor=LR_FACTOR,
        verbose=REDUCE_LR_VERBOSE,
        mode='max'
    )
    
    return model, checkpoint, early, reduce_lr