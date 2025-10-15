import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, BatchNormalization, Activation
from config.general import TRAIN_CONFIG, INSTRUMENTS


def get_model1(
    input_shape: tuple[int, int, int] | None = None
):
    if not input_shape:
        input_shape = TRAIN_CONFIG.input_shape
    num_classes = len(INSTRUMENTS)

    name = "model1"
    
    model = tf.keras.models.Sequential([
        # First convolution
        Input(shape=input_shape),
        Conv2D(16, (3,3), activation='relu'),
        MaxPooling2D(2, 2),
        
        # Second convolution
        Conv2D(32, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        
        # Third convolution
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        
        # Fourth convolution
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dropout(0.2),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    return model, name