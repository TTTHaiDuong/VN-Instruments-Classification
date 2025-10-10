import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, BatchNormalization, Activation
from config.general import TRAIN_CONFIG


def get_model1(
    cfg = TRAIN_CONFIG,
    **overrides
):
    destin_cfg = cfg.model_copy(update=overrides)
    num_classes = len(destin_cfg.class_names)

    name = "model1"
    
    model = tf.keras.models.Sequential([
        # First convolution
        Input(shape=destin_cfg.input_shape),
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

        # # First convolution
        # Conv2D(16, (3, 3), input_shape=input_shape),
        # BatchNormalization(),
        # Activation('relu'),
        # MaxPooling2D(2, 2),
        
        # # Second convolution
        # Conv2D(32, (3, 3)),
        # BatchNormalization(),
        # Activation('relu'),
        # MaxPooling2D(2, 2),
        
        # # Third convolution
        # Conv2D(64, (3, 3)),
        # BatchNormalization(),
        # Activation('relu'),
        # MaxPooling2D(2, 2),
        
        # # Fourth convolution
        # Conv2D(64, (3, 3)),
        # BatchNormalization(),
        # Activation('relu'),
        # MaxPooling2D(2, 2),

        # # Fifth convolution
        # Conv2D(128, (3, 3)),
        # BatchNormalization(),
        # Activation('relu'),
        # MaxPooling2D(2, 2),

        # Flatten(),
        # Dropout(0.5),
        # Dense(256, activation='relu'),
        # Dropout(0.3),
        # Dense(128, activation='relu'),
        # Dense(n_class, activation='softmax')