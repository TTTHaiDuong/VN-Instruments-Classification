import numpy as np
import tensorflow as tf
import config.general as cfg
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from scripts.tfrecord import parse_example as raw_parse
from typing import Literal


def from_images(train_dir, val_dir, test_dir, use_augment=False):
    """
    Load and preprocess image data from training, validation, and test directories.
    The samples are Mel-spectrogram images.
    Each directory contains subfolders named according to the class labels defined in CLASS_LABELS (config.py).

    Parameters:
        train_dir (str): Path to the training directory.
        val_dir (str): Path to the validation directory.
        test_dir (str): Path to the test directory.
        augment_data (bool): Whether to apply image data augmentation on the training set.

    Returns:
        train_generator: Data generator for training.
        val_generator: Data generator for validation.
        test_generator: Data generator for testing.
    """
    # Apply image data augmentation for the training set if specified
    aug_params = cfg.IMAGE_AUGMENTATION | {"rescale": cfg.RESCALE} if use_augment else {"rescale": cfg.RESCALE}
    train_datagen = ImageDataGenerator(**aug_params)

    train_set = train_datagen.flow_from_directory(
        train_dir,
        target_size = (cfg.INPUT_SHAPE[0], cfg.INPUT_SHAPE[1]),
        shuffle = True,
        subset = "training",
        classes = cfg.CLASS_LABELS
    )

    val_datagen = ImageDataGenerator(rescale=cfg.RESCALE)
    val_set = val_datagen.flow_from_directory(
        val_dir,
        target_size = (cfg.INPUT_SHAPE[0], cfg.INPUT_SHAPE[1]),
        classes = cfg.CLASS_LABELS,
        shuffle = False
    )

    test_datagen = ImageDataGenerator(rescale=cfg.RESCALE)
    test_set = test_datagen.flow_from_directory(
        test_dir,
        target_size = (cfg.INPUT_SHAPE[0], cfg.INPUT_SHAPE[1]),
        classes = cfg.CLASS_LABELS,
        shuffle = False
    )    

    return train_set, val_set, test_set


def load_tfrecords(files: str | list[str]):
    if isinstance(files, str):
        files = [files]

    ds = tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.AUTOTUNE)
    
    ds = ds.map(
        # loại bỏ trường id khỏi các mẫu, id chỉ dùng quản lý mẫu chứ không sử dụng cho huấn luyện
        lambda example_proto: tuple(raw_parse(example_proto)[:2]), 
        num_parallel_calls=tf.data.AUTOTUNE
    )
    return ds


def np_dataset(dataset):
    X_list, y_list = [], []
    for x, y in dataset.as_numpy_iterator():
        X_list.append(x)
        y_list.append(y)
    
    X = np.stack(X_list, axis=0)
    y = np.array(y_list)

    return X, y


def make_fold(
    dataset: tf.data.Dataset,
    n_folds: int,
    fold_idx: int,
    split: Literal["train", "val"]
):
    """
    """
    ds = dataset.enumerate()

    if split == "train":
        ds = ds.filter(lambda i, _: (i % n_folds) != fold_idx)
    elif split == "val":
        ds = ds.filter(lambda i, _: (i % n_folds) == fold_idx)
    else:
        raise ValueError("split phải là 'train' hoặc 'val'.")

    ds = ds.map(lambda _, xy: xy, num_parallel_calls=tf.data.AUTOTUNE)
    return ds