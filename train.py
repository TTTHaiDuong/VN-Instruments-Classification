import os, click
import matplotlib.pyplot as plt
import tensorflow as tf
from config.general import TRAIN_CONFIG
from build.datagen import from_images, load_tfrecords, np_dataset
from datetime import datetime
from build.callbacks import get_callbacks
from build.model1 import get_model1
from build.model2 import get_model2
import config.general as config
from typing import Literal


MODEL_REGISTRY = {
    1: get_model1,
    2: get_model2
}


def fit_model(
    model, 
    checkpoint, 
    early, 
    reducelr, 
    train_set,
    y, 
    val_set, 
    class_weight = None,
    cfg = TRAIN_CONFIG,
    verbose = False,
    **overrides
):
    """
    Train the CNN model.

    Parameters:
        model: The CNN model instance.
        checkpoint: Callback for saving the best model during training.
        early: Callback for early stopping.
        reducelr: Callback for reducing learning rate on plateau.
        train_set: Training dataset.
        val_set: Validation dataset.
        class_weight: Dictionary mapping class indices to weights, to handle class imbalance.

    Returns:
        model_history: Training history object containing loss, metrics, etc.
    """
    destin_cfg = cfg.model_copy(update=overrides)

    model.compile(
        loss = config.LOSS, 
        optimizer = config.OPTIMIZER, 
        metrics = config.METRICS,
        # weighted_metrics =
    )
    
    model_history = model.fit(
        train_set,
        y,
        validation_data = val_set, 
        batch_size = destin_cfg.batch_size, 
        validation_batch_size = cfg.validation_batch_size,
        epochs = destin_cfg.epochs, 
        callbacks = [cb for cb in [early, checkpoint, reducelr] if cb],
        verbose = verbose,
        # class_weight = class_weight
    )
    return model_history


def save_training_plot(
    history, 
    file_prefix="training_plot",
    save_at = ""
):
    """
    Plot and save the training history graph with a timestamp in the filename.

    Parameters:
        history: History object returned from model.fit().
        prefix (str): Prefix for the filename.
    """
    os.makedirs(config.TRAIN_HISTORY_PATH, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{file_prefix}_{timestamp}.png"
    full_path = os.path.join(config.TRAIN_HISTORY_PATH, filename)

    plt.figure(figsize=(12, 4))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Train vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Train vs Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(full_path, dpi=300)
    plt.close()

    print(f"Plot successfully saved at: {full_path}")


def validate_model_idx(ctx, param, value):
    if value not in MODEL_REGISTRY:
        raise click.BadParameter(
            f"Invalid index: {value}. Allowed: {list(MODEL_REGISTRY.keys())}"
        )
    return value


@click.command()
@click.option("--model_index", "-i", type=int, default=1, callback=validate_model_idx)
@click.option("--image_augment", "-a", is_flag=False)
@click.option("--training_plot", "-t", is_flag=True)
@click.option("--dataset_source", "-s", type=click.Choice(["img", "tfr"]), default="img")
@click.option("--verbose", "-v", is_flag=True)
def train(
    model_index: int, 
    image_augment: bool, 
    training_plot: bool,
    dataset_source: Literal["img", "tfr"] = "img",
    cfg = TRAIN_CONFIG,
    verbose = False
):
    print(f"\n===== Training Model {model_index} =====")

    if dataset_source == "img":
        train_set, val_set, _ = from_images(
            config.TRAIN_DATA, 
            config.VAL_DATA, 
            config.TEST_DATA, 
            image_augment
        )
    elif dataset_source == "tfr":
        train_set = load_tfrecords("tfrecord/train.tfrecord").shuffle(1000)
        X_train, y_train = np_dataset(train_set)
       
        val_set = load_tfrecords("tfrecord/val.tfrecord")
        val_set = np_dataset(val_set)

    model, model_name = MODEL_REGISTRY[model_index]()
    checkpoint, early, reducelr = get_callbacks(
        checkpoint_path=os.path.join(config.CHECKPOINT_PATH, model_name, f"{model_name}_epoch{{epoch:02d}}_{{val_macro_f1_score:.4f}}.weights.h5"),
        bestmodel_path=os.path.join(config.BEST_MODEL, f"{model_name}.h5"),
    )

    model.compile(
        loss = config.LOSS, 
        optimizer = config.OPTIMIZER, 
        metrics = config.METRICS,
    )

    model_history = model.fit(
        X_train,
        y_train,
        validation_data = val_set, 
        batch_size = TRAIN_CONFIG.batch_size, 
        validation_batch_size = cfg.validation_batch_size,
        epochs = TRAIN_CONFIG.epochs, 
        callbacks = [cb for cb in [early, checkpoint, reducelr] if cb],
        verbose = verbose,
        # class_weight = class_weight
    )

    # history = fit_model(model, check, early, reduce, train_set=train_set[0], y=train_set[1], val_set=val_set)
    
    if training_plot:
        save_training_plot(model_history)


if __name__ == "__main__":
    train()