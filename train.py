import os, click
import config.general as config
import matplotlib.pyplot as plt
from typing import Literal
from pathlib import Path
from datetime import datetime
from build.datagen import from_images, load_tfrecords, np_dataset
from build.callbacks import get_callbacks
from build.model1 import get_model1
from build.model2 import get_model2
from config.general import TRAIN_CONFIG


MODEL_REGISTRY = {
    1: get_model1,
    2: get_model2
}

def save_training_plot(
    history, 
    fpath_out: str,
    show = True
):
    """
    Plot and save the training history graph with a timestamp in the filename.

    Parameters:
        history: History object returned from model.fit().
        prefix (str): Prefix for the filename.
    """
    out_path = Path(fpath_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

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
    plt.savefig(fpath_out, dpi=300)
    
    if show:
        plt.show()
    else:
        plt.close()

    print(f"Plot successfully saved at: {fpath_out}")


def validate_model_idx(ctx, param, value):
    if value not in MODEL_REGISTRY:
        raise click.BadParameter(
            f"Invalid index: {value}. Allowed: {list(MODEL_REGISTRY.keys())}"
        )
    return value


@click.command()
@click.argument("mode", type=click.Choice(["img", "tfr"]), default="tfr")
@click.option("--model_index", "-i", type=int, default=1, callback=validate_model_idx)
@click.option("--image_augment", "-a", is_flag=False)
@click.option("--training_plot", "-t", is_flag=True, default=True)
@click.option("--verbose", "-v", is_flag=True)
def train(
    mode: Literal["img", "tfr"],
    model_index: int, 
    image_augment: bool, 
    training_plot: bool,
    cfg = TRAIN_CONFIG,
    verbose = False
):
    print(f"\n===== Training Model {model_index} =====")

    if mode == "img":
        train_set, val_set, _ = from_images(
            config.TRAIN_DATA, 
            config.VAL_DATA, 
            config.TEST_DATA, 
            image_augment
        )
    elif mode == "tfr":
        train_set = load_tfrecords("tfrecord/train.tfrecord").shuffle(1000)
        X_train, y_train = np_dataset(train_set)
       
        val_set = load_tfrecords("tfrecord/val.tfrecord")
        val_set = np_dataset(val_set)

    model, model_name = MODEL_REGISTRY[model_index]()
    checkpoint, early, reducelr = get_callbacks(
        checkpoint_path=os.path.join(config.CHECKPOINT_PATH, model_name, f"{model_name}_ep{{epoch:02d}}_{{metric_name}}_{{metric_score:.4f}}_{{timestamp:%d%m%Y_%H%M%S}}.weights.h5"),
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
    
    if training_plot:
        timestamp = datetime.now().strftime('%d%m%Y_%H%M%S')
        fname = f"{model_name}_train_plot_{timestamp}.png"
        full_path = os.path.join(config.TRAIN_HISTORY_PATH, fname)

        save_training_plot(model_history, full_path)


if __name__ == "__main__":
    train()