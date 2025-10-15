import click
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.preprocessing import label_binarize
from train import validate_model_idx
from scripts.utils import collect_data_files, limit_per_class, register
from scripts.predict import load_model, collect_mels
from scripts.preprocess_image import load_image
from config.general import MAIN_MODEL, INSTRUMENTS


EVAL_REGISTER = {}
COLORS = ["blue", "red", "green", "purple", "orange"]


@register(EVAL_REGISTER, "score")
def classification_score(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    class_names: list[str],
):
    """
    Display precision, recall, F1-score, and confusion matrix based on test_generator.
    
    Parameters:
        model: Trained model.
        test_generator: ImageDataGenerator for the test set.
        model_name: Name of the model, used in plot titles.
    """
    y_pred_classes = np.argmax(y_pred, axis=1)  # Predicted class indices: [0, 1, 2, 3]

    # Print classification report
    print(f"\nClassification Metrics:\n")
    print(metrics.classification_report(y_true, y_pred_classes, target_names=class_names, digits=4))


@register(EVAL_REGISTER, "matrix")
def confusion_matrix(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    class_names: list[str],
):
    # Compute and plot confusion matrix
    y_pred_classes = np.argmax(y_pred, axis=1)  # Predicted class indices: [0, 1, 2, 3]

    cm = metrics.confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicated")
    plt.ylabel("True")
    plt.title("Confusion Matrix")


@register(EVAL_REGISTER, "roc")
def roc_curve(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    class_names: list[str],
):
    y_true_bin = label_binarize(y_true, classes=[i for i in range(len(class_names))])
    n_classes = y_true_bin.shape[1]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_true_bin[:, i], y_pred[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(figsize=(10, 6))
    for i, color in zip(range(n_classes), COLORS):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f"ROC {class_names[i]} (AUC = {roc_auc[i]:.2f})")
    
    plt.plot(fpr["macro"], tpr["macro"], color="black", linestyle="--", lw=2,
             label=f"Macro-average ROC (AUC = {roc_auc["macro"]:.2f})")
    
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title(f"ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)


@register(EVAL_REGISTER, "pr")
def pr_curve(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    class_names: list[str], 
):
    """
    Plot the Precision-Recall Curve for a multi-class classification problem.

    Parameters:
        y_true: Ground truth labels in one-hot encoding format (shape: [n_samples, n_classes]).
        y_pred_prob: Predicted probabilities for each class (shape: [n_samples, n_classes]).
        class_names: List of class names (default: None, uses Class 0, Class 1, ...).
        title: Title of the plot.
    """
    y_true_bin = label_binarize(y_true, classes=[i for i in range(len(class_names))])
    n_classes = y_pred.shape[1]

    precision = dict()
    recall = dict()
    pr_auc = dict()

    for i in range(n_classes):
        precision[i], recall[i], _ = metrics.precision_recall_curve(y_true_bin[:, i], y_pred[:, i])
        pr_auc[i] = metrics.auc(recall[i], precision[i])
    
    plt.figure(figsize=(8, 6))

    for i, color in zip(range(n_classes), COLORS):
        plt.plot(recall[i], precision[i], color=color, lw=2,
                 label=f"PR {class_names[i]} (AUC = {pr_auc[i]:.2f})")  
          
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True)


@click.command()
@click.argument("mode", type=click.Choice(["audio", "img"]))
@click.argument("test_path", type=click.Path(exists=True, file_okay=False))
@click.option("--model_path", "-p", type=click.Path(exists=True, dir_okay=False), default=MAIN_MODEL)
@click.option("--model_index", "-i", type=int, default=1, callback=validate_model_idx)
@click.option("--plots_metrics", "-m", multiple=True, type=click.Choice(list(EVAL_REGISTER.keys())), default=("score",))
@click.option("--verbose", "-v", is_flag=True)
def eval(
    mode: str,
    test_path: str,
    model_path,
    model_index: int,
    plots_metrics: list[str],
    max_per_class: int = 100,
    shuffle = False,
    seed: int | None = None,
    class_map = INSTRUMENTS,
    verbose = False
):
    paths, labels = collect_data_files(test_path)

    paths, labels = limit_per_class(
        fpaths=paths, 
        labels=labels, 
        max_per_class=max_per_class,
        shuffle=shuffle,
        seed=seed,
        verbose=verbose
    )

    model = load_model(model_path, model_index)

    if mode == "audio":
        mels = collect_mels(paths)
        mels = np.stack(mels, axis=0)
        y_true = [class_map.label_to_index(lbl) for lbl in labels]
        y_pred = model.predict(mels, verbose=verbose) 

    elif mode == "img":
        mels = []
        for path in paths:
            mels.append(load_image(path, (128, 212)))
        
        y_true = [class_map.label_to_index(label) for label in labels]
        y_pred = model.predict(np.squeeze(np.array(mels), axis=1), verbose=verbose)

    else:
        raise ValueError("Invalid mode")

    for pm in plots_metrics:
        if pm in EVAL_REGISTER.keys():
            EVAL_REGISTER[pm](y_true, y_pred, class_map.names())
        else:
            raise ValueError("Not existing key")
    plt.show()


if __name__ == "__main__":
    eval()

    # import librosa
    # from scripts.mel_spectrogram import to_mel
    # from scripts.preprocess_image import load_image
    # from build.datagen import load_tfrecords, np_dataset

    # y, sr = librosa.load("rawdata/train/danbau/danbau000.wav")
    # audio = to_mel(y, sr)

    # tfr = load_tfrecords("tfrecord/train.tfrecord")
    # X_train, y_train = np_dataset(tfr)

    # # image = load_image("images_dataset/test/danbau/danbau400.png", (128, 212))
    # diff = np.abs(audio - X_train[0]).mean()
    # print("mean abs diff:", diff)