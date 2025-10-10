import argparse
import matplotlib.pyplot as plt
import numpy as np
import os, click
import seaborn as sns
import sklearn.metrics as metrics
from build.datagen import from_images
from config.general import *
from predict import predict_endpoint
from pathlib import Path
from sklearn.preprocessing import label_binarize
from scripts.utils import collect_files, limit_files_per_class
from config.general import INSTRUMENTS
from scripts.predict import load_model, collect_mels
from train import validate_model_idx
from scripts.preprocess_image import load_image
from typing import cast


# def plot_metrics(model, test_generator, model_name: str):
#     """
#     Display precision, recall, F1-score, and confusion matrix based on test_generator.
    
#     Parameters:
#         model: Trained model.
#         test_generator: ImageDataGenerator for the test set.
#         model_name: Name of the model, used in plot titles.
#     """
#     # Reset the generator to start from the first sample
#     test_generator.reset()
    
#     # Get true labels and predictions
#     y_true = test_generator.classes  # Ground truth labels: [0, 1, 2, 3]
#     y_pred = model.predict(test_generator)  # Probabilities: shape (n_samples, n_classes)
#     y_pred_classes = np.argmax(y_pred, axis=1)  # Predicted class indices: [0, 1, 2, 3]

#     # Print classification report
#     print(f"\n  {model_name} - Classification Metrics:\n")
#     print(metrics.classification_report(y_true, y_pred_classes, target_names=CLASS_NAMES, digits=4))

#     # Compute and plot confusion matrix
#     cm = metrics.confusion_matrix(y_true, y_pred_classes)
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
#     plt.xlabel("Predicated")
#     plt.ylabel("True")
#     title = f"{model_name} -- Confusion Matrix" if model_name else "Confusion Matrix"    
#     plt.title(title)
#     plt.show()


def classification_report(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
    model_name: str = ""
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
    print(f"\n  {model_name} - Classification Metrics:\n")
    print(metrics.classification_report(y_true, y_pred_classes, target_names=class_names, digits=4))

    # Compute and plot confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicated")
    plt.ylabel("True")
    title = f"{model_name} -- Confusion Matrix" if model_name else "Confusion Matrix"    
    plt.title(title)
    plt.show()


# def plot_roc_curve(model, test_generator, title="ROC Curve"):
#     """
#     Plot the ROC Curve for each class and the macro-average ROC from a multi-class classifier.

#     Parameters:
#         model: Trained classification model.
#         test_generator: ImageDataGenerator for the test set.
#         title (str): Title for the ROC plot.
#     """
#     # Reset the generator to start from the beginning
#     test_generator.reset()
#     y_true = test_generator.classes  # True labels: [0, 1, 2, 3]
#     y_score = model.predict(test_generator)  # Probabilities: shape (n_samples, n_classes)

#     # Binarize the labels for One-vs-Rest ROC calculation
#     y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3, 4])  # Shape: (n_samples, 4)
#     n_classes = len(CLASS_NAMES)

#     # Compute ROC curve and AUC for each class
#     fpr = dict()
#     tpr = dict()
#     roc_auc = dict()
#     for i in range(n_classes):
#         fpr[i], tpr[i], _ = metrics.roc_curve(y_true_bin[:, i], y_score[:, i])
#         roc_auc[i] = metrics.auc(fpr[i], tpr[i])

#     # Compute macro-average ROC curve and AUC
#     all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
#     mean_tpr = np.zeros_like(all_fpr)
#     for i in range(n_classes):
#         mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
#     mean_tpr /= n_classes

#     fpr["macro"] = all_fpr
#     tpr["macro"] = mean_tpr
#     roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

#     # Plot all ROC curves
#     plt.figure(figsize=(10, 6))
#     colors = ["blue", "red", "green", "purple"]
#     for i, color in zip(range(n_classes), colors):
#         plt.plot(fpr[i], tpr[i], color=color, lw=2,
#                  label=f"ROC {CLASS_NAMES[i]} (AUC = {roc_auc[i]:.2f})")
    
#     plt.plot(fpr["macro"], tpr["macro"], color="black", linestyle="--", lw=2,
#              label=f"Macro-average ROC (AUC = {roc_auc["macro"]:.2f})")
    
#     plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel("False Positive Rate (FPR)")
#     plt.ylabel("True Positive Rate (TPR)")
#     plt.title(title)
#     plt.legend(loc="lower right")
#     plt.grid(True)
#     plt.show()

#     # In AUC
#     print("AUC for each class:")
#     for i in range(n_classes):
#         print(f"{CLASS_NAMES[i]}: {roc_auc[i]:.2f}")
#     print(f"Macro-average AUC: {roc_auc["macro"]:.2f}")



# def plot_precision_recall_curve(
#     y_true: np.ndarray, 
#     y_pred_prob: np.ndarray,
#     class_names: list[str] = None, 
#     title="Precision-Recall Curve"
# ):
#     """
#     Plot the Precision-Recall Curve for a multi-class classification problem.

#     Parameters:
#         y_true: Ground truth labels in one-hot encoding format (shape: [n_samples, n_classes]).
#         y_pred_prob: Predicted probabilities for each class (shape: [n_samples, n_classes]).
#         class_names: List of class names (default: None, uses Class 0, Class 1, ...).
#         title: Title of the plot.
#     """
#     n_classes = y_true.shape[1]
#     if class_names is None:
#         class_names = [f"Class {i}" for i in range(n_classes)]
    
#     plt.figure(figsize=(8, 6))
    
#     for i in range(n_classes):
#         # Compute precision, recall, and AUC for each class
#         precision, recall, _ = metrics.precision_recall_curve(y_true[:, i], y_pred_prob[:, i])
#         auc_pr = metrics.auc(recall, precision)
        
#         # Plot the curve
#         plt.plot(recall, precision, marker=".", label=f"{class_names[i]} (AUC = {auc_pr:.2f})")
    
#     plt.xlabel("Recall")
#     plt.ylabel("Precision")
#     plt.title(title)
#     plt.legend()
#     plt.grid(True)
#     plt.show()


@click.command()
@click.argument("mode", type=click.Choice(["audio", "img"]))
@click.argument("test_path", type=click.Path(exists=True, file_okay=False))
@click.option("--model_path", "-p", type=click.Path(exists=True, dir_okay=False), default=MAIN_MODEL)
@click.option("--model_index", "-i", type=int, default=1, callback=validate_model_idx)
@click.option("--verbose", "-v", is_flag=True)
def cli(
    mode: str,
    test_path: str,
    model_path,
    model_index: int,
    max_per_class: int = 100,
    shuffle = False,
    seed: int | None = None,
    class_map = INSTRUMENTS,
    verbose = False
):
    paths, labels = collect_files(test_path)

    if None in labels:
        raise ValueError(f"Missing `label` for {paths[labels.index(None)]}")
    labels = cast(list[str], labels)

    paths, labels = limit_files_per_class(
        file_paths=paths, 
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

    classification_report(y_true, y_pred, class_map.names())



def main():
    parser = argparse.ArgumentParser(description="Model evaluation metrics.")
    parser.add_argument("-m1", "--metrics1", action="store_true", help="Display classification metrics for image-based test set.")
    parser.add_argument("-m2", "--metrics2", action="store_true", help="Display classification metrics for audio-based test set.")
    parser.add_argument("-r", "--roc", action="store_true", help="Display ROC curve.")
    parser.add_argument("-p", "--prcurve", action="store_true", help="Display Precision-Recall curve.")
    
    parser.add_argument("-m", "--model", type=str, default=MAIN_MODEL, help="Path to the trained model.")
    parser.add_argument("-i", "--index", type=str, default=1, help="Model index (1, 2) if loading from .weights.h5.")
    parser.add_argument("-t", "--test", type=str, default=r"rawdata\test", help="Path to the test audio directory.")

    args = parser.parse_args()

    def list_files_in_directory(directory):
        """List all files in a given directory."""
        return [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    file_count = 100
    model = load_model(args.model, args.index)
    model_name = Path(args.model).stem or ""

    # Load test image data
    _, _, test = from_images(TRAIN_DATA, VAL_DATA, TEST_DATA)

    # Collect test audio file paths by class
    val_file_paths = []
    val_file_paths.extend(list_files_in_directory(os.path.join(args.test, "danbau"))[:file_count])
    val_file_paths.extend(list_files_in_directory(os.path.join(args.test, "dannhi"))[:file_count])
    val_file_paths.extend(list_files_in_directory(os.path.join(args.test, "dantranh"))[:file_count])
    val_file_paths.extend(list_files_in_directory(os.path.join(args.test, "sao"))[:file_count])
    val_file_paths.extend(list_files_in_directory(os.path.join(args.test, "dantrung"))[:file_count])

    # Assign corresponding labels for evaluation
    val_labels = [0]*file_count + [1]*file_count + [2]*file_count + [3]*file_count + [4]*file_count

    # Option 1: Classification report from image-based dataset
    if args.metrics1:
        plot_metrics(model, test, model_name)

    # Option 2: Classification report from audio-based dataset
    elif args.metrics2:
        plot_audio_metrics(model, val_file_paths, val_labels)

    # Option 3: ROC curve
    elif args.roc:
        plot_roc_curve(
            model=model, 
            test_generator=test, 
            title=f"{model_name} -- ROC Curve"
        )

    # Option 4: Precision-Recall curve
    elif args.prcurve:
        y_true = []
        y_pred_prob = []

        for images, labels in test:
            preds = model.predict(images)
            y_true.append(labels)
            y_pred_prob.append(preds)

            if len(y_true) * test.batch_size >= test.samples:
                break
            
        y_true = np.concatenate(y_true, axis=0)  # Shape: (n_samples, n_class)
        y_pred_prob = np.concatenate(y_pred_prob, axis=0)
        
        plot_precision_recall_curve(
            y_true=y_true, 
            y_pred_prob=y_pred_prob, 
            class_names=CLASS_NAMES, 
            title=f"{model_name or ""} -- Precision-Recall Curve"
        )

    else:
        parser.print_help()



if __name__ == "__main__":
    cli()
    # main()

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
