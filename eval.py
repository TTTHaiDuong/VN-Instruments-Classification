from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from main import predict_predominant_instrument
import numpy as np
import tensorflow as tf
import os
import argparse
from config import *
import seaborn as sns
import matplotlib.pyplot as plt



def evaluate_accuracy(model, val_file_paths, val_labels):
    """Đánh giá độ chính xác của mô hình trên tập validation.

    Parameters:
        model: Mô hình CNN đã huấn luyện.
        val_set: Tập dữ liệu validation.
        val_labels: Nhãn của tập dữ liệu validation.

    Returns:
        accuracy: Độ chính xác của mô hình trên tập validation.
    """
    predictions = []
    for file in val_file_paths:
        class_ratios, _ = predict_predominant_instrument(model, file)
        pred_class = np.argmax(class_ratios)
        predictions.append(pred_class)
    return accuracy_score(val_labels, predictions)



def evaluate_metrics(model, val_file_paths, val_labels):
    predictions = []
    for file in val_file_paths:
        class_ratios, _ = predict_predominant_instrument(model, file)
        pred_class = np.argmax(class_ratios)
        predictions.append(pred_class)
    print(classification_report(val_labels, predictions, target_names=CLASS_NAMES))



def plot_confusion_matrix(model, validation_files, validation_labels):
    predictions = []
    for file in validation_files:
        class_ratios, _ = predict_predominant_instrument(model, file)
        pred_class = np.argmax(class_ratios)
        predictions.append(pred_class)
    cm = confusion_matrix(validation_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()



def main():
    parser = argparse.ArgumentParser(description="Các chỉ số đánh giá mô hình.")
    parser.add_argument("-a", "--accuracy", action="store_true", help="In thông tin accuracy.")
    parser.add_argument("-m", "--metrics", action="store_true", help="In ra các chỉ số đánh giá.")
    parser.add_argument("-c", "--confusion", action="store_true", help="Hiển thị ma trận nhầm lẫn.")

    args = parser.parse_args()

    def list_files_in_directory(directory):
        """Liệt kê tất cả các file trong thư mục."""
        return [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    file_count = 100
    model = tf.keras.models.load_model(r"bestmodel\model1.h5", custom_objects={"LeakyReLU": tf.keras.layers.LeakyReLU})

    val_file_paths = []
    val_file_paths.extend(list_files_in_directory(r"rawdata\test\danbau")[:file_count])
    val_file_paths.extend(list_files_in_directory(r"rawdata\test\dannhi")[:file_count])
    val_file_paths.extend(list_files_in_directory(r"rawdata\test\dantranh")[:file_count])
    val_file_paths.extend(list_files_in_directory(r"rawdata\test\sao")[:file_count])

    val_labels = [0]*file_count + [1]*file_count + [2]*file_count + [3]*file_count  # Nhãn thực tế tương ứng với các file âm thanh

    if args.accuracy:
        accuracy = evaluate_accuracy(model, val_file_paths, val_labels)
        print(f"Độ chính xác của mô hình trên tập validation: {accuracy:.2%}")

    elif args.metrics:
        evaluate_metrics(model, val_file_paths, val_labels)

    elif args.confusion:
        plot_confusion_matrix(model, val_file_paths, val_labels)

    else:
        print("Hãy chọn các tuỳ chọn -a, -m, -c.")



if __name__ == "__main__":
    main()