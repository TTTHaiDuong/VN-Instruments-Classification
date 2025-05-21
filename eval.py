import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import tensorflow as tf
from config import *
from main import predict_predominant_instrument
from build.model1 import get_model1
from build.model2 import get_model2
from build.model3 import get_model3
import sklearn.metrics as metrics
from sklearn.preprocessing import label_binarize
from build.datagen import generate_images_data



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
    return metrics.accuracy_score(val_labels, predictions)



def evaluate_metrics(model, val_file_paths, val_labels):
    predictions = []
    for file in val_file_paths:
        class_ratios, _ = predict_predominant_instrument(model, file)
        pred_class = np.argmax(class_ratios)
        predictions.append(pred_class)
    print(metrics.classification_report(val_labels, predictions, target_names=CLASS_NAMES))



def plot_metrics(model, test_generator):
    """
    Hiển thị precision, recall, F1-score và ma trận nhầm lẫn từ test_generator.
    Args:
        model: Mô hình đã huấn luyện.
        test_generator: ImageDataGenerator cho tập test.
        class_names: Tên các lớp.
    """
    # Reset generator để bắt đầu từ mẫu đầu tiên
    test_generator.reset()
    
    # Lấy nhãn thực tế và dự đoán
    y_true = test_generator.classes  # Nhãn thực: [0, 1, 2, 3]
    y_pred = model.predict(test_generator)  # Xác suất: shape (n_samples, n_classes)
    y_pred_classes = np.argmax(y_pred, axis=1)  # Lớp dự đoán: [0, 1, 2, 3]

    # In classification report
    print("Classification Report:")
    print(metrics.classification_report(y_true, y_pred_classes, target_names=CLASS_NAMES, digits=4))

    # Tính và vẽ ma trận nhầm lẫn
    cm = metrics.confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Dự đoán')
    plt.ylabel('Thực tế')
    plt.title('Ma trận nhầm lẫn')
    plt.show()



def plot_confusion_matrix(model, validation_files, validation_labels):
    predictions = []
    for file in validation_files:
        class_ratios, _ = predict_predominant_instrument(model, file)
        pred_class = np.argmax(class_ratios)
        predictions.append(pred_class)
    cm = metrics.confusion_matrix(validation_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()



def plot_roc_curve(model, test_generator):
    # Lấy nhãn thực tế và xác suất dự đoán
    test_generator.reset()
    y_true = test_generator.classes  # Nhãn thực: [0, 1, 2, 3]
    y_score = model.predict(test_generator)  # Xác suất: shape (n_samples, 4)

    # Binarize nhãn cho OvR
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3])  # Shape: (n_samples, 4)
    n_classes = len(CLASS_NAMES)

    # Tính ROC và AUC cho từng lớp
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    # Tính macro-average ROC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

    # Vẽ ROC curve
    plt.figure(figsize=(10, 6))
    colors = ['blue', 'red', 'green', 'purple']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC {CLASS_NAMES[i]} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot(fpr["macro"], tpr["macro"], color='black', linestyle='--', lw=2,
             label=f'Macro-average ROC (AUC = {roc_auc["macro"]:.2f})')
    
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve - Phân loại 4 nhạc cụ')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

    # In AUC
    print("AUC cho từng lớp:")
    for i in range(n_classes):
        print(f"{CLASS_NAMES[i]}: {roc_auc[i]:.2f}")
    print(f"Macro-average AUC: {roc_auc['macro']:.2f}")



def plot_precision_recall_curve(y_true, y_pred_prob, class_names=None, title="Precision-Recall Curve"):
    """
    Vẽ Precision-Recall Curve cho bài toán phân loại đa lớp.
    
    Parameters:
    - y_true: Nhãn thực tế (one-hot encoding, shape: [n_samples, n_classes]).
    - y_pred_prob: Xác suất dự đoán (shape: [n_samples, n_classes]).
    - class_names: Danh sách tên lớp (mặc định: None, sẽ dùng Class 0, Class 1,...).
    - title: Tiêu đề của biểu đồ.
    
    Returns:
    - None (vẽ biểu đồ trực tiếp).
    """
    n_classes = y_true.shape[1]
    if class_names is None:
        class_names = [f'Class {i}' for i in range(n_classes)]
    
    plt.figure(figsize=(8, 6))
    
    for i in range(n_classes):
        # Tính Precision, Recall và AUC cho từng lớp
        precision, recall, _ = metrics.precision_recall_curve(y_true[:, i], y_pred_prob[:, i])
        auc_pr = metrics.auc(recall, precision)
        
        # Vẽ đường cong
        plt.plot(recall, precision, marker='.', label=f'{class_names[i]} (AUC = {auc_pr:.2f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()



def main():
    parser = argparse.ArgumentParser(description="Các chỉ số đánh giá mô hình.")
    parser.add_argument("-a", "--accuracy", action="store_true", help="In thông tin accuracy.")
    parser.add_argument("-m", "--metrics", action="store_true", help="In ra các chỉ số đánh giá.")
    parser.add_argument("-m2", "--metrics2", action="store_true", help="In ra các chỉ số đánh giá, tập test là âm thanh.")
    parser.add_argument("-c", "--confusion", action="store_true", help="Hiển thị ma trận nhầm lẫn.")
    parser.add_argument("-r", "--roc", action="store_true", help="Hiển thị ROC.")
    parser.add_argument("-pr", "--pr_curve", action="store_true", help="Hiển thị PR Curve.")
    
    parser.add_argument("-i", "--input", type=str, help="Đường dẫn của mô hình cần đánh giá.")
    parser.add_argument("-t", "--test", type=str, default=r"rawdata\test", help="Đường dẫn của tập test âm thanh.")

    args = parser.parse_args()

    def list_files_in_directory(directory):
        """Liệt kê tất cả các file trong thư mục."""
        return [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    file_count = 100
    model1, _, _, _ = get_model2()
    # model1.load_weights(r"bestmodel\model1.h5")
    model = tf.keras.models.load_model(r"bestmodel\model1.h5", custom_objects={"LeakyReLU": tf.keras.layers.LeakyReLU})
    # model = model1
    val_file_paths = []
    val_file_paths.extend(list_files_in_directory(os.path.join(args.test, "danbau"))[:file_count])
    val_file_paths.extend(list_files_in_directory(os.path.join(args.test, "dannhi"))[:file_count])
    val_file_paths.extend(list_files_in_directory(os.path.join(args.test, "dantranh"))[:file_count])
    val_file_paths.extend(list_files_in_directory(os.path.join(args.test, "sao"))[:file_count])

    val_labels = [0]*file_count + [1]*file_count + [2]*file_count + [3]*file_count  # Nhãn thực tế tương ứng với các file âm thanh

    _, _, test = generate_images_data(TRAIN_DATA, VAL_DATA, TEST_DATA)

    # Chỉ số accuracy
    if args.accuracy:
        accuracy = evaluate_accuracy(model, val_file_paths, val_labels)
        print(f"Độ chính xác của mô hình trên tập validation: {accuracy:.2%}")

    # Các chỉ số như precision, recall, f1
    elif args.metrics:
        plot_metrics(model, test)

    # Các chỉ số đánh giá, hỗ trợ tập test là âm thanh
    elif args.metrics2:
        evaluate_metrics(model, val_file_paths, val_labels)

    # Ma trận nhầm lẫn, hỗ trợ tập test là âm thanh
    elif args.confusion:
        plot_confusion_matrix(model, val_file_paths, val_labels)

    # ROC và AUC
    elif args.roc:
        plot_roc_curve(model=model, test_generator=test)

    elif args.pr_curve:
        y_true = []
        y_pred_prob = []

        for images, labels in test:
            # Dự đoán xác suất cho batch hiện tại
            preds = model.predict(images)

            # Lưu nhãn và xác suất
            y_true.append(labels)
            y_pred_prob.append(preds)

            # Dừng khi đã duyệt qua toàn bộ tập test
            if len(y_true) * test.batch_size >= test.samples:
                break
            
        # Chuyển danh sách thành mảng numpy
        y_true = np.concatenate(y_true, axis=0)  # Shape: (n_samples, 4)
        y_pred_prob = np.concatenate(y_pred_prob, axis=0)
        plot_precision_recall_curve(y_true=y_true, y_pred_prob=y_pred_prob, class_names=CLASS_NAMES)

    else:
        parser.print_help()



if __name__ == "__main__":
    main()