import os
import numpy as np
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
from mel_spec import audio_to_mel_spectrogram
from config import *
from mel_spec import *
from collections import Counter
import argparse
import re
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array


def predict_predominant_instrument(model, file_path, segment_length=5, sr=22050, log=True):
    """Dự đoán nhạc cụ chính trong file âm thanh"""
    ext = os.path.splitext(file_path)[1].lower()
    predictions = []
    segment_info = []

    if ext in ['.wav', '.mp3']:
        y, sr = librosa.load(file_path, sr=sr)
        total_length = len(y) / sr
        num_segments = int(np.ceil(total_length / segment_length))

        for i in range(num_segments):
            start_sample = int(i * segment_length * sr)
            end_sample = int(min((i + 1) * segment_length * sr, len(y)))
            segment_y = y[start_sample:end_sample]
            mel_spec = audio_to_mel_spectrogram(segment_y)
            pred = model.predict(np.expand_dims(mel_spec, axis=0))
            predicted_class = np.argmax(pred, axis=1)[0]
            predictions.append(predicted_class)

            segment_info.append({
                "start_time": i * segment_length,
                "probabilities": pred.tolist()
            })

            if log:
                print(f"[AUDIO] {os.path.basename(file_path)} Đoạn {i+1}/{num_segments}: Thời điểm: {i*segment_length:.2f}s, Tỉ lệ: {np.round(pred, 3)}")

    elif ext in ['.png', '.jpg', '.jpeg']:
        img = Image.open(file_path).convert('RGB')
        img = img.resize((INPUT_SHAPE[0], INPUT_SHAPE[1]))  # hoặc theo kích thước input của model bạn
        img_array = img_to_array(img) / 255.0  # chuẩn hóa
        img_array = np.expand_dims(img_array, axis=0)

        pred = model.predict(img_array)
        predicted_class = np.argmax(pred, axis=1)[0]

        predictions.append(predicted_class)
        segment_info.append({
            "source": file_path,
            "probabilities": pred.tolist()
        })

        if log:
            print(f"[IMAGE] {os.path.basename(file_path)}: {np.round(pred, 3)}")

    class_counts = Counter(predictions)
    total_segments = len(predictions)
    class_ratios = [class_counts.get(i, 0) / total_segments * 100 for i in range(len(CLASS_NAMES))]

    if log:
        print("\nTỉ lệ dự đoán tổng cho từng lớp:")
        for class_name, ratio in zip(CLASS_NAMES, class_ratios):
            print(f"{class_name}: {ratio:.2f}%")
        print(f"Nhạc cụ chính: {CLASS_NAMES[np.argmax(class_ratios)]}")
    return class_ratios, segment_info



# Hàm hiển thị xác suất
def plot_probabilities(probabilities):
    plt.figure(figsize=(10, 4))
    plt.bar(probabilities.keys(), probabilities.values())
    plt.title('Probability of Each Instrument')
    plt.xlabel('Instrument')
    plt.ylabel('Probability')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



def print_prediction_results(predictions, class_names, top_k=4):
    """
    In tên nhạc cụ được dự đoán và xác suất từng lớp.

    Parameters:
        predictions (np.ndarray): Kết quả trả về từ model.predict(), shape (1, num_classes)
        class_names (list): Danh sách tên các lớp, ví dụ: ['dantranh', 'danbau', 'dannhi', 'sao']
        top_k (int): Số lớp muốn hiển thị (mặc định: tất cả)
    """
    # Lấy vector xác suất đầu ra
    probs = predictions[0]  # (num_classes,)
    
    # Lấy chỉ số lớp có xác suất cao nhất
    predicted_index = int(np.argmax(probs))
    predicted_label = class_names[predicted_index]
    predicted_prob = probs[predicted_index]

    print(f"Nhạc cụ được dự đoán: **{predicted_label}** (xác suất: {predicted_prob:.2%})\n")
    print("Xác suất từng lớp:")

    # Sắp xếp theo xác suất giảm dần
    sorted_indices = np.argsort(probs)[::-1]

    for i in sorted_indices[:top_k]:
        label = class_names[i]
        prob = probs[i]
        print(f"  - {label:<10}: {prob:.2%}")



def plot_prediction_probabilities(predictions, class_names):
    """
    Vẽ biểu đồ xác suất dự đoán cho từng lớp.

    Parameters:
        predictions (np.ndarray): Kết quả từ model.predict(), shape (1, num_classes)
        class_names (list): Danh sách tên các lớp, ví dụ: ['danbau', 'dannhi', 'dantranh', 'sao']
    """
    probs = predictions[0]

    # Sắp xếp theo xác suất giảm dần
    sorted_indices = np.argsort(probs)[::-1]
    sorted_labels = [class_names[i] for i in sorted_indices]
    sorted_probs = probs[sorted_indices]

    # Vẽ biểu đồ
    plt.figure(figsize=(8, 4))
    bars = plt.barh(sorted_labels, sorted_probs, color='skyblue')
    plt.xlabel('Xác suất')
    plt.title('Biểu đồ xác suất dự đoán nhạc cụ')
    plt.xlim(0, 1)

    # Hiển thị giá trị phần trăm trên từng cột
    for bar, prob in zip(bars, sorted_probs):
        plt.text(prob + 0.01, bar.get_y() + bar.get_height()/2,
                 f"{prob:.2%}", va='center')

    plt.gca().invert_yaxis()  # Lớp có xác suất cao nhất ở trên
    plt.tight_layout()
    plt.show()



def is_weight_only_file(file_path):
    return re.match(r'.+\.weights\.h5$', file_path) is not None



def predict(file_path):
    model = tf.keras.models.load_model(r"bestmodel\model1_20250519.h5",
                                       custom_objects={"LeakyReLU": tf.keras.layers.LeakyReLU}
                                       )

    most, _ = predict_predominant_instrument(model, file_path=file_path, segment_length=5, sr=22050)
    return CLASS_NAMES[np.argmax(most)]


def main():
    parser = argparse.ArgumentParser("Sử dụng mô hình đã huấn luyện để dự đoán.")
    parser.add_argument("-p", "--path", type=str, help="Đường dẫn đến file âm thanh cần dự đoán.")
    parser.add_argument("-m", "--model", type=int, help="Chỉ số mô hình muốn sử dụng: 1, 2, 3.")



if __name__ == "__main__":
    # sample_file = "C:\\Users\\tranh\\Downloads\\BaChuaThac-ChauVan-ThanhNgoan-CHAUVAN.wav"
    # predicted_instrument, probabilities = predict_and_compare_instrument(sample_file, model, label_encoder)
    
    # print(f"Predicted instrument: {predicted_instrument}")
    # print("Probabilities:")
    # for instrument, prob in probabilities.items():
    #     print(f"{instrument}: {prob:.4f}")
    
    # # Hiển thị biểu đồ xác suất
    # plot_probabilities(probabilities)
    model = tf.keras.models.load_model(r"bestmodel\model1_20250519.h5",
                                       custom_objects={"LeakyReLU": tf.keras.layers.LeakyReLU}
                                       )

    sample_file = r"test_audio\Gió Đánh Đò Đưa ( Đàn Bầu + Đàn Tranh Phương Nhung ).mp3"
    most, _ = predict_predominant_instrument(model, sample_file, segment_length=5, sr=22050)