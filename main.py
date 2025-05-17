import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from PIL import Image
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift
import matplotlib.pyplot as plt
from mel_spec import audio_to_mel_spectrogram
from build.model1 import get_model1
from config import *
from mel_spec import *
from collections import Counter
from pydub import AudioSegment



# Hàm dự đoán và so sánh nhạc cụ
def predict_instrument(model, input_file):
    mel_spec = audio_to_mel_spectrogram(
        input_file, 
        n_mels=N_MELS, 
        hop_length=HOP_LENGTH, 
        n_fft=N_FFT, 
        sr=SR, 
        duration=None, 
        input_shape=INPUT_SHAPE)
    
    print(mel_spec.shape)
    mel_spec = np.expand_dims(mel_spec, axis=0)
    return model.predict(mel_spec)



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


def predict(file_path):
    model = tf.keras.models.load_model(r"bestmodel\model1.h5")
    predictions = predict_instrument(model, file_path, ["dantranh", "danbau", "dannhi", "sao"], is_audio=True)

    predicted_index = np.argmax(predictions)  # Lấy chỉ mục của xác suất cao nhất

    # Danh sách nhãn của bạn
    labels = ["Đàn bầu", "Đàn nhị", "Đàn tranh", "Sáo"]

    return labels[predicted_index]



def predict_main_instrument(model, file_path, segment_length_ms=5000):
    """
    Dự đoán nhạc cụ chính trong một file âm thanh.

    Parameters:
        file_path (str): Đường dẫn file âm thanh.
        model: Mô hình phân loại nhạc cụ (CNN).
        predict_function: Hàm xử lý dự đoán, nhận input là đoạn audio, trả về tên nhạc cụ.
        segment_length_ms (int): Độ dài mỗi đoạn cắt (ms).

    Returns:
        Tuple(str, dict): Tên nhạc cụ phổ biến nhất và phân phối dự đoán.
    """
    audio = AudioSegment.from_file(file_path)
    duration = len(audio)
    predictions = []

    for start in range(0, duration, segment_length_ms):
        end = min(start + segment_length_ms, duration)
        segment = audio[start:end]

        # Chuyển đoạn audio sang định dạng đầu vào cho model
        mel_spec = audio_to_mel_spectrogram(
            segment, 
            n_mels=N_MELS, 
            hop_length=HOP_LENGTH, 
            n_fft=N_FFT, 
            sr=SR, 
            duration=None, 
            input_shape=INPUT_SHAPE)
    
        mel_spec = np.expand_dims(mel_spec, axis=0)
        
        predicted_index = int(np.argmax(model.predict(mel_spec)[0]))
        predicted_label = ["danbau", "dannhi", "dantranh", "sao"][predicted_index]

        predictions.append(predicted_label)

    # Đếm tần suất từng nhạc cụ
    counter = Counter(predictions)
    most_common = counter.most_common(1)[0][0] if counter else None

    return most_common, dict(counter)



if __name__ == "__main__":
    # sample_file = "C:\\Users\\tranh\\Downloads\\BaChuaThac-ChauVan-ThanhNgoan-CHAUVAN.wav"
    # predicted_instrument, probabilities = predict_and_compare_instrument(sample_file, model, label_encoder)
    
    # print(f"Predicted instrument: {predicted_instrument}")
    # print("Probabilities:")
    # for instrument, prob in probabilities.items():
    #     print(f"{instrument}: {prob:.4f}")
    
    # # Hiển thị biểu đồ xác suất
    # plot_probabilities(probabilities)
    model = tf.keras.models.load_model(r"bestmodel\model1.h5")

    sample_file = r"C:\Users\tranh\Downloads\Về quê.mp3"
    # predicted_instrument = predict_main_instrument(model, sample_file)

    print(f"Dự đoán: {predict_instrument(model, sample_file)}")
    # print_prediction_results(predicted_instrument, CLASS_NAMES, top_k=4)
    # plot_prediction_probabilities(predicted_instrument, CLASS_NAMES) 
    # print(f"Dự đoán nhạc cụ: {predicted_instrument}")
    # print(f"Tỉ lệ: {probabilities}")
    # for instrument, prob in probabilities.items():
    #     print(f"{instrument}: {prob:.4f}")