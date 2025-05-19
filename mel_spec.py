import argparse
import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from config import *
from file_utils import get_unique_filename
from pathlib import Path
from pydub import AudioSegment



def audio_to_mel_spectrogram(
    audio_input, 
    duration=None,
    start=0.0,
    hop_length=512, 
    n_fft=2048, 
    n_mels=128, 
    sr=22050, 
    input_shape=(128, 128, 3)):
    '''
    Chuyển file âm thanh hoặc AudioSegment thành Mel-spectrogram với kích thước cố định.
    
    Parameters:
        audio_input (str | ndarray | AudioSegment): Đường dẫn đến file âm thanh, mảng numpy hoặc đối tượng AudioSegment.
        duration (float): Độ dài âm thanh (giây). Nếu None, tự động tính.
        start (float): Vị trí bắt đầu lấy âm thanh (giây).
        hop_length (int): Khoảng cách giữa các khung.
        n_fft (int): Kích thước FFT.
        n_mels (int): Số lượng Mel bands.
        sr (int): Tần số lấy mẫu.
        input_shape (tuple): Kích thước đầu vào mong muốn cho mô hình.
    '''
    # Tải âm thanh
    if isinstance(audio_input, str):
        y, sr = librosa.load(audio_input, sr=sr)

    elif isinstance(audio_input, np.ndarray):
        y = audio_input
        # Đảm bảo y là mảng 1D (mono)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        # Chuẩn hóa
        y = y / np.max(np.abs(y) + 1e-10)

    elif isinstance(audio_input, AudioSegment):
        audio_input = audio_input.set_frame_rate(sr)
        y = np.array(audio_input.get_array_of_samples(), dtype=np.float32)

        if audio_input.channels > 1:
            y = y.reshape((-1, audio_input.channels)).mean(axis=1)

        y = y / np.max(np.abs(y) + 1e-10)
        sr = audio_input.frame_rate

    else:
        raise ValueError('audio_input phải là đường dẫn file str, ndarray hoặc AudioSegment.')

    # Cắt âm thanh nếu cung cấp duration
    if duration is not None:
        start_sample = int(start * sr)
        end_sample = int(start_sample + duration * sr)

        if end_sample > len(y):
            pad_length = end_sample -len(y)
            y = np.pad(y[start_sample:], (0, pad_length), mode='constant')
        else:
            y = y[start_sample:end_sample]   
    else:
        duration = len(y) / sr

    # Tính fixed_length dựa trên độ dài âm thanh
    fixed_length = int(np.floor((sr * duration) / hop_length)) + 1

    # Chuyển sang Mel-spectrogram, log decibel
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft)    
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Cắt hoặc đệm để đạt fixed_length
    if mel_spec_db.shape[1] > fixed_length:
        mel_spec_db = mel_spec_db[:, :fixed_length]
    else:
        mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, fixed_length - mel_spec_db.shape[1])), mode='constant')

    # Thêm chiều kênh
    mel_spec_db = mel_spec_db[..., np.newaxis]
    
    # Resize về input_shape
    mel_spec_db = tf.image.resize(mel_spec_db, input_shape[:2], method='bilinear', antialias=True).numpy()
    
    # Chuyển về 3 kênh nếu cần
    if input_shape[-1] == 3:
        mel_spec_db = np.repeat(mel_spec_db, 3, axis=-1)
    
    # Chuẩn hóa về [0, 1]
    mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-10)
    
    return mel_spec_db



def display_mel_spectrogram(mel_spec_db, duration=5, n_mels=128):
    '''Hiển thị Mel-spectrogram dưới dạng hình ảnh.
    
    Parameters:
        mel_spec_db (ndarray): Mel-spectrogram dạng log.
        duration (float): Độ dài âm thanh (giây).
        n_mels (int): Số lượng Mel bands.
    '''
    # Chuẩn hoá chiều của mel_spec_db
    if len(mel_spec_db.shape) == 3:
        if mel_spec_db.shape[-1] == 1:
            mel_spec_db = mel_spec_db[:, :, 0]  # Loại bỏ kênh đơn thành 2D
        elif mel_spec_db.shape[-1] == 3:
            mel_spec_db = mel_spec_db[:, :, 0]  # Lấy kênh đầu tiên để hiển thị
        else:
            raise ValueError(f'Mong đợi 1 hoặc 3 kênh, nhận được kích thước {mel_spec_db.shape}')
    elif len(mel_spec_db.shape) != 2:
        raise ValueError(f'Mong đợi mảng 2D hoặc 3D với 1/3 kênh, nhận được kích thước {mel_spec_db.shape}')
    
    plt.figure(figsize=(8, 4))
    plt.imshow(mel_spec_db, aspect='auto', origin='upper',
               extent=[0, duration, 0, n_mels], cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Mel Bands')
    plt.tight_layout()
    plt.show()
    plt.close()



def save_mel_spectrograms(
        input_path, 
        output_dir, 
        hop_length=512, 
        n_fft=2048, 
        n_mels=128, 
        sr=22050, 
        input_shape=(128, 128, 3)):
    '''
    Chuyển file hoặc các file âm thanh trong input_dir thành Mel-spectrogram 
    và lưu dưới dạng .png trong output_dir mà không ghi đè file cũ.
    
    Parameters:
        input_path (str): Thư mục chứa file âm thanh hoặc đường dẫn file âm thanh gốc.
        output_dir (str): Thư mục lưu các file Mel-spectrogram đã chuyển đổi.
        hop_length (int): Khoảng cách giữa các khung.
        n_fft (int): Kích thước FFT.
        n_mels (int): Số lượng Mel bands.
        sr (int): Tần số lấy mẫu khi số hoá tín hiệu analog (sampling rate).
        input_shape (tuple): Kích thước đầu vào của mô hình CNN (height, width, channels).
    '''
    os.makedirs(output_dir, exist_ok=True)
    
    # Lấy danh sách các lớp (subfolders), nếu không có thì coi input_dir là một lớp duy nhất
    classes = [d for d in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, d))]
    if not classes:
        classes = ['']
    
    # Nếu input là một file
    if os.path.isfile(input_path):
        if input_path.endswith(('.wav', '.mp3')):
            base_name = Path(input_path).stem
            output_path = get_unique_filename(os.path.join(output_dir, f'{base_name}.png'))

            try:
                mel_spec = audio_to_mel_spectrogram(
                    input_path,
                    n_mels=n_mels,
                    hop_length=hop_length,
                    n_fft=n_fft,
                    sr=sr,
                    input_shape=input_shape
                )
                plt.imsave(output_path, mel_spec[:, :, 0], cmap='magma')
                print(f'Đã lưu: {output_path}')

            except Exception as e:
                print(f'Lỗi khi lưu {input_path}: {e}')
        else:
            print(f'Tệp {input_path} không phải định dạng .wav hoặc .mp3.')

    # Nếu input là một thư mục
    elif os.path.isdir(input_path):
        for class_name in classes:
            class_dir = os.path.join(input_path, class_name) if class_name else input_path
            output_class_dir = os.path.join(output_dir, class_name) if class_name else output_dir
            os.makedirs(output_class_dir, exist_ok=True)

            for f in os.listdir(class_dir):
                if f.endswith(('.wav', '.mp3')):
                    file_path = os.path.join(class_dir, f)
                    base_name = Path(f).stem
                    output_path = get_unique_filename(os.path.join(output_class_dir, f'{base_name}.png'))

                    try:
                        mel_spec = audio_to_mel_spectrogram(
                            file_path,
                            n_mels=n_mels,
                            hop_length=hop_length,
                            n_fft=n_fft,
                            sr=sr,
                            input_shape=input_shape
                        )

                        plt.imsave(output_path, mel_spec[:, :, 0], cmap='magma')
                        print(f'Đã lưu: {output_path}')

                    except Exception as e:
                        print(f'Lỗi khi lưu {file_path}: {e}')



def main():
    parser = argparse.ArgumentParser(description='Chuyển đổi âm thanh thành Mel-spectrogram và lưu dưới dạng hình ảnh.')
    parser.add_argument('-s', '--save', action='store_true', help='Lưu file, hoặc các file .')
    parser.add_argument('-d', '--display', action='store_true', help='Hiển thị Mel-spectrogram từ file âm thanh.')
    parser.add_argument('-i', '--input', type=str, help='Thư mục chứa các file âm thanh hoặc file âm thanh gốc.')
    parser.add_argument('-o', '--output', type=str, help='Thư mục lưu các file Mel-spectrogram đã chuyển đổi.')
    parser.add_argument('-st', '--start', type=float, default=0.0, help='Thời điểm (giây) bắt đầu cắt nguồn âm thanh.')
    parser.add_argument('-du', '--duration', type=float, default=5.0, help='Thời lượng (giây) của nguồn âm thanh khi chuyển sang Mel-spectrogram, sử dụng để hiển thị.')

    args = parser.parse_args()
    
    # Lưu file hoặc các file âm thanh trong thư mục thành Mel-spectrogram
    if args.save:
        if not args.input or not args.output:
            print('Vui lòng cung cấp đường dẫn file hoặc thư mục đầu vào và thư mục đầu ra.')
            return
        save_mel_spectrograms(args.input, args.output)

    # Chuyển một file âm thanh thành Mel-spectrogram rồi hiển thị lên
    elif args.display:
        if not args.input:
            print('Vui lòng cung cấp đường dẫn file âm thanh.')
            return
        mel_spec = audio_to_mel_spectrogram(args.input, duration=args.duration, start=args.start)
        display_mel_spectrogram(mel_spec, duration=args.duration)
    
    else:
        print('Vui lòng chọn một trong các tùy chọn: -s (lưu) hoặc -d (hiển thị).')
        parser.print_help()



if __name__ == '__main__':
    main()