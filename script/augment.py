import argparse
import json
import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
import soundfile as sf
import sys
from pydub import AudioSegment
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *
from file_utils import get_unique_filename
from mel_spec import audio_to_mel_spectrogram, display_mel_spectrogram



def augment_audio(audio, sr=22050, methods=None):
    '''Tăng cường dữ liệu âm thanh bằng cách áp dụng tất cả phương pháp liên tiếp.

    Parameters:
        audio (str | np.ndarray | AudioSegment): Đường dẫn file, ndarray (librosa), hoặc AudioSegment.
        sr (int): Sampling rate, sử dụng cho phương thức pitch_shift.
        methods (Dict): Các phương thức làm lệch âm thanh gốc. Xem thêm tại config.py `AUDIO_AUGMENTATION`.
    '''
    if isinstance(audio, str):
        if not os.path.isfile(audio):
            raise FileNotFoundError(f'Không tìm thấy file: {audio}')
        y, _ = librosa.load(audio, sr=sr)

    elif isinstance(audio, AudioSegment):
        # Stereo -> mono
        if audio.channels > 1:
            audio = audio.set_channels(1)
        y = np.array(audio.get_array_of_samples()).astype(np.float32)
        y = y / np.iinfo(audio.array_type).max # Chuẩn hoá [-1, 1]

    elif isinstance(audio, np.ndarray):
        y = audio

    else:
        raise TypeError('Đầu vào phải là đường dẫn file, ndarray, hoặc AudioSegment.')


    if not isinstance(y, np.ndarray) or len(y) < 2:
        raise ValueError('Dữ liệu audio không hợp lệ hoặc quá ngắn.')

    if methods is None:
        methods = AUDIO_AUGMENTATION

    augmented = y.copy()

    for method, params in methods.items():
        if method == 'pitch_shift':
            n_steps = np.random.uniform(*params)
            augmented = librosa.effects.pitch_shift(augmented, sr=sr, n_steps=n_steps)

        elif method == 'time_stretch':
            rate = np.random.uniform(*params)
            try:
                augmented = librosa.effects.time_stretch(augmented, rate=rate)
            except Exception:
                continue

        elif method == 'add_noise':
            mean, std = params
            noise = np.random.normal(mean, std, len(augmented))
            augmented = augmented + noise
        
        else:
            raise Exception(f"Phương thức không hợp lệ: {method}")

    return augmented



def save_audio(y, output_file, sr=22050):
    '''
    Lưu file âm thanh đã tăng cường dưới dạng .wav.

    Parameters:
        y (np.ndarray): Dữ liệu âm thanh (được chuẩn hóa [-1, 1] trước, dtype=float32).
        sr (int): Sampling rate.
        output_path (str): Đường dẫn lưu file đầu ra.
    '''
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Kiểm tra và chuẩn hóa kiểu dữ liệu
    if not isinstance(y, np.ndarray):
        raise TypeError('Dữ liệu đầu vào phải là numpy.ndarray')
    
    if y.dtype != np.float32:
        y = y.astype(np.float32)

    # Ghi file
    sf.write(output_file, y, samplerate=sr)



def save_augment_audio(
        audio,
        ouput_file,
        sr=22050,
        methods=None):
    augmented = augment_audio(audio, sr, methods)
    save_audio(augmented, ouput_file, sr)



def save_augment_audios_to_mel_spectrogram(
        input_path, 
        output_dir, 
        hop_length=512, 
        n_fft=2048, 
        n_mels=128, 
        sr=22050, 
        input_shape=(128, 128, 3),
        methods=None):
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
        methods (Dict): Các phương thức làm lệch âm thanh gốc. Xem thêm tại config.py `AUDIO_AUGMENTATION`.
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
                augmented = augment_audio(input_path, sr, methods)
                mel_spec = audio_to_mel_spectrogram(
                    augmented,
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
                        augmented = augment_audio(file_path, sr, methods)
                        mel_spec = audio_to_mel_spectrogram(
                            augmented,
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

    

def generate_augmented_mel_dataset(
    input_dir,
    output_dir,
    class_aug_methods,
    sr=22050,
    input_shape=(128, 128, 3),
    hop_length=512,
    n_fft=2048,
    n_mels=128):
    """
    Sinh dữ liệu Mel-spectrogram từ âm thanh với augmentation theo từng lớp cụ thể.
    Nếu như file mới được đặt tên trùng với file đã tồn tại thì nó sẽ ghi đè file đó.

    Parameters:
        input_dir (str): Thư mục chứa thư mục con theo tên lớp (danbau, dantranh,...).
        output_dir (str): Thư mục lưu Mel-spectrogram đã tăng cường.
        class_aug_methods (dict): Dict ánh xạ tên lớp -> dict các augmentation tương ứng.
        sr (int): Sampling rate.
        input_shape (tuple): Kích thước đầu vào.
    """
    os.makedirs(output_dir, exist_ok=True)

    for class_name in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        aug_methods = class_aug_methods.get(class_name, None)
        output_class_dir = os.path.join(output_dir, class_name)
        os.makedirs(output_class_dir, exist_ok=True)

        for filename in os.listdir(class_path):
            if not filename.lower().endswith(('.wav', '.mp3')):
                continue

            audio_path = os.path.join(class_path, filename)
            base_name = Path(filename).stem
            output_path = os.path.join(output_class_dir, f'{base_name}.png')

            try:
                # Tăng cường âm thanh
                augmented = augment_audio(audio_path, sr=sr, methods=aug_methods)

                # Chuyển sang Mel
                mel_spec = audio_to_mel_spectrogram(
                    augmented,
                    n_mels=n_mels,
                    hop_length=hop_length,
                    n_fft=n_fft,
                    sr=sr,
                    input_shape=input_shape
                )

                # Lưu ảnh
                plt.imsave(output_path, mel_spec[:, :, 0], cmap='magma')
                print(f'Đã lưu: {output_path}')

            except Exception as e:
                print(f'Lỗi với file {audio_path}: {e}')



def main():
    parser = argparse.ArgumentParser('Xử lý các file âm thanh được làm nhiễu.')
    parser.add_argument('-au', '--audio', action='store_true', help='Lưu augmented audio thành file âm thanh.')
    parser.add_argument('-im', '--image', action='store_true', help='Lưu các augmented audio thành các Mel-spectrogram.')
    parser.add_argument('-d', '--display', action='store_true', help='Hiển thị augmented audio với Mel-spectrogram.')
    parser.add_argument('-g', '--generate', action='store_true', help='Khởi tạo tập dữ liệu với các mẫu được tăng cường.')

    parser.add_argument('-i', '--input', type=str, help='Đường dẫn file gốc.')
    parser.add_argument('-o', '--output', type=str, help='Đường dẫn file cần lưu.')
    parser.add_argument('-s', '--sr', type=int, default=22050, help='Đường dẫn file gốc.')
    parser.add_argument('-du', '--duration', type=float, default=5.0, help='Thời lượng (giây) của nguồn âm thanh khi chuyển sang Mel-spectrogram, sử dụng để hiển thị.')
    parser.add_argument('-st', '--start', type=float, default=0.0, help='Thời điểm (giây) bắt đầu cắt nguồn âm thanh.')
    parser.add_argument('-m', '--methods', type=json.loads, help='Dictionary các phương pháp tăng cường, định dạng JSON. Ví dụ: \'{"pitch_shift": [-2, 2]}\'')

    args = parser.parse_args()
    
    # Lưu một file augmented audio thành file âm thanh
    if args.audio:
        if not os.path.isfile(args.input) or not os.path.isdir(os.path.dirname(args.output)):
            print(f'Đường dẫn file gốc {args.input} hoặc tên file dự định lưu {args.output} không hợp lệ.')
            return
        
        if not args.methods:
            args.methods = AUDIO_AUGMENTATION
        save_augment_audio(args.input, args.output, args.sr, args.methods)
    
    # Lưu các file augmented audio thành các Mel-spectrogram
    elif args.image:
        if not os.path.isfile(args.input) and not os.path.isdir(args.input) or not os.path.isdir(os.path.dirname(args.output)):
            print(f'Đường dẫn file hoặc thư mục gốc {args.input} hoặc thư mục đích {args.output} không hợp lệ.')
            return
        save_augment_audios_to_mel_spectrogram(args.input, args.output, methods=args.methods)
    
    # Hiển thị Mel-spectrogram
    elif args.display:
        if not os.path.isfile(args.input):
            print(f'Đường dẫn file {args.input} không hợp lệ.')
            return
        augmented = augment_audio(args.input, args.sr, args.methods)
        mel_spec = audio_to_mel_spectrogram(augmented, duration=args.duration, start=args.start)
        display_mel_spectrogram(mel_spec, duration=args.duration)

    # Khởi tạo tập tăng cường dữ liệu
    elif args.generate:
        if not os.path.isdir(args.input) or not os.path.isdir(args.output):
            print(f'Đường dẫn thư mục nguồn {args.input} hoặc đường dẫn thư mục đích {args.output} không hợp lệ.')
            return
        
        if not args.methods:
            args.methods = AUDIO_CLASS_AUGMENTATION
        generate_augmented_mel_dataset(args.input, args.output, args.methods)

    else:
        parser.print_help()



if __name__ == "__main__":
    main()