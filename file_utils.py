import argparse
import hashlib
import librosa
import os
import shutil
from collections import defaultdict
from PIL import Image
from pydub import AudioSegment



def rename_files(dir_path, prefix, start_index=0, padding=0):
    '''
    Đổi tên tất cả các file trong thư mục thành định dạng <prefix><số thứ tự>.<extension gốc>

    Parameters:
        dir_path (str): Đường dẫn tới thư mục chứa các file cần đổi tên.
        prefix (str): Tiền tố cho tên file mới.
        start_index (int): Số thứ tự bắt đầu (mặc định là 0).
        padding (int): Số chữ số để đệm cho số thứ tự (mặc định là 0).
    '''
    counter = start_index
    for filename in os.listdir(dir_path):
        old_file_path = os.path.join(dir_path, filename)
        if os.path.isfile(old_file_path):
            _, ext = os.path.splitext(filename)  # Tách phần mở rộng
            
            # Thêm số 0 vào trước nếu cần padding
            index_str = str(counter).rjust(padding, '0') if padding > 0 else str(counter)

            new_filename = f'{prefix}{index_str}{ext}'  # Giữ lại extension
            new_file_path = os.path.join(dir_path, new_filename)

            try:
                os.rename(old_file_path, new_file_path)
                print(f'Đã đổi tên {old_file_path} thành {new_file_path}.')

            except FileExistsError:
                print(f'Đã có file trùng tên với {new_file_path}.')

            counter += 1



def get_unique_filename(base_path, padding=0):
    '''
    Trả về tên file khác nếu file đã tồn tại bằng cách thêm số thứ tự vào đằng sau.
    Chẳng hạn như: 'file001.txt', 'file002.txt'.

    Parameters:
        base_path (str): Đường dẫn tới file gốc.
        padding (int): Số chữ số để đệm cho số thứ tự (mặc định là 0).
    
    Returns:
        str: Đường dẫn tới file mới không trùng lặp.
    '''
    if not os.path.exists(base_path):
        return base_path
    
    base, ext = os.path.splitext(base_path)
    counter = 1
    while True:
        index_str = str(counter).rjust(padding, '0') if padding > 0 else str(counter)

        new_path = f'{base}{index_str}{ext}'
        if not os.path.exists(new_path):
            return new_path
        counter += 1



def split_audio_file(input_file, output_dir, segment_length=5000, padding=0):
    '''
    Cắt một file âm thanh thành các đoạn nhỏ có độ dài `segment_length` (mặc định 5s),
    và lưu chúng vào thư mục `output_dir`.
    
    Parameters:
        input_file (str): Đường dẫn đến file âm thanh gốc.
        output_dir (str): Thư mục lưu các file âm thanh đã cắt.
        segment_length (float): Độ dài mỗi đoạn cắt (mili giây).
    '''
    # Đọc file âm thanh
    try:
        audio = AudioSegment.from_file(input_file)
    except Exception as e:
        print(f'Không thể đọc file {input_file}: {e}')
        return
    
    # Kiểm tra nếu độ dài file nhỏ hơn segment_length thì bỏ qua
    if len(audio) < segment_length:
        print(f'File {input_file} quá ngắn (< {segment_length * 1000}s), bỏ qua.')
        return
    
    # Cắt file thành các đoạn nhỏ
    num_segments = len(audio) // segment_length
    for i in range(num_segments):
        start_time = i * segment_length
        end_time = start_time + segment_length
        segment = audio[start_time:end_time]
        
        # Tạo tên file mới cho đoạn cắt
        new_filename = f'{os.path.splitext(os.path.basename(input_file))[0]}_part{i+1}.wav'
        new_file_path = get_unique_filename(os.path.join(output_dir, new_filename), padding=padding)

        # Lưu đoạn âm thanh cắt ra
        segment.export(new_file_path, format='wav')
        print(f'Đã lưu file: {new_file_path}')



def split_audio_directory(input_dir, output_dir, segment_length=5000, padding=0):
    '''
    Cắt file hoặc các file âm thanh trong thư mục `input_dir` 
    thành các đoạn nhỏ có độ dài `segment_length` (mặc định 5s),
    và lưu chúng vào thư mục `output_dir`.
    
    Parameters:
        input_dir (str): Đường dẫn file hoặc thư mục chứa các file âm thanh gốc.
        output_dir (str): Thư mục lưu các file âm thanh đã cắt.
        segment_length (int): Độ dài mỗi đoạn cắt (đơn vị: mili giây, mặc định là 5000ms = 5s).
    '''
    # Tạo thư mục đích nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)

    # Nếu input_dir là file, cắt file đó
    if os.path.isfile(input_dir):
        split_audio_file(
            input_file=input_dir,
            output_dir=output_dir,
            segment_length=segment_length,
            padding=padding
        )

    elif os.path.isdir(input_dir):
        # Lấy danh sách các file trong thư mục nguồn
        for filename in os.listdir(input_dir):
            file_path = os.path.join(input_dir, filename)
            if os.path.isfile(file_path):
                split_audio_file(
                    input_file=file_path,
                    output_dir=output_dir,
                    segment_length=segment_length,
                    padding=padding
                )



def move_files(src_path, dest_dir):
    '''
    Di chuyển file hoặc toàn bộ file trong thư mục từ src_path sang dest_dir.

    Parameters:
        src_path (str): Đường dẫn đến file hoặc thư mục nguồn.
        dest_dir (str): Đường dẫn thư mục đích.
    '''
    if not os.path.exists(src_path):
        print(f'Lỗi: Đường dẫn nguồn {src_path} không tồn tại.')
        return

    os.makedirs(dest_dir, exist_ok=True)

    # Nếu src là một file
    if os.path.isfile(src_path):
        filename = os.path.basename(src_path)
        destination_file = os.path.join(dest_dir, filename)
        try:
            shutil.move(src_path, destination_file)
            print(f'Đã chuyển: {filename}')
        except Exception as e:
            print(f'Lỗi khi chuyển {filename}: {e}')

    # Nếu src là một thư mục
    elif os.path.isdir(src_path):
        for filename in os.listdir(src_path):
            source_file = os.path.join(src_path, filename)
            destination_file = os.path.join(dest_dir, filename)
            if os.path.isfile(source_file):
                try:
                    shutil.move(source_file, destination_file)
                    print(f'Đã chuyển: {filename}')
                except Exception as e:
                    print(f'Lỗi khi chuyển {filename}: {e}')
    else:
        print(f'Lỗi: {src_path} không phải là file hoặc thư mục hợp lệ.')



def count_files(dir_path):
    '''
    Đếm số lượng file trong thư mục.

    Parameters:
        dir_path (str): Đường dẫn tới thư mục.

    Returns:
        int: Số lượng file trong thư mục.
    '''
    count = 0
    for _, _, files in os.walk(dir_path):
        count += len(files)
    return count



def convert_mp3_to_wav(input_file, output_dir=None):
    '''
    Chuyển đổi tệp .mp3 sang .wav

    Parameters:
        input_path (str): Đường dẫn tới file .mp3
        output_path (str): Đường dẫn lưu file .wav (nếu không có sẽ cùng tên với file .mp3)
    '''
    if not output_dir:
        base = os.path.splitext(input_file)[0]
        output_dir = f'{base}.wav'

    try:
        audio = AudioSegment.from_mp3(input_file)
        audio.export(output_dir, format='wav')
        print(f'Đã chuyển: {input_file} → {output_dir}')
    except Exception as e:
        print(f'Lỗi khi chuyển file {input_file}: {e}')



def convert_all_mp3_to_wav(input_path, output_dir=None):
    '''
    Chuyển đổi file hoặc các file .mp3 trong một thư mục thành .wav
    
    Parameters:
        input_path (str): Đường dẫn tới file hoặc thư mục chứa các file .mp3
    '''
    if os.path.isfile(input_path):
        convert_mp3_to_wav(input_path, output_dir)
    
    elif os.path.isdir(input_path):
        for filename in os.listdir(input_path):
            if filename.endswith('.mp3'):
                input_path = os.path.join(input_path, filename)
                convert_mp3_to_wav(input_path)
    else:
        raise Exception(f"Đường dẫn {input_path} không hợp lệ.")



def split_audio_by_timestamps(input_file, output_dir, cut_points):
    '''
    Cắt file âm thanh tại các thời điểm chỉ định.

    Parameters:
        input_file (str): Đường dẫn tới file âm thanh gốc.
        output_dir (str): Thư mục để lưu các đoạn âm thanh đã cắt.
        cut_points (List[int | float]): Danh sách thời điểm (tính bằng giây) để cắt âm thanh.
    '''
    # Đảm bảo thư mục đích tồn tại
    os.makedirs(output_dir, exist_ok=True)

    # Đọc file âm thanh
    audio = AudioSegment.from_file(input_file)

    # Chuyển thời điểm từ giây sang mili giây và sắp xếp
    cut_points_ms = sorted([int(point * 1000) for point in cut_points])

    # Thêm điểm đầu và cuối
    all_points = [0] + cut_points_ms + [len(audio)]

    base_name = os.path.splitext(os.path.basename(input_file))[0]

    for i in range(len(all_points) - 1):
        start = all_points[i]
        end = all_points[i + 1]
        segment = audio[start:end]
        output_path = os.path.join(output_dir, f'{base_name}_part{i+1}.wav')
        segment.export(output_path, format='wav')
        print(f'Đã lưu đoạn {i+1}: {output_path}')



def get_audio_hash(file_path, sr=22050):
    '''
    Tính hàm băm (hash) của nội dung âm thanh.
    
    Parameters:
        file_path (str): Đường dẫn đến file âm thanh.
        sr (int): Tần số lấy mẫu để chuẩn hóa.
    
    Returns:
        str: Giá trị hash của tín hiệu âm thanh.
    '''
    try:
        # Tải tín hiệu âm thanh
        y, _ = librosa.load(file_path, sr=sr, mono=True)
        
        # Chuẩn hóa tín hiệu: cắt bỏ khoảng lặng đầu/cuối
        y, _ = librosa.effects.trim(y)
        
        # Chuyển thành byte để băm
        y_bytes = y.tobytes()
        
        # Tính hash bằng SHA-256
        return hashlib.sha256(y_bytes).hexdigest()
    except Exception as e:
        print(f'[Audio][Hashing] Lỗi xử lý {file_path}: {e}')
        return None



def get_image_hash(file_path):
    try:
        with Image.open(file_path) as img:
            img = img.convert('L').resize((64, 64))  # chuyển grayscale và resize
            data = img.tobytes()
            return hashlib.md5(data).hexdigest()
    except Exception as e:
        print(f'[Image][Hashing] Lỗi xử lý {file_path}: {e}')
        return None
    


def find_duplicate_media_files(input_dir, audio_sr=22050, audio_exts=('.wav', '.mp3'), image_exts=('.png', '.jpg', '.jpeg')):
    '''
    Tìm các file âm thanh, hình ảnh trùng nhau trong thư mục.
    
    Parameters:
        input_dir (str): Đường dẫn đến thư mục chứa file âm thanh.
        audio_sr (int): Tần số lấy mẫu âm thanh.
        audio_exts (tuple): Phần mở rộng âm thanh hỗ trợ.
        image_exts (tuple): Phần mở rộng hình ảnh hỗ trợ.

    Returns:
        dict: Dictionary chứa các nhóm file trùng nhau.
    '''
    # Dictionary để lưu hash và danh sách file tương ứng
    hash_to_files_audio = defaultdict(list)
    hash_to_files_image = defaultdict(list)
    
    # Duyệt qua tất cả file trong thư mục
    for root, _, files in os.walk(input_dir):
        for f in files:
            file_path = os.path.join(root, f)
            ext = f.lower()

            if ext.endswith(audio_exts):
                h = get_audio_hash(file_path, sr=audio_sr)
                if h:
                    hash_to_files_audio[h].append(file_path)

            elif ext.endswith(image_exts):
                h = get_image_hash(file_path)
                if h:
                    hash_to_files_image[h].append(file_path)
    
    # Lọc các hash có nhiều hơn 1 file (tức là trùng nhau)
    duplicates_audio = {h: files for h, files in hash_to_files_audio.items() if len(files) > 1}
    duplicates_image = {h: files for h, files in hash_to_files_image.items() if len(files) > 1}
    
    return {
        'audio': duplicates_audio, 
        'image': duplicates_image
    }



def remove_duplicates(input_dir):
    '''
    Xóa các file âm thanh trùng nhau trong thư mục.
    '''
    duplicates = find_duplicate_media_files(
            input_dir=input_dir,
            audio_sr=22050,
        )
    
    for file_paths in duplicates.values():
        for f in file_paths[1:]:
            os.remove(f)
            print(f'Đã xóa: {f}')



def print_duplicates(duplicates):
    '''
    In danh sách các file trùng nhau.
    '''
    audio_duplicates = duplicates.get('audio', {})
    image_duplicates = duplicates.get('image', {})

    if not audio_duplicates and not image_duplicates:
        print('Không tìm thấy file trùng.')
        return

    if audio_duplicates:
        print('Các file âm thanh trùng nhau:')
        for hash_value, files in audio_duplicates.items():
            print(f'\nHash: {hash_value}')
            for f in files:
                print(f'  - {f}')
    else:
        print('Không có file âm thanh trùng nhau.')

    print('\n' + '='*50 + '\n')

    if image_duplicates:
        print('Các file hình ảnh trùng nhau:')
        for hash_value, files in image_duplicates.items():
            print(f'\nHash: {hash_value}')
            for f in files:
                print(f'  - {f}')
    else:
        print('Không có file hình ảnh trùng nhau.')



def main():
    parser = argparse.ArgumentParser(description='Tiện ích làm việc với các file, thư mục sử dụng cho chuẩn bị dataset.')
    parser.add_argument('-r', '--rename', type=str, nargs=2, help='[prefix, start_index]: Đổi tên các file wav đã cắt thành dãy tên có thứ tự. Tham số là số thứ tự bắt đầu.')
    parser.add_argument('-s', '--split', type=int, help='Độ đài mỗi đoạn cắt (giây). Cắt các file wav thành các đoạn nhỏ hơn.')
    parser.add_argument('-m', '--move', action='store_true', help='Di chuyển các file đã cắt sang thư mục khác.')
    parser.add_argument('-c', '--count', action='store_true', help='Đếm số lượng file trong thư mục.')
    parser.add_argument('-cv', '--cover', action='store_true', help='Chuyển các file mp3 sang wav.')
    parser.add_argument('-p', '--points', type=str, nargs='+', help='Các điểm cắt (định dạng "mm:ss") để cắt file âm thanh.')
    parser.add_argument('-dup', '--duplicate', type=str, help='Tìm các file trùng lập trong thư mục.')
    parser.add_argument('-rm', '--remove', type=str, help='Xoá các file trùng lập trong thư mục. Lưu ý cần sử dụng -dup trước khi quyết định.')

    parser.add_argument('-i', '--input', type=str, help='Đường dẫn đầu vào.')
    parser.add_argument('-o', '--output', type=str, help='Đường dẫn đầu ra.')
    parser.add_argument('-pd', '--padding', type=int, default=0, help='Số lượng đệm số 0 khi đặt tên hoặc đổi tên các file.')

    args = parser.parse_args()

    # Đổi tên hàng loạt các file trong thư mục
    if args.rename:
        if not os.path.isdir(args.input):
            print(f'Đường dẫn {args.input} không phải là thư mục hợp lệ.')
            return
        rename_files(args.input, args.rename[0], int(args.rename[1]), args.padding)

    # Cắt một file hoặc các file âm thanh trong thư mục thành các đoạn
    elif args.split:
        if not os.path.isdir(args.input) and os.path.isdir(args.input) or not os.path.isdir(args.output):
            print(f'Đường dẫn đầu vào {args.input} hoặc đường dẫn đầu ra {args.output} không hợp lệ.')
            return
        split_audio_directory(
            input_dir=args.input,
            output_dir=args.output,
            segment_length=args.split,
            padding=args.padding
        )

    # Di chuyển các file trong thư mục sang vị trí khác
    elif args.move:
        if not os.path.isdir(args.input) or not os.path.isdir(args.output):
            print(f'Đường dẫn đầu vào {args.input} hoặc đường dẫn đầu ra {args.output} không hợp lệ.')
            return
        move_files(args.input, args.output)

    # Đếm số lượng các file trong thư mục
    elif args.count:
        if not os.path.isdir(args.input):
            print(f'Đường dẫn {args.input} không phải là thư mục hợp lệ.')
            return
        dir_name = os.path.basename(args.input)
        print(f'Số lượng file trong thư mục {dir_name}: {count_files(args.input)}')

    # Chuyển định dạng file hoặc các file mp3 trong thư mục sang wav
    elif args.cover:
        if not os.path.isdir(args.input) and not os.path.isfile(args.input):
            print(f'Đường dẫn {args.input} không hợp lệ.')
            return
        convert_all_mp3_to_wav(args.input, args.output)

    # Cắt file âm thanh tại các điểm được cung cấp
    elif args.points:
        if not os.path.isfile(args.input) or not os.path.isdir(args.output):
            print(f'Đường dẫn file nguồn {args.input} hoặc thư mục đích {args.output} không hợp lệ.')
            return
        
        os.makedirs(args.output, exist_ok=True)
        
        time_seconds = [int(m) * 60 + int(s) for m, s in (t.split(':') for t in args.points)]

        cut_points = [float(x) for x in time_seconds]
        split_audio_by_timestamps(
            input_file=args.input,
            output_dir=args.output,
            cut_points=cut_points
        )

    # In ra các file trùng lập trong một thư mục
    elif args.duplicate:
        if not os.path.isdir(args.duplicate):
            print(f'Đường dẫn {args.duplicate} không phải là thư mục hợp lệ.')
            return
        print_duplicates(
            find_duplicate_media_files(input_dir=args.duplicate)
        )
    
    # Xoá các file trùng lập chỉ giữ lại bản chính trong thư mục
    elif args.remove:
        if not os.path.isdir(args.remove):
            print(f'Đường dẫn {args.remove} không phải là thư mục hợp lệ.')
            return
        remove_duplicates(args.remove)

    else:
        parser.print_help()



if __name__ == '__main__':
    main()    