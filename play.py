import argparse
import keyboard
import os
import sys
import time
import vlc
from pathlib import Path
from send2trash import send2trash

parser = argparse.ArgumentParser(description="Phát các file âm thanh .wav trong một thư mục. Công cụ kiểm tra làm sạch dataset.")
parser.add_argument("-i", "--input", type=str, required=True, help="Đường dẫn thư mục chứa các file âm thanh.")
args = parser.parse_args()

if not os.path.isdir(args.input):
    raise Exception("Đường dẫn thư mục không hợp lệ.")

# === Cấu hình ===
AUDIO_EXT = [".wav"]
SKIP_TIME = 1        # số giây tua mỗi lần
SPEED_STEP = 0.25

# === Biến trạng thái ===
current_index = 0
speed = 1.0             # Tốc độ phát
playing = False
folder_path = args.input
deleted = 0

# === Tải danh sách file âm thanh ===
audio_files = sorted([f for f in os.listdir(folder_path) if Path(f).suffix in AUDIO_EXT])

# VLC player global
player = None

def load_audio(index):
    global player, speed
    if player:
        player.stop()
    path = os.path.join(folder_path, audio_files[index])
    media = vlc.Media(path)
    player = vlc.MediaPlayer()
    player.set_media(media)
    player.play()
    time.sleep(0.1)  # Đợi player khởi tạo
    player.set_rate(speed)

    time.sleep(0.1)  # Thêm thời gian chờ để đảm bảo VLC cập nhật metadata
    duration = player.get_length() / 1000  # đơn vị: giây
    duration_str = f"{duration:.2f}s" if duration > 0 else "unknown"
    print(f"\n>> Playing: {audio_files[index]} (Speed: {speed:.2f}x, Duration: {duration_str})")

def play_audio(start_sec):
    global playing
    if player is None:
        return
    player.set_time(int(start_sec * 1000))  # VLC dùng milisecond
    player.play()
    playing = True

def stop_audio():
    global playing
    if player:
        player.pause()
    playing = False

def play_new():
    load_audio(current_index)
    play_audio(0)

def skip(seconds):
    if player is None:
        return

    duration = player.get_length() / 1000
    if duration <= 0:
        return

    state = player.get_state()

    if state == vlc.State.Ended:
        new_pos = 0 if seconds > 0 else max(0, duration - 1)
        print(f" - {audio_files[current_index]} Current: {new_pos:.2f}s")

        player.stop()
        player.set_time(int(new_pos * 1000))
        player.play()
        time.sleep(0.1)
        return

    current_time = player.get_time() / 1000
    new_pos = current_time + seconds

    # Giới hạn biên
    if new_pos < 0:
        new_pos = 0
    elif new_pos > duration - 0.01:
        new_pos = duration - 0.01

    player.set_time(int(new_pos * 1000))
    print(f" - {audio_files[current_index]} Current: {new_pos:.2f}s")

def delete_current_file():
    global current_index, player, deleted
    if player:
        player.stop()
        del player
        player = None
        time.sleep(0.1)  # Đợi hệ thống giải phóng file    
    
    path = os.path.join(folder_path, audio_files[current_index])
    send2trash(path)
    print(f"[Del] Đã chuyển vào thùng rác: {audio_files[current_index]}")
    deleted += 1
    audio_files.pop(current_index)
    # Xử lý trường hợp hết file
    if not audio_files:
        print("Không còn file nào để phát.")
        sys.exit(0)
    if current_index >= len(audio_files):
        current_index = len(audio_files) - 1
    play_new()

def change_file(next=True):
    global current_index
    stop_audio()
    current_index = (current_index + (1 if next else -1)) % len(audio_files)
    play_new()

def jump_to_file():
    global current_index
    stop_audio()
    print("\nNhập tên file (không cần phần mở rộng): ", end="", flush=True)
    filename = sys.stdin.readline().strip()
    if not filename:
        print("Tên file không hợp lệ.")
        return
    matched_indices = [i for i, f in enumerate(audio_files) if Path(f).stem == filename]
    if matched_indices:
        current_index = matched_indices[0]
        play_new()
        print(f"Chuyển đến file: {audio_files[current_index]}")
    else:
        print(f"Không tìm thấy file: {filename}")

# === Bắt đầu ===
load_audio(current_index)

print(f"""
      \n[🎧] Controls:
      \n←/→: tua {SKIP_TIME} giây
      \nShift+←/→: bài trước/sau
      \nTab: nhảy tới file cụ thể
      \n+/-: tăng giảm tốc độ
      \nSpace: play/pause
      \nDelete: xoá file hiện tại
      \nEsc: thoát chương trình
      """)

while True:
    try:
        if keyboard.is_pressed('esc'):
            if player:
                player.stop()
            print(f"[Del] Bạn đã xoá {deleted} files")
            break

        elif keyboard.is_pressed('space'):
            if playing:
                stop_audio()
            else:
                if player:
                    player.play()
                    playing = True
            time.sleep(0.3)

        elif keyboard.is_pressed('right') and not keyboard.is_pressed('shift'):
            skip(SKIP_TIME)
            time.sleep(0.2)

        elif keyboard.is_pressed('left') and not keyboard.is_pressed('shift'):
            skip(-SKIP_TIME)
            time.sleep(0.2)

        elif keyboard.is_pressed('right') and keyboard.is_pressed('shift'):
            change_file(next=True)
            time.sleep(0.5)

        elif keyboard.is_pressed('left') and keyboard.is_pressed('shift'):
            change_file(next=False)
            time.sleep(0.5)

        elif keyboard.is_pressed('+'):
            speed = min(speed + SPEED_STEP, 4.0)  # Giới hạn tốc độ max
            if player:
                player.set_rate(speed)
            print(f" (+) speed: {speed:.2f}x")
            time.sleep(0.3)

        elif keyboard.is_pressed('-'):
            speed = max(0.1, speed - SPEED_STEP)
            if player:
                player.set_rate(speed)
            print(f" (-) speed: {speed:.2f}x")
            time.sleep(0.3)

        elif keyboard.is_pressed('delete'):
            delete_current_file()
            time.sleep(0.5)

        elif keyboard.is_pressed('tab'):
            jump_to_file()
            time.sleep(0.5)

    except Exception as e:
        print(f"Lỗi: {e}")
        break