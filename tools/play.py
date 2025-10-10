# """
#     WAV AUDIO PLAYER
#     =============================================

#     Dependency:
#         Make sure your computer has VLC Media Player installed.
#         Download from:   https://www.videolan.org/vlc/
#         Add the libvlc.dll path to your system environment variables.

#     Run in terminal:
#         python play.py -i <path\to\folder>
# """

# import argparse
# import keyboard
# import os
# import shutil
# import sys
# import textwrap
# import time
# import vlc
# from pathlib import Path
# from send2trash import send2trash

# parser = argparse.ArgumentParser(description="Play .wav audio files in a folder.")
# parser.add_argument("-i", "--input", type=str, required=True, help="The path to the folder containing the audio files.")
# parser.add_argument("-t", "--skiptime", type=int, default=1, help="")
# parser.add_argument("-s", "--speedstep", type=float, default=0.25)
# args = parser.parse_args()

# if not os.path.isdir(args.input):
#     raise Exception("Invalid folder path.")

# # === Config ===
# AUDIO_EXT = [".wav", ".mp3", ".ogg"]
# SKIP_TIME = args.skiptime
# SPEED_STEP = args.speedstep

# # === States ===
# vlc_instance = vlc.Instance("--no-audio-time-stretch", "--quiet")
# folder_path = args.input
# player = None
# current_index = 0
# speed = 1.0
# progress_state = None
# deleted = 0                 # Number of files deleted

# # === Load audio files ===
# audio_files = sorted([f for f in os.listdir(folder_path) if Path(f).suffix in AUDIO_EXT])

# def load_audio(index):
#     global player
#     release_player()

#     # Ignore PCR
#     path = os.path.join(folder_path, audio_files[index])
#     media = vlc_instance.media_new(path)
#     player = vlc_instance.media_player_new()
#     player.set_media(media)
#     player.play()

# def release_player():
#     global player
#     if player:
#         player.release()
#         player = None
#         time.sleep(0.1) # Wait for file release

# def play_audio(start_sec):
#     if not player: return
#     player.set_time(int(start_sec * 1000))  # VLC use milisecond

# def play_new(current_pos=0):
#     load_audio(current_index)
#     play_audio(current_pos)

# def skip(seconds):
#     if not player: return

#     duration = player.get_length() / 1000
#     if duration <= 0: return

#     state = player.get_state()
#     if state == vlc.State.Ended:
#         player.stop()
#         player.play()
#         if seconds > 0: player.set_time(0)
#         else: player.set_time(int(duration + seconds) * 1000)
#         return

#     current_time = player.get_time() / 1000
#     new_pos = current_time + seconds
#     new_pos = max(0, min(new_pos, duration))

#     player.set_time(int(new_pos * 1000))

# def seek_by_percentage(position):
#     if not player: return

#     state = player.get_state()
#     if state == vlc.State.Ended:
#         player.stop()
#         player.play()
    
#     if 0 <= position <= 1:
#         player.set_position(position)

# def delete_current_file():
#     global current_index, player, deleted
#     release_player()
    
#     path = os.path.join(folder_path, audio_files[current_index])
#     send2trash(path)
#     print(f"\n[Del] Moved to the recycle bin: {audio_files[current_index]}")
#     deleted += 1
#     audio_files.pop(current_index)
#     # When no files
#     if not audio_files:
#         print("No more files to play.")
#         sys.exit(0)
#     if current_index >= len(audio_files):
#         current_index = len(audio_files) - 1
    
#     play_new()

# def change_file(next=True):
#     global current_index
#     release_player()
#     current_index = (current_index + (1 if next else -1)) % len(audio_files)
#     play_new()

# def jump_to_file():
#     global current_index
#     release_player()
    
#     print(f"\nAvailable files ({len(audio_files)}):")
#     if len(audio_files) <= 20:
#         for idx, file in enumerate(audio_files):
#             print(f"[{idx + 1}] {Path(file).stem}")
    
#     print("\nEnter file number or name (without extension): ", end="", flush=True)
#     user_input = sys.stdin.readline().strip()
    
#     if not user_input:
#         print("Operation cancelled.")
#         return
    
#     try:
#         selected_index = int(user_input) - 1
#         if 0 <= selected_index < len(audio_files):
#             current_index = selected_index
#             print(f"\nJumped to file [{selected_index + 1}]: {audio_files[current_index]}")
#             play_new()
#         else:
#             print(f"\nInvalid number. Please enter between 1 and {len(audio_files)}")
    
#     except ValueError:
#         matched_indices = [i for i, f in enumerate(audio_files) 
#                           if Path(f).stem.lower() == user_input.lower()]
        
#         if matched_indices:
#             current_index = matched_indices[0]
#             print(f"\nJumped to file: {audio_files[current_index]}")
#             play_new()
#         else:
#             print(f"\nFile not found: {user_input}")

# def print_progress_bar(length=40):
#     if not player: return
    
#     duration = player.get_length() / 1000
#     current_time = player.get_time() / 1000

#     if duration <= 0 or current_time < 0: return

#     global progress_state
#     if progress_state and progress_state == [current_index, current_time, duration, speed]: return
#     progress_state = [current_index, current_time, duration, speed]
    
#     progress = current_time / duration if duration > 0 else 0
#     progress = max(0, min(1, progress))
    
#     terminal_width = shutil.get_terminal_size().columns
    
#     max_filename_length = max(10, terminal_width - 150)
    
#     filled_length = int(length * progress)
#     bar_length = max(10, min(length, terminal_width - 50))
#     bar = '█' * filled_length + '-' * (bar_length - filled_length)
    
#     current_str = time.strftime('%M:%S', time.gmtime(current_time))
#     duration_str = time.strftime('%M:%S', time.gmtime(duration))

#     filename = audio_files[current_index]
#     if len(filename) > max_filename_length:
#         filename_display = filename[:max_filename_length-3] + '...'
#     else:
#         filename_display = filename.ljust(max_filename_length)
    
#     display_str = f">> ({current_index + 1}) {filename_display} |{bar}| {current_str}/{duration_str} | Speed: {speed:.2f}x"
    
#     display_str = display_str[:terminal_width-1]
    
#     sys.stdout.write('\r')
#     sys.stdout.write(display_str)
#     sys.stdout.flush()
    
# # === Start ===
# print(textwrap.dedent(f"""
#     [🎧] Controls:
#     ──────────────────────────────
#     0-9:        Jump to 0%-90%
#     ←/→:        Seek {SKIP_TIME} seconds
#     Shift+←/→:  Next/previous file
#     Tab:        Jump to file
#     +/-:        Increase or decrease speed
#     Space:      Play/pause
#     Delete:     Delete current file
#     Esc:        Exit
# """))

# load_audio(current_index)

# while True:
#     try:
#         print_progress_bar()

#         if keyboard.is_pressed('esc'):
#             if player:
#                 player.stop()
#             print(f"\n[Del] Total files deleted: {deleted} files")
#             break

#         elif keyboard.is_pressed('space'):
#             if not player: continue
#             if player.is_playing():
#                 player.pause()
#             else:
#                 player.play()
                  
#         elif keyboard.is_pressed('right') and not keyboard.is_pressed('shift'):
#             skip(SKIP_TIME)
            
#         elif keyboard.is_pressed('left') and not keyboard.is_pressed('shift'):
#             skip(-SKIP_TIME)            

#         elif keyboard.is_pressed('right+shift'):
#             change_file(next=True)
            
#         elif keyboard.is_pressed('left+shift'):
#             change_file(next=False)            

#         elif keyboard.is_pressed('+'):
#             speed = min(speed + SPEED_STEP, 4.0)  # Max speed limit
#             if player:
#                 player.set_rate(speed)

#         elif keyboard.is_pressed('-'):
#             speed = max(0.1, speed - SPEED_STEP)
#             if player:
#                 player.set_rate(speed)

#         elif keyboard.is_pressed('delete'):
#             delete_current_file()

#         elif keyboard.is_pressed('tab'):
#             jump_to_file()    
        
#         else:
#             for num in range(10):
#                 if keyboard.is_pressed(str(num)):
#                     seek_by_percentage(num / 10)
        
#         time.sleep(0.1)
            
#     except Exception as e:
#         print(f"Error: {e}")
#         break

import argparse
import os
import shutil
import sys
import textwrap
import time
import vlc
from pathlib import Path
from send2trash import send2trash
from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.widgets import Frame, TextArea
from prompt_toolkit.formatted_text import HTML

# === Parse args ===
parser = argparse.ArgumentParser(description="Play .wav audio files in a folder.")
parser.add_argument("-i", "--input", type=str, required=True, help="The path to the folder containing the audio files.")
parser.add_argument("-t", "--skiptime", type=int, default=1)
parser.add_argument("-s", "--speedstep", type=float, default=0.25)
args = parser.parse_args()

if not os.path.isdir(args.input):
    raise Exception("Invalid folder path.")

# === Config ===
AUDIO_EXT = [".wav", ".mp3", ".ogg"]
SKIP_TIME = args.skiptime
SPEED_STEP = args.speedstep

# === States ===
vlc_instance = vlc.Instance("--no-audio-time-stretch", "--quiet")
folder_path = args.input
player = None
current_index = 0
speed = 1.0
progress_state = None
deleted = 0

# === Load audio files ===
audio_files = sorted([f for f in os.listdir(folder_path) if Path(f).suffix in AUDIO_EXT])

def release_player():
    global player
    if player:
        player.release()
        player = None
        time.sleep(0.1)

def load_audio(index):
    global player
    release_player()
    path = os.path.join(folder_path, audio_files[index])
    media = vlc_instance.media_new(path)
    player = vlc_instance.media_player_new()
    player.set_media(media)
    player.play()

def play_audio(start_sec):
    if not player: return
    player.set_time(int(start_sec * 1000))

def play_new(current_pos=0):
    load_audio(current_index)
    play_audio(current_pos)

def skip(seconds):
    if not player: return
    duration = player.get_length() / 1000
    if duration <= 0: return
    state = player.get_state()
    if state == vlc.State.Ended:
        player.stop(); player.play()
        if seconds > 0: player.set_time(0)
        else: player.set_time(int(duration + seconds) * 1000)
        return
    current_time = player.get_time() / 1000
    new_pos = max(0, min(current_time + seconds, duration))
    player.set_time(int(new_pos * 1000))

def seek_by_percentage(position):
    if not player: return
    state = player.get_state()
    if state == vlc.State.Ended:
        player.stop(); player.play()
    if 0 <= position <= 1:
        player.set_position(position)

def delete_current_file():
    global current_index, deleted
    release_player()
    path = os.path.join(folder_path, audio_files[current_index])
    send2trash(path)
    deleted += 1
    audio_files.pop(current_index)
    if not audio_files:
        print("No more files to play.")
        sys.exit(0)
    if current_index >= len(audio_files):
        current_index = len(audio_files) - 1
    play_new()

def change_file(next=True):
    global current_index
    release_player()
    current_index = (current_index + (1 if next else -1)) % len(audio_files)
    play_new()

def jump_to_file():
    global current_index
    release_player()
    print(f"\nAvailable files ({len(audio_files)}):")
    for idx, file in enumerate(audio_files[:20]):
        print(f"[{idx+1}] {Path(file).stem}")
    print("Enter file number: ", end="", flush=True)
    user_input = sys.stdin.readline().strip()
    if not user_input:
        print("Cancelled."); return
    try:
        idx = int(user_input) - 1
        if 0 <= idx < len(audio_files):
            current_index = idx; play_new()
    except ValueError:
        matches = [i for i,f in enumerate(audio_files) if Path(f).stem.lower()==user_input.lower()]
        if matches:
            current_index = matches[0]; play_new()

# === UI ===
text_area = TextArea(focusable=False)
frame = Frame(text_area, title="🎧 WAV AUDIO PLAYER")

kb = KeyBindings()

@kb.add("escape")
def _(event):
    global player
    if player: player.stop()
    print(f"\n[Del] Total files deleted: {deleted} files")
    event.app.exit()

@kb.add("space")
def _(event):
    if not player: return
    if player.is_playing(): player.pause()
    else: player.play()

@kb.add("right")
def _(event): skip(SKIP_TIME)

@kb.add("left")
def _(event): skip(-SKIP_TIME)

@kb.add("s-right")
def _(event): change_file(next=True)

@kb.add("s-left")
def _(event): change_file(next=False)

@kb.add("+")
def _(event):
    global speed
    speed = min(speed + SPEED_STEP, 4.0)
    if player: player.set_rate(speed)

@kb.add("-")
def _(event):
    global speed
    speed = max(0.1, speed - SPEED_STEP)
    if player: player.set_rate(speed)

@kb.add("delete")
def _(event): delete_current_file()

@kb.add("tab")
def _(event): jump_to_file()

for num in range(10):
    @kb.add(str(num))
    def _(event, n=num):
        seek_by_percentage(n/10)

# === Progress update ===
def refresh():
    if not player: return
    duration = player.get_length() / 1000
    current_time = player.get_time() / 1000
    if duration <= 0 or current_time < 0: return
    progress = max(0, min(current_time/duration, 1))
    filled = int(40*progress)
    bar = "█"*filled + "-"*(40-filled)
    filename = audio_files[current_index]
    info = f">> ({current_index+1}) {filename} |{bar}| {time.strftime('%M:%S', time.gmtime(current_time))}/{time.strftime('%M:%S', time.gmtime(duration))} | Speed: {speed:.2f}x"
    text_area.text = info

# === Start ===
print(textwrap.dedent(f"""
    [🎧] Controls:
    ──────────────────────────────
    0-9:        Jump to 0%-90%
    ←/→:        Seek {SKIP_TIME} seconds
    Shift+←/→:  Next/previous file
    Tab:        Jump to file
    +/-:        Increase or decrease speed
    Space:      Play/pause
    Delete:     Delete current file
    Esc:        Exit
"""))

load_audio(current_index)

app = Application(layout=Layout(frame), key_bindings=kb, full_screen=False, refresh_interval=0.5, after_render=lambda _: refresh())
app.run()