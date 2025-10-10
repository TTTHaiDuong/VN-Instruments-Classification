import click, os, librosa
import matplotlib.pyplot as plt
from pathlib import Path
from scripts.mel_spectrogram import to_mel
from scripts.utils import collect_files
from config.general import MEL_CONFIG


def plot_mel(
    mel_db,
    mel_cfg = MEL_CONFIG,
    **overrides
):
    """
    Display a Mel-spectrogram as an image.

    Parameters:
        mel_spec_db (ndarray): Log-scaled Mel-spectrogram.
        duration (float): Duration of the audio (in seconds).
        n_mels (int): Number of Mel bands.
    """
    destin_cfg = mel_cfg.model_copy(update=overrides)

    # Normalize the shape of mel_spec_db
    if len(mel_db.shape) == 3:
        if mel_db.shape[-1] == 1:
            mel_db = mel_db[:, :, 0]  # Remove single channel to get 2D
        elif mel_db.shape[-1] == 3:
            mel_db = mel_db[:, :, 0]   # Take the first channel for visualization
        else:
            raise ValueError(f"Expected 1 or 3 channels, got shape {mel_db.shape}")
    elif len(mel_db.shape) != 2:
        raise ValueError(f"Expected a 2D or 3D array with 1 or 3 channels, got shape {mel_db.shape}")
    
    plt.figure(figsize=(8, 4))
    plt.imshow(mel_db, aspect="auto", origin="upper",
               extent=(0, destin_cfg.duration, 0, destin_cfg.n_mels), cmap="magma")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Mel-spectrogram")
    plt.xlabel("Time (s)")
    plt.ylabel("Mel Bands")
    plt.tight_layout()
    plt.show()
    plt.close()


@click.group()
def cli():
    pass


@cli.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("out_path", type=click.Path())
def save(
    input_path: str, 
    out_path: str
):
    """
    Converts an audio file or a directory of audio files into Mel-spectrograms 
    and saves them as .png images in the output directory without overwriting existing files.

    Parameters:
        input_path (str): Path to an audio file or directory containing audio files.
        output_dir (str): Directory where the generated Mel-spectrogram images will be saved.
        hop_length (int): Number of samples between successive frames.
        n_fft (int): FFT window size.
        n_mels (int): Number of Mel frequency bands.
        sr (int): Sampling rate for audio signal.
        input_shape (tuple): Desired input shape for CNN models (height, width, channels).
    """
    paths, labels = collect_files(input_path)

    if not bool(Path(out_path).suffix):
        os.makedirs(out_path, exist_ok=True)

    for path, label in zip(paths, labels):
        if path.endswith((".wav")):
            base_name = Path(path).stem
            # nếu như input_path là một file cụ thể thì cho phép đặt tên trực tiếp bằng out_path
            # không thì sử dụng tên file input_path để đặt
            out_dir = os.path.join(out_path, label or "")
            out_file = (
                out_path 
                if len(paths) == 1 and bool(Path(out_path).suffix) 
                else os.path.join(out_dir, f"{base_name}.png")
            )
            os.makedirs(out_dir, exist_ok=True)

            y, sr = librosa.load(path)
            mel = to_mel(y, sr)
            try:
                plt.imsave(out_file, mel[:, :, 0], cmap="magma")
                print(f"Saved: {out_file}")
            except Exception as e:
                print(f"Error processing: {e}")
        else:
            print(f"Unsupported audio format: {path}")
    

@cli.command()
@click.argument("file_path", type=click.Path(exists=True, dir_okay=False))
def display(file_path):
    y, sr = librosa.load(file_path)
    mel_spec = to_mel(y, sr)
    plot_mel(mel_spec)


if __name__ == "__main__":
    cli()