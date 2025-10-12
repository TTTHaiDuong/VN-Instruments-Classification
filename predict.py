import click, librosa
import matplotlib.pyplot as plt
import numpy as np
import scripts.predict as pred 
from typing import BinaryIO
from train import validate_model_idx
from scripts.utils import to_mm_ss
from matplotlib.ticker import FuncFormatter
from config.general import MEL_CONFIG, INSTRUMENTS, MAIN_MODEL, TRAIN_CONFIG


def plot_segment_probabilities(
    segments_info: list[pred.SegmentInfo], 
    class_names: list[str], 
    threshold: float
):
    times = [seg["start"] for seg in segments_info]
    probs = np.array([seg["probs"] for seg in segments_info])  # shape: (segments, classes)

    plt.figure(figsize=(12, 6))
    
    for class_idx, class_name in enumerate(class_names):
        plt.plot(times, probs[:, class_idx], marker='o', label=class_name)

    # Threshold line (confidence)
    plt.axhline(y=threshold, color='gray', linestyle='--', linewidth=1.5, label=f'Threshold {threshold}')

    ax = plt.gca()
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: to_mm_ss(x)))

    plt.xlabel("Time (mm:ss)")
    plt.ylabel("Probability")
    plt.title("Class Probabilities Over Time")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def summary_result_report(
    mean_probs,
    voting_ratios,
    class_names
):
    predicted_class_mean = class_names[np.argmax(mean_probs)]
    predicted_class_vote = class_names[np.argmax(voting_ratios)]

    print("\n[Averaged Softmax Probabilities]")
    for name, ratio in zip(class_names, mean_probs):
        print(f"{name}: {ratio:.2f}%")

    print(f"Predicted (Avg): {predicted_class_mean}")
    if np.any(voting_ratios):
        print("\n[Voting Based on Confidence Threshold]")
        for name, ratio in zip(class_names, voting_ratios):
            print(f"{name}: {ratio:.2f}%")
        print(f"Predicted (Voting): {predicted_class_vote}")
    else:
        print("\nNo segments passed the confidence threshold.")


def predict_endpoint(
    model_path: str,
    model_index: int, 
    file_or_buffer: str | BinaryIO, 
    class_names: list[str],
    threshold: float,
    segment_len: float | None = None,
    verbose = False
):
    """
    Predict the predominant instrument from an audio input (file path or buffer).

    Parameters:
        file_or_buffer (str or io.BytesIO): Path to the audio file (".wav" or ".mp3") or a binary buffer (e.g. BytesIO).
        model_index (int): Which model architecture to use (1 or 2).
        model_file (str): Path to the model file. Either a full model ".h5" or weights ".weights.h5".

    Returns:
        str: The name of the predicted instrument with the highest average probability.
        mean_probs (np.ndarray): Averaged softmax probabilities per class.
        voting_ratios (np.ndarray): Percentage of confident votes per class.
        segment_info (List[Dict]): Info for each audio segment.
    
    Raises:
        ValueError: If model_index is invalid or model_file has unsupported extension.
        TypeError: If input is neither a string nor a BytesIO buffer.
    """
    if not segment_len:
        segment_len = MEL_CONFIG.duration
    
    y, sr = librosa.load(file_or_buffer)

    model = pred.load_model(model_path, model_index)

    mels = pred.split_to_mels(y, sr, segment_len)
    mels = np.stack(mels, axis=0)
    
    segments_info = pred.from_audio(
        model=model, 
        mels=mels,
        segment_len=segment_len,
        threshold=threshold,
        verbose=verbose
    )
    mean_probs, voting_ratios = pred.mean_voting_probs(segments_info)

    predicted_class_mean = class_names[np.argmax(mean_probs)]
    predicted_class_vote = class_names[np.argmax(voting_ratios)]

    if verbose:
        print("\n[Averaged Softmax Probabilities]")
        for name, ratio in zip(class_names, mean_probs):
            print(f"{name}: {ratio:.2f}%")
        print(f"Predicted (Avg): {predicted_class_mean}")

        if np.any(voting_ratios):
            print("\n[Voting Based on Confidence Threshold]")
            for name, ratio in zip(class_names, voting_ratios):
                print(f"{name}: {ratio:.2f}%")
            print(f"Predicted (Voting): {predicted_class_vote}")
        else:
            print("\nNo segments passed the confidence threshold.")

    print(f"\nPredicted main instrument: {predicted_class_vote}")
    return predicted_class_vote, mean_probs, voting_ratios, segments_info


@click.command()
@click.argument("mode", type=click.Choice(["audio", "img"]))
@click.argument("file_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--model_path", "-p", type=click.Path(exists=True, dir_okay=False), default=MAIN_MODEL)
@click.option("--model_index", "-i", type=int, default=1, callback=validate_model_idx)
@click.option("--display_chart", "-d", is_flag=True)
@click.option("--threshold", "-t", type=float, default=0.6)
def main(
    mode: str,
    file_path: str,
    model_path: str, 
    model_index: int, 
    display_chart: bool, 
    threshold: float, 
    input_shape = TRAIN_CONFIG.input_shape,
    class_names: list[str] = INSTRUMENTS.names(),
    verbose = True
):
    if mode == "audio":
        result = predict_endpoint(
            model_path=model_path,
            model_index=model_index, 
            file_or_buffer=file_path, 
            class_names=class_names, 
            threshold=threshold, 
            verbose=verbose
        )
        if display_chart:
            plot_segment_probabilities(result[3], class_names, threshold)
    
    elif mode == "img":
        model = pred.load_model(model_path, model_index)
        probs = pred.from_img(model, file_path, input_shape, verbose)
        class_name = class_names[np.argmax(probs, axis=0)]
        print("Predicted class:", class_name, "| Probs:", np.round(probs, 3))


if __name__ == "__main__":
    main()