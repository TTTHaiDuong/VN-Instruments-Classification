import librosa
import numpy as np
import tensorflow as tf
from typing import TypedDict
from collections import Counter
from rich.progress import Progress
from scripts.mel_spectrogram import to_mel
from scripts.preprocess_image import load_image
from scripts.utils import print_table, to_mm_ss # để log
from config.general import INSTRUMENTS # để log
from train import MODEL_REGISTRY


class SegmentInfo(TypedDict, total=True):
    """Lưu trữ thông tin sau khi dự đoán từng đoạn của một nguồn âm thanh.
    """
    true_conf: bool
    class_idx: int
    start: float
    probs: list[float]


def split_to_mels(
    y: np.ndarray, 
    sr_in: float | int, 
    segment_len: float
):
    """Cắt một đoạn nhạc dài thành các đoạn bằng nhau để sử dụng cho dự đoán.
    Đoạn cuối cùng chiều dài có thể không bằng `segment_len`.
    """
    total_duration = len(y) / sr_in
    num_segments = int(np.ceil(total_duration / segment_len))

    mels = []
    for i in range(num_segments):
        start = int(i * segment_len * sr_in)
        end = int(min((i + 1) * segment_len * sr_in, len(y)))
        segment_y = y[start:end]
        mel = to_mel(segment_y, sr_in)
        mels.append(mel)
    
    return mels


def collect_mels(fpaths: list[str]) -> list[np.ndarray]:
    """Chuyển các file âm thanh thành melsp.
    Các file âm thanh này là mẫu thuộc dataset, tất cả nên chuẩn hoá thời lượng trước.
    """
    with Progress() as progress:
        task = progress.add_task(
            "[magenta]Extracting Mel-spectrograms...", total=len(fpaths)
        )
        
        mels = []
        for fp in fpaths:
            y, sr = librosa.load(fp)
            mel = to_mel(y, sr)
            mels.append(mel)

            progress.update(task, advance=1)
    
    return mels


def load_model(
    model_fpath: str, 
    model_index: int | None = None
):
    """Load file mô hình đã huấn luyện dùng để dự đoán.
    Hỗ trợ file full cấu hình `.h5` và file chỉ lưu trọng số `.weight.h5`
    (file `.weight.h5` phải cung cấp thêm tham số `model_index`)
    """
    if model_fpath.lower().endswith(".weights.h5"):
        if not model_index:
            raise ValueError("Missing `model_index` when load .weight.h5 file.")
        
        model, _ = MODEL_REGISTRY[model_index]()
        print(f"Loading model weights from: {model_fpath}")
        model.load_weights(model_fpath)

    elif model_fpath.lower().endswith(".h5"):
        print(f"Loading full model from: {model_fpath}")
        model = tf.keras.models.load_model(
            model_fpath,
            custom_objects={"LeakyReLU": tf.keras.layers.LeakyReLU}
        )
    else:
        raise ValueError(f"{model_fpath} must end with '.weights.h5' or '.h5'")
    
    return model


def from_audio(
    model, 
    mels: np.ndarray | list[np.ndarray],
    segment_len: float,
    threshold: float,
    verbose = False
) -> list[SegmentInfo]:
    """Dự đoán các melsp và cho ra các `SegmentInfo`.
    """
    preds = model.predict(mels, verbose=verbose)

    segments_info = []
    logs = []

    for i, pred in enumerate(preds):
        predicted_idx = int(np.argmax(pred))
        conf = float(np.max(pred))

        segments_info.append({
            "true_conf": conf >= threshold,
            "class_idx": predicted_idx,
            "start": i * segment_len,
            "probs": pred.tolist(),
        })

        if verbose:
            logs.append({
                "Segment": i + 1,
                "mm:ss": to_mm_ss(int(i * segment_len)),
                "Class": INSTRUMENTS.index_to_name(predicted_idx),
                "Conf": f"{conf:.2f}",
                "Passed": "✓" if conf >= threshold else "",
                "Probs": np.round(pred[0], 3)
            })
    if verbose:
        print_table(logs, "Segments Prediction Detail")

    return segments_info


def from_img(
    model,
    file_path: str,
    target_size: tuple[int, int],
    verbose = False
):
    img = load_image(file_path, target_size)
    pred = model.predict(img, verbose=verbose)
    return pred[0]


def mean_voting_probs(
    segments_info: list[SegmentInfo],
) -> tuple[np.ndarray, np.ndarray]:
    """
    """
    mean_probs = np.mean([seg["probs"] for seg in segments_info], axis=0)

    pass_classes = [seg["class_idx"] for seg in segments_info if seg["true_conf"]]
    vote_counts = Counter(pass_classes)
    total_votes = sum(vote_counts.values())
    voting_ratios = np.array([
        (vote_counts.get(i, 0) / total_votes) if total_votes > 0 else 0
        for i in range(len(mean_probs))
    ])

    return mean_probs, voting_ratios