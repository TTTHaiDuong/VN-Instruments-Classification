import numpy as np
import tensorflow as tf
from scripts.utils import print_table


def _bytes_feature(value: tf.Tensor | np.ndarray | str | bytes) -> tf.train.Feature:
    """Trả về bytes list (dùng cho mảng numpy)"""
    # không sửa thứ tự các mệnh đề if
    if isinstance(value, tf.Tensor):
        value = value.numpy()
    if isinstance(value, np.ndarray):
        value = value.tobytes()
    if isinstance(value, str):
        value = value.encode("utf-8")
    if not isinstance(value, (bytes, bytearray)):
        raise TypeError(f"Expected bytes-like, got {type(value)}")
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(mel: np.ndarray, label: int, file_id: str = "") -> tf.train.Example:
    feature = {
        "mel": _bytes_feature(mel.tobytes()),      # lưu mảng float32 dạng bytes
        "shape": _int64_feature(mel.shape[0]),     # chiều cao H
        "width": _int64_feature(mel.shape[1]),     # chiều rộng W
        "channels": _int64_feature(mel.shape[2]),  # số kênh C
        "label": _int64_feature(label),
        "id": _bytes_feature(file_id)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def write_tfrecord(
    mels: list[np.ndarray], 
    class_idxs: list[int], 
    filename: str, 
    ids: list[str] | None = None,
    verbose: bool = False
):
    logs = []

    with tf.io.TFRecordWriter(filename) as writer:
        for i, (ml, lbl) in enumerate(zip(mels, class_idxs)):
            file_id = ids[i] if ids is not None else ""
            example = serialize_example(ml.astype(np.float32), lbl, file_id)
            writer.write(example.SerializeToString())

            # Ghi log
            if verbose and (i == 0 or ml.shape != mels[i-1].shape or lbl != class_idxs[i-1]):
                logs.append({
                    "Index": i,
                    "Sample ID": file_id if file_id else None,
                    "Mel shape": ml.shape,
                    "Label": lbl,
                })

    if verbose:
        total = len(class_idxs)
        summary = [{"Label": "TOTAL", "Count": total}]

        # đếm theo label
        unique, counts = np.unique(class_idxs, return_counts=True)
        for u, c in zip(unique, counts):
            summary.append({
                "Label": u,
                "Count": c,
            })

        print_table(logs, title="TFRecord Write Log")
        print()
        print_table(summary, title="Summary")


def parse_example(example_proto) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    feature_description = {
        "mel": tf.io.FixedLenFeature([], tf.string),
        "shape": tf.io.FixedLenFeature([], tf.int64),
        "width": tf.io.FixedLenFeature([], tf.int64),
        "channels": tf.io.FixedLenFeature([], tf.int64),
        "label": tf.io.FixedLenFeature([], tf.int64),
        "id": tf.io.FixedLenFeature([], tf.string)
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)

    # khôi phục mel từ bytes
    H = parsed["shape"]
    W = parsed["width"]
    C = parsed["channels"]
    mel = tf.io.decode_raw(parsed["mel"], tf.float32)
    mel = tf.reshape(mel, (H, W, C))

    label = parsed["label"]
    file_id = parsed["id"]

    return mel, label, file_id