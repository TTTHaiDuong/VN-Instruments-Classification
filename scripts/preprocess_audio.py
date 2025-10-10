import numpy as np


def is_mono(y: np.ndarray) -> bool:
    return y.ndim == 1 or (y.ndim == 2 and y.shape[0] == 1)


def to_mono(y: np.ndarray) -> np.ndarray:
    # Accept shapes: (n,) or (n_channels, n) or (n, n_channels)
    if y.ndim == 1:
        return y
    if y.ndim == 2:
        # common: (n_channels, n_samples) OR (n_samples, n_channels)
        if y.shape[0] <= y.shape[1]:
            return np.mean(y, axis=0)
        else:
            return np.mean(y, axis=1)

    raise ValueError(f"Unexpected audio shape: {y.shape}")


def match_target_rms(y: np.ndarray, target_rms=0.1):
    rms = np.sqrt(np.mean(y**2))
    scaling_factor = target_rms / (rms + 1e-9)
    return y * scaling_factor


def normalize_dbfs(y: np.ndarray, target_dbfs: float = -20.0) -> np.ndarray:
    """Chuẩn hoá âm lượng dBFS
    """
    # tránh lỗi khi tín hiệu toàn số 0
    rms = np.sqrt(np.mean(y**2)) if np.any(y) else 1e-6  
    current_dbfs = 20 * np.log10(rms)
    gain = 10 ** ((target_dbfs - current_dbfs) / 20)
    return y * gain


def normalize_amplitude(y: np.ndarray) -> np.ndarray:
    if np.issubdtype(y.dtype, np.integer):
        max_val = np.iinfo(y.dtype).max
        y = y.astype(np.float32) / max_val
    else:
        y = y.astype(np.float32)
        # if signal is >1.0 or < -1.0, scale by max abs
        max_abs = np.max(np.abs(y)) if y.size else 1.0
        if max_abs > 1.0:
            y = y / max_abs
    return y


def pad_or_trim(y: np.ndarray, target_len: int, mode: str = "constant") -> np.ndarray:
    """Đệm hoặc cắt tỉa bới mẫu âm thanh để chuyển thành melsp sử dụng cho huấn luyện.
    """
    if len(y) > target_len:
        # center crop
        start = (len(y) - target_len) // 2
        return y[start:start + target_len]
    elif len(y) < target_len:
        pad_len = target_len - len(y)
        left = pad_len // 2
        right = pad_len - left
        return np.pad(y, (left, right), mode=mode) # type: ignore
    return y


def pre_emphasis(y: np.ndarray, coef: float = 0.97) -> np.ndarray:
    return np.append(y[0], y[1:] - coef * y[:-1])