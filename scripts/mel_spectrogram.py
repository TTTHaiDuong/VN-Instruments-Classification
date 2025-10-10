import numpy as np
import librosa
import scripts.preprocess_audio as ppa
from types import SimpleNamespace
from config.general import MEL_CONFIG, TRAIN_CONFIG


NORMALIZE_REGISTRY = {}

def _register_normalize(name):
    def decorator(func):
        NORMALIZE_REGISTRY[name] = func
        return func
    return decorator


@_register_normalize("minmax_0_1")
def _minmax_0_1(S: np.ndarray, top_db: float) -> np.ndarray:
    # assume S in [-top_db, 0]
    return (S + top_db) / top_db


@_register_normalize("zscore")
def _zscore(S: np.ndarray, _) -> np.ndarray:
    mu = S.mean()
    sigma = S.std() if S.std() > 0 else 1.0
    return (S - mu) / sigma


def _preprocess_audio(
    y: np.ndarray,
    sr_in: int | float,
    cfg
) -> np.ndarray:
    # 1. resample
    if sr_in != cfg.sr:
        y = librosa.resample(y.astype(np.float32), orig_sr=sr_in, target_sr=cfg.sr)

    # 2. mono
    y = ppa.to_mono(y)

    # 3. normalize volume
    y = ppa.normalize_dbfs(y, cfg.target_dbfs)

    # 4. normalize amplitude
    y = ppa.normalize_amplitude(y)
    
    # 5. pre-emphasis
    if cfg.pre_emphasis:
        y = ppa.pre_emphasis(y, coef=cfg.pre_emphasis)

    # 6. pad/trim
    expected_len = int(cfg.sr * cfg.duration)
    return ppa.pad_or_trim(y, expected_len, mode=cfg.pad_mode)  


def _process_mel(
    y: np.ndarray,
    cfg,
    return_db: bool = True
) -> np.ndarray:
    S = librosa.feature.melspectrogram(
        y=y,
        sr=cfg.sr,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        fmin=cfg.fmin,
        fmax=cfg.fmax or cfg.sr // 2,
        power=2.0
    )

    if return_db:
        return librosa.power_to_db(S, ref=np.max, top_db=cfg.top_db)
    return S


def _fix_shape(
    S: np.ndarray, 
    H: int, 
    W: int, 
    fill_val: float | None = None
) -> np.ndarray:
    """Resize/pad spectrogram to (H, W)"""
    if fill_val is None:
        fill_val = S.min() if S.size > 0 else 0.0

    S_fixed = np.full((H, W), fill_val, dtype=np.float32)
    h = min(H, S.shape[0])
    w = min(W, S.shape[1])
    S_fixed[:h, :w] = S[:h, :w]
    return S_fixed


def _postprocess_mel(
    S: np.ndarray, 
    cfg
) -> np.ndarray:
    """Normalize, fix shape, and add channels"""
    # 1. normalize
    if cfg.normalize in NORMALIZE_REGISTRY:
        S = NORMALIZE_REGISTRY[cfg.normalize](S, top_db=cfg.top_db)
    S = S.astype(np.float32)

    # 2. enforce shape
    H, W, C = cfg.input_shape
    S = _fix_shape(S, H, W, fill_val=S.min())

    # 3. channel handling
    if C == 1:
        S_out = np.expand_dims(S, axis=-1)  # (H, W, 1)
    elif C == 3:
        # Option A: replicate
        # S_out = np.repeat(S[..., np.newaxis], 3, axis=-1)

        # Option B: mel + delta + delta-delta
        delta = librosa.feature.delta(S)
        delta2 = librosa.feature.delta(S, order=2)
        S_out = np.stack([S, delta, delta2], axis=-1)
    else:
        raise ValueError(f"Unsupported channel count: {C}")

    return S_out


def to_mel(
    y: np.ndarray,
    sr_in: int | float,
    mel_cfg = MEL_CONFIG,
    train_cfg = TRAIN_CONFIG,
    to_tensor: bool = False,  # if True, returns shape (C, H, W) e.g. for pytorch
    **overrides
) -> np.ndarray:
    destin_cfg = mel_cfg.model_dump() | train_cfg.model_dump() | (overrides or {})
    destin_cfg = SimpleNamespace(**destin_cfg)

    # 1. audio preprocessing
    y = _preprocess_audio(y, sr_in, destin_cfg)
    
    # 2. mel spectrogram
    S = _process_mel(y, destin_cfg)
    
    # 3. normalize + fix shape + channel
    S_out = _postprocess_mel(S, destin_cfg)

    # 4. return (C, H, W) if tensor (PyTorch)
    if to_tensor:
        return np.transpose(S_out, (2, 0, 1))
    return S_out


if __name__ == "__main__":
    y, sr = librosa.load("rawdata/train/danbau/danbau000.wav")
    y = to_mel(y, sr)
    print(y)