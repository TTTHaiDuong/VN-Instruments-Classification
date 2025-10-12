import librosa, os
import numpy as np
from typing import TypedDict
from scripts.utils import register


class BackgroundNoiseParams(TypedDict):
    noise_path: str
    amplitude: tuple[float, float]


class AUGMethods(TypedDict, total=False):
    pitch_shift:      tuple[float, float]
    time_stretch:     tuple[float, float]
    add_noise:        tuple[float, float]
    volume:           tuple[float, float]
    time_mask:        tuple[float, float]
    background_noise: BackgroundNoiseParams


AUGMENT_REGISTRY = {}


@register(AUGMENT_REGISTRY, "pitch_shift")
def _pitch_shift(y, sr, params):
    n_steps = np.random.uniform(*params)
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)


@register(AUGMENT_REGISTRY, "time_stretch")
def _time_stretch(y, sr, params):
    rate = np.random.uniform(*params)
    try:
        return librosa.effects.time_stretch(y, rate=rate)
    except Exception:
        return y
    

@register(AUGMENT_REGISTRY, "add_noise")
def _add_noise(y, sr, params):
    mean, std = params
    noise = np.random.normal(mean, std, len(y))
    return y + noise
    

@register(AUGMENT_REGISTRY, "volume")
def _volume(y, sr, params):
    min_db, max_db = params
    db_change = np.random.uniform(min_db, max_db)
    factor = 10.0 ** (db_change / 20.0)
    return y * factor


@register(AUGMENT_REGISTRY, "time_mask")
def _time_mask(y, sr, params):
    min_duration, max_duration = params
    duration = np.random.uniform(min_duration, max_duration)
    mask_samples = int(duration * sr)
    if mask_samples >= len(y):
        return y
    start = np.random.randint(0, len(y) - mask_samples)
    y_masked = y.copy()
    y_masked[start:start + mask_samples] = 0
    return y_masked


@register(AUGMENT_REGISTRY, "background_noise")
def _background_noise(y, sr, params):
    noise_path = params["noise_path"]
    amplitude_range = params["amplitude"]

    if not os.path.isfile(noise_path):
        raise FileNotFoundError(f"Noise file not found: {noise_path}")

    noise, _ = librosa.load(noise_path, sr=sr)

    # Make sure noise have the same length
    if len(noise) > len(y):
        noise = noise[:len(y)]
    else:
        noise = np.tile(noise, int(np.ceil(len(y) / len(noise))))[:len(y)]

    noise_amplitude = np.random.uniform(*amplitude_range)
    return y + noise_amplitude * noise


def augment_audio(
    audio: np.ndarray,
    methods: AUGMethods,
    sr: int | float | None = None,
) -> np.ndarray:
    """
    
    """
    if not isinstance(audio, np.ndarray):
        raise TypeError("`audio` must be a numpy.ndarray")

    if audio.ndim != 1:
        raise ValueError("`audio` must be 1D (mono)")

    # những kỹ thuật như time_stretch cần 2 mẫu trở lên
    if audio.size < 2:
        raise ValueError("`audio` is too short")
        
    augmented = audio.copy()
    if not methods:
        return augmented
    
    for method, params in methods.items():
        if method not in AUGMENT_REGISTRY:
            raise Exception(f"Invalid augmentation method: {method}")
        augmented = AUGMENT_REGISTRY[method](augmented, sr, params)

    return augmented


def _manual_test(input_path, methods, out_path):
    import soundfile as sf

    y, sr = librosa.load(input_path)
    out = augment_audio(y, methods, sr)
    sf.write(out_path, out, sr)


if __name__ == "__main__":
    _manual_test(
        methods={
            "pitch_shift": (-2, 2),
            # "time_stretch": (0.8, 1.2),
            # "add_noise": (0.0, 0.01),
            # "volume": (-6, 6),
            # "time_mask": (0.1, 0.5),
            # "background_noise": {
            #     "noise_path": librosa.example("trumpet"),
            #     "amplitude": (0.1, 0.5)
            # }
        },
        input_path=r"rawdata\train\danbau\danbau000.wav",
        out_path=r"test\test.wav"
    )