import click, random, re, yaml, librosa
import matplotlib.pyplot as plt
import soundfile as sf
from typing import Literal
from pathlib import Path
from scripts.augment import augment_audio, AUGMethods
from scripts.utils import collect_data_files
from scripts.mel_spectrogram import to_mel
from config.general import MEL_CONFIG


def safe_str(x):
    if isinstance(x, dict):
        parts = [f"{k}_{safe_str(v)}" for k, v in x.items()]
        return "__".join(parts)
    elif isinstance(x, (list, tuple)):
        return "_".join([safe_str(i) for i in x])
    elif isinstance(x, float):
        return f"{x:.6g}"  # tránh 1e-05 kiểu loạn
    else:
        return str(x)


def params_to_str(params):
    """Chuyển params dict/list/tuple thành chuỗi để sử dụng cho tên file.

        Với dict:
        ```
        >>> params_to_str({"shift": 2, "stretch": { "cfg": (1, 2) }})
        >>> # shift_2__stretch_cfg_1_2
        ```
        Với list:
        ```
        >>> params_to_str(["noise", "0.01", "db"])
        >>> # noise_0.01_db
        ```
        Với tuple:
        ```
        >>> params_to_str((0.5, 22050))
        >>> # 0.5_22050
        ```
    """
    if not params:
        return ""
    param_str = safe_str(params)
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", param_str)


def save_augmented_audio(
    fpath_in: str, 
    fpath_out: str,
    methods: AUGMethods,
    sr_to: int | float | None = None
):
    """"""    
    y, sr = librosa.load(fpath_in, sr=sr_to)
    augmented = augment_audio(y, methods, sr=sr)

    out_path = Path(fpath_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ext = out_path.suffix.lower()

    if ext == ".png":
        mel_spec = to_mel(augmented, sr_in=sr)
        plt.imsave(out_path, mel_spec[:, :, 0], cmap="magma")
    elif ext == ".wav":
        sf.write(out_path, augmented, sr)
    else:
        raise ValueError(f"Không hỗ trợ định dạng: {ext}")

    print(f"Saved: {out_path}")


def generate_augment_plan(
    fpaths: list, 
    labels: list, 
    augment_plan: dict
) -> tuple[list[str], list[AUGMethods]]:
    """
    Sinh danh sách (filepath, label, methods_dict) từ AUGMENT_PLAN.

    Args:
        file_list (list): [(file_path, class_name), ...]
        augment_plan (dict): AUGMENT_PLAN định nghĩa cho từng lớp.

    Returns:
        list: [(filepath, label, {method: params, ...}), ...]
    """
    fpaths_out, tasks = [], []

    # gom file theo class từ list
    fpaths_by_class = {} # { "danbau": ["danbau001.wav", "danbau002.wav"], "dannhi": [] }
    for f, cls in zip(fpaths, labels):
        fpaths_by_class.setdefault(cls, []).append(f)

    # cls: "danbau", cfg: { "allow_overlap": bool, methods: [] }
    for cls, cfg in augment_plan.items():
        if cls not in fpaths_by_class:
            continue

        files = fpaths_by_class[cls]
        allow_overlap = cfg.get("allow_overlap", False)

        # method_cfg: { "pitch_shift": (x, y), "time_stretch": (...) }
        for count, method_cfg in cfg["methods"]:
            if not method_cfg: 
                continue

            if allow_overlap:
                chosen_files = [random.choice(files) for _ in range(count)]
            else:
                chosen_files = random.sample(files, min(count, len(files)))

            for f in chosen_files:
                fpaths_out.append(f)
                tasks.append(method_cfg)

    return fpaths_out, tasks


@click.group()
def cli():
    pass


@cli.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("dpath_out", type=click.Path(exists=True, file_okay=False))
@click.argument("plan_fpath", type=click.Path(exists=True, dir_okay=False))
@click.option("--mode", "-m", type=click.Choice(["img", "audio"]), default="audio")
def save(
    input_path: str,
    dpath_out: str,
    plan_fpath: str,
    mode: Literal["audio", "img"] = "audio",
    sr = MEL_CONFIG.sr
):
    """
    Convert audio file(s) from input_path into augmented Mel-spectrogram images
    according to AUGMENT_PLAN.
    """
    fpaths, labels = collect_data_files(input_path, ".wav")

    with open(plan_fpath, "r", encoding="utf-8") as f:
        augment_plan = yaml.safe_load(f)
    aug_fpaths, tasks = generate_augment_plan(fpaths, labels, augment_plan)

    for f, methods in zip(aug_fpaths, tasks):
        base_name = Path(f).stem
        suffix = "__" + params_to_str(methods) if methods else ""
        ext = ".wav" if mode == "audio" else ".png"

        fpath_out = str(Path(dpath_out) / f"{base_name}{suffix}{ext}")

        try:
            save_augmented_audio(
                fpath_in=f,
                fpath_out=fpath_out,
                methods=methods,
                sr_to=sr
            )
        except Exception as e:
            print(f"Error saving {f}: {e}")


if __name__ == "__main__":
    cli()
    # a = {
    #     "danbau": {
    #         "allow_overlap": False,
    #         "methods": [
    #             (5, {"pitch_shift": (-0.5, 0.5), "time_stretch": ()}),
    #             (1, {"add_noise": (0, 0.01)})
    #         ]
    #     },
    #     "dannhi": {
    #         "allow_overlap": False,
    #         "methods": [
    #             (3, {"add_noise": (0, 0.01)})
    #         ]
    #     }
    # }

    # # print("\n".join(map(str, collect_audio_files("rawdata/train"))))
    # # print("\n".join(map(str, collect_audio_files("rawdata/train/danbau", "danbau"))))
    # # print("\n".join(map(str, collect_audio_files("rawdata/train/danbau/danbau001.wav", "danbau"))))
    # save_augment(
    #     "rawdata/train",
    #     "test",
    #     a
    # )
    # print("\n".join(map(str, generate_augment_plan(
    #     [
    #         ("rawdata/train/danbau/danbau001.wav", "danbau"),
    #         ("rawdata/train/danbau/danbau002.wav", "danbau"),
    #         ("rawdata/train/danbau/danbau003.wav", "danbau"),
    #         ("rawdata/train/danbau/danbau004.wav", "danbau"),
    #         ("rawdata/train/danbau/dannhi005.wav", "dannhi"),
    #         ("rawdata/train/danbau/dannhi006.wav", "dannhi"),
    #     ],
    #     a
    # ))))