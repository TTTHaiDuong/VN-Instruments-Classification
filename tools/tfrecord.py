import click
import librosa
import scripts.tfrecord as tfr
from scripts.utils import collect_files
from scripts.mel_spectrogram import to_mel
from config.general import INSTRUMENTS
from rich.progress import Progress


def save(
    input_path: str,
    out_path: str,
    class_map,
    label: str | None = None,
    verbose: bool = False
):
    paths, labels = collect_files(input_path, label)

    mels, labels,  = [], []
    with Progress() as progress:
        task = progress.add_task(
            "[magenta]Extracting Mel-spectrograms...", total=len(paths)
        )

        for path, label in zip(paths, labels):
            if not label or not isinstance(class_map._label_to_index[label], int):
                raise ValueError(f"Invalid class name {path} -> {label}")

            y, sr = librosa.load(path)
            mels.append(to_mel(y, sr))
            labels.append(class_map.label_to_index(label))

            progress.update(task, advance=1)

    tfr.write_tfrecord(mels, labels, out_path, None, verbose)


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("out_path", type=click.Path(dir_okay=False))
@click.option("--label", "-l", type=click.Choice(INSTRUMENTS.labels()))
@click.option("--verbose", "-v", is_flag=True)
def main(
    input_path: str,
    out_path: str,
    class_map = INSTRUMENTS,
    label: str | None = None,
    verbose: bool = False
):
    save(input_path, out_path, class_map, label, verbose)


if __name__ == "__main__":
    main()