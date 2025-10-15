import click
import librosa
import scripts.tfrecord as tfr
from scripts.utils import collect_data_files
from scripts.mel_spectrogram import to_mel
from rich.progress import Progress
from config.general import INSTRUMENTS


def make_tfrecord(
    input_path: str,
    fpath_out: str,
    class_idxs_map: dict[str, int],
    verbose: bool = False
):
    fpaths, labels = collect_data_files(input_path, ".wav")

    mels, class_idxs,  = [], []
    with Progress() as progress:
        task = progress.add_task(
            "[magenta]Extracting Mel-spectrograms...", total=len(fpaths)
        )

        for path, label in zip(fpaths, labels):
            if not label in class_idxs_map.keys():
                raise ValueError(f"Invalid class label {path} -> {label}")

            y, sr = librosa.load(path)
            mels.append(to_mel(y, sr))
            class_idxs.append(class_idxs_map[label])

            progress.update(task, advance=1)

    tfr.write_tfrecord(mels, class_idxs, fpath_out, None, verbose)


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("out_path", type=click.Path(dir_okay=False))
@click.option("--verbose", "-v", is_flag=True)
def cli(
    input_path: str,
    out_path: str,
    class_map = INSTRUMENTS,
    verbose: bool = False
):
    make_tfrecord(input_path, out_path, class_map.indexes_map(), verbose)


if __name__ == "__main__":
    cli()
    # save(
    #     input_path="rawdata/train",
    #     out_path="test/train.tfrecord",
    #     class_map=INSTRUMENTS,
    #     verbose=True
    # )