import click
import os
from pydub import AudioSegment


def normalize_dbfs(audio, target_dbfs = -20.0):
    change_in_dBFS = target_dbfs - audio.dBFS
    return audio.apply_gain(change_in_dBFS)


@click.command()
@click.argument("dir_path", type=click.Path(exists=True, file_okay=False))
@click.argument("output_dir", type=click.Path(file_okay=False))
@click.option("--target_dbfs", "-t", type=float, default=-20.0)
def normalize_volume(
    dir_path: str, 
    output_dir: str, 
    target_dbfs
):
    os.makedirs(output_dir, exist_ok=True)

    for f in os.listdir(dir_path):
        filepath = os.path.join(dir_path, f)

        audio = AudioSegment.from_file(filepath)
        normalized_audio = normalize_dbfs(audio, target_dbfs)

        outpath = os.path.join(output_dir, f)
        normalized_audio.export(outpath, format="wav")


if __name__ == "__main__":
    normalize_volume()