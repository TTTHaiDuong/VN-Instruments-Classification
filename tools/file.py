import click, os
from pathlib import Path
from typing import Callable


@click.group()
def cli():
    """Tiện ích xử lý file âm thanh."""
    pass


@cli.command()
@click.argument("dir_path", type=click.Path(exists=True, file_okay=False))
@click.argument("prefix", type=str)
@click.option("--start", "-s", type=int, default=0)
@click.option("--padding", "-p", type=int, default=3)
def rename(
    dir_path: str | Path,
    prefix: str,
    start: int = 0,
    padding: int = 0,
    predicate: Callable[[str], bool] = lambda f: True
):
    """
    Rename all files in the specified folder to the format [prefix][index].[original_extension],
    sorted by original filename in alphabetical order.

    Parameters:
        dir_path (str): Path to the folder containing the files to rename.
        prefix (str): Prefix for the new filenames.
        start (int): Starting index (default is 0).
        padding (int): Number of digits to pad the index with (default is 0).
        predicate (Callable): Function to filter which files to rename. Defaults to all files.
    """
    dir_path = Path(dir_path)

    # Sort files by name in alphabetical order and filter by `predicate`
    file_paths = sorted(
        f for f in dir_path.iterdir()
        if f.is_file() and predicate(f.name)
    )

    counter = start
    for i, f in enumerate(file_paths):
        index_str = str(counter).rjust(padding, "0")
        new_file = dir_path / f"{prefix}{index_str}{f.suffix}"

        try:
            f.rename(new_file)
            print(f"{i}. Renamed {f} → {new_file}")
        except FileExistsError:
            print(f"{i}. At {f}: File {new_file} already exists. Skipped.")

        counter += 1


def move_files(
    src_path: str, 
    dest_dir: str,
    predicate: Callable[[str], bool] = lambda f: True
):
    """
    Move a file or all files from a source directory to a destination directory.

    Parameters:
        src_path (str): Path to the source file or directory.
        dest_dir (str): Path to the destination directory.
    """
    if not os.path.exists(src_path):
        print(f"Error: Source path {src_path} does not exist.")
        return

    os.makedirs(dest_dir, exist_ok=True)

    # If the source is a file
    if os.path.isfile(src_path):
        filename = os.path.basename(src_path)
        destination_file = os.path.join(dest_dir, filename)
        try:
            shutil.move(src_path, destination_file)
            print(f"Moved: {filename}")
        except Exception as e:
            print(f"Error moving {filename}: {e}")

    # If the source is a directory
    elif os.path.isdir(src_path):
        for filename in os.listdir(src_path):
            if not predicate(filename):
                continue
            source_file = os.path.join(src_path, filename)
            destination_file = os.path.join(dest_dir, filename)
            if os.path.isfile(source_file):
                try:
                    shutil.move(source_file, destination_file)
                    print(f"Moved: {filename}")
                except Exception as e:
                    print(f"Error moving {filename}: {e}")
    else:
        print(f"Error: {src_path} is not a valid file or directory.")



def count(dir_path: str):
    """Count the number of files in a directory."""
    count = 0
    for _, _, files in os.walk(dir_path):
        count += len(files)
    return count