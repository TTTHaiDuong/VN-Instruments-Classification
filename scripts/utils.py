import random, os
from rich.console import Console
from rich.table import Table
from collections import defaultdict
from pathlib import Path


class SafeDict(dict):
    """Dict an toàn cho format string — không lỗi khi thiếu key."""
    def __missing__(self, key):
        # Có thể return "" hoặc tên placeholder gốc
        return f"{{{key}}}"


def to_mm_ss(seconds: int) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def unique_filename(base_path: str, padding=0):
    """
    Returns a unique file path if the specified file already exists by appending a numeric suffix.
    For example: "file001.txt", "file002.txt".

    Parameters:
        base_path (str): The base file path.
        padding (int): Number of digits to pad the index with (default is 0).

    Returns:
        str: A unique file path that does not conflict with existing files.
    """
    if not os.path.exists(base_path):
        return base_path
    
    base, ext = os.path.splitext(base_path)
    counter = 1
    while True:
        index_str = str(counter).rjust(padding, "0")

        new_path = f"{base}_{index_str}{ext}"
        if not os.path.exists(new_path):
            return new_path
        counter += 1


def collect_data_files(
    input_path: str,
    file_format: str | tuple[str, ...] | None = None
) -> tuple[list[str], list[str]]:
    """Trả về danh sách các đường dẫn mẫu âm thanh và nhãn (file_path, class_name).

    Args:
        input_path (str): Đường dẫn file, hoặc thư mục, có thể là:
        - Đường dẫn file âm thanh cụ thể
        - Đường dẫn thư mục chứa các loại âm thanh cùng nhãn
        - Đường dẫn thư mục chứa các loại âm thanh được phân loại ra từng thư mục riêng (danbau, dannhi,...)
    """
    src_path = Path(input_path)

    if file_format:
        if isinstance(file_format, str):
            file_format = (file_format,)
        file_format = tuple(
            ext if ext.startswith(".") else f".{ext}" for ext in file_format
        )

    if src_path.is_file():
        return [str(src_path)], [src_path.parent.name]

    fpaths, labels = [], []
    subdirs = [d for d in src_path.iterdir() if d.is_dir()] or [src_path]

    for cls in subdirs:
        for f in cls.iterdir():
            if f.is_file() and (not file_format or f.suffix in file_format):
                fpaths.append(str(f))
                labels.append(cls.name)

    return fpaths, labels


def limit_per_class(
    fpaths: list[str], 
    labels: list[str], 
    max_per_class: int, 
    shuffle = False, 
    seed: int | None = None,
    verbose = False
):
    """
    Giới hạn số file tối đa cho mỗi lớp.

    Parameters
    ----------
    file_paths : list of str
        Danh sách đường dẫn file.
    labels : list of str
        Danh sách nhãn (cùng chiều với file_paths).
    max_per_class : int
        Số lượng tối đa file giữ lại cho mỗi lớp.
    shuffle : bool
        Nếu True thì random chọn file, nếu False thì lấy theo thứ tự ban đầu.
    seed : int or None
        Seed cho random (đảm bảo reproducibility).

    Returns
    -------
    filtered_paths : list of str
        Danh sách file đã được lọc.
    filtered_labels : list of str
        Danh sách label tương ứng với file đã được lọc.
    """
    # Gom file theo nhãn
    fpaths_by_label = defaultdict(list)
    for f, label in zip(fpaths, labels):
        fpaths_by_label[label].append(f)

    filtered_fpaths, filtered_labels = [], []

    logs = []
    # Lọc file cho từng nhãn
    for label, paths in fpaths_by_label.items():
        if shuffle:
            random.seed(seed)
            sampled_paths = random.sample(paths, min(max_per_class, len(paths)))
        else:
            sampled_paths = paths[:max_per_class]

        samples_len = len(sampled_paths)
        filtered_fpaths.extend(sampled_paths)
        filtered_labels.extend([label] * samples_len)

        if verbose:
            logs.append({
                "Label": label, 
                "Colon": ":", 
                "Count": samples_len
            })

    if verbose:
        print_table(logs, "Samples Found", show_header=False, box=False, column_styles={ "Label": {"justify": "right" }})

    return filtered_fpaths, filtered_labels


def print_table(
    rows: list[dict],
    title: str,
    column_styles: dict[str, dict] | None = None,
    **overrides
):
    default_style = dict(
        title=title,
        padding=(0, 2), # (top/bottom, left/right) => mỗi cột có 2 space bên lề
        show_lines=False,
        show_edge=False
    )

    table = Table(**{**default_style, **overrides})

    columns = list(rows[0].keys())

    default_col_style = {"justify": "center", "no_wrap": False}

    for col in columns:
        col_style = {**default_col_style, **(column_styles.get(col, {}) if column_styles else {})}
        table.add_column(str(col), **col_style)

    for row in rows:
        values = [str(row.get(col, "")) for col in columns]
        table.add_row(*values)

    Console().print(table)


def register(registry, name):
    def decorator(func):
        registry[name] = func
        return func
    return decorator