import click
import yt_dlp
import os
import re


def sanitize_filename(value: str, max_length=50):
    """Loại bỏ ký tự không hợp lệ, giới hạn tên file."""
    value = str(value)
    # Loại bỏ các ký tự cấm trong Windows
    value = re.sub(r'[<>:"/\\|?*]', '', value)
    # Thay nhiều khoảng trắng bằng 1 dấu _
    value = re.sub(r"\s+", "_", value.strip())
    # Giới hạn độ dài tên file (tránh quá 260 ký tự path limit của Windows)
    return value[:max_length]


def download_audio(urls, out_dir="downloads", audio_format="wav", skip=True):
    os.makedirs(out_dir, exist_ok=True)

    def progress_hook(d):
        if d["status"] == "finished":
            click.echo(f"\n✅ Đã tải xong: {d['filename']}")
        elif d["status"] == "error":
            click.echo(f"\n⚠️ Lỗi khi tải: {d.get('filename','unknown')}")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(out_dir, "%(title)s.%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": audio_format,
                "preferredquality": "0",
            }
        ],
        "noplaylist": True,
        "overwrites": not skip,
        "progress_hooks": [progress_hook],
        "windowsfilenames": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for url in urls:
            info = ydl.extract_info(url, download=False)
            # sanitize title ngay tại đây
            info["title"] = sanitize_filename(info["title"])
            ydl.process_info(info)


@click.group()
def cli():
    """Tool tải audio từ YouTube, TikTok, Facebook... bằng yt-dlp."""
    pass


@cli.command()
@click.argument("urls", nargs=-1)
@click.option("-o", "--out-dir", default="downloads", help="Thư mục lưu file audio.")
@click.option("-f", "--format", "audio_format", default="mp3", help="Định dạng audio (mp3, wav, flac...).")
@click.option("--no-skip", is_flag=True, help="Tải lại và ghi đè file đã tồn tại.")
def from_links(urls, out_dir, audio_format, no_skip):
    """Tải audio trực tiếp từ danh sách URL"""
    if not urls:
        click.echo("Bạn phải nhập ít nhất 1 URL.")
        return
    download_audio(list(urls), out_dir, audio_format, skip=not no_skip)


@cli.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("-o", "--out-dir", default="downloads", help="Thư mục lưu file audio.")
@click.option("-f", "--format", "audio_format", default="mp3", help="Định dạng audio (mp3, wav, flac...).")
@click.option("--no-skip", is_flag=True, help="Tải lại và ghi đè file đã tồn tại.")
def from_file(file, out_dir, audio_format, no_skip):
    """Tải audio từ file chứa danh sách URL"""
    with open(file, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]
    if not urls:
        click.echo("File rỗng hoặc không có URL hợp lệ.")
        return
    download_audio(urls, out_dir, audio_format, skip=not no_skip)


if __name__ == "__main__":
    cli()
