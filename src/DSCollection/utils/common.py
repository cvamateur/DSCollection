import os
import sys
import contextlib

_IMG_EXT = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp')
_VID_EXT = (".mp4", ".mkv", ".flv", ".avi", ".xmv", ".webm")


def check_path(path: str, existence: bool = True):
    path = os.path.abspath(os.path.expanduser(path))
    if os.path.exists(path) != existence:
        abj = "not" if existence else "already"
        Error = FileNotFoundError if existence else FileExistsError
        prefix = f"Path {abj} exist: "
        raise Error(prefix + path)
    return path


def is_image(path: str) -> bool:
    return path.lower().endswith(_IMG_EXT)


def is_video(path: str) -> bool:
    return path.lower().endswith(_VID_EXT)


@contextlib.contextmanager
def silent():
    def _to_null(*args, **kwargs):
        pass

    stdout_write = sys.stdout.write
    stderr_write = sys.stderr.write
    sys.stdout.write = _to_null
    sys.stderr.write = _to_null
    yield
    sys.stdout.write = stdout_write
    sys.stderr.write = stderr_write

