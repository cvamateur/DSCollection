import os
import sys
import contextlib


def check_path(path: str, existence: bool = True):
    path = os.path.abspath(os.path.expanduser(path))
    if os.path.exists(path) != existence:
        abj = "not" if existence else "already"
        Error = FileNotFoundError if existence else FileExistsError
        prefix = f"Path {abj} exist: "
        raise Error(prefix + path)
    return path


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

