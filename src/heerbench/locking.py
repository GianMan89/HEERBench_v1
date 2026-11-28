"""Non-blocking advisory file locks on Windows and POSIX."""
from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path


@contextmanager
def exclusive_file_lock(path: Path, message: str, *, error_type=RuntimeError):
    """Hold an OS-managed exclusive lock until exit, including on exceptions.

    Windows uses a one-byte range at offset zero; Windows permits locking
    beyond EOF, so empty lock files from previous runs work unchanged.
    Never remove the lock file while other processes may be using it.
    """
    with path.open("a+b") as stream:
        fd = stream.fileno()
        if os.name == "nt":
            import msvcrt

            def acquire():
                stream.seek(0)
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)

            def release():
                stream.seek(0)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            def acquire():
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

            def release():
                fcntl.flock(fd, fcntl.LOCK_UN)

        try:
            acquire()
        except OSError as exc:
            raise error_type(f"{message} (Lock error: {exc})") from exc
        try:
            yield
        finally:
            release()
