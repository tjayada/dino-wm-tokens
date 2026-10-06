"""Where things live: a fast local data root, an optional persistent mirror, and the results folder."""

import os
import shutil
from pathlib import Path


def data_root() -> Path:
    """`$STABLEWM_HOME`: datasets/, checkpoints/, features/ (fast local disk)."""
    return Path(os.environ.get("STABLEWM_HOME", Path.home() / ".stable-wm"))


def persist_root() -> Path | None:
    """`$DWT_PERSIST`: a mirror that survives sessions (e.g. a Google Drive folder), or None."""
    value = os.environ.get("DWT_PERSIST")
    return Path(value) if value else None


def results_root() -> Path:
    return Path(os.environ.get("DWT_RESULTS", "results"))


def persist(path: Path, kind: str) -> None:
    """Copy a file or folder under `data_root()/<kind>/` to the mirror, if there is one."""
    root = persist_root()
    if root is None:
        return
    dest = root / kind / path.relative_to(data_root() / kind)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if path.is_dir():
        shutil.copytree(path, dest, dirs_exist_ok=True)
    else:
        shutil.copy(path, dest)


def restore(path: Path, kind: str) -> bool:
    """Bring a file or folder back from the mirror if it is missing locally. Returns whether it exists now."""
    if path.exists():
        return True
    root = persist_root()
    if root is None:
        return False
    src = root / kind / path.relative_to(data_root() / kind)
    if not src.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, path)
    else:
        shutil.copy(src, path)
    return True
