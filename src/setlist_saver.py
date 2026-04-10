"""setlist_saver.py — Save a DJ setlist to the Auto-dj setlists directory."""

from __future__ import annotations

import shutil
from pathlib import Path


def save_setlist(track_paths: list[str], setlist_name: str, setlists_dir: Path) -> str:
    """Copy ordered track files into a numbered subfolder of *setlists_dir*.

    Args:
        track_paths: Absolute file paths in setlist order.
        setlist_name: Used as the output subfolder name (sanitised for the
            file system).
        setlists_dir: Parent directory that holds all saved setlists
            (created automatically if it does not exist).

    Returns:
        Absolute path of the created output folder.

    Raises:
        OSError: If a file cannot be read or written.
    """
    safe_name = (
        "".join(c if c.isalnum() or c in " _-" else "_" for c in setlist_name).strip()
    ) or "Setlist"

    output_dir = setlists_dir / safe_name
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, src_path_str in enumerate(track_paths, start=1):
        src = Path(src_path_str)
        if not src.is_file():
            continue
        dest_name = f"{i:02d} - {src.name}"
        shutil.copy2(src, output_dir / dest_name)

    return str(output_dir)
