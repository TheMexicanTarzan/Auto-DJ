"""setlist_saver.py — Copy a DJ setlist's audio files into a user-chosen directory."""

from __future__ import annotations

import shutil
import tkinter as tk
from pathlib import Path
from tkinter import filedialog


def save_setlist(track_paths: list[str], setlist_name: str) -> str:
    """Prompt the user for a destination folder and copy tracks into a numbered subfolder.

    Opens a native OS folder-picker dialog (hidden Tk root window).

    Args:
        track_paths: Absolute file paths in setlist order.
        setlist_name: Used as the output subfolder name.

    Returns:
        Absolute path of the created output folder.

    Raises:
        RuntimeError: If the user cancels the dialog without choosing a folder.
        OSError: If a file cannot be copied.
    """
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes("-topmost", True)

    dest_parent = filedialog.askdirectory(
        parent=root,
        title="Select destination folder for setlist",
    )
    root.destroy()

    if not dest_parent:
        raise RuntimeError("Cancelled by user.")

    # Sanitise name for use as a filesystem directory
    safe_name = (
        "".join(c if c.isalnum() or c in " _-" else "_" for c in setlist_name).strip()
    ) or "Setlist"

    output_dir = Path(dest_parent) / safe_name
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, src_path_str in enumerate(track_paths, start=1):
        src = Path(src_path_str)
        if not src.is_file():
            continue
        dest_name = f"{i:02d} - {src.name}"
        shutil.copy2(src, output_dir / dest_name)

    return str(output_dir)
