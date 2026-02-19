"""
utils.py — File-system utilities for the DJ Mixing Pathfinding System.

Provides a directory scanner that recursively finds audio files,
deduplicates them by content hash, and returns fully-analysed Song objects.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from src.models import Song

logger = logging.getLogger(__name__)

# Audio extensions we recognise (all lower-case for comparison)
SUPPORTED_EXTENSIONS: set[str] = {
    ".mp3",
    ".wav",
    ".flac",
    ".ogg",
    ".m4a",
    ".aac",
    ".wma",
    ".aiff",
}


def _file_hash(path: Path, chunk_size: int = 8192) -> str:
    """
    Compute a SHA-256 hash of a file's contents.

    Reading in chunks keeps memory usage constant regardless of file size.
    Two files with identical bytes will always produce the same hash,
    making this a reliable duplicate detector.
    """
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            sha.update(chunk)
    return sha.hexdigest()


def scan_directory(directory: str | Path) -> list[Song]:
    """
    Recursively scan *directory* for audio files and return a deduplicated
    list of Song objects, each fully analysed (BPM, key, embedding).

    Deduplication:
        We hash every file's contents with SHA-256.  If two files share the
        same hash they are byte-identical, so we keep only the first one we
        encounter and log a warning for the duplicate.

    Args:
        directory: Root folder to scan.

    Returns:
        A list of Song instances, one per unique audio file found.

    Raises:
        NotADirectoryError: If *directory* doesn't exist or isn't a folder.
    """
    root = Path(directory).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Not a valid directory: {root}")

    seen_hashes: dict[str, Path] = {}  # hash → first path we saw it at
    songs: list[Song] = []
    skipped = 0

    # Walk the tree, sorted for deterministic ordering across runs
    audio_files = sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    logger.info("Found %d audio file(s) in '%s'.", len(audio_files), root)

    for path in audio_files:
        # --- Duplicate check (by content hash) ---
        file_hash = _file_hash(path)
        if file_hash in seen_hashes:
            logger.warning(
                "Skipping duplicate: '%s' (identical to '%s').",
                path,
                seen_hashes[file_hash],
            )
            skipped += 1
            continue

        seen_hashes[file_hash] = path

        # --- Analyse and build Song object ---
        try:
            song = Song.from_file(path)
            songs.append(song)
        except Exception:
            logger.exception("Failed to analyse '%s', skipping.", path)

    logger.info(
        "Scan complete: %d song(s) loaded, %d duplicate(s) skipped.",
        len(songs),
        skipped,
    )
    return songs
