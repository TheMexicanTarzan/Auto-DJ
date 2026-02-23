"""
utils.py — File-system utilities for the DJ Mixing Pathfinding System.

Provides a directory scanner that recursively finds audio files,
deduplicates them by content hash, and returns fully-analysed Song objects.
"""

from __future__ import annotations

import hashlib
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def _file_hash(path: Path, chunk_size: int = 65_536) -> str:
    """
    Compute a SHA-256 hash of a file's contents.

    Reading in 64 KiB chunks (up from 8 KiB) better utilises OS read-ahead
    buffers and reduces syscall overhead for large audio files while keeping
    memory usage constant.
    """
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            sha.update(chunk)
    return sha.hexdigest()


def scan_directory(
    directory: str | Path,
    progress_callback: callable | None = None,
) -> list[Song]:
    """
    Recursively scan *directory* for audio files and return a deduplicated
    list of Song objects, each fully analysed (BPM, key, embedding).

    Deduplication:
        We hash every file's contents with SHA-256.  If two files share the
        same hash they are byte-identical, so we keep only the first one we
        encounter and log a warning for the duplicate.

    Parallelism:
        Audio analysis is the dominant cost.  After the (fast, sequential)
        deduplication pass, unique files are analysed in parallel using a
        thread pool.  The CLAP model and librosa release the GIL during
        their C/CUDA kernels, so threads provide genuine concurrency here.

    Args:
        directory: Root folder to scan.
        progress_callback: Optional ``fn(current, total, filename)``
            invoked after each file is processed (whether success, skip,
            or failure).  Useful for UI progress reporting.

    Returns:
        A list of Song instances, one per unique audio file found.

    Raises:
        NotADirectoryError: If *directory* doesn't exist or isn't a folder.
    """
    root = Path(directory).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Not a valid directory: {root}")

    seen_hashes: dict[str, Path] = {}  # hash → first path we saw it at
    unique_paths: list[Path] = []
    skipped = 0

    # Walk the tree, sorted for deterministic ordering across runs
    audio_files = sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    total = len(audio_files)
    logger.info("Found %d audio file(s) in '%s'.", total, root)

    # --- Phase 1: fast sequential deduplication pass ---
    for path in audio_files:
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
        unique_paths.append(path)

    # --- Phase 2: parallel audio analysis ---
    # Use min of CPU count and unique files to avoid over-subscription.
    # Cap at 4 workers — each analysis loads a full audio file into memory
    # and the CLAP model is shared, so more threads increase contention.
    max_workers = min(4, len(unique_paths), os.cpu_count() or 1)
    songs: list[Song] = []

    if max_workers <= 1 or len(unique_paths) <= 1:
        # Fall back to sequential for trivial cases (avoids pool overhead)
        for idx, path in enumerate(unique_paths, 1):
            try:
                songs.append(Song.from_file(path))
            except Exception:
                logger.exception("Failed to analyse '%s', skipping.", path)
            if progress_callback:
                progress_callback(idx + skipped, total, path.name)
    else:
        # Map futures → original index so we can report progress in order
        completed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_path = {
                pool.submit(Song.from_file, p): p for p in unique_paths
            }
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                completed += 1
                try:
                    songs.append(future.result())
                except Exception:
                    logger.exception("Failed to analyse '%s', skipping.", path)
                if progress_callback:
                    progress_callback(completed + skipped, total, path.name)

    logger.info(
        "Scan complete: %d song(s) loaded, %d duplicate(s) skipped.",
        len(songs),
        skipped,
    )
    return songs
