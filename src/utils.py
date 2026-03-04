"""
utils.py — File-system utilities for the DJ Mixing Pathfinding System.

Provides a directory scanner that recursively finds audio files,
deduplicates them by content hash, and returns fully-analysed Song objects.
"""

from __future__ import annotations

import hashlib
import logging
import os
from multiprocessing import Pool
from pathlib import Path

from src.models import Song, analyse_audio, compute_embeddings_batch

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


def _analyse_one(path_str: str) -> tuple | None:
    """
    Wrapper for multiprocessing: analyse a single audio file.

    Returns an AudioAnalysis on success, None on failure.
    Must be a top-level function for pickling.
    """
    try:
        return analyse_audio(path_str)
    except Exception:
        logger.exception("Failed to analyse '%s', skipping.", path_str)
        return None


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
        Audio analysis is split into two phases:
        - Phase 2a: BPM + key detection run in a multiprocessing pool
          (CPU-bound librosa work benefits from true parallelism).
          Workers are capped at 3 to limit memory (each child loads
          audio into memory independently).
        - Phase 2b: CLAP embeddings are computed in the main process
          using batched forward passes (avoids duplicating the ~600 MB
          model across worker processes).

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

    # --- Phase 2a: parallel BPM + key analysis (multiprocessing) ---
    # Cap at 3 workers — each child loads full audio into memory and
    # librosa / numpy are CPU-bound, so true parallelism helps.
    max_workers = min(3, len(unique_paths), os.cpu_count() or 1)
    analyses = []

    path_strs = [str(p) for p in unique_paths]

    if max_workers <= 1 or len(unique_paths) <= 1:
        # Fall back to sequential for trivial cases (avoids pool overhead)
        for idx, path_str in enumerate(path_strs, 1):
            result = _analyse_one(path_str)
            if result is not None:
                analyses.append(result)
            if progress_callback:
                progress_callback(idx + skipped, total, Path(path_str).name)
    else:
        completed = 0
        with Pool(processes=max_workers) as pool:
            for result in pool.imap_unordered(_analyse_one, path_strs):
                completed += 1
                if result is not None:
                    analyses.append(result)
                fname = result.filename if result is not None else "unknown"
                if progress_callback:
                    progress_callback(completed + skipped, total, fname)

    if not analyses:
        logger.info(
            "Scan complete: 0 song(s) loaded, %d duplicate(s) skipped.",
            skipped,
        )
        return []

    # --- Phase 2b: batch CLAP embeddings (main process) ---
    logger.info(
        "Computing CLAP embeddings for %d track(s) in batches...",
        len(analyses),
    )
    audio_list = [(a.audio, a.sr) for a in analyses]
    embeddings = compute_embeddings_batch(audio_list, batch_size=8)

    # --- Build Song objects ---
    songs: list[Song] = []
    for analysis, embedding in zip(analyses, embeddings):
        songs.append(Song(
            file_path=analysis.file_path,
            filename=analysis.filename,
            bpm=analysis.bpm,
            key=analysis.key,
            embedding=embedding,
            beat_times=analysis.beat_times,
            downbeat_times=analysis.downbeat_times,
        ))

    logger.info(
        "Scan complete: %d song(s) loaded, %d duplicate(s) skipped.",
        len(songs),
        skipped,
    )
    return songs
