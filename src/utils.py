"""
utils.py — File-system utilities for the DJ Mixing Pathfinding System.

Provides a directory scanner that recursively finds audio files,
deduplicates them by content hash, and returns fully-analysed Song objects.
"""

from __future__ import annotations

import hashlib
import logging
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

# Number of songs to analyse before computing embeddings and freeing
# the raw audio.  Keeps peak memory bounded (~5 waveforms + CLAP model).
_CHUNK_SIZE = 5


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

    Memory-bounded chunked processing:
        Songs are processed in small chunks (default 5) to keep memory
        usage bounded on constrained systems (e.g. 4 GB WSL instances):

        For each chunk:
          1. Analyse audio sequentially (BPM, key, beats via madmom/essentia).
          2. Compute CLAP embeddings for the chunk in a batched forward pass.
          3. Build Song objects and discard raw waveform data.

        This ensures at most ~5 waveforms are held in memory at once,
        rather than the entire library.

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

    # --- Phase 2: chunked analyse → embed → build Song loop ---
    songs: list[Song] = []
    processed = 0

    for chunk_start in range(0, len(unique_paths), _CHUNK_SIZE):
        chunk_paths = unique_paths[chunk_start : chunk_start + _CHUNK_SIZE]

        # 2a. Sequential audio analysis (BPM, key, beats)
        chunk_analyses = []
        for path in chunk_paths:
            processed += 1
            try:
                analysis = analyse_audio(str(path))
                chunk_analyses.append(analysis)
            except Exception:
                logger.exception("Failed to analyse '%s', skipping.", path)
            if progress_callback:
                progress_callback(processed + skipped, total, path.name)

        if not chunk_analyses:
            continue

        # 2b. Batch CLAP embeddings for this chunk (main process)
        audio_list = [(a.audio, a.sr) for a in chunk_analyses]
        embeddings = compute_embeddings_batch(audio_list, batch_size=_CHUNK_SIZE)

        # 2c. Build Song objects — raw audio is released when
        #     chunk_analyses goes out of scope at next iteration.
        for analysis, embedding in zip(chunk_analyses, embeddings):
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
