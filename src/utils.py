"""
utils.py — File-system utilities for the DJ Mixing Pathfinding System.

Provides a directory scanner that recursively finds audio files,
deduplicates them by content hash, and returns fully-analysed Song objects.
"""

from __future__ import annotations

import hashlib
import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
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

# Maximum number of parallel audio-analysis worker processes.
# Capped at 8 to avoid excessive memory usage on constrained systems.
_N_ANALYSIS_WORKERS: int = min(os.cpu_count() or 1, 8)

# Spawn context used for ProcessPoolExecutor: safe when forking from a
# multi-threaded uvicorn process or after CUDA has been initialised.
_MP_SPAWN_CTX = multiprocessing.get_context("spawn")


def _fingerprint_match(fp_a: str, fp_b: str, threshold: float = 0.8) -> bool:
    """
    Compare two Chromaprint fingerprints and return True if they
    represent the same audio content (similarity >= *threshold*).

    Uses bitwise Hamming similarity over the decoded integer arrays.
    Returns False if either fingerprint is empty or cannot be decoded.
    """
    if not fp_a or not fp_b:
        return False

    try:
        import chromaprint
    except ImportError:
        return False

    try:
        raw_a, _ = chromaprint.decode_fingerprint(fp_a)
        raw_b, _ = chromaprint.decode_fingerprint(fp_b)
    except Exception:
        return False

    if not raw_a or not raw_b:
        return False

    # Compare over the overlapping portion
    length = min(len(raw_a), len(raw_b))
    if length == 0:
        return False

    matching_bits = 0
    total_bits = length * 32  # each element is a 32-bit integer
    for a, b in zip(raw_a[:length], raw_b[:length]):
        # XOR gives bits that differ; popcount gives the count
        xor = a ^ b
        matching_bits += 32 - bin(xor & 0xFFFFFFFF).count("1")

    similarity = matching_bits / total_bits
    return similarity >= threshold


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


def _under_excluded(path: Path, exclude_dirs: set[Path]) -> bool:
    """Return True if *path* is inside any of the *exclude_dirs* subtrees."""
    for excl in exclude_dirs:
        try:
            path.relative_to(excl)
            return True
        except ValueError:
            pass
    return False


def scan_directory(
    directory: str | Path,
    progress_callback: callable | None = None,
    exclude_dirs: set[Path] | None = None,
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

    _excluded: set[Path] = {e.resolve() for e in (exclude_dirs or set())}

    # Walk the tree, sorted for deterministic ordering across runs
    audio_files = sorted(
        p for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_EXTENSIONS
        and not _under_excluded(p, _excluded)
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

    # Reverse map: path → hash (for attaching hashes to Song objects)
    path_to_hash: dict[Path, str] = {p: h for h, p in seen_hashes.items()}

    # --- Phase 2: pipelined analyse → embed → build Song loop ---
    #
    # The next chunk's analysis jobs are submitted to the process pool
    # *before* compute_embeddings_batch blocks the main thread on the GPU.
    # This lets CPU workers for chunk N+1 run in parallel with GPU embedding
    # of chunk N, hiding the ~5–8 s analysis cost inside the ~52 s embed.
    #
    # Peak memory across all processes: up to 2 × _CHUNK_SIZE waveforms
    # (current chunk in the main process + next chunk in worker processes).
    songs: list[Song] = []
    processed = 0

    chunks = [
        unique_paths[s : s + _CHUNK_SIZE]
        for s in range(0, len(unique_paths), _CHUNK_SIZE)
    ]

    with ProcessPoolExecutor(
        max_workers=_N_ANALYSIS_WORKERS,
        mp_context=_MP_SPAWN_CTX,
    ) as executor:
        if chunks:
            # Pre-submit the first chunk so workers start immediately.
            current_futures = [
                executor.submit(analyse_audio, str(p)) for p in chunks[0]
            ]

            for chunk_idx, chunk_paths in enumerate(chunks):
                # Submit the NEXT chunk's analysis jobs before embedding the
                # current chunk — CPU workers overlap with GPU inference.
                if chunk_idx + 1 < len(chunks):
                    next_futures = [
                        executor.submit(analyse_audio, str(p))
                        for p in chunks[chunk_idx + 1]
                    ]
                else:
                    next_futures = []

                # Collect current chunk's analysis results (submission order
                # preserved so embeddings align with analyses below).
                chunk_analyses = []
                for path, future in zip(chunk_paths, current_futures):
                    processed += 1
                    try:
                        analysis = future.result()
                        chunk_analyses.append(analysis)
                    except Exception:
                        logger.exception("Failed to analyse '%s', skipping.", path)
                    if progress_callback:
                        progress_callback(processed + skipped, total, path.name)

                if chunk_analyses:
                    # Embed current chunk — next chunk's workers run in parallel.
                    audio_list = [(a.audio, a.sr) for a in chunk_analyses]
                    embeddings = compute_embeddings_batch(audio_list)

                    # Build Song objects; raw audio released when chunk_analyses
                    # goes out of scope at the next iteration.
                    for analysis, embedding in zip(chunk_analyses, embeddings):
                        songs.append(Song(
                            file_path=analysis.file_path,
                            filename=analysis.filename,
                            bpm=analysis.bpm,
                            key=analysis.key,
                            embedding=embedding,
                            beat_times=analysis.beat_times,
                            downbeat_times=analysis.downbeat_times,
                            content_hash=path_to_hash.get(Path(analysis.file_path), ""),
                            fingerprint=analysis.fingerprint,
                            duration_sec=analysis.duration_sec,
                        ))

                current_futures = next_futures

    # --- Phase 3: fingerprint-based deduplication ---
    # Catches cross-format / cross-album duplicates that SHA-256 misses
    # (e.g. same song as .mp3 and .flac with different binary content).
    accepted: list[Song] = []
    fp_dupes = 0
    for song in songs:
        if not song.fingerprint:
            accepted.append(song)
            continue
        is_dup = False
        for existing in accepted:
            if existing.fingerprint and _fingerprint_match(song.fingerprint, existing.fingerprint):
                logger.warning(
                    "Skipping fingerprint duplicate: '%s' (matches '%s').",
                    song.filename,
                    existing.filename,
                )
                fp_dupes += 1
                is_dup = True
                break
        if not is_dup:
            accepted.append(song)

    logger.info(
        "Scan complete: %d song(s) loaded, %d hash duplicate(s) skipped, "
        "%d fingerprint duplicate(s) skipped.",
        len(accepted),
        skipped,
        fp_dupes,
    )
    return accepted


def discover_changes(
    directory: str | Path,
    known_hashes: set[str],
    exclude_dirs: set[Path] | None = None,
) -> tuple[list[Path], set[str]]:
    """
    Scan *directory* for audio files and compare against *known_hashes*
    to find new files and detect removed files.

    Returns:
        A tuple of (new_paths, removed_hashes) where:
          - new_paths: list of Paths to files whose content hash is not
            in *known_hashes* (genuinely new content).
          - removed_hashes: set of hashes from *known_hashes* that no
            longer match any file on disk.
    """
    root = Path(directory).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Not a valid directory: {root}")

    _excluded: set[Path] = {e.resolve() for e in (exclude_dirs or set())}

    audio_files = sorted(
        p for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_EXTENSIONS
        and not _under_excluded(p, _excluded)
    )

    on_disk_hashes: set[str] = set()
    new_paths: list[Path] = []
    seen: set[str] = set()  # dedup within this scan

    for path in audio_files:
        h = _file_hash(path)
        if h in seen:
            logger.warning("Skipping duplicate: '%s'.", path)
            continue
        seen.add(h)
        on_disk_hashes.add(h)
        if h not in known_hashes:
            new_paths.append(path)

    removed_hashes = known_hashes - on_disk_hashes

    logger.info(
        "Directory diff: %d new file(s), %d removed file(s).",
        len(new_paths),
        len(removed_hashes),
    )
    return new_paths, removed_hashes


def analyse_new_songs(
    paths: list[Path],
    progress_callback: callable | None = None,
    known_fingerprints: dict[str, str] | None = None,
) -> list[Song]:
    """
    Analyse a list of audio file paths and return Song objects.

    Same chunked pipeline as scan_directory but without the discovery
    and deduplication steps (callers are expected to pass already-
    deduplicated paths from ``discover_changes``).

    Args:
        paths: Audio file paths to analyse.
        progress_callback: Optional ``fn(current, total, filename)``.
        known_fingerprints: Optional dict mapping fingerprint → file_path
            for songs already in the graph.  New songs whose fingerprint
            matches an existing one are skipped as duplicates.
    """
    if not paths:
        return []

    songs: list[Song] = []
    total = len(paths)
    processed = 0

    # Build path→hash map for these files
    path_to_hash: dict[Path, str] = {}
    for p in paths:
        path_to_hash[p] = _file_hash(p)

    chunks = [paths[s : s + _CHUNK_SIZE] for s in range(0, len(paths), _CHUNK_SIZE)]

    with ProcessPoolExecutor(
        max_workers=_N_ANALYSIS_WORKERS,
        mp_context=_MP_SPAWN_CTX,
    ) as executor:
        if chunks:
            current_futures = [
                executor.submit(analyse_audio, str(p)) for p in chunks[0]
            ]

            for chunk_idx, chunk_paths in enumerate(chunks):
                if chunk_idx + 1 < len(chunks):
                    next_futures = [
                        executor.submit(analyse_audio, str(p))
                        for p in chunks[chunk_idx + 1]
                    ]
                else:
                    next_futures = []

                chunk_analyses = []
                for path, future in zip(chunk_paths, current_futures):
                    processed += 1
                    try:
                        analysis = future.result()
                        chunk_analyses.append(analysis)
                    except Exception:
                        logger.exception("Failed to analyse '%s', skipping.", path)
                    if progress_callback:
                        progress_callback(processed, total, path.name)

                if chunk_analyses:
                    audio_list = [(a.audio, a.sr) for a in chunk_analyses]
                    embeddings = compute_embeddings_batch(audio_list)

                    for analysis, embedding in zip(chunk_analyses, embeddings):
                        songs.append(Song(
                            file_path=analysis.file_path,
                            filename=analysis.filename,
                            bpm=analysis.bpm,
                            key=analysis.key,
                            embedding=embedding,
                            beat_times=analysis.beat_times,
                            downbeat_times=analysis.downbeat_times,
                            content_hash=path_to_hash.get(Path(analysis.file_path), ""),
                            fingerprint=analysis.fingerprint,
                            duration_sec=analysis.duration_sec,
                        ))

                current_futures = next_futures

    # Fingerprint dedup against existing graph songs
    if known_fingerprints:
        accepted: list[Song] = []
        for song in songs:
            if song.fingerprint:
                for known_fp, known_path in known_fingerprints.items():
                    if _fingerprint_match(song.fingerprint, known_fp):
                        logger.warning(
                            "Skipping fingerprint duplicate: '%s' (matches existing '%s').",
                            song.filename,
                            Path(known_path).name,
                        )
                        break
                else:
                    accepted.append(song)
            else:
                accepted.append(song)
        songs = accepted

    logger.info("Analysed %d new song(s).", len(songs))
    return songs
