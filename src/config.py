"""
config.py — Shared configuration for the DJ Mixing Pathfinding System.
"""

from __future__ import annotations

import os
from pathlib import Path

# Resolve the project root relative to this file (Auto-DJ/)
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Root directory containing audio files.  Override with the SONGS_DIRECTORY
# environment variable, or change the default below to point at your
# music library.  The Dash app loads the graph from this directory on
# startup (or from the JSON cache if it already exists).
SONGS_DIRECTORY: str = str(Path(os.environ.get(
    "SONGS_DIRECTORY",
    str(_PROJECT_ROOT / ".." / 'Playlists'),
)).resolve())

# Directory where Auto-DJ generated setlists are saved.
# Kept as a subfolder of SONGS_DIRECTORY so setlists travel with the library,
# but explicitly excluded from graph analysis to avoid re-scanning the copies.
SETLISTS_DIRECTORY: Path = Path(SONGS_DIRECTORY) / "Auto-dj setlists"

# Disk cache for the serialised graph.  If this file exists the app
# skips the expensive directory-scan + embedding step entirely.
CACHE_PATH: Path = Path(
    os.environ.get(
        "CACHE_PATH",
        str(_PROJECT_ROOT / "cache" / "dj_graph_cache.pkl"),
    )
)

# Legacy JSON cache path — used for automatic migration to pickle.
_LEGACY_JSON_CACHE: Path = _PROJECT_ROOT / "cache" / "dj_graph_cache.json"

# Separate cache for the UMAP-reduced graph (32-dim cosine embeddings).
# Stored alongside the main cache so both can coexist without conflict.
UMAP_CACHE_PATH: Path = CACHE_PATH.parent / "dj_graph_umap_cache.pkl"

# ---------------------------------------------------------------------------
# Audio chunking / embedding parameters
# ---------------------------------------------------------------------------

# Phrase-aligned chunking bounds (seconds).  Chunks shorter than the minimum
# are merged with their neighbours; chunks longer than the maximum are split
# with a fixed stride.
CHUNK_MIN_SEC: int = 12
CHUNK_MAX_SEC: int = 30

# Estimated RAM consumed per parallel audio-analysis worker process.
# Used by _optimal_worker_count() in utils.py to cap workers on low-RAM
# systems (e.g. 4 GB WSL instances or shared CI machines).
RAM_PER_WORKER_GB: float = 1.5


