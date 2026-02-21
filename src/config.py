"""
config.py — Shared configuration for the DJ Mixing Pathfinding System.
"""

from __future__ import annotations

from pathlib import Path

# Root directory containing audio files.  Change this to point at your
# music library.  The Dash app loads the graph from this directory on
# startup (or from the JSON cache if it already exists).
SONGS_DIRECTORY: str = r"C:\Users\pelon\OneDrive\Escritorio"

# Disk cache for the serialised graph.  If this file exists the app
# skips the expensive directory-scan + embedding step entirely.
CACHE_PATH: Path = Path(r"C:\Users\pelon\OneDrive\Escritorio\DJ\Auto-DJ\cache\dj_graph_cache.json")
