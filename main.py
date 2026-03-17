"""
main.py — CLI entry point for the DJ Mixing Pathfinding System.

Usage:
    python main.py <music_directory> <source_filename> <target_filename>

Example:
    python main.py ./my_music "track_a.mp3" "track_z.mp3"

This script:
    1. Scans the given directory for audio files.
    2. Analyses each track (BPM, key, CLAP embedding).
    3. Builds a weighted undirected graph of possible transitions.
    4. Finds the smoothest mix path from source to target via Dijkstra.
    5. Prints the path with per-hop transition details.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.config import CACHE_PATH, _LEGACY_JSON_CACHE
from src.graph import DJGraph, NoPathError
from src.metrics import calculate_weight
from src.utils import analyse_new_songs, discover_changes, scan_directory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("auto-dj")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Find the smoothest DJ mix path between two tracks.",
    )
    parser.add_argument(
        "directory",
        help="Root directory containing audio files to scan.",
    )
    parser.add_argument(
        "source",
        help="Filename of the starting track (e.g. 'track_a.mp3').",
    )
    parser.add_argument(
        "target",
        help="Filename of the destination track (e.g. 'track_z.mp3').",
    )
    return parser


def _print_path(path: list, total_cost: float) -> None:
    """Pretty-print the mix path with per-hop details."""
    print("\n" + "=" * 60)
    print("  SHORTEST MIX PATH")
    print("=" * 60)

    for i, song in enumerate(path):
        prefix = "  START " if i == 0 else f"  [{i}]   "
        print(f"{prefix} {song.filename}")
        print(f"          BPM: {song.bpm}  |  Key: {song.key}")

        # Print the transition cost to the next song
        if i < len(path) - 1:
            hop_cost = calculate_weight(song, path[i + 1])
            print(f"            ↓  cost: {hop_cost:.4f}")

    print("-" * 60)
    print(f"  Total path cost: {total_cost:.4f}")
    print(f"  Hops:            {len(path) - 1}")
    print("=" * 60 + "\n")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    # --- 1. Load from cache or scan & build --------------------------------
    if CACHE_PATH.exists():
        logger.info("Loading graph from cache '%s'...", CACHE_PATH)
        dj_graph = DJGraph.load_from_pickle(CACHE_PATH)
    elif _LEGACY_JSON_CACHE.exists():
        logger.info("Migrating legacy JSON cache to pickle...")
        dj_graph = DJGraph.load_from_json(_LEGACY_JSON_CACHE)
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        dj_graph.save_to_pickle(CACHE_PATH)
    else:
        logger.info("Scanning '%s' for audio files...", args.directory)
        songs = scan_directory(args.directory)

        if not songs:
            logger.error("No audio files found in '%s'.", args.directory)
            return 1

        logger.info("Loaded %d track(s).", len(songs))

        # --- 2. Build graph -----------------------------------------------
        logger.info("Building mixing graph...")
        dj_graph = DJGraph.build(songs)

        # --- Save to cache for future runs --------------------------------
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        dj_graph.save_to_pickle(CACHE_PATH)

    # --- Incremental sync: detect new/removed songs -----------------------
    if CACHE_PATH.exists():
        logger.info("Checking for library changes in '%s'...", args.directory)
        try:
            new_paths, removed_hashes = discover_changes(
                args.directory, dj_graph.known_hashes,
            )
        except (NotADirectoryError, OSError) as exc:
            logger.warning("Cannot scan directory for changes: %s", exc)
            new_paths, removed_hashes = [], set()

        changed = False

        if removed_hashes:
            logger.info("Removing %d deleted song(s)...", len(removed_hashes))
            dj_graph.remove_songs(removed_hashes)
            changed = True

        if new_paths:
            logger.info("Analysing %d new song(s)...", len(new_paths))
            new_songs = analyse_new_songs(
                new_paths,
                known_fingerprints=dj_graph.known_fingerprints,
            )
            if new_songs:
                dj_graph.add_songs_incremental(new_songs)
                changed = True

        if changed:
            CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            dj_graph.save_to_pickle(CACHE_PATH)
            logger.info("Cache updated.")

    logger.info(
        "Graph ready: %d node(s), %d edge(s).",
        dj_graph.num_nodes,
        dj_graph.num_edges,
    )

    # --- 3. Find shortest path --------------------------------------------
    try:
        path, cost = dj_graph.get_shortest_path(args.source, args.target)
    except KeyError as exc:
        logger.error("%s", exc)
        return 1
    except NoPathError as exc:
        logger.error("No path found: %s", exc)
        return 1

    # --- 4. Display results -----------------------------------------------
    _print_path(path, cost)
    return 0


if __name__ == "__main__":
    sys.exit(main())
