"""
app.py — FastAPI + Sigma.js web interface for the DJ Mixing Pathfinding System.

Run with:
    python app.py

The application starts the web server immediately and loads the mixing
graph in the background (from the pickle cache if available, otherwise by
scanning SONGS_DIRECTORY).  A progress bar is shown while analysis runs.

Once loaded, users can:

    1. Select a start and destination song from searchable inputs.
    2. Click "Find Path" to compute the shortest (smoothest) mix path.
    3. See the path highlighted on a Sigma.js WebGL graph visualisation.
    4. Click any node to inspect its top-5 nearest neighbours.
"""

from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import orjson
import uvicorn
from fastapi import FastAPI
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.config import CACHE_PATH, SONGS_DIRECTORY, _LEGACY_JSON_CACHE
from src.graph import DJGraph, NoPathError
from src.metrics import calculate_weight
from src.utils import analyse_new_songs, discover_changes, scan_directory

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("auto-dj-web")

# ---------------------------------------------------------------------------
# Background graph loader
# ---------------------------------------------------------------------------

# Shared mutable state protected by a lock.
_graph_lock = threading.Lock()
_graph_state: dict = {
    "graph": None,          # DJGraph | None
    "ready": False,
    "progress": 0,          # 0-100
    "total": 0,
    "current": 0,
    "current_file": "",
    "error": None,          # str | None
}


def _progress_callback(current: int, total: int, filename: str) -> None:
    with _graph_lock:
        _graph_state["current"] = current
        _graph_state["total"] = total
        _graph_state["current_file"] = filename
        _graph_state["progress"] = int(current / total * 100) if total else 0


def _load_graph_background() -> None:
    """Load graph in a background thread, updating _graph_state."""
    try:
        if CACHE_PATH.exists():
            logger.info("Loading graph from cache '%s'...", CACHE_PATH)
            with _graph_lock:
                _graph_state["current_file"] = "cache"
                _graph_state["progress"] = 50
            graph = DJGraph.load_from_pickle(CACHE_PATH)
        elif _LEGACY_JSON_CACHE.exists():
            logger.info(
                "Migrating legacy JSON cache '%s' to pickle...",
                _LEGACY_JSON_CACHE,
            )
            with _graph_lock:
                _graph_state["current_file"] = "cache"
                _graph_state["progress"] = 50
            graph = DJGraph.load_from_json(_LEGACY_JSON_CACHE)
            CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            graph.save_to_pickle(CACHE_PATH)
        else:
            logger.info(
                "No cache found. Scanning '%s' for audio files...",
                SONGS_DIRECTORY,
            )
            try:
                songs = scan_directory(
                    SONGS_DIRECTORY,
                    progress_callback=_progress_callback,
                )
            except (NotADirectoryError, OSError) as exc:
                logger.warning(
                    "Cannot scan directory: %s. "
                    "Starting with an empty graph — update SONGS_DIRECTORY "
                    "in src/config.py.",
                    exc,
                )
                songs = []

            if not songs:
                logger.warning(
                    "No audio files found in '%s'. "
                    "Starting with an empty graph — update SONGS_DIRECTORY "
                    "in src/config.py.",
                    SONGS_DIRECTORY,
                )

            logger.info("Building mixing graph from %d track(s)...", len(songs))
            graph = DJGraph.build(songs)
            if songs:
                CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
                graph.save_to_pickle(CACHE_PATH)

        # --- Incremental sync: detect new/removed songs ---
        try:
            new_paths, removed_hashes = discover_changes(
                SONGS_DIRECTORY, graph.known_hashes,
            )
        except (NotADirectoryError, OSError) as exc:
            logger.warning("Cannot scan directory for changes: %s", exc)
            new_paths, removed_hashes = [], set()

        sync_changed = False

        if removed_hashes:
            logger.info("Removing %d deleted song(s)...", len(removed_hashes))
            graph.remove_songs(removed_hashes)
            sync_changed = True

        if new_paths:
            logger.info("Analysing %d new song(s)...", len(new_paths))
            new_songs = analyse_new_songs(new_paths, progress_callback=_progress_callback)
            if new_songs:
                graph.add_songs_incremental(new_songs)
                sync_changed = True

        if sync_changed:
            CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            graph.save_to_pickle(CACHE_PATH)
            logger.info("Cache updated with library changes.")

        with _graph_lock:
            _graph_state["graph"] = graph
            _graph_state["ready"] = True
            _graph_state["progress"] = 100

        logger.info(
            "Graph ready: %d node(s), %d edge(s).",
            graph.num_nodes,
            graph.num_edges,
        )

    except Exception as exc:
        logger.exception("Failed to build graph.")
        with _graph_lock:
            _graph_state["error"] = str(exc)


def _start_loader_thread() -> None:
    """Launch the background graph-loader thread (called once)."""
    thread = threading.Thread(target=_load_graph_background, daemon=True)
    thread.start()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TOP_K_NEIGHBOURS = 5


def _get_graph() -> DJGraph | None:
    """Return the graph if ready, else None."""
    with _graph_lock:
        return _graph_state["graph"] if _graph_state["ready"] else None


def _get_progress() -> dict:
    """Return a snapshot of loading progress."""
    with _graph_lock:
        return {
            "ready": _graph_state["ready"],
            "progress": _graph_state["progress"],
            "current": _graph_state["current"],
            "total": _graph_state["total"],
            "current_file": _graph_state["current_file"],
            "error": _graph_state["error"],
        }


def _orjson_response(data: object, status_code: int = 200) -> Response:
    """Return a fast JSON response using orjson."""
    return Response(
        content=orjson.dumps(data),
        media_type="application/json",
        status_code=status_code,
    )


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _lifespan(application: FastAPI):
    _start_loader_thread()
    yield


app = FastAPI(title="Auto-DJ Mix Pathfinder", lifespan=_lifespan)


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.get("/api/status")
def api_status():
    """Return current loading progress."""
    return _orjson_response(_get_progress())


@app.get("/api/graph")
def api_graph():
    """Return the full graph (nodes + edges) for the frontend."""
    graph = _get_graph()
    if graph is None:
        return _orjson_response({"error": "Graph not ready"}, status_code=503)

    coords = graph.layout_coords

    nodes = []
    for song in graph.songs:
        x, y = coords.get(song.file_path, (0.0, 0.0))
        nodes.append({
            "id": song.file_path,
            "label": song.filename,
            "x": x,
            "y": y,
            "bpm": song.bpm,
            "key": song.key,
        })

    edges = []
    for edge in graph.graph.es:
        src_name = graph.graph.vs[edge.source]["name"]
        dst_name = graph.graph.vs[edge.target]["name"]
        edges.append({
            "source": src_name,
            "target": dst_name,
            "weight": edge["weight"],
        })

    return _orjson_response({
        "nodes": nodes,
        "edges": edges,
        "num_nodes": graph.num_nodes,
        "num_edges": graph.num_edges,
    })


@app.get("/api/songs")
def api_songs():
    """Return a lightweight song list for dropdown population."""
    graph = _get_graph()
    if graph is None:
        return _orjson_response({"error": "Graph not ready"}, status_code=503)

    songs = [
        {"id": s.file_path, "label": f"{s.filename}  ({s.bpm} BPM, {s.key})"}
        for s in graph.songs
    ]
    return _orjson_response(songs)


class PathRequest(BaseModel):
    start: str
    end: str


@app.post("/api/path")
def api_path(req: PathRequest):
    """Compute shortest path between two songs."""
    graph = _get_graph()
    if graph is None:
        return _orjson_response({"error": "Graph not ready"}, status_code=503)

    if not req.start or not req.end:
        return _orjson_response(
            {"error": "Please select both a start and destination song."},
            status_code=400,
        )

    if req.start == req.end:
        try:
            song = graph.get_song(req.start)
        except KeyError as exc:
            return _orjson_response({"error": str(exc)}, status_code=404)
        return _orjson_response({
            "path_nodes": [song.file_path],
            "path_edges": [],
            "summary": f"Start and destination are the same track:\n  {song.filename}",
            "total_cost": 0.0,
        })

    try:
        path, total_cost = graph.get_shortest_path(req.start, req.end)
    except NoPathError:
        return _orjson_response({
            "error": (
                "No mixable path exists between these two tracks.\n"
                "The BPM gap at every intermediate step is too large."
            ),
        }, status_code=404)
    except KeyError as exc:
        return _orjson_response({"error": f"Song not found: {exc}"}, status_code=404)

    # Build text summary
    lines = ["SHORTEST MIX PATH", "=" * 40]
    path_node_ids = [s.file_path for s in path]
    path_edges = []

    for i, song in enumerate(path):
        prefix = "START" if i == 0 else f"  [{i}]"
        lines.append(f"{prefix}  {song.filename}")
        lines.append(f"        BPM: {song.bpm}  |  Key: {song.key}")

        if i < len(path) - 1:
            hop_cost = calculate_weight(song, path[i + 1])
            lines.append(f"          -> cost: {hop_cost:.4f}")
            path_edges.append([song.file_path, path[i + 1].file_path])

    lines.append("-" * 40)
    lines.append(f"Total cost: {total_cost:.4f}  |  Hops: {len(path) - 1}")

    return _orjson_response({
        "path_nodes": path_node_ids,
        "path_edges": path_edges,
        "summary": "\n".join(lines),
        "total_cost": total_cost,
    })


@app.get("/api/neighbors/{node_id:path}")
def api_neighbors(node_id: str):
    """Return a node's metadata and its top-K nearest neighbours."""
    graph = _get_graph()
    if graph is None:
        return _orjson_response({"error": "Graph not ready"}, status_code=503)

    try:
        song = graph.get_song(node_id)
    except KeyError:
        return _orjson_response({"error": f"Unknown node: {node_id}"}, status_code=404)

    neighbours = graph.neighbours(node_id)
    top_k = neighbours[:_TOP_K_NEIGHBOURS]

    return _orjson_response({
        "node": {
            "id": song.file_path,
            "label": song.filename,
            "bpm": song.bpm,
            "key": song.key,
        },
        "neighbors": [
            {
                "id": nbr.file_path,
                "label": nbr.filename,
                "bpm": nbr.bpm,
                "key": nbr.key,
                "cost": round(cost, 4),
            }
            for nbr, cost in top_k
        ],
    })


# ---------------------------------------------------------------------------
# Static files — serve the frontend
# ---------------------------------------------------------------------------

app.mount("/static", StaticFiles(directory="static"), name="static")
# Serve index.html at the root
app.mount("/", StaticFiles(directory="static", html=True), name="root")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _start_loader_thread()
    uvicorn.run(app, host="127.0.0.1", port=8050)
