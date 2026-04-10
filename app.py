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
from fastapi import FastAPI, Request
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from collections import defaultdict

from src.config import CACHE_PATH, SETLISTS_DIRECTORY, SONGS_DIRECTORY, _LEGACY_JSON_CACHE
from src.graph import DJGraph, NoPathError
from src.setlist_saver import save_setlist
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
    "version": 0,           # bumped on every graph mutation
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
                    exclude_dirs={SETLISTS_DIRECTORY},
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

        # Mark graph as ready BEFORE the incremental sync so the UI is
        # usable immediately.  The sync will update the graph in-place
        # (protected by _graph_lock) if any changes are found.
        with _graph_lock:
            _graph_state["graph"] = graph
            _graph_state["ready"] = True
            _graph_state["progress"] = 100
            _graph_state["version"] += 1

        logger.info(
            "Graph ready: %d node(s), %d edge(s).",
            graph.num_nodes,
            graph.num_edges,
        )

        # --- Incremental sync: detect new/removed songs ---
        # Runs AFTER the graph is already serving requests.
        try:
            new_paths, removed_hashes = discover_changes(
                SONGS_DIRECTORY, graph.known_hashes,
                exclude_dirs={SETLISTS_DIRECTORY},
            )
        except (NotADirectoryError, OSError) as exc:
            logger.warning("Cannot scan directory for changes: %s", exc)
            new_paths, removed_hashes = [], set()

        # --- Analyse new songs BEFORE mutating the graph ---
        # This keeps the mutation window (where API reads could see
        # inconsistent state) as short as possible.
        new_songs: list = []
        if new_paths:
            logger.info("Analysing %d new song(s)...", len(new_paths))
            new_songs = analyse_new_songs(
                new_paths,
                progress_callback=_progress_callback,
                known_fingerprints=graph.known_fingerprints,
            )

        # --- Apply all mutations in a tight block ---
        sync_changed = False
        if removed_hashes:
            logger.info("Removing %d deleted song(s)...", len(removed_hashes))
            graph.remove_songs(removed_hashes)
            sync_changed = True

        if new_songs:
            graph.add_songs_incremental(new_songs)
            sync_changed = True

        if sync_changed:
            CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            graph.save_to_pickle(CACHE_PATH)
            with _graph_lock:
                _graph_state["version"] += 1
            logger.info("Cache updated with library changes.")

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

_DEFAULT_TOP_K = 10


def _get_graph() -> DJGraph | None:
    """Return the graph if ready, else None."""
    with _graph_lock:
        return _graph_state["graph"] if _graph_state["ready"] else None


def _get_graph_version() -> int:
    """Return the current graph version counter."""
    with _graph_lock:
        return _graph_state["version"]


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


def _build_directory_tree(graph: DJGraph) -> dict:
    """Build a nested directory tree from song file paths.

    Returns a dict like:
        {"name": "root", "children": [{"name": "Genre A", "children": [...]}]}

    Each leaf directory (one that directly contains songs) also has a
    ``"count"`` key with the number of songs it holds.
    """
    base = Path(SONGS_DIRECTORY).resolve()
    dir_counts: dict[str, int] = defaultdict(int)

    for song in graph.songs:
        try:
            rel = Path(song.file_path).resolve().relative_to(base)
        except ValueError:
            # Song outside SONGS_DIRECTORY — put in root
            rel = Path(Path(song.file_path).name)
        parent = str(rel.parent) if rel.parent != Path(".") else "."
        dir_counts[parent] += 1

    # Build nested tree from flat directory paths
    tree: dict = {"name": "All Songs", "children": [], "count": 0}
    nodes_map: dict[str, dict] = {".": tree}

    for dir_path in sorted(dir_counts):
        parts = dir_path.split("/") if dir_path != "." else []
        current_path = "."
        parent_node = tree
        for part in parts:
            current_path = part if current_path == "." else current_path + "/" + part
            if current_path not in nodes_map:
                child: dict = {"name": part, "path": current_path, "children": [], "count": 0}
                parent_node["children"].append(child)
                nodes_map[current_path] = child
            parent_node = nodes_map[current_path]
        parent_node["count"] = dir_counts[dir_path]

    # Roll up counts to parent directories
    def _rollup(node: dict) -> int:
        total = node.get("count", 0)
        for child in node.get("children", []):
            total += _rollup(child)
        node["count"] = total
        return total

    _rollup(tree)
    return tree


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
def api_graph(request: Request):
    """Return the full graph (nodes + edges) for the frontend."""
    graph = _get_graph()
    if graph is None:
        return _orjson_response({"error": "Graph not ready"}, status_code=503)

    etag = f'W/"{_get_graph_version()}"'
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304)

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

    # Bulk-extract edge attributes to avoid per-edge igraph lookups
    g = graph.graph
    ecount = g.ecount()
    if ecount > 0:
        edge_list = g.get_edgelist()
        vertex_names = g.vs["name"]
        weights = g.es["weight"]
        has_edge_type = "edge_type" in g.es.attributes()
        edge_types = g.es["edge_type"] if has_edge_type else None

        edges = [None] * ecount
        type_counts: dict[str, int] = {}
        for i in range(ecount):
            src_idx, dst_idx = edge_list[i]
            etype = (edge_types[i] or "direct") if edge_types else "direct"
            edges[i] = {
                "source": vertex_names[src_idx],
                "target": vertex_names[dst_idx],
                "weight": weights[i],
                "edge_type": etype,
            }
            type_counts[etype] = type_counts.get(etype, 0) + 1
    else:
        edges = []
        type_counts = {}

    directory_tree = _build_directory_tree(graph)

    return Response(
        content=orjson.dumps({
            "nodes": nodes,
            "edges": edges,
            "num_nodes": graph.num_nodes,
            "num_edges": graph.num_edges,
            "edge_type_counts": type_counts,
            "songs_directory": str(Path(SONGS_DIRECTORY).resolve()),
            "directory_tree": directory_tree,
        }),
        media_type="application/json",
        headers={"ETag": etag},
    )


@app.get("/api/songs")
def api_songs(request: Request):
    """Return a lightweight song list for dropdown population."""
    graph = _get_graph()
    if graph is None:
        return _orjson_response({"error": "Graph not ready"}, status_code=503)

    etag = f'W/"{_get_graph_version()}"'
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304)

    songs = [
        {"id": s.file_path, "label": f"{s.filename}  ({s.bpm} BPM, {s.key})"}
        for s in graph.songs
    ]
    return Response(
        content=orjson.dumps(songs),
        media_type="application/json",
        headers={"ETag": etag},
    )


class PathRequest(BaseModel):
    start: str
    end: str
    waypoints: list[str] | None = None  # ordered intermediate stops
    allowed_types: list[str] | None = None
    excluded_dirs: list[str] | None = None


@app.post("/api/path")
def api_path(req: PathRequest):
    """Compute the shortest mix path, optionally through one or more waypoints."""
    graph = _get_graph()
    if graph is None:
        return _orjson_response({"error": "Graph not ready"}, status_code=503)

    if not req.start or not req.end:
        return _orjson_response(
            {"error": "Please select both a start and destination song."},
            status_code=400,
        )

    # Full ordered list of stops: [start, ...waypoints, end]
    stops = [req.start] + (req.waypoints or []) + [req.end]

    allowed = set(req.allowed_types) if req.allowed_types else None
    excluded = set(req.excluded_dirs) if req.excluded_dirs else None

    full_path: list = []
    path_edges: list = []
    total_cost = 0.0

    for i in range(len(stops) - 1):
        seg_start, seg_end = stops[i], stops[i + 1]

        if seg_start == seg_end:
            # Same node – ensure it appears once then move on
            if not full_path:
                try:
                    full_path.append(graph.get_song(seg_start))
                except KeyError as exc:
                    return _orjson_response({"error": str(exc)}, status_code=404)
            continue

        try:
            seg_path, seg_cost = graph.get_shortest_path(
                seg_start, seg_end,
                allowed_types=allowed,
                excluded_dirs=excluded,
                songs_directory=SONGS_DIRECTORY,
            )
        except NoPathError:
            stop_num = i + 1
            return _orjson_response({
                "error": (
                    f"No mixable path exists between stop {stop_num} and "
                    f"stop {stop_num + 1}. "
                    "The BPM gap at every intermediate step is too large."
                ),
            }, status_code=404)
        except KeyError as exc:
            return _orjson_response({"error": f"Song not found: {exc}"}, status_code=404)

        # Concatenate segments – the junction node was already added as the
        # last node of the previous segment, so skip seg_path[0].
        if full_path:
            full_path.extend(seg_path[1:])
        else:
            full_path.extend(seg_path)

        total_cost += seg_cost

        for j in range(len(seg_path) - 1):
            path_edges.append([seg_path[j].file_path, seg_path[j + 1].file_path])

    # Degenerate case: all stops resolved to the same node
    if not full_path:
        try:
            full_path = [graph.get_song(req.start)]
        except KeyError as exc:
            return _orjson_response({"error": str(exc)}, status_code=404)

    n = len(full_path)
    hops = n - 1
    summary = f"{n} track{'s' if n != 1 else ''} · {hops} hop{'s' if hops != 1 else ''}"

    return _orjson_response({
        "path_nodes": [s.file_path for s in full_path],
        "path_edges": path_edges,
        "summary": summary,
        "total_cost": total_cost,
    })


class SetlistRequest(BaseModel):
    min_bpm: float = 0.0
    max_bpm: float = 999.0
    target_duration_min: float = 60.0   # desired total set length in minutes
    starting_key: str | None = None     # key of the first track (optional)
    set_key: str | None = None          # key constraint for all tracks (optional)
    allowed_types: list[str] | None = None


@app.post("/api/setlist")
def api_setlist(req: SetlistRequest):
    """Generate a randomised continuous setlist matching the given criteria."""
    graph = _get_graph()
    if graph is None:
        return _orjson_response({"error": "Graph not ready"}, status_code=503)

    if req.min_bpm > req.max_bpm:
        return _orjson_response(
            {"error": "min_bpm must be ≤ max_bpm."}, status_code=400
        )

    target_sec = req.target_duration_min * 60.0
    allowed = set(req.allowed_types) if req.allowed_types else None

    try:
        songs = graph.generate_setlist(
            min_bpm=req.min_bpm,
            max_bpm=req.max_bpm,
            target_sec=target_sec,
            starting_key=req.starting_key or None,
            set_key=req.set_key or None,
            allowed_types=allowed,
        )
    except ValueError as exc:
        return _orjson_response({"error": str(exc)}, status_code=400)

    # Build path_edges for graph highlighting (consecutive pairs only)
    path_edges: list = []
    for i in range(len(songs) - 1):
        a_id, b_id = songs[i].file_path, songs[i + 1].file_path
        try:
            va = graph.graph.vs.find(name=a_id)
            vb = graph.graph.vs.find(name=b_id)
            eid = graph.graph.get_eid(va.index, vb.index, directed=False, error=False)
            if eid >= 0:
                path_edges.append([a_id, b_id])
        except Exception:
            pass

    _FALLBACK = 210.0
    total_sec = sum(
        s.duration_sec if s.duration_sec > 0 else _FALLBACK for s in songs
    )
    total_min = total_sec / 60.0
    n = len(songs)
    summary = (
        f"{n} track{'s' if n != 1 else ''}"
        f" \u00B7 ~{total_min:.0f} min"
    )

    return _orjson_response({
        "path_nodes": [s.file_path for s in songs],
        "path_edges": path_edges,
        "summary": summary,
    })


@app.get("/api/neighbors/{node_id:path}")
def api_neighbors(node_id: str, k: int = _DEFAULT_TOP_K, types: str | None = None):
    """Return a node's metadata and its top-K nearest neighbours."""
    graph = _get_graph()
    if graph is None:
        return _orjson_response({"error": "Graph not ready"}, status_code=503)

    try:
        song = graph.get_song(node_id)
    except KeyError:
        return _orjson_response({"error": f"Unknown node: {node_id}"}, status_code=404)

    allowed_types = set(types.split(",")) if types else None
    neighbours = graph.neighbours(node_id, allowed_types=allowed_types)
    top_k = neighbours[:max(1, min(k, 50))]

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
                "edge_type": etype,
            }
            for nbr, cost, etype in top_k
        ],
    })


class SaveSetlistRequest(BaseModel):
    setlist_name: str
    track_paths: list[str]


@app.post("/api/save_setlist")
def api_save_setlist(req: SaveSetlistRequest):
    """Copy the setlist tracks into a numbered subfolder of SETLISTS_DIRECTORY."""
    if not req.track_paths:
        return _orjson_response({"error": "No tracks provided."}, status_code=400)

    try:
        output_dir = save_setlist(req.track_paths, req.setlist_name, SETLISTS_DIRECTORY)
    except OSError as exc:
        return _orjson_response(
            {"error": f"Could not write files: {exc}"}, status_code=500
        )

    return _orjson_response({
        "success": True,
        "output_dir": output_dir,
        "message": f"Saved to {output_dir}",
    })


class RecalculateRequest(BaseModel):
    harmonic: float = 0.35
    tempo: float = 0.25
    semantic: float = 0.40
    double_penalty: float = 0.0
    triplet_penalty: float = 0.0


@app.post("/api/recalculate")
def api_recalculate(req: RecalculateRequest):
    """Recalculate all edges using cached song data with new weights."""
    graph = _get_graph()
    if graph is None:
        return _orjson_response({"error": "Graph not ready"}, status_code=503)

    weights = {
        "harmonic": req.harmonic,
        "tempo": req.tempo,
        "semantic": req.semantic,
    }
    type_penalties = {
        "double": req.double_penalty,
        "triplet": req.triplet_penalty,
    }

    graph.recalculate_edges(weights=weights, type_penalties=type_penalties)

    with _graph_lock:
        _graph_state["version"] += 1

    # Persist to cache
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    graph.save_to_pickle(CACHE_PATH)

    return _orjson_response({
        "num_edges": graph.num_edges,
        "message": "Edges recalculated successfully.",
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
    # _start_loader_thread() is called by the FastAPI lifespan handler —
    # do NOT call it here or the graph will be loaded twice.
    uvicorn.run(app, host="127.0.0.1", port=8050)
