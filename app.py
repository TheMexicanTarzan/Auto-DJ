"""
app.py — Plotly Dash web interface for the DJ Mixing Pathfinding System.

Run with:
    python app.py

The application starts the web server immediately and loads the mixing
graph in the background (from the JSON cache if available, otherwise by
scanning SONGS_DIRECTORY).  A progress bar is shown while analysis runs.

Once loaded, users can:

    1. Select a start and destination song from searchable dropdowns.
    2. Click "Find Path" to compute the shortest (smoothest) mix path.
    3. See the path highlighted on a Cytoscape graph visualisation.
    4. Click any node to inspect its top-5 nearest neighbours.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import dash_cytoscape as cyto
import networkx as nx
from dash import Dash, Input, Output, State, callback, html, dcc, ctx

from src.config import CACHE_PATH, SONGS_DIRECTORY
from src.graph import DJGraph
from src.metrics import calculate_weight
from src.utils import scan_directory

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
            graph = DJGraph.load_from_json(CACHE_PATH)
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
                graph.save_to_json(CACHE_PATH)

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


# Start background loading immediately.
_loader_thread = threading.Thread(target=_load_graph_background, daemon=True)
_loader_thread.start()

# ---------------------------------------------------------------------------
# Cytoscape helpers
# ---------------------------------------------------------------------------

_MAX_WEIGHT = 1.0
_TOP_K_NEIGHBOURS = 5


def _get_graph() -> DJGraph | None:
    """Return the graph if ready, else None."""
    with _graph_lock:
        return _graph_state["graph"] if _graph_state["ready"] else None


def _get_progress() -> dict:
    """Return a snapshot of loading progress."""
    with _graph_lock:
        return dict(_graph_state)


def _build_cytoscape_elements(graph: DJGraph) -> list[dict]:
    """Convert the DJGraph into Cytoscape node + edge dicts."""
    elements: list[dict] = []

    for song in graph.songs:
        elements.append({
            "data": {
                "id": song.file_path,
                "label": song.filename,
                "bpm": song.bpm,
                "key": song.key,
            },
        })

    for src, dst, data in graph.graph.edges(data=True):
        w = data["weight"]
        elements.append({
            "data": {
                "source": src,
                "target": dst,
                "weight": round(w, 4),
                "norm_weight": min(w / _MAX_WEIGHT, 1.0),
            },
        })

    return elements


# ---------------------------------------------------------------------------
# Cytoscape stylesheet
# ---------------------------------------------------------------------------

_stylesheet = [
    # Default node
    {
        "selector": "node",
        "style": {
            "label": "data(label)",
            "background-color": "#6c7a89",
            "color": "#ffffff",
            "font-size": "10px",
            "text-valign": "bottom",
            "text-halign": "center",
            "width": 30,
            "height": 30,
            "border-width": 2,
            "border-color": "#4a5568",
        },
    },
    # Default edge
    {
        "selector": "edge",
        "style": {
            "width": 1,
            "line-color": "#cbd5e0",
            "target-arrow-color": "#cbd5e0",
            "target-arrow-shape": "triangle",
            "curve-style": "bezier",
            "opacity": 0.3,
        },
    },
    # Highlighted path node
    {
        "selector": ".path-node",
        "style": {
            "background-color": "#e53e3e",
            "border-color": "#c53030",
            "border-width": 3,
            "width": 40,
            "height": 40,
            "font-weight": "bold",
            "font-size": "12px",
            "color": "#ffffff",
            "z-index": 10,
        },
    },
    # Highlighted path edge
    {
        "selector": ".path-edge",
        "style": {
            "line-color": "#e53e3e",
            "target-arrow-color": "#e53e3e",
            "width": 3,
            "opacity": 1.0,
            "z-index": 10,
        },
    },
    # Selected / clicked node
    {
        "selector": ":selected",
        "style": {
            "background-color": "#3182ce",
            "border-color": "#2b6cb0",
            "border-width": 3,
        },
    },
]

# ---------------------------------------------------------------------------
# Dash app layout
# ---------------------------------------------------------------------------

app = Dash(__name__)
app.title = "Auto-DJ Mix Pathfinder"

app.layout = html.Div(
    style={
        "fontFamily": "'Segoe UI', Roboto, sans-serif",
        "backgroundColor": "#1a202c",
        "color": "#e2e8f0",
        "minHeight": "100vh",
        "padding": "0",
        "margin": "0",
    },
    children=[
        # Interval timer that polls loading progress
        dcc.Interval(
            id="progress-interval",
            interval=1000,  # 1 second
            n_intervals=0,
        ),
        # Hidden store for graph-ready flag
        dcc.Store(id="graph-ready", data=False),
        # Header
        html.Div(
            id="header",
            style={
                "background": "linear-gradient(135deg, #2d3748, #4a5568)",
                "padding": "20px 30px",
                "borderBottom": "2px solid #e53e3e",
            },
            children=[
                html.H1(
                    "Auto-DJ Mix Pathfinder",
                    style={"margin": "0", "fontSize": "28px", "color": "#e2e8f0"},
                ),
                html.P(
                    id="header-stats",
                    children="Loading tracks...",
                    style={"margin": "5px 0 0 0", "color": "#a0aec0", "fontSize": "14px"},
                ),
            ],
        ),
        # Loading overlay
        html.Div(
            id="loading-overlay",
            style={
                "display": "flex",
                "flexDirection": "column",
                "alignItems": "center",
                "justifyContent": "center",
                "height": "calc(100vh - 90px)",
                "padding": "40px",
            },
            children=[
                html.H2(
                    "Analysing your music library...",
                    style={"color": "#e2e8f0", "marginBottom": "20px"},
                ),
                html.Div(
                    style={
                        "width": "60%",
                        "maxWidth": "500px",
                        "backgroundColor": "#2d3748",
                        "borderRadius": "8px",
                        "overflow": "hidden",
                        "height": "30px",
                        "border": "1px solid #4a5568",
                    },
                    children=[
                        html.Div(
                            id="progress-bar",
                            style={
                                "width": "0%",
                                "height": "100%",
                                "backgroundColor": "#e53e3e",
                                "transition": "width 0.5s ease",
                                "borderRadius": "8px",
                            },
                        ),
                    ],
                ),
                html.P(
                    id="progress-text",
                    children="Starting up...",
                    style={
                        "color": "#a0aec0",
                        "marginTop": "12px",
                        "fontSize": "14px",
                    },
                ),
            ],
        ),
        # Main content: sidebar + graph (hidden until ready)
        html.Div(
            id="main-content",
            style={"display": "none", "height": "calc(100vh - 90px)"},
            children=[
                # ---- Left sidebar (controls + details) ----
                html.Div(
                    style={
                        "width": "360px",
                        "minWidth": "360px",
                        "backgroundColor": "#2d3748",
                        "padding": "20px",
                        "overflowY": "auto",
                        "borderRight": "1px solid #4a5568",
                    },
                    children=[
                        # -- Pathfinding controls --
                        html.H3(
                            "Pathfinding",
                            style={"marginTop": "0", "color": "#e2e8f0"},
                        ),
                        html.Label(
                            "Start Song",
                            style={"fontSize": "13px", "color": "#a0aec0"},
                        ),
                        dcc.Dropdown(
                            id="start-dropdown",
                            options=[],
                            placeholder="Search for a track...",
                            searchable=True,
                            style={
                                "marginBottom": "12px",
                                "backgroundColor": "#4a5568",
                                "color": "#1a202c",
                            },
                        ),
                        html.Label(
                            "Destination Song",
                            style={"fontSize": "13px", "color": "#a0aec0"},
                        ),
                        dcc.Dropdown(
                            id="end-dropdown",
                            options=[],
                            placeholder="Search for a track...",
                            searchable=True,
                            style={
                                "marginBottom": "12px",
                                "backgroundColor": "#4a5568",
                                "color": "#1a202c",
                            },
                        ),
                        html.Button(
                            "Find Path",
                            id="find-path-btn",
                            n_clicks=0,
                            style={
                                "width": "100%",
                                "padding": "10px",
                                "backgroundColor": "#e53e3e",
                                "color": "#ffffff",
                                "border": "none",
                                "borderRadius": "6px",
                                "cursor": "pointer",
                                "fontSize": "15px",
                                "fontWeight": "bold",
                            },
                        ),
                        # -- Path results --
                        html.Div(
                            id="path-output",
                            style={
                                "marginTop": "18px",
                                "padding": "14px",
                                "backgroundColor": "#1a202c",
                                "borderRadius": "6px",
                                "fontSize": "13px",
                                "whiteSpace": "pre-wrap",
                                "maxHeight": "260px",
                                "overflowY": "auto",
                                "border": "1px solid #4a5568",
                            },
                            children="Select two songs and click Find Path.",
                        ),
                        html.Hr(style={"borderColor": "#4a5568", "margin": "20px 0"}),
                        # -- Node details panel --
                        html.H3(
                            "Track Details",
                            style={"marginTop": "0", "color": "#e2e8f0"},
                        ),
                        html.Div(
                            id="node-details",
                            style={
                                "padding": "14px",
                                "backgroundColor": "#1a202c",
                                "borderRadius": "6px",
                                "fontSize": "13px",
                                "whiteSpace": "pre-wrap",
                                "maxHeight": "280px",
                                "overflowY": "auto",
                                "border": "1px solid #4a5568",
                            },
                            children="Click a node on the graph to see details.",
                        ),
                    ],
                ),
                # ---- Graph area ----
                html.Div(
                    style={"flex": "1", "position": "relative"},
                    children=[
                        cyto.Cytoscape(
                            id="cyto-graph",
                            elements=[],
                            layout={"name": "cose", "animate": False},
                            stylesheet=_stylesheet,
                            style={"width": "100%", "height": "100%"},
                            minZoom=0.2,
                            maxZoom=3.0,
                            responsive=True,
                        ),
                    ],
                ),
            ],
        ),
    ],
)

# ---------------------------------------------------------------------------
# Callback — Progress polling + transition to main UI
# ---------------------------------------------------------------------------


@callback(
    Output("progress-bar", "style"),
    Output("progress-text", "children"),
    Output("loading-overlay", "style"),
    Output("main-content", "style"),
    Output("header-stats", "children"),
    Output("cyto-graph", "elements"),
    Output("start-dropdown", "options"),
    Output("end-dropdown", "options"),
    Output("progress-interval", "disabled"),
    Output("graph-ready", "data"),
    Input("progress-interval", "n_intervals"),
    State("graph-ready", "data"),
)
def update_progress(n_intervals, already_ready):
    """Poll background loader and flip to main UI when done."""
    progress = _get_progress()

    bar_style = {
        "width": f'{progress["progress"]}%',
        "height": "100%",
        "backgroundColor": "#e53e3e",
        "transition": "width 0.5s ease",
        "borderRadius": "8px",
    }

    if progress.get("error"):
        return (
            bar_style,
            f'Error: {progress["error"]}',
            {"display": "flex", "flexDirection": "column", "alignItems": "center",
             "justifyContent": "center", "height": "calc(100vh - 90px)", "padding": "40px"},
            {"display": "none", "height": "calc(100vh - 90px)"},
            "Error loading tracks",
            [],
            [],
            [],
            True,
            False,
        )

    if progress["ready"]:
        graph = _get_graph()
        elements = _build_cytoscape_elements(graph) if graph else []
        song_options = [
            {"label": f"{s.filename}  ({s.bpm} BPM, {s.key})", "value": s.file_path}
            for s in graph.songs
        ] if graph else []
        stats = (
            f"{graph.num_nodes} tracks loaded | "
            f"{graph.num_edges} possible transitions"
        ) if graph else "No tracks loaded"

        return (
            bar_style,
            "Done!",
            {"display": "none"},
            {"display": "flex", "height": "calc(100vh - 90px)"},
            stats,
            elements,
            song_options,
            song_options,
            True,   # stop polling
            True,
        )

    # Still loading
    current = progress["current"]
    total = progress["total"]
    pct = progress["progress"]
    fname = progress["current_file"]

    if total > 0:
        text = f"Analysing track {current} of {total} ({pct}%) — {fname}"
    else:
        text = "Scanning for audio files..."

    return (
        bar_style,
        text,
        {"display": "flex", "flexDirection": "column", "alignItems": "center",
         "justifyContent": "center", "height": "calc(100vh - 90px)", "padding": "40px"},
        {"display": "none", "height": "calc(100vh - 90px)"},
        "Loading tracks...",
        [],
        [],
        [],
        False,
        False,
    )


# ---------------------------------------------------------------------------
# Callback — Pathfinding
# ---------------------------------------------------------------------------


@callback(
    Output("path-output", "children"),
    Output("cyto-graph", "elements", allow_duplicate=True),
    Input("find-path-btn", "n_clicks"),
    State("start-dropdown", "value"),
    State("end-dropdown", "value"),
    State("graph-ready", "data"),
    prevent_initial_call=True,
)
def find_path(n_clicks: int, start_fp: str | None, end_fp: str | None, ready: bool):
    """Compute shortest path and highlight it on the graph."""
    graph = _get_graph()
    if not graph or not ready:
        return "Graph is still loading, please wait...", []

    if not start_fp or not end_fp:
        return "Please select both a start and destination song.", _build_cytoscape_elements(graph)

    if start_fp == end_fp:
        song = graph.get_song(start_fp)
        elements = _apply_path_classes(graph, [start_fp], set())
        return f"Start and destination are the same track:\n  {song.filename}", elements

    try:
        path, total_cost = graph.get_shortest_path(start_fp, end_fp)
    except nx.NetworkXNoPath:
        return (
            "No mixable path exists between these two tracks.\n"
            "The BPM gap at every intermediate step is too large.",
            _build_cytoscape_elements(graph),
        )
    except KeyError as exc:
        return f"Song not found: {exc}", _build_cytoscape_elements(graph)

    # Build text summary
    lines = ["SHORTEST MIX PATH", "=" * 40]
    path_node_ids = [s.file_path for s in path]
    path_edge_set: set[tuple[str, str]] = set()

    for i, song in enumerate(path):
        prefix = "START" if i == 0 else f"  [{i}]"
        lines.append(f"{prefix}  {song.filename}")
        lines.append(f"        BPM: {song.bpm}  |  Key: {song.key}")

        if i < len(path) - 1:
            hop_cost = calculate_weight(song, path[i + 1])
            lines.append(f"          -> cost: {hop_cost:.4f}")
            path_edge_set.add((song.file_path, path[i + 1].file_path))

    lines.append("-" * 40)
    lines.append(f"Total cost: {total_cost:.4f}  |  Hops: {len(path) - 1}")

    elements = _apply_path_classes(graph, path_node_ids, path_edge_set)
    return "\n".join(lines), elements


def _apply_path_classes(
    graph: DJGraph,
    path_node_ids: list[str],
    path_edge_set: set[tuple[str, str]],
) -> list[dict]:
    """Return elements with path classes applied."""
    base = _build_cytoscape_elements(graph)
    path_nodes = set(path_node_ids)
    elements = []

    for el in base:
        new_el = {**el, "data": {**el["data"]}}
        classes = []

        if "source" not in el["data"]:
            # Node
            if el["data"]["id"] in path_nodes:
                classes.append("path-node")
        else:
            # Edge
            edge_key = (el["data"]["source"], el["data"]["target"])
            if edge_key in path_edge_set:
                classes.append("path-edge")

        if classes:
            new_el["classes"] = " ".join(classes)

        elements.append(new_el)

    return elements


# ---------------------------------------------------------------------------
# Callback — Node inspection (click a node)
# ---------------------------------------------------------------------------


@callback(
    Output("node-details", "children"),
    Input("cyto-graph", "tapNodeData"),
    State("graph-ready", "data"),
    prevent_initial_call=True,
)
def inspect_node(node_data: dict | None, ready: bool):
    """Show metadata and top-K neighbours for a clicked node."""
    graph = _get_graph()
    if not graph or not ready:
        return "Graph is still loading, please wait..."

    if node_data is None:
        return "Click a node on the graph to see details."

    file_path = node_data["id"]
    try:
        song = graph.get_song(file_path)
    except KeyError:
        return f"Unknown node: {file_path}"

    neighbours = graph.neighbours(file_path)
    top_k = neighbours[:_TOP_K_NEIGHBOURS]

    lines = [
        f"{song.filename}",
        f"BPM: {song.bpm}  |  Key: {song.key}",
        f"Path: {song.file_path}",
        "",
        f"Top {_TOP_K_NEIGHBOURS} Mixable Tracks:",
        "-" * 36,
    ]

    if not top_k:
        lines.append("  (no compatible neighbours)")
    else:
        for rank, (nbr, cost) in enumerate(top_k, 1):
            lines.append(
                f"  {rank}. {nbr.filename}\n"
                f"     BPM: {nbr.bpm} | Key: {nbr.key}\n"
                f"     Transition cost: {cost:.4f}"
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)
