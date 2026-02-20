"""
app.py — Plotly Dash web interface for the DJ Mixing Pathfinding System.

Run with:
    python app.py

The application loads the mixing graph once at startup (from the JSON
cache if available, otherwise by scanning SONGS_DIRECTORY), then serves
an interactive UI where users can:

    1. Select a start and destination song from searchable dropdowns.
    2. Click "Find Path" to compute the shortest (smoothest) mix path.
    3. See the path highlighted on a Cytoscape graph visualisation.
    4. Click any node to inspect its top-5 nearest neighbours.
"""

from __future__ import annotations

import logging
from pathlib import Path

import dash_cytoscape as cyto
import networkx as nx
from dash import Dash, Input, Output, State, callback, html, dcc

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
# Graph initialisation (runs once at import time)
# ---------------------------------------------------------------------------


def _load_graph() -> DJGraph:
    """Load graph from cache or build from scratch."""
    if CACHE_PATH.exists():
        logger.info("Loading graph from cache '%s'...", CACHE_PATH)
        return DJGraph.load_from_json(CACHE_PATH)

    logger.info("No cache found. Scanning '%s' for audio files...", SONGS_DIRECTORY)
    try:
        songs = scan_directory(SONGS_DIRECTORY)
    except (NotADirectoryError, OSError) as exc:
        logger.warning(
            "Cannot scan directory: %s. "
            "Starting with an empty graph — update SONGS_DIRECTORY in src/config.py.",
            exc,
        )
        return DJGraph.build([])

    if not songs:
        logger.warning(
            "No audio files found in '%s'. "
            "Starting with an empty graph — update SONGS_DIRECTORY in src/config.py.",
            SONGS_DIRECTORY,
        )
        return DJGraph.build([])

    logger.info("Building mixing graph from %d track(s)...", len(songs))
    graph = DJGraph.build(songs)
    graph.save_to_json(CACHE_PATH)
    return graph


dj_graph: DJGraph = _load_graph()
logger.info(
    "Graph ready: %d node(s), %d edge(s).",
    dj_graph.num_nodes,
    dj_graph.num_edges,
)

# ---------------------------------------------------------------------------
# Cytoscape element builders
# ---------------------------------------------------------------------------

# Maximum edge weight used to scale visual properties.  Edge weights
# live in [0, 1] so we cap at 1.0 for normalisation.
_MAX_WEIGHT = 1.0

# Maximum number of neighbours shown in the details panel.
_TOP_K_NEIGHBOURS = 5


def _build_cytoscape_elements() -> list[dict]:
    """Convert the DJGraph into Cytoscape node + edge dicts."""
    elements: list[dict] = []

    for song in dj_graph.songs:
        elements.append({
            "data": {
                "id": song.file_path,
                "label": song.filename,
                "bpm": song.bpm,
                "key": song.key,
            },
        })

    for src, dst, data in dj_graph.graph.edges(data=True):
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


_base_elements = _build_cytoscape_elements()

# ---------------------------------------------------------------------------
# Dropdown options (sorted by filename)
# ---------------------------------------------------------------------------

_song_options = [
    {"label": f"{s.filename}  ({s.bpm} BPM, {s.key})", "value": s.file_path}
    for s in dj_graph.songs
]

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
        # Header
        html.Div(
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
                    f"{dj_graph.num_nodes} tracks loaded | "
                    f"{dj_graph.num_edges} possible transitions",
                    style={"margin": "5px 0 0 0", "color": "#a0aec0", "fontSize": "14px"},
                ),
            ],
        ),
        # Main content: sidebar + graph
        html.Div(
            style={"display": "flex", "height": "calc(100vh - 90px)"},
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
                            options=_song_options,
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
                            options=_song_options,
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
                            elements=_base_elements,
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
# Callback 1 — Pathfinding
# ---------------------------------------------------------------------------


@callback(
    Output("path-output", "children"),
    Output("cyto-graph", "elements"),
    Input("find-path-btn", "n_clicks"),
    State("start-dropdown", "value"),
    State("end-dropdown", "value"),
    prevent_initial_call=True,
)
def find_path(n_clicks: int, start_fp: str | None, end_fp: str | None):
    """Compute shortest path and highlight it on the graph."""
    if not start_fp or not end_fp:
        return "Please select both a start and destination song.", _base_elements

    if start_fp == end_fp:
        song = dj_graph.get_song(start_fp)
        # Highlight just the single node
        elements = _apply_path_classes([start_fp], set())
        return f"Start and destination are the same track:\n  {song.filename}", elements

    try:
        path, total_cost = dj_graph.get_shortest_path(start_fp, end_fp)
    except nx.NetworkXNoPath:
        return (
            "No mixable path exists between these two tracks.\n"
            "The BPM gap at every intermediate step is too large.",
            _base_elements,
        )
    except KeyError as exc:
        return f"Song not found: {exc}", _base_elements

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

    elements = _apply_path_classes(path_node_ids, path_edge_set)
    return "\n".join(lines), elements


def _apply_path_classes(
    path_node_ids: list[str],
    path_edge_set: set[tuple[str, str]],
) -> list[dict]:
    """Return a copy of the base elements with path classes applied."""
    path_nodes = set(path_node_ids)
    elements = []

    for el in _base_elements:
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
# Callback 2 — Node inspection (click a node)
# ---------------------------------------------------------------------------


@callback(
    Output("node-details", "children"),
    Input("cyto-graph", "tapNodeData"),
    prevent_initial_call=True,
)
def inspect_node(node_data: dict | None):
    """Show metadata and top-K neighbours for a clicked node."""
    if node_data is None:
        return "Click a node on the graph to see details."

    file_path = node_data["id"]
    try:
        song = dj_graph.get_song(file_path)
    except KeyError:
        return f"Unknown node: {file_path}"

    neighbours = dj_graph.neighbours(file_path)
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
    app.run(debug=True, host="0.0.0.0", port=8050)
