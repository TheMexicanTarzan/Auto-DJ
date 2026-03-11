"""
graph.py — Mixing graph and pathfinding for the DJ Mixing Pathfinding System.

This is the final piece of the pipeline.  It takes the fully-analysed Song
objects and the transition-cost function from metrics.py, wires them into a
weighted undirected graph via igraph, and exposes Dijkstra-based pathfinding
so we can answer: "What is the smoothest sequence of transitions from
Song A to Song B?"

Architecture
------------
    Songs  ──►  DJGraph.build()  ──►  igraph Graph (undirected)
                                          │
                   query ──►  get_shortest_path(src, dst)
                                          │
                                     list[Song]  (ordered mix path)

Each *node* in the graph is a Song, keyed by its file_path (the only
guaranteed-unique identifier).  Each *undirected edge* {A, B} carries the
transition weight computed from the three symmetric cost components
(harmonic distance, tempo penalty, semantic distance).

Because all three metrics live in a metric space (distance(A, B) ==
distance(B, A)), a single undirected edge per pair fully captures the
transition cost.  This halves edge count, memory, and Dijkstra traversal
work compared to a directed representation.

Edges with infinite weight (unmixable tempo gap) are simply not created,
which is equivalent to "no road exists between these two nodes" in a
road-network analogy.  Dijkstra's algorithm naturally handles this:
if no finite-cost path exists, it reports the target as unreachable.
"""

from __future__ import annotations

import base64
import json
import logging
import pickle
from pathlib import Path

import igraph
import numpy as np

from src.metrics import batch_calculate_weights, batch_calculate_weights_incremental
from src.models import Song

logger = logging.getLogger(__name__)


class NoPathError(Exception):
    """Raised when no finite-cost path exists between two songs."""


class DJGraph:
    """
    A weighted undirected graph where nodes are songs and edge weights
    represent transition cost (lower = smoother mix).

    All cost components are symmetric, so a single undirected edge per
    pair is sufficient and halves storage and traversal work.

    Attributes:
        graph:    The underlying igraph Graph (undirected).
        _songs:   Lookup dict mapping file_path → Song for quick access.
        _layout:  Cached 2D layout coordinates {file_path: (x, y)}.
    """

    def __init__(self) -> None:
        self.graph: igraph.Graph = igraph.Graph(directed=False)
        self._songs: dict[str, Song] = {}
        self._filename_index: dict[str, Song] = {}  # filename → Song (O(1) lookup)
        self._layout: dict[str, tuple[float, float]] = {}
        self._sorted_songs_cache: list[Song] | None = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        songs: list[Song],
        *,
        weights: dict[str, float] | None = None,
    ) -> DJGraph:
        """
        Factory that creates a fully-wired DJGraph from a list of songs.

        For every unordered pair {A, B} where A ≠ B, we compute the
        transition cost.  If the cost is finite (i.e. the pair is
        mixable), we add an undirected edge {A, B} with that cost as
        the weight.  Infinite-cost pairs are skipped — they simply
        have no connecting edge.

        Complexity: O(n²) edge evaluations for n songs.  This is
        inherent to building a dense graph; for very large libraries
        a k-nearest-neighbours pre-filter could reduce this.

        Args:
            songs:   List of Song objects to become nodes.
            weights: Optional metric blending coefficients forwarded
                     to calculate_weight().

        Returns:
            A fully constructed DJGraph instance.
        """
        instance = cls()
        instance._add_songs(songs)
        instance._add_edges(weights=weights)
        instance.compute_layout()
        return instance

    def _add_songs(self, songs: list[Song]) -> None:
        """Register each song as a node using batch vertex addition."""
        if not songs:
            return

        for song in songs:
            self._songs[song.file_path] = song
            self._filename_index[song.filename] = song

        # Batch vertex addition — much faster than per-vertex add_vertex()
        n = len(songs)
        self.graph.add_vertices(n)
        self.graph.vs["name"] = [s.file_path for s in songs]
        self.graph.vs["filename"] = [s.filename for s in songs]
        self.graph.vs["bpm"] = [s.bpm for s in songs]
        self.graph.vs["key"] = [s.key for s in songs]

        self._sorted_songs_cache = None  # invalidate cache
        logger.info("Added %d node(s) to the graph.", n)

    def _add_edges(self, *, weights: dict[str, float] | None = None) -> None:
        """
        Evaluate every unordered song pair and add finite-cost edges.

        Uses vectorised numpy operations (batch_calculate_weights) to
        compute the full N×N weight matrix in bulk, replacing the O(n²)
        Python loop with BLAS-backed matrix multiplication and numpy
        broadcasting.  This yields ~50-100× speedup for libraries of
        500+ songs.
        """
        songs = list(self._songs.values())
        n = len(songs)

        if n < 2:
            logger.info("Added 0 edge(s) to the graph (fewer than 2 songs).")
            return

        # Compute the full weight matrix in one vectorised pass
        weight_matrix, finite_mask = batch_calculate_weights(songs, weights=weights)

        # Extract (row, col) indices of finite-weight pairs
        src_indices, dst_indices = np.where(finite_mask)

        edge_count = len(src_indices)
        skipped = n * (n - 1) // 2 - edge_count

        if edge_count > 0:
            # Build edge list using igraph vertex indices (which match
            # the insertion order from _add_songs)
            edge_list = list(zip(src_indices.tolist(), dst_indices.tolist()))
            weight_list = weight_matrix[src_indices, dst_indices].tolist()

            self.graph.add_edges(edge_list)
            self.graph.es["weight"] = weight_list

        logger.info(
            "Added %d edge(s) to the graph (%d unmixable pair(s) skipped).",
            edge_count,
            skipped,
        )

    # ------------------------------------------------------------------
    # Incremental updates
    # ------------------------------------------------------------------

    @property
    def known_hashes(self) -> set[str]:
        """Return the set of content hashes for all songs in the graph."""
        return {s.content_hash for s in self._songs.values() if s.content_hash}

    def add_songs_incremental(
        self,
        new_songs: list[Song],
        *,
        weights: dict[str, float] | None = None,
    ) -> None:
        """
        Add new songs to an existing graph, computing only the edges
        involving the new songs (new×existing + new×new).

        This is O(n_new × n_total) instead of O(n_total²).
        """
        if not new_songs:
            return

        existing_songs = list(self._songs.values())
        n_old = len(existing_songs)

        # Build index mapping: existing song file_path → igraph vertex index
        old_vertex_indices = {}
        if n_old > 0:
            for i, v in enumerate(self.graph.vs):
                old_vertex_indices[v["name"]] = v.index

        # Register new songs as vertices
        first_new_idx = self.graph.vcount()
        self.graph.add_vertices(len(new_songs))
        for i, song in enumerate(new_songs):
            v_idx = first_new_idx + i
            self.graph.vs[v_idx]["name"] = song.file_path
            self.graph.vs[v_idx]["filename"] = song.filename
            self.graph.vs[v_idx]["bpm"] = song.bpm
            self.graph.vs[v_idx]["key"] = song.key
            self._songs[song.file_path] = song
            self._filename_index[song.filename] = song

        # Compute edges incrementally
        cross_edges, cross_weights, int_edges, int_weights = (
            batch_calculate_weights_incremental(
                new_songs, existing_songs, weights=weights,
            )
        )

        # Translate local indices to igraph vertex indices
        edge_list: list[tuple[int, int]] = []
        weight_list: list[float] = []

        # Cross edges: (new_local_idx, existing_local_idx) → (igraph_idx, igraph_idx)
        existing_v_indices = [old_vertex_indices[s.file_path] for s in existing_songs] if n_old > 0 else []
        for new_local, old_local in cross_edges:
            edge_list.append((first_new_idx + new_local, existing_v_indices[old_local]))
        weight_list.extend(cross_weights)

        # Internal edges: (new_local_i, new_local_j)
        for ni, nj in int_edges:
            edge_list.append((first_new_idx + ni, first_new_idx + nj))
        weight_list.extend(int_weights)

        if edge_list:
            existing_edge_count = self.graph.ecount()
            self.graph.add_edges(edge_list)
            for i, w in enumerate(weight_list):
                self.graph.es[existing_edge_count + i]["weight"] = w

        self._sorted_songs_cache = None
        self.compute_layout()

        logger.info(
            "Incremental update: added %d song(s), %d edge(s).",
            len(new_songs),
            len(edge_list),
        )

    def remove_songs(self, hashes_to_remove: set[str]) -> None:
        """
        Remove songs whose content_hash is in *hashes_to_remove*.

        Deletes the corresponding vertices (and all incident edges)
        from the igraph graph.
        """
        if not hashes_to_remove:
            return

        # Find vertex indices to remove
        vertices_to_delete = []
        songs_to_delete = []
        for song in self._songs.values():
            if song.content_hash in hashes_to_remove:
                try:
                    v = self.graph.vs.find(name=song.file_path)
                    vertices_to_delete.append(v.index)
                except ValueError:
                    pass
                songs_to_delete.append(song)

        if not songs_to_delete:
            return

        # Remove from internal dicts
        for song in songs_to_delete:
            self._songs.pop(song.file_path, None)
            self._filename_index.pop(song.filename, None)
            self._layout.pop(song.file_path, None)

        # Delete vertices (igraph removes incident edges automatically)
        # Must delete in reverse index order to avoid index shifting
        self.graph.delete_vertices(sorted(vertices_to_delete, reverse=True))

        self._sorted_songs_cache = None

        logger.info(
            "Removed %d song(s) and their edges from the graph.",
            len(songs_to_delete),
        )

    # ------------------------------------------------------------------
    # Layout computation
    # ------------------------------------------------------------------

    def compute_layout(
        self, algorithm: str = "fruchterman_reingold",
    ) -> dict[str, tuple[float, float]]:
        """
        Compute 2D layout coordinates using an igraph layout algorithm.

        Args:
            algorithm: Layout algorithm name (e.g. 'fruchterman_reingold',
                       'kamada_kawai', 'drl', 'lgl').

        Returns:
            Dict mapping file_path → (x, y) coordinates.
        """
        if self.graph.vcount() == 0:
            self._layout = {}
            return self._layout

        layout = self.graph.layout(algorithm)

        self._layout = {}
        for v in self.graph.vs:
            self._layout[v["name"]] = (layout[v.index][0], layout[v.index][1])

        return self._layout

    @property
    def layout_coords(self) -> dict[str, tuple[float, float]]:
        """Cached layout coordinates. Computed on first access if needed."""
        if not self._layout and self.graph.vcount() > 0:
            self.compute_layout()
        return self._layout

    # ------------------------------------------------------------------
    # Serialization (caching)
    # ------------------------------------------------------------------

    def save_to_json(self, path: str | Path) -> None:
        """
        Serialize the graph to a JSON file.

        Embeddings are stored as base64-encoded float32 blobs rather than
        JSON number arrays.  This reduces file size by ~4× and speeds up
        both serialization and deserialization substantially.  The compact
        (no-indent) format further reduces I/O overhead.
        """
        path = Path(path)

        songs_data = []
        for song in self._songs.values():
            emb_bytes = song.embedding.astype(np.float32).tobytes()
            songs_data.append({
                "file_path": song.file_path,
                "filename": song.filename,
                "bpm": song.bpm,
                "key": song.key,
                "embedding_b64": base64.b64encode(emb_bytes).decode("ascii"),
                "content_hash": song.content_hash,
            })

        edges_data = []
        for edge in self.graph.es:
            edges_data.append({
                "source": self.graph.vs[edge.source]["name"],
                "target": self.graph.vs[edge.target]["name"],
                "weight": edge["weight"],
            })

        layout_data = {k: list(v) for k, v in self.layout_coords.items()}

        payload = {
            "songs": songs_data,
            "edges": edges_data,
            "layout": layout_data,
        }
        # Compact JSON (no indent) — much faster to write and ~30% smaller
        path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        logger.info("Graph saved to '%s'.", path)

    @classmethod
    def load_from_json(cls, path: str | Path) -> DJGraph:
        """
        Reconstruct a DJGraph from a JSON cache file previously
        written by ``save_to_json()``.

        Song objects and their numpy embeddings are fully restored so
        that pathfinding and weight queries work identically to a
        freshly-built graph.
        """
        path = Path(path)
        raw = json.loads(path.read_text(encoding="utf-8"))

        instance = cls()

        # Rebuild Song objects (supports both legacy list and b64 embeddings)
        song_list = []
        for s in raw["songs"]:
            if "embedding_b64" in s:
                emb = np.frombuffer(
                    base64.b64decode(s["embedding_b64"]), dtype=np.float32
                ).copy()
            else:
                emb = np.array(s["embedding"], dtype=np.float32)
            song = Song(
                file_path=s["file_path"],
                filename=s["filename"],
                bpm=s["bpm"],
                key=s["key"],
                embedding=emb,
                content_hash=s.get("content_hash", ""),
            )
            instance._songs[song.file_path] = song
            instance._filename_index[song.filename] = song
            song_list.append(song)

        # Batch vertex addition
        if song_list:
            instance.graph.add_vertices(len(song_list))
            instance.graph.vs["name"] = [s.file_path for s in song_list]
            instance.graph.vs["filename"] = [s.filename for s in song_list]
            instance.graph.vs["bpm"] = [s.bpm for s in song_list]
            instance.graph.vs["key"] = [s.key for s in song_list]

        # Rebuild edges.
        if raw["edges"]:
            edge_list = [(e["source"], e["target"]) for e in raw["edges"]]
            weight_list = [e["weight"] for e in raw["edges"]]
            instance.graph.add_edges(edge_list)
            instance.graph.es["weight"] = weight_list

        # Restore layout if present, otherwise compute it.
        if "layout" in raw and raw["layout"]:
            instance._layout = {k: tuple(v) for k, v in raw["layout"].items()}
        else:
            instance.compute_layout()

        logger.info(
            "Graph loaded from '%s': %d node(s), %d edge(s).",
            path,
            instance.num_nodes,
            instance.num_edges,
        )
        return instance

    # ------------------------------------------------------------------
    # Pickle serialization (faster binary format)
    # ------------------------------------------------------------------

    def save_to_pickle(self, path: str | Path) -> None:
        """
        Serialize the graph to a pickle file.

        Pickle natively handles numpy arrays without conversion, making
        it ~10x faster than JSON for cache load/save.
        """
        path = Path(path)

        songs_data = []
        for song in self._songs.values():
            songs_data.append({
                "file_path": song.file_path,
                "filename": song.filename,
                "bpm": song.bpm,
                "key": song.key,
                "embedding": song.embedding,
                "content_hash": song.content_hash,
            })

        edges_data = []
        for edge in self.graph.es:
            edges_data.append({
                "source": self.graph.vs[edge.source]["name"],
                "target": self.graph.vs[edge.target]["name"],
                "weight": edge["weight"],
            })

        layout_data = {k: v for k, v in self.layout_coords.items()}

        payload = {
            "songs": songs_data,
            "edges": edges_data,
            "layout": layout_data,
        }

        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Graph saved to '%s'.", path)

    @classmethod
    def load_from_pickle(cls, path: str | Path) -> DJGraph:
        """
        Reconstruct a DJGraph from a pickle cache file previously
        written by ``save_to_pickle()``.
        """
        path = Path(path)
        with open(path, "rb") as f:
            raw = pickle.load(f)  # noqa: S301

        instance = cls()

        song_list = []
        for s in raw["songs"]:
            emb = np.asarray(s["embedding"], dtype=np.float32)
            song = Song(
                file_path=s["file_path"],
                filename=s["filename"],
                bpm=s["bpm"],
                key=s["key"],
                embedding=emb,
                content_hash=s.get("content_hash", ""),
            )
            instance._songs[song.file_path] = song
            instance._filename_index[song.filename] = song
            song_list.append(song)

        if song_list:
            instance.graph.add_vertices(len(song_list))
            instance.graph.vs["name"] = [s.file_path for s in song_list]
            instance.graph.vs["filename"] = [s.filename for s in song_list]
            instance.graph.vs["bpm"] = [s.bpm for s in song_list]
            instance.graph.vs["key"] = [s.key for s in song_list]

        if raw["edges"]:
            edge_list = [(e["source"], e["target"]) for e in raw["edges"]]
            weight_list = [e["weight"] for e in raw["edges"]]
            instance.graph.add_edges(edge_list)
            instance.graph.es["weight"] = weight_list

        if "layout" in raw and raw["layout"]:
            instance._layout = {k: tuple(v) for k, v in raw["layout"].items()}
        else:
            instance.compute_layout()

        logger.info(
            "Graph loaded from '%s': %d node(s), %d edge(s).",
            path,
            instance.num_nodes,
            instance.num_edges,
        )
        return instance

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_song(self, identifier: str) -> Song:
        """
        Look up a Song by file_path or filename.

        Both lookups are O(1) dict operations.

        Raises:
            KeyError: If no song matches.
        """
        # Fast path: direct file_path match
        if identifier in self._songs:
            return self._songs[identifier]

        # Fast path: filename match via dedicated index
        if identifier in self._filename_index:
            return self._filename_index[identifier]

        raise KeyError(
            f"No song found for '{identifier}'. "
            f"Available: {[s.filename for s in self._songs.values()]}"
        )

    def get_shortest_path(
        self,
        source_id: str,
        target_id: str,
    ) -> tuple[list[Song], float]:
        """
        Find the lowest-cost sequence of transitions from source to target
        using Dijkstra's algorithm.

        Dijkstra's algorithm is the right choice here because:
          - All edge weights are non-negative (costs are in [0, 1]).
          - We want the single-source shortest path, not all-pairs.
          - igraph's C-based implementation is highly optimised.

        Args:
            source_id: file_path or filename of the starting track.
            target_id: file_path or filename of the destination track.

        Returns:
            A tuple of (path, total_cost) where:
              - path is an ordered list of Song objects from source to target.
              - total_cost is the sum of edge weights along that path.

        Raises:
            KeyError:      If source or target is not in the graph.
            NoPathError:   If no finite-cost route exists between
                           the two songs.
        """
        source = self.get_song(source_id)
        target = self.get_song(target_id)

        src_v = self.graph.vs.find(name=source.file_path)
        tgt_v = self.graph.vs.find(name=target.file_path)

        # Self-path: zero cost, single node
        if src_v.index == tgt_v.index:
            return [source], 0.0

        # If no edges exist, there is definitely no path
        if self.graph.ecount() == 0:
            raise NoPathError(
                f"No path between '{source.filename}' and '{target.filename}'"
            )

        paths = self.graph.get_shortest_paths(
            src_v.index,
            to=tgt_v.index,
            weights="weight",
            output="vpath",
        )

        if not paths or not paths[0]:
            raise NoPathError(
                f"No path between '{source.filename}' and '{target.filename}'"
            )

        node_indices = paths[0]
        song_path = [self._songs[self.graph.vs[i]["name"]] for i in node_indices]

        # Compute total cost from edge weights
        total_cost = 0.0
        for i in range(len(node_indices) - 1):
            eid = self.graph.get_eid(node_indices[i], node_indices[i + 1])
            total_cost += self.graph.es[eid]["weight"]

        return song_path, total_cost

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    @property
    def num_nodes(self) -> int:
        return self.graph.vcount()

    @property
    def num_edges(self) -> int:
        return self.graph.ecount()

    @property
    def songs(self) -> list[Song]:
        """All songs in the graph, ordered by filename (cached)."""
        if self._sorted_songs_cache is None:
            self._sorted_songs_cache = sorted(
                self._songs.values(), key=lambda s: s.filename
            )
        return self._sorted_songs_cache

    def edge_weight(self, source_id: str, target_id: str) -> float:
        """Return the weight of a specific edge, or inf if no edge exists."""
        src = self.get_song(source_id)
        dst = self.get_song(target_id)
        src_v = self.graph.vs.find(name=src.file_path)
        dst_v = self.graph.vs.find(name=dst.file_path)
        eid = self.graph.get_eid(
            src_v.index, dst_v.index, error=False,
        )
        if eid < 0:
            return float("inf")
        return self.graph.es[eid]["weight"]

    def neighbours(self, song_id: str) -> list[tuple[Song, float]]:
        """
        Return all songs reachable in one hop from the given song,
        sorted by transition cost (cheapest first).
        """
        song = self.get_song(song_id)
        v = self.graph.vs.find(name=song.file_path)
        result = []
        for eid in self.graph.incident(v, mode="all"):
            edge = self.graph.es[eid]
            # In an undirected graph, pick the *other* endpoint
            other_idx = edge.target if edge.source == v.index else edge.source
            other_v = self.graph.vs[other_idx]
            result.append((self._songs[other_v["name"]], edge["weight"]))
        return sorted(result, key=lambda x: x[1])
