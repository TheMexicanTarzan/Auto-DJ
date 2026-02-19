"""
graph.py — Mixing graph and pathfinding for the DJ Mixing Pathfinding System.

This is the final piece of the pipeline.  It takes the fully-analysed Song
objects and the transition-cost function from metrics.py, wires them into a
weighted directed graph via NetworkX, and exposes Dijkstra-based pathfinding
so we can answer: "What is the smoothest sequence of transitions from
Song A to Song B?"

Architecture
------------
    Songs  ──►  DJGraph.build()  ──►  NetworkX DiGraph
                                          │
                   query ──►  get_shortest_path(src, dst)
                                          │
                                     list[Song]  (ordered mix path)

Each *node* in the graph is a Song, keyed by its file_path (the only
guaranteed-unique identifier).  Each *directed edge* A→B carries the
transition weight returned by calculate_weight(A, B).

We use a directed graph because, although the current cost function is
symmetric, real-world DJ transitions are not always reversible at equal
quality (e.g. energy ramp-ups vs. ramp-downs).  Building directed edges
from the start means the graph is ready for asymmetric cost functions
without structural changes.

Edges with infinite weight (unmixable tempo gap) are simply not created,
which is equivalent to "no road exists between these two nodes" in a
road-network analogy.  Dijkstra's algorithm naturally handles this:
if no finite-cost path exists, it reports the target as unreachable.
"""

from __future__ import annotations

import logging
import math
from itertools import permutations

import networkx as nx

from src.metrics import calculate_weight
from src.models import Song

logger = logging.getLogger(__name__)


class DJGraph:
    """
    A weighted directed graph where nodes are songs and edge weights
    represent transition cost (lower = smoother mix).

    Attributes:
        graph:    The underlying NetworkX DiGraph.
        _songs:   Lookup dict mapping file_path → Song for quick access.
    """

    def __init__(self) -> None:
        self.graph: nx.DiGraph = nx.DiGraph()
        self._songs: dict[str, Song] = {}

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

        For every ordered pair (A, B) where A ≠ B, we compute the
        transition cost.  If the cost is finite (i.e. the pair is
        mixable), we add a directed edge A→B with that cost as the
        weight.  Infinite-cost pairs are skipped — they simply have
        no connecting edge.

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
        return instance

    def _add_songs(self, songs: list[Song]) -> None:
        """Register each song as a node, storing it in our lookup dict."""
        for song in songs:
            self._songs[song.file_path] = song
            # Attach song metadata to the node so NetworkX visualisation
            # tools can display it if needed.
            self.graph.add_node(
                song.file_path,
                filename=song.filename,
                bpm=song.bpm,
                key=song.key,
            )

        logger.info("Added %d node(s) to the graph.", len(songs))

    def _add_edges(self, *, weights: dict[str, float] | None = None) -> None:
        """
        Evaluate every ordered song pair and add finite-cost edges.

        We iterate over all permutations (not combinations) because
        the graph is directed: the cost of A→B may differ from B→A
        with future asymmetric metrics.
        """
        edge_count = 0
        skipped = 0
        songs = list(self._songs.values())

        for song_a, song_b in permutations(songs, 2):
            cost = calculate_weight(song_a, song_b, weights=weights)

            if math.isinf(cost):
                skipped += 1
                continue

            self.graph.add_edge(
                song_a.file_path,
                song_b.file_path,
                weight=cost,
            )
            edge_count += 1

        logger.info(
            "Added %d edge(s) to the graph (%d unmixable pair(s) skipped).",
            edge_count,
            skipped,
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_song(self, identifier: str) -> Song:
        """
        Look up a Song by file_path or filename.

        Tries an exact file_path match first (O(1) dict lookup).
        Falls back to a linear scan over filenames so callers can
        use the shorter, friendlier name.

        Raises:
            KeyError: If no song matches.
        """
        # Fast path: direct file_path match
        if identifier in self._songs:
            return self._songs[identifier]

        # Slow path: match by filename
        for song in self._songs.values():
            if song.filename == identifier:
                return song

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
          - NetworkX's implementation runs in O((V + E) log V) with a
            min-heap, which is efficient for our graph sizes.

        Args:
            source_id: file_path or filename of the starting track.
            target_id: file_path or filename of the destination track.

        Returns:
            A tuple of (path, total_cost) where:
              - path is an ordered list of Song objects from source to target.
              - total_cost is the sum of edge weights along that path.

        Raises:
            KeyError:  If source or target is not in the graph.
            nx.NetworkXNoPath:  If no finite-cost route exists between
                                the two songs (e.g. tempo gap is too large
                                at every intermediate step).
        """
        source = self.get_song(source_id)
        target = self.get_song(target_id)

        # Dijkstra returns the list of node IDs (file_paths) on the
        # shortest path.  The 'weight' parameter tells it which edge
        # attribute to minimise.
        node_path = nx.dijkstra_path(
            self.graph,
            source=source.file_path,
            target=target.file_path,
            weight="weight",
        )
        total_cost = nx.dijkstra_path_length(
            self.graph,
            source=source.file_path,
            target=target.file_path,
            weight="weight",
        )

        # Convert node IDs back to Song objects
        song_path = [self._songs[fp] for fp in node_path]

        return song_path, total_cost

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    @property
    def num_nodes(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def num_edges(self) -> int:
        return self.graph.number_of_edges()

    @property
    def songs(self) -> list[Song]:
        """All songs in the graph, ordered by filename."""
        return sorted(self._songs.values(), key=lambda s: s.filename)

    def edge_weight(self, source_id: str, target_id: str) -> float:
        """Return the weight of a specific edge, or inf if no edge exists."""
        src = self.get_song(source_id)
        dst = self.get_song(target_id)
        try:
            return self.graph[src.file_path][dst.file_path]["weight"]
        except KeyError:
            return float("inf")

    def neighbours(self, song_id: str) -> list[tuple[Song, float]]:
        """
        Return all songs reachable in one hop from the given song,
        sorted by transition cost (cheapest first).
        """
        song = self.get_song(song_id)
        result = []
        for _, neighbour_fp, data in self.graph.edges(song.file_path, data=True):
            result.append((self._songs[neighbour_fp], data["weight"]))
        return sorted(result, key=lambda x: x[1])
