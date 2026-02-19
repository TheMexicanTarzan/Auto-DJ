"""
Unit tests for the DJGraph class and pathfinding logic.

All tests use synthetic Song objects with controlled BPM, key, and
embeddings so that transition costs are deterministic and predictable.
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pytest

from src.graph import DJGraph
from src.models import Song


# =========================================================================
# Helpers
# =========================================================================


def _song(
    name: str,
    bpm: float = 120.0,
    key: str = "C major",
    embedding: np.ndarray | None = None,
) -> Song:
    """Build a minimal Song with a unique file_path derived from *name*."""
    if embedding is None:
        # Deterministic embedding seeded by name so each song is unique
        rng = np.random.RandomState(hash(name) % 2**31)
        embedding = rng.randn(512).astype(np.float32)
    return Song(
        file_path=f"/music/{name}",
        filename=name,
        bpm=bpm,
        key=key,
        embedding=embedding,
    )


# A reusable embedding so "identical vibe" tests can share it
_SHARED_EMB = np.random.RandomState(42).randn(512).astype(np.float32)


# =========================================================================
# DJGraph construction
# =========================================================================


class TestGraphBuild:
    def test_empty_list_produces_empty_graph(self):
        g = DJGraph.build([])
        assert g.num_nodes == 0
        assert g.num_edges == 0

    def test_single_song_has_no_edges(self):
        g = DJGraph.build([_song("solo.mp3")])
        assert g.num_nodes == 1
        assert g.num_edges == 0

    def test_two_compatible_songs_get_bidirectional_edges(self):
        a = _song("a.mp3", bpm=120, key="C major", embedding=_SHARED_EMB)
        b = _song("b.mp3", bpm=122, key="G major", embedding=_SHARED_EMB)
        g = DJGraph.build([a, b])

        assert g.num_nodes == 2
        # Both A→B and B→A should exist (directed graph)
        assert g.num_edges == 2

    def test_unmixable_tempo_produces_no_edge(self):
        # 100 vs 130 BPM = 30% diff — way over the 8% threshold
        a = _song("slow.mp3", bpm=100, key="C major", embedding=_SHARED_EMB)
        b = _song("fast.mp3", bpm=130, key="C major", embedding=_SHARED_EMB)
        g = DJGraph.build([a, b])

        assert g.num_nodes == 2
        assert g.num_edges == 0

    def test_node_metadata_stored(self):
        s = _song("meta.mp3", bpm=140, key="D minor")
        g = DJGraph.build([s])

        node_data = g.graph.nodes[s.file_path]
        assert node_data["filename"] == "meta.mp3"
        assert node_data["bpm"] == 140
        assert node_data["key"] == "D minor"

    def test_edge_weight_is_finite_and_positive(self):
        a = _song("x.mp3", bpm=125, key="A minor", embedding=_SHARED_EMB)
        b = _song("y.mp3", bpm=126, key="A minor", embedding=_SHARED_EMB)
        g = DJGraph.build([a, b])

        w = g.edge_weight("x.mp3", "y.mp3")
        assert 0.0 <= w < float("inf")


# =========================================================================
# Song lookup
# =========================================================================


class TestGetSong:
    def test_lookup_by_file_path(self):
        s = _song("track.mp3")
        g = DJGraph.build([s])
        assert g.get_song("/music/track.mp3") is s

    def test_lookup_by_filename(self):
        s = _song("track.mp3")
        g = DJGraph.build([s])
        assert g.get_song("track.mp3") is s

    def test_unknown_raises_key_error(self):
        g = DJGraph.build([_song("a.mp3")])
        with pytest.raises(KeyError, match="nope"):
            g.get_song("nope.mp3")


# =========================================================================
# Shortest path (Dijkstra)
# =========================================================================


class TestShortestPath:
    def test_direct_neighbours(self):
        """Two compatible songs → path is just [source, target]."""
        a = _song("start.mp3", bpm=120, key="C major", embedding=_SHARED_EMB)
        b = _song("end.mp3", bpm=121, key="C major", embedding=_SHARED_EMB)
        g = DJGraph.build([a, b])

        path, cost = g.get_shortest_path("start.mp3", "end.mp3")
        assert len(path) == 2
        assert path[0].filename == "start.mp3"
        assert path[1].filename == "end.mp3"
        assert cost >= 0.0

    def test_multi_hop_path(self):
        """
        A → B → C where A and C are NOT directly connected (tempo too far),
        but both are compatible with B.  Dijkstra should route through B.

             A (100 BPM)  ─→  B (105 BPM)  ─→  C (110 BPM)
                 ╳ direct A→C (10% gap = unmixable)
        """
        emb = _SHARED_EMB
        a = _song("a.mp3", bpm=100, key="C major", embedding=emb)
        b = _song("b.mp3", bpm=105, key="C major", embedding=emb)
        c = _song("c.mp3", bpm=110, key="C major", embedding=emb)
        g = DJGraph.build([a, b, c])

        # A→C directly is unmixable (10%), but A→B→C is fine
        path, cost = g.get_shortest_path("a.mp3", "c.mp3")
        assert len(path) == 3
        assert path[0].filename == "a.mp3"
        assert path[1].filename == "b.mp3"
        assert path[2].filename == "c.mp3"

    def test_no_path_raises(self):
        """Completely disconnected songs should raise NetworkXNoPath."""
        a = _song("island_a.mp3", bpm=80, embedding=_SHARED_EMB)
        b = _song("island_b.mp3", bpm=160, embedding=_SHARED_EMB)
        g = DJGraph.build([a, b])

        with pytest.raises(nx.NetworkXNoPath):
            g.get_shortest_path("island_a.mp3", "island_b.mp3")

    def test_path_to_self(self):
        """A song to itself is a zero-cost, single-node path."""
        s = _song("self.mp3", bpm=128, key="E minor", embedding=_SHARED_EMB)
        g = DJGraph.build([s])

        path, cost = g.get_shortest_path("self.mp3", "self.mp3")
        assert len(path) == 1
        assert cost == 0.0

    def test_chooses_lower_cost_route(self):
        """
        Given two routes to the same target, Dijkstra should pick
        the one with lower total cost.

        A ──(high cost)──► C
        A ──(low)──► B ──(low)──► C

        We control cost via key clash: A and C have a tritone clash,
        but B is harmonically compatible with both.
        """
        emb = _SHARED_EMB
        a = _song("a.mp3", bpm=120, key="C major", embedding=emb)
        b = _song("b.mp3", bpm=120, key="G major", embedding=emb)
        c = _song("c.mp3", bpm=120, key="D major", embedding=emb)
        g = DJGraph.build([a, b, c])

        # Both direct and via-B paths exist; check the algorithm picks
        # the cheapest total cost
        path, cost = g.get_shortest_path("a.mp3", "c.mp3")
        direct_cost = g.edge_weight("a.mp3", "c.mp3")
        assert cost <= direct_cost


# =========================================================================
# Neighbours
# =========================================================================


class TestNeighbours:
    def test_returns_sorted_by_cost(self):
        emb = _SHARED_EMB
        a = _song("hub.mp3", bpm=120, key="C major", embedding=emb)
        b = _song("close.mp3", bpm=120, key="G major", embedding=emb)
        c = _song("far.mp3", bpm=120, key="F# major", embedding=emb)
        g = DJGraph.build([a, b, c])

        nbrs = g.neighbours("hub.mp3")
        # close.mp3 (perfect fifth from C) should come before
        # far.mp3 (tritone from C)
        assert len(nbrs) == 2
        assert nbrs[0][0].filename == "close.mp3"
        assert nbrs[1][0].filename == "far.mp3"
        assert nbrs[0][1] <= nbrs[1][1]

    def test_isolated_node_has_no_neighbours(self):
        a = _song("alone.mp3", bpm=60, embedding=_SHARED_EMB)
        b = _song("other.mp3", bpm=200, embedding=_SHARED_EMB)
        g = DJGraph.build([a, b])

        assert g.neighbours("alone.mp3") == []


# =========================================================================
# Edge weight query
# =========================================================================


class TestEdgeWeight:
    def test_existing_edge(self):
        a = _song("p.mp3", bpm=120, key="C major", embedding=_SHARED_EMB)
        b = _song("q.mp3", bpm=121, key="C major", embedding=_SHARED_EMB)
        g = DJGraph.build([a, b])
        assert g.edge_weight("p.mp3", "q.mp3") < float("inf")

    def test_missing_edge_returns_inf(self):
        a = _song("s.mp3", bpm=80, embedding=_SHARED_EMB)
        b = _song("t.mp3", bpm=160, embedding=_SHARED_EMB)
        g = DJGraph.build([a, b])
        assert math.isinf(g.edge_weight("s.mp3", "t.mp3"))


# =========================================================================
# Properties
# =========================================================================


class TestProperties:
    def test_songs_sorted_by_filename(self):
        songs = [_song("z.mp3"), _song("a.mp3"), _song("m.mp3")]
        # Give them similar BPMs so edges exist
        for s in songs:
            s.bpm = 120
        g = DJGraph.build(songs)
        names = [s.filename for s in g.songs]
        assert names == sorted(names)
