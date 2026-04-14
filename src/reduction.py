"""
reduction.py — UMAP dimensionality reduction for semantic embeddings.

Projects the high-dimensional MERT embeddings into a compact 32-dim manifold
space using UMAP with cosine distance.  The reduced embeddings are
L2-normalised so that the existing cosine-similarity edge-weight pipeline
works unchanged on the new space.

Usage
-----
    from src.reduction import build_umap_graph

    umap_graph = build_umap_graph(
        original_graph,
        n_neighbors=15,
        min_dist=0.1,
    )

Why 32 dimensions?
------------------
32 dimensions is a practical sweet spot: enough to retain harmonic and
timbral manifold structure that UMAP extracts from the 768-dim MERT space,
but small enough that concentration-of-measure effects (which dilute cosine
similarity in very high dimensions) are substantially reduced.  The output
is L2-normalised so cosine similarity remains the natural distance measure.
"""

from __future__ import annotations

import logging
from dataclasses import replace

import numpy as np

logger = logging.getLogger(__name__)

def build_umap_graph(
    source_graph: "DJGraph",  # noqa: F821 — forward ref; graph imported lazily
    *,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    n_components: int = 32,
    weights: dict[str, float] | None = None,
    type_penalties: dict[str, float] | None = None,
) -> "DJGraph":
    """
    Fit UMAP on the source graph's song embeddings, substitute the reduced
    (32-dim, L2-normalised) embeddings into copies of the songs, build a
    fresh DJGraph with UMAP-derived edges, and return it.

    The source graph is never modified.

    Args:
        source_graph:   A fully-built DJGraph whose Song objects carry the
                        original (768-dim) MERT embeddings.
        n_neighbors:    UMAP local neighbourhood size.  Lower values (5–15)
                        produce tighter local clusters; higher values (50–100)
                        preserve more global structure.
        min_dist:       Minimum distance between points in the low-dim output
                        space (0.0–1.0).  Smaller values pack similar songs
                        more tightly; larger values spread them more evenly.
        n_components:   Output dimensionality (default 32).  Higher values
                        preserve more information; lower values compress
                        further and reduce concentration-of-measure effects.
        weights:        Metric blend coefficients forwarded to DJGraph.build
                        (harmonic / tempo / semantic).  None = defaults.
        type_penalties: Additive type penalties forwarded to DJGraph.build
                        (double / triplet).  None = defaults.

    Returns:
        A new DJGraph whose Song objects carry n_components-dim UMAP
        embeddings and whose edges are computed from cosine similarity
        in that space.

    Raises:
        ImportError:  If umap-learn is not installed.
        ValueError:   If the source graph contains no songs.
    """
    try:
        from umap import UMAP
    except ImportError as exc:
        raise ImportError(
            "umap-learn is required for UMAP reduction. "
            "Install it with: pip install umap-learn"
        ) from exc

    # Local import avoids circular dependency (graph → metrics → models).
    from src.graph import DJGraph

    songs = source_graph.songs  # sorted list[Song]
    if not songs:
        raise ValueError("Source graph contains no songs.")

    embeddings = np.stack([s.embedding for s in songs])  # (N, D)
    n_songs, n_dims = embeddings.shape
    logger.info(
        "Fitting UMAP: %d songs × %d-dim → %d-dim  "
        "(n_neighbors=%d, min_dist=%.2f) …",
        n_songs, n_dims, n_components, n_neighbors, min_dist,
    )

    reducer = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric="cosine",
        random_state=42,
        low_memory=True,
    )
    reduced = reducer.fit_transform(embeddings)  # (N, 32), float64

    # L2-normalise so cosine similarity in the reduced space is well-defined.
    norms = np.linalg.norm(reduced, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    reduced = (reduced / norms).astype(np.float32)

    logger.info("UMAP fit complete. Building graph from UMAP embeddings …")

    # Swap each song's embedding for its UMAP-reduced counterpart.
    umap_songs = [replace(s, embedding=reduced[i]) for i, s in enumerate(songs)]

    # DJGraph.build computes all edges and the FR layout from scratch.
    return DJGraph.build(umap_songs, weights=weights, type_penalties=type_penalties)
