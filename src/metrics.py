"""
metrics.py — Transition cost functions for the DJ Mixing Pathfinding System.

This module answers the question: "How smoothly can a DJ transition from
Song A to Song B?"  The answer is a single float — a *weight* — that will
later become the edge cost in our mixing graph.

The weight is composed of three independent penalties:

    1. Harmonic distance   — are the two keys musically compatible?
    2. Tempo penalty       — are the BPMs close enough to beat-match?
    3. Semantic distance   — do the songs "sound" alike (via CLAP embeddings)?

A weight of 0.0 means a perfect transition.  A weight of float('inf')
means the transition is physically impossible (e.g. tempos too far apart).
"""

from __future__ import annotations

import math

import numpy as np

from src.models import Song

# =========================================================================
# 1. Harmonic Distance  (Circle of Fifths)
# =========================================================================
#
# BACKGROUND
# ----------
# The Circle of Fifths arranges all 12 pitch classes so that *adjacent*
# notes are a perfect fifth apart.  Keys that are close on this circle
# share many notes and blend well; keys on opposite sides (a tritone
# apart) clash harshly.
#
# The circle looks like this (clockwise):
#
#            C
#        F       G
#      Bb          D
#        Eb      A
#          Ab  E
#            Db/C#
#              |
#            F#/Gb     (tritone — max distance from C)
#
# We assign each pitch class a *position* on this circle (0–11), then
# measure the shortest arc between two positions.  The maximum possible
# arc is 6 (the tritone, directly opposite on the 12-point circle).
#
# RELATIVE MAJOR / MINOR
# ----------------------
# Every minor key has a *relative major* that shares the exact same set
# of notes (e.g. A minor ↔ C major).  DJs treat these as harmonically
# identical, so we map every key to its major equivalent before measuring
# distance.  The relative major is always 3 semitones above the minor
# root:  root_semitone_of_minor + 3 = root_semitone_of_relative_major.
#
# PENALTY MAPPING
# ---------------
#   Circle distance 0  → 0.0   (same key / relative major-minor)
#   Circle distance 1  → 0.17  (perfect fifth — very smooth)
#   Circle distance 2  → 0.33
#   Circle distance 3  → 0.50
#   Circle distance 4  → 0.67
#   Circle distance 5  → 0.83
#   Circle distance 6  → 1.0   (tritone — maximum clash)
#
# We simply divide the arc length by 6 to get a value in [0.0, 1.0].
# =========================================================================

# Pitch class name → semitone index (C = 0, C# = 1, ... B = 11).
# Must match the _PITCH_CLASSES list used in models.py.
_SEMITONE_INDEX: dict[str, int] = {
    "C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5,
    "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11,
}


def _key_to_fifths_position(key: str) -> int:
    """
    Convert a key string like 'A minor' or 'G major' into a position
    (0–11) on the Circle of Fifths.

    Steps:
        1. Parse the root note and mode from the string.
        2. If the key is minor, convert to its relative major
           (3 semitones up) so that relative pairs map to the same spot.
        3. Convert the semitone index to a Circle-of-Fifths position
           using the formula:  position = (semitone * 7) % 12
           (multiplying by 7 mod 12 is the mathematical definition of
           walking around the circle in fifth-steps).
    """
    parts = key.strip().split()
    if len(parts) != 2:
        raise ValueError(f"Expected 'Note mode' format (e.g. 'C major'), got: '{key}'")

    root, mode = parts[0], parts[1].lower()
    if root not in _SEMITONE_INDEX:
        raise ValueError(f"Unknown pitch class: '{root}'")

    semitone = _SEMITONE_INDEX[root]

    # Normalise minor → relative major so that e.g. 'A minor' and
    # 'C major' collapse to the same position (both = 0).
    if mode == "minor":
        semitone = (semitone + 3) % 12

    # Semitone → Circle of Fifths position.
    # Each fifth is 7 semitones, so position = (semitone * 7) % 12.
    return (semitone * 7) % 12


def harmonic_distance(key_a: str, key_b: str) -> float:
    """
    Return the harmonic distance between two keys as a float in [0.0, 1.0].

    0.0 = same key (or relative major/minor) — perfect harmonic match.
    1.0 = tritone — maximum harmonic clash.

    The distance is the shortest arc on the 12-point Circle of Fifths,
    normalised by dividing by the maximum possible arc (6).
    """
    pos_a = _key_to_fifths_position(key_a)
    pos_b = _key_to_fifths_position(key_b)

    # Shortest arc on a circle of 12 points: ranges from 0 to 6
    raw_distance = abs(pos_a - pos_b)
    arc = min(raw_distance, 12 - raw_distance)

    # Normalise to [0, 1]
    return arc / 6.0


# =========================================================================
# 2. Tempo Penalty
# =========================================================================
#
# DJs beat-match by nudging one deck's pitch fader.  Most hardware allows
# roughly ±8 % adjustment.  Beyond that the track audibly warps, so we
# treat a >8 % BPM gap as *unmixable* (infinite cost).
#
# Within the 8 % window we apply a simple linear penalty:
#
#     penalty = percent_diff / 8.0          (range: 0.0 – 1.0)
#
# where percent_diff = |bpm_a - bpm_b| / min(bpm_a, bpm_b) * 100.
#
# Using *min* in the denominator is the stricter (more conservative)
# choice — it measures how far the slower track would need to speed up.
# =========================================================================

# Maximum BPM difference (as a percentage) before we call it unmixable
_MAX_BPM_DIFF_PERCENT: float = 8.0


def is_compatible(song_a: Song, song_b: Song) -> bool:
    """
    Quick BPM-based compatibility check between two songs.

    Returns True if the BPM difference is within the mixable threshold
    (≤ 8%), False otherwise.  This is a lightweight gate intended to be
    called *before* the more expensive ``calculate_weight()`` so that
    incompatible pairs can be skipped without computing harmonic or
    semantic distances.
    """
    if song_a.bpm <= 0 or song_b.bpm <= 0:
        return False

    diff = abs(song_a.bpm - song_b.bpm)
    percent_diff = (diff / min(song_a.bpm, song_b.bpm)) * 100.0
    return percent_diff <= _MAX_BPM_DIFF_PERCENT


def tempo_penalty(bpm_a: float, bpm_b: float) -> float:
    """
    Return a tempo-compatibility penalty in [0.0, 1.0], or float('inf')
    if the BPM gap exceeds the mixable threshold.

    Args:
        bpm_a: Tempo of the first track (beats per minute).
        bpm_b: Tempo of the second track (beats per minute).

    Returns:
        0.0        — identical tempos.
        (0, 1.0]   — linearly scaled within the 8 % window.
        inf        — BPM difference exceeds 8 %, unmixable.

    Raises:
        ValueError: If either BPM is zero or negative.
    """
    if bpm_a <= 0 or bpm_b <= 0:
        raise ValueError(f"BPM must be positive, got {bpm_a} and {bpm_b}")

    # Percentage difference relative to the slower track
    diff = abs(bpm_a - bpm_b)
    percent_diff = (diff / min(bpm_a, bpm_b)) * 100.0

    if percent_diff > _MAX_BPM_DIFF_PERCENT:
        return float("inf")

    # Linear scale: 0 % → 0.0, 8 % → 1.0
    return percent_diff / _MAX_BPM_DIFF_PERCENT


# =========================================================================
# 3. Semantic (Embedding) Distance
# =========================================================================
#
# Our Song objects carry a CLAP embedding — a 512-dimensional vector that
# encodes the sonic "vibe" of the track (genre, energy, instrumentation).
#
# Cosine similarity measures angular closeness between two vectors,
# regardless of magnitude:
#
#     cosine_sim = (A · B) / (‖A‖ × ‖B‖)       range: [-1, 1]
#
# We convert this to a *distance* (lower = more similar):
#
#     cosine_distance = 1 - cosine_sim            range: [0, 2]
#
# Then we clamp and normalise to [0, 1] by dividing by 2.  In practice,
# CLAP embeddings are non-negative so similarity stays in [0, 1] and
# distance in [0, 1] naturally, but the clamp handles edge cases.
# =========================================================================


def semantic_distance(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """
    Return the cosine distance between two embedding vectors,
    normalised to [0.0, 1.0].

    0.0 = identical sonic character.
    1.0 = maximally dissimilar.

    Raises:
        ValueError: If either embedding is empty or zero-norm.
    """
    if emb_a.size == 0 or emb_b.size == 0:
        raise ValueError("Cannot compute distance with an empty embedding")

    norm_a = np.linalg.norm(emb_a)
    norm_b = np.linalg.norm(emb_b)

    if norm_a == 0.0 or norm_b == 0.0:
        raise ValueError("Cannot compute cosine distance with a zero-norm vector")

    cosine_sim = float(np.dot(emb_a, emb_b) / (norm_a * norm_b))

    # Cosine distance in [0, 2]; normalise to [0, 1]
    distance = (1.0 - cosine_sim) / 2.0
    return float(np.clip(distance, 0.0, 1.0))


def batch_semantic_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute the full N×N pairwise cosine distance matrix in one vectorised
    operation.  This replaces O(n²) individual ``semantic_distance()`` calls
    with a single matrix multiplication + broadcast, yielding orders-of-
    magnitude speedup for large song libraries.

    Args:
        embeddings: (N, D) array where each row is a song's embedding.

    Returns:
        (N, N) float32 array of pairwise distances in [0.0, 1.0].
        Entry [i][j] is the normalised cosine distance from song i to song j.
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Avoid division by zero for any zero-norm vector (treat as max distance)
    safe_norms = np.where(norms == 0, 1.0, norms)
    normed = embeddings / safe_norms

    # Cosine similarity matrix via single GEMM call: shape (N, N)
    sim_matrix = normed @ normed.T

    # Convert to distance in [0, 1] and clamp floating-point noise
    dist_matrix = (1.0 - sim_matrix) / 2.0
    np.clip(dist_matrix, 0.0, 1.0, out=dist_matrix)
    return dist_matrix.astype(np.float32)


# =========================================================================
# 4. Composite Weight
# =========================================================================
#
# The final edge weight is a weighted sum of the three penalties.
# Default weights reflect DJ priorities:
#
#   - Tempo is king:  if tempos don't match, nothing else matters.
#     (Handled by the inf short-circuit, not the weight itself.)
#   - Harmonic compatibility is next most important for a clean blend.
#   - Sonic similarity (vibe) matters for flow but is more subjective.
#
# The default coefficients (0.35 harmonic, 0.25 tempo, 0.40 semantic)
# can be tuned per use-case.  The result is in [0.0, 1.0] when finite.
# =========================================================================

# Default blending coefficients — must sum to 1.0
DEFAULT_WEIGHTS: dict[str, float] = {
    "harmonic": 0.35,
    "tempo": 0.25,
    "semantic": 0.40,
}


def calculate_weight(
    song_a: Song,
    song_b: Song,
    *,
    weights: dict[str, float] | None = None,
) -> float:
    """
    Calculate the overall transition cost (edge weight) between two songs.

    This is the primary entry point for the metrics module.  It combines
    harmonic, tempo, and semantic penalties into a single float.

    Args:
        song_a:  Source track.
        song_b:  Destination track.
        weights: Optional dict overriding the default blending coefficients.
                 Keys: 'harmonic', 'tempo', 'semantic'.  Values should sum
                 to 1.0 for the result to stay in [0, 1].

    Returns:
        A float in [0.0, 1.0] for mixable pairs, or float('inf') if the
        tempo gap makes the transition impossible.
    """
    w = weights or DEFAULT_WEIGHTS

    # --- Tempo check (hard gate) ---
    t_penalty = tempo_penalty(song_a.bpm, song_b.bpm)
    if math.isinf(t_penalty):
        return float("inf")

    # --- Harmonic and semantic components ---
    h_distance = harmonic_distance(song_a.key, song_b.key)
    s_distance = semantic_distance(song_a.embedding, song_b.embedding)

    # Weighted sum
    composite = (
        w["harmonic"] * h_distance
        + w["tempo"] * t_penalty
        + w["semantic"] * s_distance
    )

    return composite


# =========================================================================
# 5. Vectorised batch edge computation
# =========================================================================
#
# For graph construction we need weights for all O(n²/2) unordered pairs.
# Computing them one-at-a-time with calculate_weight() is dominated by
# Python loop overhead and redundant numpy calls.  The functions below
# do everything in bulk using numpy broadcasting / GEMM, giving ~50-100×
# speedup on libraries of 500+ songs.
# =========================================================================


def batch_tempo_penalty_matrix(bpms: np.ndarray) -> np.ndarray:
    """
    Vectorised NxN tempo penalty matrix.

    Args:
        bpms: 1-D array of BPM values (length N).

    Returns:
        (N, N) float array.  Entry [i][j] is the tempo penalty for
        transitioning from song i to song j.  Unmixable pairs get np.inf.
    """
    bpm_col = bpms[:, np.newaxis]  # (N, 1)
    bpm_row = bpms[np.newaxis, :]  # (1, N)

    diff = np.abs(bpm_col - bpm_row)
    min_bpm = np.minimum(bpm_col, bpm_row)

    # Avoid division by zero for invalid BPMs
    safe_min = np.where(min_bpm > 0, min_bpm, 1.0)
    pct_diff = (diff / safe_min) * 100.0

    penalty = pct_diff / _MAX_BPM_DIFF_PERCENT
    # Hard gate: unmixable if diff > threshold or either BPM <= 0
    unmixable = (pct_diff > _MAX_BPM_DIFF_PERCENT) | (bpm_col <= 0) | (bpm_row <= 0)
    penalty = np.where(unmixable, np.inf, penalty)

    return penalty


def batch_harmonic_distance_matrix(keys: list[str]) -> np.ndarray:
    """
    Vectorised NxN harmonic distance matrix.

    Args:
        keys: List of key strings (length N), e.g. ['C major', 'A minor', ...].

    Returns:
        (N, N) float array of harmonic distances in [0.0, 1.0].
    """
    positions = np.array([_key_to_fifths_position(k) for k in keys])
    pos_col = positions[:, np.newaxis]
    pos_row = positions[np.newaxis, :]

    raw = np.abs(pos_col - pos_row)
    arc = np.minimum(raw, 12 - raw)
    return arc / 6.0


def batch_calculate_weights(
    songs: list[Song],
    *,
    weights: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the full NxN weight matrix and return only the upper-triangle
    entries (unordered pairs) using vectorised numpy operations.  This is
    the batch replacement for calling ``calculate_weight()`` inside a
    nested loop.

    Because all three cost components (harmonic distance, tempo penalty,
    semantic distance) are symmetric, weight(A, B) == weight(B, A).  We
    exploit this by zeroing the lower triangle + diagonal in the finite
    mask so that callers only see each unordered pair once.

    Returns:
        A tuple of (weight_matrix, finite_mask) where:
          - weight_matrix: (N, N) float array of composite transition costs.
            Unmixable pairs contain np.inf.
          - finite_mask: (N, N) boolean array — True only in the upper
            triangle where the weight is finite (i.e. the pair has a
            valid edge).
    """
    w = weights or DEFAULT_WEIGHTS
    n = len(songs)

    if n == 0:
        empty = np.empty((0, 0), dtype=np.float32)
        return empty, np.empty((0, 0), dtype=bool)

    bpms = np.array([s.bpm for s in songs], dtype=np.float64)
    keys = [s.key for s in songs]
    embeddings = np.array([s.embedding for s in songs], dtype=np.float32)

    # --- Tempo (hard gate) ---
    tempo_mat = batch_tempo_penalty_matrix(bpms)
    finite_mask = np.isfinite(tempo_mat)

    # Exclude self-loops AND lower triangle (symmetric costs → keep
    # only the upper triangle so each unordered pair appears once).
    finite_mask[np.tril_indices(n)] = False

    # --- Harmonic ---
    harmonic_mat = batch_harmonic_distance_matrix(keys)

    # --- Semantic ---
    semantic_mat = batch_semantic_distance_matrix(embeddings)

    # --- Composite weighted sum ---
    weight_matrix = (
        w["harmonic"] * harmonic_mat
        + w["tempo"] * tempo_mat
        + w["semantic"] * semantic_mat
    )

    # Force non-finite entries to inf
    weight_matrix = np.where(finite_mask, weight_matrix, np.inf)

    return weight_matrix.astype(np.float32), finite_mask


def batch_calculate_weights_incremental(
    new_songs: list[Song],
    existing_songs: list[Song],
    *,
    weights: dict[str, float] | None = None,
) -> tuple[list[tuple[int, int]], list[float], list[tuple[int, int]], list[float]]:
    """
    Compute edge weights for new songs against existing songs, and
    among new songs themselves.  Avoids recomputing existing×existing
    edges.

    Args:
        new_songs:      List of newly added Song objects.
        existing_songs: List of Song objects already in the graph.
        weights:        Optional metric blending coefficients.

    Returns:
        A tuple of (cross_edges, cross_weights, internal_edges, internal_weights):
          - cross_edges:    (new_idx, existing_idx) pairs with finite weight.
          - cross_weights:  Corresponding weights.
          - internal_edges: (new_idx_i, new_idx_j) pairs among new songs (upper tri).
          - internal_weights: Corresponding weights.
        All indices are *local* — new_idx is relative to new_songs,
        existing_idx is relative to existing_songs.
    """
    w = weights or DEFAULT_WEIGHTS
    n_new = len(new_songs)
    n_old = len(existing_songs)

    if n_new == 0:
        return [], [], [], []

    new_bpms = np.array([s.bpm for s in new_songs], dtype=np.float64)
    new_keys = [s.key for s in new_songs]
    new_embs = np.array([s.embedding for s in new_songs], dtype=np.float32)

    cross_edges: list[tuple[int, int]] = []
    cross_weights: list[float] = []

    # --- Cross edges: new × existing ---
    if n_old > 0:
        old_bpms = np.array([s.bpm for s in existing_songs], dtype=np.float64)
        old_keys = [s.key for s in existing_songs]
        old_embs = np.array([s.embedding for s in existing_songs], dtype=np.float32)

        # Tempo: (n_new, n_old)
        new_col = new_bpms[:, np.newaxis]
        old_row = old_bpms[np.newaxis, :]
        diff = np.abs(new_col - old_row)
        min_bpm = np.minimum(new_col, old_row)
        safe_min = np.where(min_bpm > 0, min_bpm, 1.0)
        pct_diff = (diff / safe_min) * 100.0
        tempo_mat = pct_diff / _MAX_BPM_DIFF_PERCENT
        unmixable = (pct_diff > _MAX_BPM_DIFF_PERCENT) | (new_col <= 0) | (old_row <= 0)
        tempo_mat = np.where(unmixable, np.inf, tempo_mat)

        finite_mask = np.isfinite(tempo_mat)

        # Harmonic: (n_new, n_old)
        new_pos = np.array([_key_to_fifths_position(k) for k in new_keys])
        old_pos = np.array([_key_to_fifths_position(k) for k in old_keys])
        raw = np.abs(new_pos[:, np.newaxis] - old_pos[np.newaxis, :])
        harmonic_mat = np.minimum(raw, 12 - raw) / 6.0

        # Semantic: (n_new, n_old)
        new_norms = np.linalg.norm(new_embs, axis=1, keepdims=True)
        old_norms = np.linalg.norm(old_embs, axis=1, keepdims=True)
        safe_new = np.where(new_norms == 0, 1.0, new_norms)
        safe_old = np.where(old_norms == 0, 1.0, old_norms)
        normed_new = new_embs / safe_new
        normed_old = old_embs / safe_old
        sim_mat = normed_new @ normed_old.T
        semantic_mat = np.clip((1.0 - sim_mat) / 2.0, 0.0, 1.0)

        weight_mat = (
            w["harmonic"] * harmonic_mat
            + w["tempo"] * tempo_mat
            + w["semantic"] * semantic_mat
        )

        rows, cols = np.where(finite_mask)
        cross_edges = list(zip(rows.tolist(), cols.tolist()))
        cross_weights = weight_mat[rows, cols].tolist()

    # --- Internal edges: new × new (upper triangle) ---
    internal_edges: list[tuple[int, int]] = []
    internal_weights: list[float] = []

    if n_new >= 2:
        int_weight_mat, int_finite = batch_calculate_weights(new_songs, weights=weights)
        rows, cols = np.where(int_finite)
        internal_edges = list(zip(rows.tolist(), cols.tolist()))
        internal_weights = int_weight_mat[rows, cols].tolist()

    return cross_edges, cross_weights, internal_edges, internal_weights


# =========================================================================
# 6. Multi-type tempo edge computation (direct / double / triplet)
# =========================================================================
#
# DJs can also mix tracks whose tempos are related by integer/simple
# ratios:
#
#   - **Double tempo** (2× or 0.5×):  A 140 BPM track over a 70 BPM
#     track (or vice-versa).  Common in drum-and-bass ↔ half-time,
#     or dubstep ↔ garage transitions.
#
#   - **Triplet** (3/2× or 3/4×):  A 150 BPM track over a 100 BPM
#     track.  Creates a polyrhythmic "triplet feel" blend.
#
# The same 8 % tolerance is applied *after* scaling one BPM by the
# relevant factor.
# =========================================================================

EDGE_TYPE_NONE: int = 0
EDGE_TYPE_DIRECT: int = 1
EDGE_TYPE_DOUBLE: int = 2
EDGE_TYPE_TRIPLET: int = 3

EDGE_TYPE_LABELS: dict[int, str] = {
    EDGE_TYPE_DIRECT: "direct",
    EDGE_TYPE_DOUBLE: "double",
    EDGE_TYPE_TRIPLET: "triplet",
}

# (scaling_factor, edge_type) — checked in order; direct is first so it
# wins whenever the raw BPMs already match.
_TEMPO_FACTORS: list[tuple[float, int]] = [
    (1.0, EDGE_TYPE_DIRECT),
    (2.0, EDGE_TYPE_DOUBLE),
    (0.5, EDGE_TYPE_DOUBLE),
    (1.5, EDGE_TYPE_TRIPLET),
    (0.75, EDGE_TYPE_TRIPLET),
]

DEFAULT_TYPE_PENALTIES: dict[str, float] = {
    "double": 0.0,
    "triplet": 0.0,
}


def batch_tempo_penalty_matrix_multi(
    bpms: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorised NxN tempo penalty matrix considering direct, double-tempo,
    and triplet tempo relationships.

    For each pair the function tries every scaling factor and keeps the
    one that yields the lowest penalty (i.e. the best-matching tempo
    relationship).

    Returns:
        (penalty_matrix, edge_type_matrix) where:
          - penalty_matrix: (N, N) float — tempo penalty for the
            best-matching relationship.  ``np.inf`` if no relationship
            is within the 8 % threshold.
          - edge_type_matrix: (N, N) int8 — relationship type
            (0 = none, 1 = direct, 2 = double, 3 = triplet).
    """
    n = len(bpms)
    bpm_col = bpms[:, np.newaxis]  # (N, 1)
    bpm_row = bpms[np.newaxis, :]  # (1, N)

    best_penalty = np.full((n, n), np.inf)
    edge_types = np.zeros((n, n), dtype=np.int8)

    for factor, type_id in _TEMPO_FACTORS:
        scaled = factor * bpm_row
        diff = np.abs(bpm_col - scaled)
        min_bpm = np.minimum(bpm_col, scaled)
        safe_min = np.where(min_bpm > 0, min_bpm, 1.0)
        pct_diff = (diff / safe_min) * 100.0
        penalty = pct_diff / _MAX_BPM_DIFF_PERCENT
        mixable = (pct_diff <= _MAX_BPM_DIFF_PERCENT) & (bpm_col > 0) & (scaled > 0)

        update = mixable & (penalty < best_penalty)
        best_penalty = np.where(update, penalty, best_penalty)
        edge_types = np.where(update, type_id, edge_types)

    return best_penalty, edge_types


def _cross_tempo_penalty_multi(
    new_bpms: np.ndarray,
    old_bpms: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cross-matrix variant of :func:`batch_tempo_penalty_matrix_multi`
    for (n_new, n_old) pairs.
    """
    new_col = new_bpms[:, np.newaxis]
    old_row = old_bpms[np.newaxis, :]

    best_penalty = np.full((len(new_bpms), len(old_bpms)), np.inf)
    edge_types = np.zeros((len(new_bpms), len(old_bpms)), dtype=np.int8)

    for factor, type_id in _TEMPO_FACTORS:
        scaled = factor * old_row
        diff = np.abs(new_col - scaled)
        min_bpm = np.minimum(new_col, scaled)
        safe_min = np.where(min_bpm > 0, min_bpm, 1.0)
        pct_diff = (diff / safe_min) * 100.0
        penalty = pct_diff / _MAX_BPM_DIFF_PERCENT
        mixable = (pct_diff <= _MAX_BPM_DIFF_PERCENT) & (new_col > 0) & (scaled > 0)

        update = mixable & (penalty < best_penalty)
        best_penalty = np.where(update, penalty, best_penalty)
        edge_types = np.where(update, type_id, edge_types)

    return best_penalty, edge_types


def batch_calculate_weights_multi(
    songs: list[Song],
    *,
    weights: dict[str, float] | None = None,
    type_penalties: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Like :func:`batch_calculate_weights` but produces edges for direct,
    double-tempo and triplet relationships.

    Returns:
        (weight_matrix, finite_mask, edge_type_matrix).
    """
    w = weights or DEFAULT_WEIGHTS
    tp = type_penalties or DEFAULT_TYPE_PENALTIES
    n = len(songs)

    if n == 0:
        empty = np.empty((0, 0), dtype=np.float32)
        return empty, np.empty((0, 0), dtype=bool), np.empty((0, 0), dtype=np.int8)

    bpms = np.array([s.bpm for s in songs], dtype=np.float64)
    keys = [s.key for s in songs]
    embeddings = np.array([s.embedding for s in songs], dtype=np.float32)

    # --- Tempo (multi-type) ---
    tempo_mat, edge_types = batch_tempo_penalty_matrix_multi(bpms)
    finite_mask = np.isfinite(tempo_mat)
    finite_mask[np.tril_indices(n)] = False

    # --- Harmonic ---
    harmonic_mat = batch_harmonic_distance_matrix(keys)

    # --- Semantic ---
    semantic_mat = batch_semantic_distance_matrix(embeddings)

    # --- Composite weighted sum ---
    weight_matrix = (
        w["harmonic"] * harmonic_mat
        + w["tempo"] * tempo_mat
        + w["semantic"] * semantic_mat
    )

    # --- Type-specific additive penalties ---
    type_penalty_mat = np.where(
        edge_types == EDGE_TYPE_DOUBLE, tp.get("double", 0.0),
        np.where(edge_types == EDGE_TYPE_TRIPLET, tp.get("triplet", 0.0), 0.0),
    )
    weight_matrix = weight_matrix + type_penalty_mat

    # Force non-finite entries to inf
    weight_matrix = np.where(finite_mask, weight_matrix, np.inf)

    return weight_matrix.astype(np.float32), finite_mask, edge_types


def batch_calculate_weights_incremental_multi(
    new_songs: list[Song],
    existing_songs: list[Song],
    *,
    weights: dict[str, float] | None = None,
    type_penalties: dict[str, float] | None = None,
) -> tuple[
    list[tuple[int, int]], list[float], list[str],
    list[tuple[int, int]], list[float], list[str],
]:
    """
    Like :func:`batch_calculate_weights_incremental` but produces
    multi-type edges and returns edge-type labels.

    Returns:
        (cross_edges, cross_weights, cross_types,
         internal_edges, internal_weights, internal_types)
    """
    w = weights or DEFAULT_WEIGHTS
    tp = type_penalties or DEFAULT_TYPE_PENALTIES
    n_new = len(new_songs)
    n_old = len(existing_songs)

    if n_new == 0:
        return [], [], [], [], [], []

    new_bpms = np.array([s.bpm for s in new_songs], dtype=np.float64)
    new_keys = [s.key for s in new_songs]
    new_embs = np.array([s.embedding for s in new_songs], dtype=np.float32)

    cross_edges: list[tuple[int, int]] = []
    cross_weights: list[float] = []
    cross_types: list[str] = []

    # --- Cross edges: new × existing ---
    if n_old > 0:
        old_bpms = np.array([s.bpm for s in existing_songs], dtype=np.float64)
        old_keys = [s.key for s in existing_songs]
        old_embs = np.array([s.embedding for s in existing_songs], dtype=np.float32)

        # Tempo (multi-type)
        tempo_mat, type_mat = _cross_tempo_penalty_multi(new_bpms, old_bpms)
        finite_mask = np.isfinite(tempo_mat)

        # Harmonic
        new_pos = np.array([_key_to_fifths_position(k) for k in new_keys])
        old_pos = np.array([_key_to_fifths_position(k) for k in old_keys])
        raw = np.abs(new_pos[:, np.newaxis] - old_pos[np.newaxis, :])
        harmonic_mat = np.minimum(raw, 12 - raw) / 6.0

        # Semantic
        new_norms = np.linalg.norm(new_embs, axis=1, keepdims=True)
        old_norms = np.linalg.norm(old_embs, axis=1, keepdims=True)
        safe_new = np.where(new_norms == 0, 1.0, new_norms)
        safe_old = np.where(old_norms == 0, 1.0, old_norms)
        normed_new = new_embs / safe_new
        normed_old = old_embs / safe_old
        sim_mat = normed_new @ normed_old.T
        semantic_mat = np.clip((1.0 - sim_mat) / 2.0, 0.0, 1.0)

        weight_mat = (
            w["harmonic"] * harmonic_mat
            + w["tempo"] * tempo_mat
            + w["semantic"] * semantic_mat
        )

        # Type penalties
        type_penalty_mat = np.where(
            type_mat == EDGE_TYPE_DOUBLE, tp.get("double", 0.0),
            np.where(type_mat == EDGE_TYPE_TRIPLET, tp.get("triplet", 0.0), 0.0),
        )
        weight_mat = weight_mat + type_penalty_mat

        rows, cols = np.where(finite_mask)
        cross_edges = list(zip(rows.tolist(), cols.tolist()))
        cross_weights = weight_mat[rows, cols].tolist()
        cross_types = [
            EDGE_TYPE_LABELS.get(int(type_mat[r, c]), "direct")
            for r, c in cross_edges
        ]

    # --- Internal edges: new × new (upper triangle) ---
    internal_edges: list[tuple[int, int]] = []
    internal_weights: list[float] = []
    internal_types: list[str] = []

    if n_new >= 2:
        int_weight_mat, int_finite, int_type_mat = batch_calculate_weights_multi(
            new_songs, weights=weights, type_penalties=type_penalties,
        )
        rows, cols = np.where(int_finite)
        internal_edges = list(zip(rows.tolist(), cols.tolist()))
        internal_weights = int_weight_mat[rows, cols].tolist()
        internal_types = [
            EDGE_TYPE_LABELS.get(int(int_type_mat[r, c]), "direct")
            for r, c in internal_edges
        ]

    return cross_edges, cross_weights, cross_types, internal_edges, internal_weights, internal_types
