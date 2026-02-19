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
