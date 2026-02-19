"""
Unit tests for the transition-cost metrics module.

Covers harmonic distance, tempo penalty, semantic distance,
and the composite calculate_weight function.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.metrics import (
    DEFAULT_WEIGHTS,
    _key_to_fifths_position,
    calculate_weight,
    harmonic_distance,
    semantic_distance,
    tempo_penalty,
)
from src.models import Song


# =========================================================================
# Helpers
# =========================================================================


def _make_song(
    bpm: float = 120.0,
    key: str = "C major",
    embedding: np.ndarray | None = None,
) -> Song:
    """Shortcut to build a Song with only the fields metrics care about."""
    if embedding is None:
        embedding = np.random.randn(512).astype(np.float32)
    return Song(
        file_path="/fake/path.mp3",
        filename="path.mp3",
        bpm=bpm,
        key=key,
        embedding=embedding,
    )


# =========================================================================
# 1. Circle of Fifths position mapping
# =========================================================================


class TestKeyToFifthsPosition:
    """Verify the semitone → Circle-of-Fifths mapping."""

    def test_c_major_is_zero(self):
        assert _key_to_fifths_position("C major") == 0

    def test_g_major_is_one(self):
        # G is a perfect fifth above C → position 1
        assert _key_to_fifths_position("G major") == 1

    def test_f_sharp_major_is_six(self):
        # F# is the tritone from C → position 6 (opposite side)
        assert _key_to_fifths_position("F# major") == 6

    def test_relative_minor_maps_to_same_position(self):
        # A minor is the relative minor of C major
        assert _key_to_fifths_position("A minor") == _key_to_fifths_position("C major")

    def test_e_minor_maps_to_g_major(self):
        # E minor ↔ G major (relative pair)
        assert _key_to_fifths_position("E minor") == _key_to_fifths_position("G major")

    def test_d_minor_maps_to_f_major(self):
        # D minor ↔ F major
        assert _key_to_fifths_position("D minor") == _key_to_fifths_position("F major")

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match="Expected"):
            _key_to_fifths_position("Cmajor")

    def test_unknown_pitch_raises(self):
        with pytest.raises(ValueError, match="Unknown pitch"):
            _key_to_fifths_position("X major")


# =========================================================================
# 2. Harmonic distance
# =========================================================================


class TestHarmonicDistance:
    def test_same_key_is_zero(self):
        assert harmonic_distance("C major", "C major") == 0.0

    def test_relative_minor_is_zero(self):
        # A minor ↔ C major → same position → distance 0
        assert harmonic_distance("C major", "A minor") == 0.0

    def test_perfect_fifth(self):
        # C major → G major = 1 step on the circle
        dist = harmonic_distance("C major", "G major")
        assert dist == pytest.approx(1.0 / 6.0)

    def test_tritone_is_one(self):
        # C major → F# major = 6 steps (max) → normalised to 1.0
        assert harmonic_distance("C major", "F# major") == pytest.approx(1.0)

    def test_symmetry(self):
        d1 = harmonic_distance("D major", "A major")
        d2 = harmonic_distance("A major", "D major")
        assert d1 == pytest.approx(d2)

    def test_two_steps(self):
        # C → D = 2 steps on the circle of fifths
        assert harmonic_distance("C major", "D major") == pytest.approx(2.0 / 6.0)

    def test_wraps_around_circle(self):
        # F major (pos 11) → G major (pos 1) — shortest arc is 2, not 10
        assert harmonic_distance("F major", "G major") == pytest.approx(2.0 / 6.0)

    def test_all_same_key_pairs_are_zero(self):
        for note in ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"):
            assert harmonic_distance(f"{note} major", f"{note} major") == 0.0


# =========================================================================
# 3. Tempo penalty
# =========================================================================


class TestTempoPenalty:
    def test_identical_bpm_is_zero(self):
        assert tempo_penalty(128.0, 128.0) == 0.0

    def test_within_threshold_linear(self):
        # 120 → 126 = 5% diff (relative to 120). 5/8 = 0.625
        penalty = tempo_penalty(120.0, 126.0)
        assert penalty == pytest.approx(5.0 / 8.0)

    def test_exactly_at_threshold(self):
        # 100 → 108 = exactly 8%. Should be 1.0, not inf.
        assert tempo_penalty(100.0, 108.0) == pytest.approx(1.0)

    def test_above_threshold_is_inf(self):
        # 100 → 110 = 10% > 8%
        assert math.isinf(tempo_penalty(100.0, 110.0))

    def test_symmetry(self):
        # The slower track is always the denominator, so order doesn't
        # matter for the percentage, but let's confirm.
        assert tempo_penalty(120.0, 125.0) == pytest.approx(tempo_penalty(125.0, 120.0))

    def test_zero_bpm_raises(self):
        with pytest.raises(ValueError, match="positive"):
            tempo_penalty(0.0, 120.0)

    def test_negative_bpm_raises(self):
        with pytest.raises(ValueError, match="positive"):
            tempo_penalty(120.0, -5.0)

    def test_small_difference(self):
        # 128 → 129 ≈ 0.78% → penalty ≈ 0.78/8 ≈ 0.098
        penalty = tempo_penalty(128.0, 129.0)
        expected = (1.0 / 128.0) * 100.0 / 8.0
        assert penalty == pytest.approx(expected)


# =========================================================================
# 4. Semantic distance
# =========================================================================


class TestSemanticDistance:
    def test_identical_vectors_is_zero(self):
        v = np.array([1.0, 2.0, 3.0])
        assert semantic_distance(v, v) == pytest.approx(0.0)

    def test_opposite_vectors_is_one(self):
        v = np.array([1.0, 0.0, 0.0])
        w = np.array([-1.0, 0.0, 0.0])
        # cosine_sim = -1 → distance = (1 - (-1)) / 2 = 1.0
        assert semantic_distance(v, w) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        v = np.array([1.0, 0.0])
        w = np.array([0.0, 1.0])
        # cosine_sim = 0 → distance = (1 - 0) / 2 = 0.5
        assert semantic_distance(v, w) == pytest.approx(0.5)

    def test_symmetry(self):
        a = np.random.randn(512)
        b = np.random.randn(512)
        assert semantic_distance(a, b) == pytest.approx(semantic_distance(b, a))

    def test_empty_embedding_raises(self):
        with pytest.raises(ValueError, match="empty"):
            semantic_distance(np.array([]), np.array([1.0]))

    def test_zero_norm_raises(self):
        with pytest.raises(ValueError, match="zero-norm"):
            semantic_distance(np.zeros(3), np.array([1.0, 2.0, 3.0]))

    def test_result_in_unit_range(self):
        # Random vectors should always produce a result in [0, 1]
        for _ in range(50):
            a = np.random.randn(512)
            b = np.random.randn(512)
            d = semantic_distance(a, b)
            assert 0.0 <= d <= 1.0


# =========================================================================
# 5. Composite weight (calculate_weight)
# =========================================================================


class TestCalculateWeight:
    def test_identical_songs_near_zero(self):
        emb = np.random.randn(512).astype(np.float32)
        a = _make_song(bpm=128.0, key="C major", embedding=emb)
        b = _make_song(bpm=128.0, key="C major", embedding=emb)
        w = calculate_weight(a, b)
        assert w == pytest.approx(0.0)

    def test_unmixable_tempo_returns_inf(self):
        a = _make_song(bpm=100.0, key="C major")
        b = _make_song(bpm=130.0, key="C major")
        assert math.isinf(calculate_weight(a, b))

    def test_inf_short_circuits_other_metrics(self):
        # Even with clashing keys, the tempo gate should return inf
        # before we even look at harmony or embeddings
        a = _make_song(bpm=80.0, key="C major")
        b = _make_song(bpm=120.0, key="F# major")
        assert math.isinf(calculate_weight(a, b))

    def test_harmonic_clash_increases_weight(self):
        emb = np.random.randn(512).astype(np.float32)
        same_key = _make_song(bpm=120.0, key="C major", embedding=emb)
        clash = _make_song(bpm=120.0, key="F# major", embedding=emb)
        match = _make_song(bpm=120.0, key="C major", embedding=emb)

        w_match = calculate_weight(same_key, match)
        w_clash = calculate_weight(same_key, clash)
        assert w_clash > w_match

    def test_custom_weights(self):
        emb = np.random.randn(512).astype(np.float32)
        a = _make_song(bpm=120.0, key="C major", embedding=emb)
        b = _make_song(bpm=125.0, key="G major", embedding=emb)

        # Put all weight on tempo — harmonic and semantic ignored
        w = calculate_weight(a, b, weights={"harmonic": 0.0, "tempo": 1.0, "semantic": 0.0})
        assert w == pytest.approx(tempo_penalty(120.0, 125.0))

    def test_result_in_unit_range_for_mixable_pair(self):
        a = _make_song(bpm=126.0, key="D major")
        b = _make_song(bpm=128.0, key="A major")
        w = calculate_weight(a, b)
        assert 0.0 <= w <= 1.0

    def test_symmetry(self):
        a = _make_song(bpm=120.0, key="E minor")
        b = _make_song(bpm=122.0, key="G major")
        assert calculate_weight(a, b) == pytest.approx(calculate_weight(b, a))

    def test_default_weights_sum_to_one(self):
        total = sum(DEFAULT_WEIGHTS.values())
        assert total == pytest.approx(1.0)
