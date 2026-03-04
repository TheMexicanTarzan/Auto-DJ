"""
Unit tests for the Song model and directory scanner.

These tests use mocking to avoid requiring actual audio files, the
CLAP model weights, or heavy analysis libraries (madmom, essentia) at
test time.
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.models import (
    Song,
    _detect_beats_and_downbeats,
    _estimate_key,
    _FLAT_TO_SHARP,
    _PITCH_CLASSES,
)
from src.utils import SUPPORTED_EXTENSIONS, _file_hash, scan_directory


# =========================================================================
# Song model tests
# =========================================================================


class TestSongDataclass:
    """Basic construction and repr."""

    def test_defaults(self):
        song = Song(file_path="/tmp/a.mp3", filename="a.mp3")
        assert song.bpm == 0.0
        assert song.key == ""
        assert song.embedding.size == 0
        assert song.beat_times == []
        assert song.downbeat_times == []

    def test_repr_empty_embedding(self):
        song = Song(file_path="/tmp/a.mp3", filename="a.mp3")
        r = repr(song)
        assert "a.mp3" in r
        assert "(empty)" in r

    def test_repr_with_embedding(self):
        emb = np.zeros(512)
        song = Song(file_path="/x.wav", filename="x.wav", bpm=128.0, key="A minor", embedding=emb)
        r = repr(song)
        assert "128.0" in r
        assert "A minor" in r
        assert "(512,)" in r

    def test_beat_fields(self):
        """New beat_times / downbeat_times fields work correctly."""
        song = Song(
            file_path="/tmp/t.mp3",
            filename="t.mp3",
            bpm=120.0,
            beat_times=[0.0, 0.5, 1.0, 1.5],
            downbeat_times=[0.0, 1.0],
        )
        assert len(song.beat_times) == 4
        assert len(song.downbeat_times) == 2


class TestSongFromFile:
    """Song.from_file() with mocked audio I/O and ML model."""

    @patch("src.models._get_clap_model")
    @patch("src.models._estimate_key")
    @patch("src.models._detect_beats_and_downbeats")
    @patch("src.models.librosa")
    def test_from_file_happy_path(
        self, mock_librosa, mock_beats, mock_key, mock_clap
    ):
        # Set up librosa mock (I/O only)
        dummy_audio = np.random.randn(44100).astype(np.float32)
        mock_librosa.load.return_value = (dummy_audio, 44100)
        mock_librosa.resample.return_value = dummy_audio

        # Set up madmom beat detection mock
        mock_beats.return_value = (
            125.0,
            [0.0, 0.48, 0.96, 1.44],
            [0.0, 0.96],
        )

        # Set up essentia key detection mock
        mock_key.return_value = "A minor"

        # Set up CLAP mock
        fake_embedding = np.random.randn(512).astype(np.float32)
        mock_model = MagicMock()
        mock_model.get_audio_features.return_value = MagicMock(
            squeeze=MagicMock(return_value=MagicMock(cpu=MagicMock(return_value=MagicMock(numpy=MagicMock(return_value=fake_embedding)))))
        )
        mock_processor = MagicMock()
        mock_clap.return_value = (mock_model, mock_processor)

        # Create a real temporary file so Path.is_file() passes
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp.write(b"\x00" * 100)
            tmp_path = tmp.name

        song = Song.from_file(tmp_path)

        assert song.bpm == 125.0
        assert song.key == "A minor"
        assert song.embedding.shape == (512,)
        assert song.filename.endswith(".mp3")
        assert len(song.beat_times) == 4
        assert len(song.downbeat_times) == 2

    def test_from_file_missing_raises(self):
        with pytest.raises(FileNotFoundError):
            Song.from_file("/nonexistent/track.mp3")

    @patch("src.models.librosa")
    def test_from_file_too_short_raises(self, mock_librosa):
        """Audio shorter than 1 second should raise ValueError."""
        # 0.5 seconds at 44100 Hz
        short_audio = np.random.randn(22050).astype(np.float32)
        mock_librosa.load.return_value = (short_audio, 44100)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(b"\x00" * 100)
            tmp_path = tmp.name

        with pytest.raises(ValueError, match="Audio too short"):
            Song.from_file(tmp_path)

    @patch("src.models.librosa")
    def test_from_file_silent_raises(self, mock_librosa):
        """Effectively-silent audio should raise ValueError."""
        silent_audio = np.zeros(44100, dtype=np.float32)
        mock_librosa.load.return_value = (silent_audio, 44100)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(b"\x00" * 100)
            tmp_path = tmp.name

        with pytest.raises(ValueError, match="silent"):
            Song.from_file(tmp_path)


# =========================================================================
# Key estimation tests  (Essentia KeyExtractor)
# =========================================================================


class TestKeyEstimation:
    @patch.dict(
        "sys.modules",
        {
            "essentia": MagicMock(),
            "essentia.standard": MagicMock(),
        },
    )
    def test_returns_valid_key_string(self):
        """_estimate_key wraps Essentia and returns 'Note mode' format."""
        import sys

        mock_es = sys.modules["essentia.standard"]
        mock_extractor = MagicMock(return_value=("A", "minor", 0.82))
        mock_es.KeyExtractor.return_value = mock_extractor

        y = np.random.randn(44100).astype(np.float32)
        key = _estimate_key(y, sr=22050)

        parts = key.split()
        assert len(parts) == 2
        assert parts[0] in _PITCH_CLASSES
        assert parts[1] in ("major", "minor")
        assert key == "A minor"

    @patch.dict(
        "sys.modules",
        {
            "essentia": MagicMock(),
            "essentia.standard": MagicMock(),
        },
    )
    def test_normalises_flats_to_sharps(self):
        """Flat notation from Essentia is converted to sharp notation."""
        import sys

        mock_es = sys.modules["essentia.standard"]
        mock_extractor = MagicMock(return_value=("Bb", "minor", 0.70))
        mock_es.KeyExtractor.return_value = mock_extractor

        y = np.random.randn(44100).astype(np.float32)
        key = _estimate_key(y, sr=44100)

        assert key == "A# minor"


# =========================================================================
# Beat / downbeat detection tests  (madmom)
# =========================================================================


class TestBeatDetection:
    @patch.dict(
        "sys.modules",
        {
            "madmom": MagicMock(),
            "madmom.features": MagicMock(),
            "madmom.features.beats": MagicMock(),
        },
    )
    def test_returns_bpm_and_beat_grid(self):
        """_detect_beats_and_downbeats returns BPM, beats, and downbeats."""
        import sys

        mock_beats_mod = sys.modules["madmom.features.beats"]

        # Mock RNNDownBeatProcessor
        fake_activations = np.random.rand(100, 4)
        mock_proc = MagicMock(return_value=fake_activations)
        mock_beats_mod.RNNDownBeatProcessor.return_value = mock_proc

        # Mock DBNDownBeatTrackingProcessor — simulate 120 BPM in 4/4
        fake_beats = np.array([
            [0.0, 1], [0.5, 2], [1.0, 3], [1.5, 4],
            [2.0, 1], [2.5, 2], [3.0, 3], [3.5, 4],
        ])
        mock_dbn = MagicMock(return_value=fake_beats)
        mock_beats_mod.DBNDownBeatTrackingProcessor.return_value = mock_dbn

        bpm, beat_times, downbeat_times = _detect_beats_and_downbeats(
            "/fake/path.mp3"
        )

        assert bpm == pytest.approx(120.0, abs=1.0)
        assert len(beat_times) == 8
        assert len(downbeat_times) == 2
        assert downbeat_times[0] == 0.0
        assert downbeat_times[1] == 2.0

    @patch.dict(
        "sys.modules",
        {
            "madmom": MagicMock(),
            "madmom.features": MagicMock(),
            "madmom.features.beats": MagicMock(),
        },
    )
    def test_empty_beats_returns_zero(self):
        """If madmom finds no beats, return 0.0 BPM and empty lists."""
        import sys

        mock_beats_mod = sys.modules["madmom.features.beats"]

        mock_proc = MagicMock(return_value=np.array([]))
        mock_beats_mod.RNNDownBeatProcessor.return_value = mock_proc

        empty_beats = np.array([]).reshape(0, 2)
        mock_dbn = MagicMock(return_value=empty_beats)
        mock_beats_mod.DBNDownBeatTrackingProcessor.return_value = mock_dbn

        bpm, beat_times, downbeat_times = _detect_beats_and_downbeats(
            "/fake/path.mp3"
        )

        assert bpm == 0.0
        assert beat_times == []
        assert downbeat_times == []


# =========================================================================
# Utility / scanner tests
# =========================================================================


class TestFileHash:
    def test_identical_content_same_hash(self, tmp_path):
        a = tmp_path / "a.bin"
        b = tmp_path / "b.bin"
        data = b"identical content"
        a.write_bytes(data)
        b.write_bytes(data)
        assert _file_hash(a) == _file_hash(b)

    def test_different_content_different_hash(self, tmp_path):
        a = tmp_path / "a.bin"
        b = tmp_path / "b.bin"
        a.write_bytes(b"content A")
        b.write_bytes(b"content B")
        assert _file_hash(a) != _file_hash(b)


class TestScanDirectory:
    def test_not_a_directory_raises(self):
        with pytest.raises(NotADirectoryError):
            scan_directory("/nonexistent/path")

    @patch("src.utils.Song.from_file")
    def test_finds_audio_files(self, mock_from_file, tmp_path):
        """Create dummy audio files and verify the scanner finds them."""
        mock_from_file.side_effect = lambda p: Song(
            file_path=str(p), filename=Path(p).name, bpm=120.0, key="C major"
        )

        # Create files with supported extensions
        (tmp_path / "track1.mp3").write_bytes(b"fake mp3 data aaa")
        (tmp_path / "track2.wav").write_bytes(b"fake wav data bbb")
        (tmp_path / "notes.txt").write_bytes(b"not audio")

        songs = scan_directory(tmp_path)

        assert len(songs) == 2
        names = {s.filename for s in songs}
        assert "track1.mp3" in names
        assert "track2.wav" in names

    @patch("src.utils.Song.from_file")
    def test_deduplicates_identical_files(self, mock_from_file, tmp_path):
        """Byte-identical files should be deduplicated."""
        mock_from_file.side_effect = lambda p: Song(
            file_path=str(p), filename=Path(p).name
        )

        content = b"same bytes"
        (tmp_path / "original.mp3").write_bytes(content)
        (tmp_path / "copy.mp3").write_bytes(content)

        songs = scan_directory(tmp_path)
        assert len(songs) == 1

    @patch("src.utils.Song.from_file")
    def test_recursive_scan(self, mock_from_file, tmp_path):
        """Files in nested subdirectories should be found."""
        mock_from_file.side_effect = lambda p: Song(
            file_path=str(p), filename=Path(p).name
        )

        sub = tmp_path / "genre" / "artist"
        sub.mkdir(parents=True)
        (sub / "deep_track.flac").write_bytes(b"flac data")

        songs = scan_directory(tmp_path)
        assert len(songs) == 1
        assert songs[0].filename == "deep_track.flac"

    @patch("src.utils.Song.from_file")
    def test_skips_failed_analysis(self, mock_from_file, tmp_path):
        """If Song.from_file raises, the scanner should skip that file."""
        mock_from_file.side_effect = RuntimeError("analysis failed")
        (tmp_path / "bad.mp3").write_bytes(b"corrupt")

        songs = scan_directory(tmp_path)
        assert len(songs) == 0


class TestSupportedExtensions:
    def test_common_formats_included(self):
        for ext in (".mp3", ".wav", ".flac", ".ogg", ".m4a"):
            assert ext in SUPPORTED_EXTENSIONS
