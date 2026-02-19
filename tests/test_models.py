"""
Unit tests for the Song model and directory scanner.

These tests use mocking to avoid requiring actual audio files or the
CLAP model weights at test time.
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.models import Song, _estimate_key, _PITCH_CLASSES
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


class TestSongFromFile:
    """Song.from_file() with mocked audio I/O and ML model."""

    @patch("src.models._get_clap_model")
    @patch("src.models.librosa")
    def test_from_file_happy_path(self, mock_librosa, mock_clap):
        # Set up librosa mocks
        dummy_audio = np.random.randn(44100).astype(np.float32)
        mock_librosa.load.return_value = (dummy_audio, 44100)
        mock_librosa.beat.beat_track.return_value = (np.array([125.0]), None)
        mock_librosa.feature.chroma_cqt.return_value = np.random.rand(12, 100)
        mock_librosa.resample.return_value = dummy_audio

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
        assert song.key != ""  # some key will be estimated
        assert song.embedding.shape == (512,)
        assert song.filename.endswith(".mp3")

    def test_from_file_missing_raises(self):
        with pytest.raises(FileNotFoundError):
            Song.from_file("/nonexistent/track.mp3")


# =========================================================================
# Key estimation tests
# =========================================================================


class TestKeyEstimation:
    def test_returns_valid_key_string(self):
        # Feed a synthetic chroma profile and verify the output format
        y = np.random.randn(44100).astype(np.float32)
        key = _estimate_key(y, sr=22050)

        parts = key.split()
        assert len(parts) == 2
        assert parts[0] in _PITCH_CLASSES
        assert parts[1] in ("major", "minor")


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
