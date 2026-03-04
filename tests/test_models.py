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

from src.models import Song, _estimate_key, _PITCH_CLASSES, analyse_audio, compute_embeddings_batch
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

    def test_from_file_happy_path(self):
        import sys

        # Build a mock librosa module
        mock_librosa = MagicMock()
        dummy_audio = np.random.randn(44100).astype(np.float32)
        mock_librosa.load.return_value = (dummy_audio, 22050)
        mock_librosa.beat.beat_track.return_value = (np.array([125.0]), None)
        mock_librosa.feature.chroma_cqt.return_value = np.random.rand(12, 100)
        mock_librosa.resample.return_value = dummy_audio

        # Set up CLAP mock — build a chain: outputs.squeeze().cpu().numpy() -> embedding
        fake_embedding = np.random.randn(512).astype(np.float32)
        mock_outputs = MagicMock()
        del mock_outputs.pooler_output  # ensure hasattr check fails
        mock_outputs.squeeze.return_value.cpu.return_value.numpy.return_value = fake_embedding
        mock_model = MagicMock()
        mock_model.get_audio_features.return_value = mock_outputs
        mock_processor = MagicMock()

        # Create a real temporary file so Path.is_file() passes
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp.write(b"\x00" * 100)
            tmp_path = tmp.name

        mock_torch = MagicMock()

        with patch.dict(sys.modules, {"librosa": mock_librosa, "librosa.beat": mock_librosa.beat, "librosa.feature": mock_librosa.feature, "torch": mock_torch}), \
             patch("src.models._get_clap_model", return_value=(mock_model, mock_processor)):
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
        import sys

        # _estimate_key does `import librosa` internally; mock it
        mock_librosa = MagicMock()
        mock_librosa.feature.chroma_cqt.return_value = np.random.rand(12, 100)

        y = np.random.randn(44100).astype(np.float32)
        with patch.dict(sys.modules, {"librosa": mock_librosa, "librosa.feature": mock_librosa.feature}):
            key = _estimate_key(y, sr=22050)

        parts = key.split()
        assert len(parts) == 2
        assert parts[0] in _PITCH_CLASSES
        assert parts[1] in ("major", "minor")


# =========================================================================
# analyse_audio tests
# =========================================================================


class TestAnalyseAudio:
    """Test the top-level analyse_audio() function used by multiprocessing."""

    def test_returns_audio_analysis(self):
        import sys

        mock_librosa = MagicMock()
        dummy_audio = np.random.randn(22050).astype(np.float32)
        mock_librosa.load.return_value = (dummy_audio, 22050)
        mock_librosa.beat.beat_track.return_value = (np.array([130.0]), None)
        mock_librosa.feature.chroma_cqt.return_value = np.random.rand(12, 50)

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp.write(b"\x00" * 100)
            tmp_path = tmp.name

        with patch.dict(sys.modules, {"librosa": mock_librosa, "librosa.beat": mock_librosa.beat, "librosa.feature": mock_librosa.feature}):
            result = analyse_audio(tmp_path)

        assert result.bpm == 130.0
        assert result.key != ""
        assert result.audio is dummy_audio
        assert result.sr == 22050

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            analyse_audio("/nonexistent/track.mp3")


# =========================================================================
# Batch CLAP embedding tests
# =========================================================================


class TestComputeEmbeddingsBatch:
    """Test batched CLAP embedding computation."""

    def test_batch_returns_correct_count(self):
        import sys

        mock_librosa = MagicMock()
        mock_librosa.resample.side_effect = lambda y, **kw: y

        fake_output = np.random.randn(3, 512).astype(np.float32)

        # Build a mock that mimics: outputs.cpu().numpy() -> fake_output
        mock_outputs = MagicMock()
        mock_outputs.cpu.return_value.numpy.return_value = fake_output
        # hasattr(outputs, "pooler_output") should be False so we use raw tensor
        del mock_outputs.pooler_output

        mock_model = MagicMock()
        mock_model.get_audio_features.return_value = mock_outputs
        mock_processor = MagicMock(return_value={"input_features": MagicMock()})

        mock_torch = MagicMock()

        audio_list = [
            (np.random.randn(22050).astype(np.float32), 22050)
            for _ in range(3)
        ]

        with patch.dict(sys.modules, {"librosa": mock_librosa, "torch": mock_torch}), \
             patch("src.models._get_clap_model", return_value=(mock_model, mock_processor)):
            embeddings = compute_embeddings_batch(audio_list, batch_size=8)

        assert len(embeddings) == 3
        for emb in embeddings:
            assert emb.shape == (512,)

    def test_empty_input_returns_empty(self):
        result = compute_embeddings_batch([])
        assert result == []


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

    @patch("src.utils.compute_embeddings_batch")
    @patch("src.utils.analyse_audio")
    def test_finds_audio_files(self, mock_analyse, mock_batch_embed, tmp_path):
        """Create dummy audio files and verify the scanner finds them."""
        from src.models import AudioAnalysis

        def fake_analyse(p):
            path = Path(p)
            return AudioAnalysis(
                file_path=str(path), filename=path.name,
                bpm=120.0, key="C major",
                audio=np.zeros(100, dtype=np.float32), sr=22050,
            )

        mock_analyse.side_effect = fake_analyse
        mock_batch_embed.return_value = [
            np.zeros(512, dtype=np.float32) for _ in range(2)
        ]

        (tmp_path / "track1.mp3").write_bytes(b"fake mp3 data aaa")
        (tmp_path / "track2.wav").write_bytes(b"fake wav data bbb")
        (tmp_path / "notes.txt").write_bytes(b"not audio")

        songs = scan_directory(tmp_path)

        assert len(songs) == 2
        names = {s.filename for s in songs}
        assert "track1.mp3" in names
        assert "track2.wav" in names

    @patch("src.utils.compute_embeddings_batch")
    @patch("src.utils.analyse_audio")
    def test_deduplicates_identical_files(self, mock_analyse, mock_batch_embed, tmp_path):
        """Byte-identical files should be deduplicated."""
        from src.models import AudioAnalysis

        def fake_analyse(p):
            path = Path(p)
            return AudioAnalysis(
                file_path=str(path), filename=path.name,
                bpm=120.0, key="C major",
                audio=np.zeros(100, dtype=np.float32), sr=22050,
            )

        mock_analyse.side_effect = fake_analyse
        mock_batch_embed.return_value = [np.zeros(512, dtype=np.float32)]

        content = b"same bytes"
        (tmp_path / "original.mp3").write_bytes(content)
        (tmp_path / "copy.mp3").write_bytes(content)

        songs = scan_directory(tmp_path)
        assert len(songs) == 1

    @patch("src.utils.compute_embeddings_batch")
    @patch("src.utils.analyse_audio")
    def test_recursive_scan(self, mock_analyse, mock_batch_embed, tmp_path):
        """Files in nested subdirectories should be found."""
        from src.models import AudioAnalysis

        def fake_analyse(p):
            path = Path(p)
            return AudioAnalysis(
                file_path=str(path), filename=path.name,
                bpm=120.0, key="C major",
                audio=np.zeros(100, dtype=np.float32), sr=22050,
            )

        mock_analyse.side_effect = fake_analyse
        mock_batch_embed.return_value = [np.zeros(512, dtype=np.float32)]

        sub = tmp_path / "genre" / "artist"
        sub.mkdir(parents=True)
        (sub / "deep_track.flac").write_bytes(b"flac data")

        songs = scan_directory(tmp_path)
        assert len(songs) == 1
        assert songs[0].filename == "deep_track.flac"

    @patch("src.utils.analyse_audio")
    def test_skips_failed_analysis(self, mock_analyse, tmp_path):
        """If analyse_audio raises, the scanner should skip that file."""
        mock_analyse.side_effect = RuntimeError("analysis failed")
        (tmp_path / "bad.mp3").write_bytes(b"corrupt")

        songs = scan_directory(tmp_path)
        assert len(songs) == 0


class TestSupportedExtensions:
    def test_common_formats_included(self):
        for ext in (".mp3", ".wav", ".flac", ".ogg", ".m4a"):
            assert ext in SUPPORTED_EXTENSIONS
