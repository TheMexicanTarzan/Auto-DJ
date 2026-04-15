"""
Unit tests for the Song model and directory scanner.

These tests use mocking to avoid requiring actual audio files, the
CLAP model weights, or heavy analysis libraries (essentia) at test time.
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.models import (
    AudioAnalysis,
    Song,
    analyse_audio,
    compute_embeddings_batch,
    _detect_beats_and_downbeats,
    _detect_intro_type,
    _phrase_chunks,
    _select_embed_dtype,
    _estimate_key,
    _FLAT_TO_SHARP,
    _PITCH_CLASSES,
)
from src.utils import SUPPORTED_EXTENSIONS, _file_hash, _optimal_worker_count, scan_directory


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

    def test_from_file_happy_path(self):
        import sys

        dummy_audio = np.random.randn(44100).astype(np.float32)

        fake_embedding = np.random.randn(512).astype(np.float32)
        mock_outputs = MagicMock()
        del mock_outputs.pooler_output
        mock_outputs.squeeze.return_value.cpu.return_value.numpy.return_value = fake_embedding
        mock_model = MagicMock()
        mock_model.get_audio_features.return_value = mock_outputs
        mock_processor = MagicMock()

        mock_torch = MagicMock()

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp.write(b"\x00" * 100)
            tmp_path = tmp.name

        with patch.dict(sys.modules, {"torch": mock_torch}), \
             patch("src.models._load_audio", return_value=(dummy_audio, 44100)), \
             patch("src.models._resample", side_effect=lambda y, **kw: y), \
             patch("src.models._get_clap_model", return_value=(mock_model, mock_processor)), \
             patch("src.models._detect_beats_and_downbeats", return_value=(125.0, [0.0, 0.48, 0.96, 1.44], [0.0, 0.96])), \
             patch("src.models._estimate_key", return_value="A minor"):
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

    def test_from_file_too_short_raises(self):
        """Audio shorter than 1 second should raise ValueError."""
        short_audio = np.random.randn(22050).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(b"\x00" * 100)
            tmp_path = tmp.name

        with patch("src.models._load_audio", return_value=(short_audio, 44100)):
            with pytest.raises(ValueError, match="Audio too short"):
                Song.from_file(tmp_path)

    def test_from_file_silent_raises(self):
        """Effectively-silent audio should raise ValueError."""
        silent_audio = np.zeros(44100, dtype=np.float32)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(b"\x00" * 100)
            tmp_path = tmp.name

        with patch("src.models._load_audio", return_value=(silent_audio, 44100)):
            with pytest.raises(ValueError, match="silent"):
                Song.from_file(tmp_path)


# =========================================================================
# Key estimation tests  (Essentia KeyExtractor)
# =========================================================================


class TestKeyEstimation:
    def test_returns_valid_key_string(self):
        """_estimate_key wraps Essentia and returns 'Note mode' format."""
        import sys

        mock_es = MagicMock()
        mock_extractor_instance = MagicMock(return_value=("A", "minor", 0.82))
        mock_es.KeyExtractor = MagicMock(return_value=mock_extractor_instance)

        # The parent package mock must have .standard pointing to our mock_es,
        # otherwise `import essentia.standard as es` gets a different MagicMock.
        mock_essentia_pkg = MagicMock()
        mock_essentia_pkg.standard = mock_es

        saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k.startswith("essentia") and k in sys.modules}
        try:
            sys.modules["essentia"] = mock_essentia_pkg
            sys.modules["essentia.standard"] = mock_es

            y = np.random.randn(44100).astype(np.float32)
            key = _estimate_key(y, sr=22050)
        finally:
            for k in list(sys.modules):
                if k.startswith("essentia"):
                    del sys.modules[k]
            sys.modules.update(saved)

        parts = key.split()
        assert len(parts) == 2
        assert parts[0] in _PITCH_CLASSES
        assert parts[1] in ("major", "minor")
        assert key == "A minor"

    def test_normalises_flats_to_sharps(self):
        """Flat notation from Essentia is converted to sharp notation."""
        import sys

        mock_es = MagicMock()
        mock_extractor_instance = MagicMock(return_value=("Bb", "minor", 0.70))
        mock_es.KeyExtractor = MagicMock(return_value=mock_extractor_instance)

        mock_essentia_pkg = MagicMock()
        mock_essentia_pkg.standard = mock_es

        saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k.startswith("essentia") and k in sys.modules}
        try:
            sys.modules["essentia"] = mock_essentia_pkg
            sys.modules["essentia.standard"] = mock_es

            y = np.random.randn(44100).astype(np.float32)
            key = _estimate_key(y, sr=44100)
        finally:
            for k in list(sys.modules):
                if k.startswith("essentia"):
                    del sys.modules[k]
            sys.modules.update(saved)

        assert key == "A# minor"


# =========================================================================
# Beat / BPM detection tests  (Essentia RhythmExtractor2013)
# =========================================================================


class TestBeatDetection:
    def test_returns_bpm_and_beat_grid(self):
        """_detect_beats_and_downbeats returns BPM and beat times."""
        import sys

        # Simulate RhythmExtractor2013 returning 120 BPM with 8 beats
        fake_beats = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
        mock_extractor = MagicMock(return_value=(120.0, fake_beats, 0.9, np.array([]), np.array([])))

        mock_es = MagicMock()
        mock_es.RhythmExtractor2013 = MagicMock(return_value=mock_extractor)

        mock_essentia_pkg = MagicMock()
        mock_essentia_pkg.standard = mock_es

        saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k.startswith("essentia") and k in sys.modules}
        try:
            sys.modules["essentia"] = mock_essentia_pkg
            sys.modules["essentia.standard"] = mock_es

            y = np.random.randn(44100).astype(np.float32)
            bpm, beat_times, downbeat_times = _detect_beats_and_downbeats(y, sr=44100)
        finally:
            for k in list(sys.modules):
                if k.startswith("essentia"):
                    del sys.modules[k]
            sys.modules.update(saved)

        assert bpm == pytest.approx(120.0, abs=1.0)
        assert len(beat_times) == 8
        assert downbeat_times == []

    def test_empty_beats_returns_zero(self):
        """If no beats are found, return 0.0 BPM and empty lists."""
        import sys

        mock_extractor = MagicMock(return_value=(0.0, np.array([]), 0.0, np.array([]), np.array([])))

        mock_es = MagicMock()
        mock_es.RhythmExtractor2013 = MagicMock(return_value=mock_extractor)

        mock_essentia_pkg = MagicMock()
        mock_essentia_pkg.standard = mock_es

        saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k.startswith("essentia") and k in sys.modules}
        try:
            sys.modules["essentia"] = mock_essentia_pkg
            sys.modules["essentia.standard"] = mock_es

            y = np.random.randn(44100).astype(np.float32)
            bpm, beat_times, downbeat_times = _detect_beats_and_downbeats(y, sr=44100)
        finally:
            for k in list(sys.modules):
                if k.startswith("essentia"):
                    del sys.modules[k]
            sys.modules.update(saved)

        assert bpm == 0.0
        assert beat_times == []
        assert downbeat_times == []


# =========================================================================
# analyse_audio tests
# =========================================================================


class TestAnalyseAudio:
    """Test the top-level analyse_audio() function used by multiprocessing."""

    def test_returns_audio_analysis(self):
        dummy_audio = np.random.randn(44100).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp.write(b"\x00" * 100)
            tmp_path = tmp.name

        with patch("src.models._load_audio", return_value=(dummy_audio, 44100)), \
             patch("src.models._detect_beats_and_downbeats", return_value=(130.0, [0.0, 0.46], [0.0])), \
             patch("src.models._estimate_key", return_value="C major"):
            result = analyse_audio(tmp_path)

        assert result.bpm == 130.0
        assert result.key == "C major"
        assert result.audio is dummy_audio
        assert result.sr == 44100
        assert len(result.beat_times) == 2
        assert len(result.downbeat_times) == 1
        # intro_is_rhythmic defaults True when first beat is at 0.0 s
        assert result.intro_is_rhythmic is True

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

        # Use batch_size=1 (via _PROBED_BATCH_SIZE patch) so each chunk is
        # processed individually → _forward always receives a 1-item batch.
        fake_chunk_emb = np.zeros((1, 512), dtype=np.float32)

        mock_outputs = MagicMock()
        # model(**inputs).last_hidden_state.mean(dim=1).cpu().float().numpy()
        mock_outputs.last_hidden_state.mean.return_value \
            .cpu.return_value.float.return_value.numpy.return_value = fake_chunk_emb

        mock_model = MagicMock(return_value=mock_outputs)
        mock_processor = MagicMock(return_value={"input_values": MagicMock()})
        mock_torch = MagicMock()
        mock_torch.float32 = object()  # sentinel — != _clap_embed_dtype (None)

        # 5-tuple: (waveform, sr, beat_times, bpm, intro_is_rhythmic)
        audio_list = [
            (np.random.randn(22050).astype(np.float32), 22050, [], 0.0, True)
            for _ in range(3)
        ]

        with patch.dict(sys.modules, {"torch": mock_torch}), \
             patch("src.models._resample", side_effect=lambda y, **kw: y), \
             patch("src.models._clap_device", None), \
             patch("src.models._clap_embed_dtype", None), \
             patch("src.models._PROBED_BATCH_SIZE", 1), \
             patch("src.models._get_clap_model", return_value=(mock_model, mock_processor)):
            embeddings = compute_embeddings_batch(audio_list)

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
        def fake_analyse(p):
            path = Path(p)
            return AudioAnalysis(
                file_path=str(path), filename=path.name,
                bpm=120.0, key="C major",
                audio=np.zeros(100, dtype=np.float32), sr=44100,
                beat_times=[0.0, 0.5], downbeat_times=[0.0],
                fingerprint="", duration_sec=3.0,
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
        def fake_analyse(p):
            path = Path(p)
            return AudioAnalysis(
                file_path=str(path), filename=path.name,
                bpm=120.0, key="C major",
                audio=np.zeros(100, dtype=np.float32), sr=44100,
                beat_times=[], downbeat_times=[],
                fingerprint="",
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
        def fake_analyse(p):
            path = Path(p)
            return AudioAnalysis(
                file_path=str(path), filename=path.name,
                bpm=120.0, key="C major",
                audio=np.zeros(100, dtype=np.float32), sr=44100,
                beat_times=[], downbeat_times=[],
                fingerprint="",
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


# =========================================================================
# _select_embed_dtype tests
# =========================================================================


class TestSelectEmbedDtype:
    """_select_embed_dtype selects the right precision for the device."""

    def _make_device(self, type_: str):
        dev = MagicMock()
        dev.type = type_
        return dev

    def test_cpu_returns_float32(self):
        import sys
        mock_torch = MagicMock()
        mock_torch.float32 = "float32"
        mock_torch.float16 = "float16"
        mock_torch.bfloat16 = "bfloat16"
        with patch.dict(sys.modules, {"torch": mock_torch}):
            result = _select_embed_dtype(self._make_device("cpu"))
        assert result == mock_torch.float32

    def test_cuda_ampere_returns_bfloat16(self):
        import sys
        mock_torch = MagicMock()
        mock_torch.float32 = "float32"
        mock_torch.float16 = "float16"
        mock_torch.bfloat16 = "bfloat16"
        mock_torch.cuda.get_device_capability.return_value = (8, 0)
        with patch.dict(sys.modules, {"torch": mock_torch}):
            result = _select_embed_dtype(self._make_device("cuda"))
        assert result == mock_torch.bfloat16

    def test_cuda_turing_returns_float16(self):
        import sys
        mock_torch = MagicMock()
        mock_torch.float32 = "float32"
        mock_torch.float16 = "float16"
        mock_torch.bfloat16 = "bfloat16"
        mock_torch.cuda.get_device_capability.return_value = (7, 5)
        with patch.dict(sys.modules, {"torch": mock_torch}):
            result = _select_embed_dtype(self._make_device("cuda"))
        assert result == mock_torch.float16

    def test_cuda_old_returns_float32(self):
        import sys
        mock_torch = MagicMock()
        mock_torch.float32 = "float32"
        mock_torch.float16 = "float16"
        mock_torch.bfloat16 = "bfloat16"
        mock_torch.cuda.get_device_capability.return_value = (6, 1)
        with patch.dict(sys.modules, {"torch": mock_torch}):
            result = _select_embed_dtype(self._make_device("cuda"))
        assert result == mock_torch.float32

    def test_mps_returns_bfloat16(self):
        import sys
        mock_torch = MagicMock()
        mock_torch.float32 = "float32"
        mock_torch.bfloat16 = "bfloat16"
        with patch.dict(sys.modules, {"torch": mock_torch}):
            result = _select_embed_dtype(self._make_device("mps"))
        assert result == mock_torch.bfloat16


# =========================================================================
# _detect_intro_type tests
# =========================================================================


class TestDetectIntroType:
    """_detect_intro_type classifies pre-beat intros correctly."""

    def test_short_intro_returns_true(self):
        """Intro ≤ 5 s is always classified as rhythmic (no analysis needed)."""
        y = np.random.randn(44100 * 10).astype(np.float32)
        result = _detect_intro_type(y, 44100, beat_times=[3.0, 3.5, 4.0], bpm=120.0)
        assert result is True

    def test_no_beats_returns_true(self):
        y = np.random.randn(44100).astype(np.float32)
        assert _detect_intro_type(y, 44100, beat_times=[], bpm=0.0) is True

    def test_zero_bpm_returns_true(self):
        y = np.random.randn(44100).astype(np.float32)
        assert _detect_intro_type(y, 44100, beat_times=[10.0], bpm=0.0) is True

    def test_rhythmic_intro_detected(self):
        """Audio with energy only on expected beat positions → True."""
        sr = 44100
        bpm = 120.0
        beat_interval = 60.0 / bpm  # 0.5 s

        # Build ~20 s of silence, then place impulses at expected beat positions.
        total_samples = int(20 * sr)
        y = np.zeros(total_samples, dtype=np.float32)

        # First real beat at 10 s; expected beats extrapolate backward.
        first_beat = 10.0
        beat_times = [first_beat + i * beat_interval for i in range(10)]

        t = first_beat - beat_interval
        win = int(0.01 * sr)  # small impulse window
        while t >= 0.0:
            center = int(t * sr)
            y[max(0, center - win): center + win] = 1.0
            t -= beat_interval

        result = _detect_intro_type(y, sr, beat_times=beat_times, bpm=bpm)
        assert result is True  # ratio >> 2.0

    def test_silent_intro_detected_as_beatless(self):
        """Completely silent intro → ratio ≈ 1 → False (genuinely beatless)."""
        sr = 44100
        bpm = 120.0
        beat_interval = 60.0 / bpm

        # 20 s of white noise uniformly distributed (no beat accent pattern).
        rng = np.random.default_rng(seed=0)
        y = rng.standard_normal(int(20 * sr)).astype(np.float32) * 0.01

        first_beat = 10.0
        beat_times = [first_beat + i * beat_interval for i in range(10)]

        # No impulses → on-beat and off-beat RMS are both ~equal (noise floor).
        result = _detect_intro_type(y, sr, beat_times=beat_times, bpm=bpm)
        # With uniform noise the ratio should be ≈ 1.0 → below threshold of 2.0
        assert result is False


# =========================================================================
# _phrase_chunks tests
# =========================================================================


class TestPhraseChunks:
    """_phrase_chunks produces correct phrase-aligned audio segments."""

    SR = 24_000  # MERT target sample rate

    def _beat_times(self, bpm: float, n_beats: int, offset: float = 0.0) -> list[float]:
        interval = 60.0 / bpm
        return [offset + i * interval for i in range(n_beats)]

    def test_fallback_fixed_stride_when_few_beats(self):
        """< 32 beats → fixed-stride fallback, no phrase splitting."""
        from src.config import CHUNK_MAX_SEC
        sr = self.SR
        # 60 s of audio, only 10 beats → fallback to 30 s chunks
        y = np.zeros(60 * sr, dtype=np.float32)
        beats = self._beat_times(120.0, 10, offset=0.5)
        chunks = _phrase_chunks(y, sr, beats, bpm=120.0, intro_is_rhythmic=True)
        # Expect two 30 s chunks
        assert len(chunks) == 2
        assert all(len(c) <= CHUNK_MAX_SEC * sr for c in chunks)

    def test_phrase_split_on_32_beats(self):
        """≥ 32 beats → split at every 32nd beat."""
        sr = self.SR
        bpm = 120.0
        # 90 s of audio, 180 beats → 5+ phrases of ~16 s each (32 beats × 0.5 s)
        y = np.zeros(int(90 * sr), dtype=np.float32)
        beats = self._beat_times(bpm, 180, offset=0.0)
        chunks = _phrase_chunks(y, sr, beats, bpm=bpm, intro_is_rhythmic=True)
        # 180 beats → split points at indices 0, 32, 64, 96, 128, 160 → 5 phrases
        assert len(chunks) >= 2
        # All chunks should respect max (30 s) and be ≥ 1 s
        assert all(len(c) >= sr for c in chunks)
        assert all(len(c) <= 30 * sr for c in chunks)

    def test_beatless_intro_dropped(self):
        """intro_is_rhythmic=False → pre-beat region is discarded."""
        sr = self.SR
        bpm = 120.0
        # 10 s intro + 64 beats of content
        intro_len = int(10 * sr)
        beat_offset = 10.0
        beats = self._beat_times(bpm, 64, offset=beat_offset)
        y = np.ones(int(beat_offset * sr) + len(beats) * int(60 / bpm * sr) + sr,
                    dtype=np.float32)
        # Mark intro region with a distinct value to verify it's excluded
        y[:intro_len] = 99.0

        chunks = _phrase_chunks(y, sr, beats, bpm=bpm, intro_is_rhythmic=False)
        # None of the chunks should start with the intro marker
        for c in chunks:
            assert not np.any(c == 99.0), "Beatless intro leaked into a chunk"

    def test_rhythmic_intro_prepended_when_short(self):
        """Short rhythmic intro (< 12 s) is prepended to phrase 1."""
        from src.config import CHUNK_MIN_SEC
        sr = self.SR
        bpm = 120.0
        # 5 s intro (< min_sec=12) + 64 beats of content
        intro_sec = 5
        beat_offset = float(intro_sec)
        beats = self._beat_times(bpm, 64, offset=beat_offset)
        duration = int((beat_offset + 64 * (60 / bpm) + 1) * sr)
        y = np.zeros(duration, dtype=np.float32)

        chunks = _phrase_chunks(y, sr, beats, bpm=bpm, intro_is_rhythmic=True)
        # First chunk should be longer than a lone 16 s phrase (it includes the 5 s intro)
        phrase_samples = int(32 * (60 / bpm) * sr)
        assert len(chunks[0]) >= phrase_samples  # intro prepended

    def test_long_rhythmic_intro_is_own_chunk(self):
        """Long rhythmic intro (≥ 12 s) becomes its own chunk."""
        from src.config import CHUNK_MIN_SEC
        sr = self.SR
        bpm = 120.0
        intro_sec = 15  # > CHUNK_MIN_SEC=12
        beat_offset = float(intro_sec)
        beats = self._beat_times(bpm, 64, offset=beat_offset)
        duration = int((beat_offset + 64 * (60 / bpm) + 1) * sr)
        y = np.zeros(duration, dtype=np.float32)

        chunks = _phrase_chunks(y, sr, beats, bpm=bpm, intro_is_rhythmic=True)
        # All chunks should be ≥ 12 s (min clamp), first is the standalone intro
        assert all(len(c) >= CHUNK_MIN_SEC * sr for c in chunks)

    def test_max_clamp_splits_long_phrases(self):
        """A single phrase > 30 s is split with fixed stride."""
        from src.config import CHUNK_MAX_SEC
        sr = self.SR
        bpm = 30.0  # 32 beats × 2 s = 64 s per phrase → exceeds 30 s max
        beats = self._beat_times(bpm, 64, offset=0.0)
        duration = int((64 * (60 / bpm) + 1) * sr)
        y = np.zeros(duration, dtype=np.float32)

        chunks = _phrase_chunks(y, sr, beats, bpm=bpm, intro_is_rhythmic=True)
        assert all(len(c) <= CHUNK_MAX_SEC * sr for c in chunks)


# =========================================================================
# _optimal_worker_count tests
# =========================================================================


class TestOptimalWorkerCount:
    """_optimal_worker_count respects CPU and RAM constraints."""

    def test_returns_at_least_one(self):
        result = _optimal_worker_count()
        assert result >= 1

    def test_capped_at_eight(self):
        with patch("os.cpu_count", return_value=64):
            try:
                import psutil
                with patch.object(psutil, "cpu_count", return_value=32), \
                     patch.object(psutil, "virtual_memory",
                                  return_value=MagicMock(available=256 * 1024 ** 3)):
                    result = _optimal_worker_count()
            except ImportError:
                result = _optimal_worker_count()
        assert result <= 8

    def test_ram_cap_limits_workers(self):
        """With only 1.5 GB available, at most 1 worker should be allowed."""
        try:
            import psutil
            with patch.object(psutil, "cpu_count", return_value=8), \
                 patch.object(psutil, "virtual_memory",
                              return_value=MagicMock(available=int(1.5 * 1024 ** 3))):
                result = _optimal_worker_count()
            assert result == 1
        except ImportError:
            pytest.skip("psutil not installed")
