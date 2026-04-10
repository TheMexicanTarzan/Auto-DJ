"""
models.py — Core data model for the DJ Mixing Pathfinding System.

Defines the Song class, which represents a single audio track as a node
in our future mixing graph. Each song carries its own metadata (BPM, key)
and a semantic embedding vector that captures the 'vibe' of the track.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np

logger = logging.getLogger(__name__)


class AudioAnalysis(NamedTuple):
    """Intermediate result from audio analysis (before CLAP embedding).

    Produced by ``analyse_audio()`` in multiprocessing workers; the CLAP
    embedding step runs separately in the main process via batch inference.
    """
    file_path: str
    filename: str
    bpm: float
    key: str
    audio: np.ndarray   # mono waveform at native sample rate
    sr: int
    beat_times: list     # all detected beat times (seconds)
    downbeat_times: list # downbeat ("1") times (seconds)
    fingerprint: str     # Chromaprint audio fingerprint
    duration_sec: float  # total audio duration in seconds

# ---------------------------------------------------------------------------
# CLAP model singleton — loaded once, shared across all Song instances.
# We use laion/clap-htsat-unfused for general-purpose audio embeddings.
#
# Heavy libraries (essentia, torch, transformers) are imported lazily
# inside the functions that need them so that cached-graph startups stay
# fast.
# ---------------------------------------------------------------------------

_clap_model = None
_clap_processor = None
_clap_device = None

# Length of each audio segment fed to MERT in one forward pass.
# 30 s @ 24 kHz = 720 000 samples → keeps peak VRAM well below 2 GB.
_EMBED_CHUNK_SEC = 30


def _get_clap_model():
    """Lazy-load the MERT model and feature extractor on first use.

    When a CUDA-capable GPU is available the model is moved to it
    automatically, giving a significant speed-up for embedding inference.
    """
    global _clap_model, _clap_processor, _clap_device

    if _clap_model is None:
        import torch
        from transformers import AutoFeatureExtractor, AutoModel

        local_path = str(Path(__file__).resolve().parent / "mert")
        logger.info("Loading MERT model from '%s' (this only happens once)...", local_path)
        _clap_processor = AutoFeatureExtractor.from_pretrained(
            local_path, trust_remote_code=True
        )
        _clap_model = AutoModel.from_pretrained(local_path, trust_remote_code=True)
        _clap_model.eval()  # inference mode — no gradient tracking needed

        _clap_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _clap_model.to(_clap_device)
        if _clap_device.type == "cuda":
            _clap_model.half()  # float16 halves weight memory (~380 MB → ~190 MB)
        logger.info("MERT model loaded on device: %s", _clap_device)

    return _clap_model, _clap_processor


# ---------------------------------------------------------------------------
# Audio loading helper  (Essentia MonoLoader — backed by ffmpeg)
# ---------------------------------------------------------------------------

_DEFAULT_SR = 44100  # Standard sample rate used for analysis


def _load_audio(path: str | Path, sr: int = _DEFAULT_SR) -> tuple[np.ndarray, int]:
    """
    Load an audio file as a mono waveform using Essentia's ``MonoLoader``.

    Uses ffmpeg under the hood (via Essentia's C++ bindings), which is
    significantly faster than librosa's Python fallback chain and natively
    handles m4a/aac without PySoundFile warnings.

    Args:
        path: Path to the audio file.
        sr:   Target sample rate (default 44100 Hz).

    Returns:
        (y, sr) — mono waveform as float32 numpy array, and the sample rate.
    """
    import essentia.standard as es

    loader = es.MonoLoader(filename=str(path), sampleRate=float(sr))
    y = loader()
    return y, sr


def _resample(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio using Essentia's ``Resample`` algorithm."""
    if orig_sr == target_sr:
        return y
    import essentia.standard as es

    resampler = es.Resample(inputSampleRate=float(orig_sr),
                            outputSampleRate=float(target_sr))
    return resampler(y.astype(np.float32))


# ---------------------------------------------------------------------------
# Key detection helper  (Essentia KeyExtractor)
# ---------------------------------------------------------------------------

# Canonical pitch-class names used throughout the application (sharp notation).
_PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Essentia may return flat notation (e.g. "Bb"); normalise to sharps so that
# downstream code (_SEMITONE_INDEX in metrics.py) always receives a known name.
_FLAT_TO_SHARP: dict[str, str] = {
    "Db": "C#",
    "Eb": "D#",
    "Gb": "F#",
    "Ab": "G#",
    "Bb": "A#",
}


def _estimate_key(y: np.ndarray, sr: int) -> str:
    """
    Estimate the musical key of an audio signal using Essentia's
    ``KeyExtractor`` algorithm.

    Returns a string like ``'C major'`` or ``'A minor'``.
    """
    import essentia.standard as es

    key_extractor = es.KeyExtractor(sampleRate=float(sr))
    key, scale, _strength = key_extractor(y.astype(np.float32))

    # Normalise flat notation → sharp notation
    key = _FLAT_TO_SHARP.get(key, key)

    return f"{key} {scale}"


# ---------------------------------------------------------------------------
# Beat / BPM detection helper  (Essentia RhythmExtractor2013)
# ---------------------------------------------------------------------------

_RHYTHM_SR = 44100  # RhythmExtractor2013 requires 44100 Hz input


def _detect_beats_and_downbeats(
        y: np.ndarray,
        sr: int,
) -> tuple[float, list[float], list[float]]:
    """
    Detect BPM and the full beat grid using Essentia's
    ``RhythmExtractor2013`` algorithm (multifeature method).

    Args:
        y:  Mono audio waveform as a numpy array.
        sr: Sample rate of the audio.

    Returns:
        bpm:            Estimated tempo in beats per minute.
        beat_times:     List of all detected beat times in seconds.
        downbeat_times: Always empty — RhythmExtractor2013 does not
                        distinguish downbeats from other beats.
    """
    import essentia.standard as es

    # RhythmExtractor2013 requires 44100 Hz input; resample if needed.
    if sr != _RHYTHM_SR:
        y = _resample(y, orig_sr=sr, target_sr=_RHYTHM_SR)

    audio = y.astype(np.float32)

    rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
    bpm, beats, beats_confidence, _, _ = rhythm_extractor(audio)

    beat_times: list[float] = beats.tolist() if len(beats) > 0 else []

    return float(bpm), beat_times, []


# ---------------------------------------------------------------------------
# Audio fingerprinting  (Chromaprint via pyacoustid)
# ---------------------------------------------------------------------------

def _compute_fingerprint(y: np.ndarray, sr: int) -> str:
    """
    Compute a Chromaprint audio fingerprint from a mono waveform.

    Returns the fingerprint as a compact encoded string.  Returns an
    empty string if the Chromaprint library is unavailable so that
    fingerprinting degrades gracefully.
    """
    try:
        import chromaprint
    except ImportError:
        logger.warning(
            "chromaprint not available — skipping audio fingerprinting. "
            "Install pyacoustid and libchromaprint for deduplication support."
        )
        return ""

    # Chromaprint expects 16-bit signed PCM at the native sample rate.
    max_val = np.max(np.abs(y))
    if max_val > 0:
        pcm = (y / max_val * 32767).astype(np.int16)
    else:
        return ""

    _, encoded = chromaprint.get_fingerprint(pcm.tobytes(), sr, 1)
    return encoded


def analyse_audio(path: str | Path) -> AudioAnalysis:
    """
    Load an audio file and compute BPM, key, beats, and downbeats
    (everything *except* the CLAP embedding).

    This is a top-level function so it can be pickled for use with
    ``multiprocessing.Pool``.  The CLAP embedding step is deliberately
    excluded — it runs in the main process via batch inference to avoid
    duplicating the ~600 MB model across worker processes.
    """
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    logger.info("Analysing '%s'...", path.name)

    y, sr = _load_audio(path)

    # Validate audio
    duration_sec = len(y) / sr
    duration = duration_sec
    if duration < 1.0:
        raise ValueError(
            f"Audio too short ({duration:.2f}s): {path.name}. "
            "At least 1 second of audio is required."
        )

    rms = float(np.sqrt(np.mean(y**2)))
    if rms < 1e-4:
        raise ValueError(
            f"Audio appears to be silent (RMS={rms:.6f}): {path.name}"
        )

    # Beat / tempo detection (Essentia)
    bpm, beat_times, downbeat_times = _detect_beats_and_downbeats(y, sr)

    # Key estimation (Essentia)
    key = _estimate_key(y, sr)

    # Audio fingerprint (Chromaprint)
    fingerprint = _compute_fingerprint(y, sr)

    return AudioAnalysis(
        file_path=str(path),
        filename=path.name,
        bpm=round(bpm, 2),
        key=key,
        audio=y,
        sr=sr,
        beat_times=beat_times,
        downbeat_times=downbeat_times,
        fingerprint=fingerprint,
        duration_sec=duration_sec,
    )


def compute_embeddings_batch(
    audio_list: list[tuple[np.ndarray, int]],
    batch_size: int = 8,
) -> list[np.ndarray]:
    """
    Compute CLAP embeddings for multiple audio signals in batched
    forward passes.

    Args:
        audio_list: List of (waveform, sample_rate) tuples.
        batch_size: Number of audio signals per forward pass.

    Returns:
        List of 1-D numpy arrays (one 512-dim embedding per input).
    """
    if not audio_list:
        return []

    import torch

    model, processor = _get_clap_model()
    target_sr = 24_000  # MERT was trained at 24 kHz

    # Resample all audio to 24 kHz in parallel.  Essentia's C++ Resample
    # releases the GIL, so threads genuinely run concurrently.  Order is
    # preserved by ThreadPoolExecutor.map().
    def _resample_one(args: tuple[np.ndarray, int]) -> np.ndarray:
        y, sr = args
        return _resample(y, orig_sr=sr, target_sr=target_sr) if sr != target_sr else y

    with ThreadPoolExecutor() as pool:
        resampled = list(pool.map(_resample_one, audio_list))

    embeddings: list[np.ndarray] = []
    chunk_samples = _EMBED_CHUNK_SEC * target_sr  # e.g. 30 s × 24 000 = 720 000

    for y in resampled:
        # Split the waveform into fixed-length segments.  Drop any trailing
        # segment shorter than 1 s — too short to produce a reliable embedding.
        # If the whole track is shorter than 1 s, fall back to the full audio.
        chunks = [
            y[s : s + chunk_samples]
            for s in range(0, len(y), chunk_samples)
            if len(y[s : s + chunk_samples]) >= target_sr
        ]
        if not chunks:
            chunks = [y]

        chunk_embs: list[np.ndarray] = []
        chunk_lengths: list[int] = []
        for chunk in chunks:
            inputs = processor(
                chunk,
                sampling_rate=target_sr,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(_clap_device) for k, v in inputs.items()}
            if _clap_device.type == "cuda":
                inputs = {k: v.half() if v.is_floating_point() else v
                          for k, v in inputs.items()}
            with torch.no_grad():
                out = model(**inputs)
            # Mean-pool over the time axis → (hidden_dim,)
            chunk_embs.append(out.last_hidden_state.mean(dim=1).squeeze().cpu().float().numpy())
            chunk_lengths.append(len(chunk))

        # Weighted average — longer chunks contribute proportionally more.
        weights = np.array(chunk_lengths, dtype=np.float64)
        weights /= weights.sum()
        embeddings.append(np.average(chunk_embs, axis=0, weights=weights))

    return embeddings


# ---------------------------------------------------------------------------
# Song dataclass
# ---------------------------------------------------------------------------


@dataclass
class Song:
    """
    Represents a single audio track in the DJ library.

    Attributes:
        file_path:       Absolute path to the audio file on disk.
        filename:        Just the file name (e.g. 'track.mp3').
        bpm:             Estimated tempo in beats per minute.
        key:             Estimated musical key (e.g. 'A minor').
        embedding:       512-dim CLAP embedding capturing the track's sonic character.
        beat_times:      List of all detected beat times in seconds.
        downbeat_times:  List of downbeat ("1") times in seconds — essential
                         for DJ phrasing and phrase-aligned mixing.
    """

    file_path: str
    filename: str
    bpm: float = 0.0
    key: str = ""
    embedding: np.ndarray = field(default_factory=lambda: np.array([]))
    beat_times: list[float] = field(default_factory=list)
    downbeat_times: list[float] = field(default_factory=list)
    content_hash: str = ""
    fingerprint: str = ""
    duration_sec: float = 0.0   # total audio duration; 0 means unknown

    # ------------------------------------------------------------------
    # Factory method — the primary way to create a Song from a file
    # ------------------------------------------------------------------

    @classmethod
    def from_file(cls, path: str | Path) -> Song:
        """
        Analyse an audio file and return a fully-populated Song instance.

        Steps:
            1. Load the audio with Essentia MonoLoader (mono, 44100 Hz).
            2. Validate the audio (reject files that are too short or silent).
            3. Estimate BPM and beat grid via Essentia's RhythmExtractor2013.
            4. Estimate the musical key via Essentia's KeyExtractor.
            5. Extract a CLAP embedding for semantic similarity.
        """
        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")

        logger.info("Analysing '%s'...", path.name)

        # --- 1. Load audio ------------------------------------------------
        y, sr = _load_audio(path)

        # --- 2. Validate audio --------------------------------------------
        duration_sec = len(y) / sr
        duration = duration_sec
        if duration < 1.0:
            raise ValueError(
                f"Audio too short ({duration:.2f}s): {path.name}. "
                "At least 1 second of audio is required."
            )

        rms = float(np.sqrt(np.mean(y**2)))
        if rms < 1e-4:
            raise ValueError(
                f"Audio appears to be silent (RMS={rms:.6f}): {path.name}"
            )

        # --- 3. Beat / tempo detection (Essentia) -------------------------
        bpm, beat_times, downbeat_times = _detect_beats_and_downbeats(y, sr)

        # --- 4. Key estimation (Essentia) ---------------------------------
        key = _estimate_key(y, sr)

        # --- 5. Audio fingerprint (Chromaprint) ---------------------------
        fingerprint = _compute_fingerprint(y, sr)

        # --- 6. CLAP embedding --------------------------------------------
        embedding = cls._compute_embedding(y, sr)

        return cls(
            file_path=str(path),
            filename=path.name,
            bpm=round(bpm, 2),
            key=key,
            embedding=embedding,
            beat_times=beat_times,
            downbeat_times=downbeat_times,
            fingerprint=fingerprint,
            duration_sec=duration_sec,
        )

    # ------------------------------------------------------------------
    # Embedding helper
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_embedding(y: np.ndarray, sr: int) -> np.ndarray:
        """
        Generate a fixed-size semantic embedding for the audio signal
        using the MERT model.

        MERT expects audio at 24 kHz, so we resample if needed.
        Returns a 1-D numpy array (hidden_dim dimensions, typically 768).
        """
        import torch

        model, processor = _get_clap_model()

        # MERT expects 24 kHz input
        target_sr = 24_000
        if sr != target_sr:
            y = _resample(y, orig_sr=sr, target_sr=target_sr)

        # Split into fixed-length chunks to avoid CUDA OOM on long tracks.
        chunk_samples = _EMBED_CHUNK_SEC * target_sr  # 720 000 @ 24 kHz
        chunks = [
            y[s : s + chunk_samples]
            for s in range(0, len(y), chunk_samples)
            if len(y[s : s + chunk_samples]) >= target_sr
        ]
        if not chunks:
            chunks = [y]

        chunk_embs: list[np.ndarray] = []
        chunk_lengths: list[int] = []
        for chunk in chunks:
            inputs = processor(chunk, sampling_rate=target_sr, return_tensors="pt")
            inputs = {k: v.to(_clap_device) for k, v in inputs.items()}
            if _clap_device.type == "cuda":
                inputs = {k: v.half() if v.is_floating_point() else v
                          for k, v in inputs.items()}
            with torch.no_grad():
                out = model(**inputs)
            chunk_embs.append(out.last_hidden_state.mean(dim=1).squeeze().cpu().float().numpy())
            chunk_lengths.append(len(chunk))

        # Weighted average — longer chunks contribute proportionally more.
        weights = np.array(chunk_lengths, dtype=np.float64)
        weights /= weights.sum()
        return np.average(chunk_embs, axis=0, weights=weights)

    # ------------------------------------------------------------------
    # Readable representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        emb_shape = self.embedding.shape if self.embedding.size > 0 else "(empty)"
        return (
            f"Song(filename='{self.filename}', "
            f"bpm={self.bpm}, key='{self.key}', "
            f"embedding={emb_shape})"
        )
