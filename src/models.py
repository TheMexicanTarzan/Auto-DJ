"""
models.py — Core data model for the DJ Mixing Pathfinding System.

Defines the Song class, which represents a single audio track as a node
in our future mixing graph. Each song carries its own metadata (BPM, key)
and a semantic embedding vector that captures the 'vibe' of the track.
"""

from __future__ import annotations

import logging
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


def _get_clap_model():
    """Lazy-load the CLAP model and processor on first use."""
    global _clap_model, _clap_processor

    if _clap_model is None:
        from transformers import ClapModel, ClapProcessor

        local_path = str(Path(__file__).resolve().parent / "clap")
        logger.info("Loading CLAP model from '%s' (this only happens once)...", local_path)
        _clap_processor = ClapProcessor.from_pretrained(local_path)
        _clap_model = ClapModel.from_pretrained(local_path)
        _clap_model.eval()  # inference mode — no gradient tracking needed

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
    duration = len(y) / sr
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

    return AudioAnalysis(
        file_path=str(path),
        filename=path.name,
        bpm=round(bpm, 2),
        key=key,
        audio=y,
        sr=sr,
        beat_times=beat_times,
        downbeat_times=downbeat_times,
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
    target_sr = 48_000

    # Resample all audio to 48 kHz (CLAP's expected rate)
    resampled = []
    for y, sr in audio_list:
        if sr != target_sr:
            y = _resample(y, orig_sr=sr, target_sr=target_sr)
        resampled.append(y)

    embeddings: list[np.ndarray] = []

    for i in range(0, len(resampled), batch_size):
        batch = resampled[i : i + batch_size]
        inputs = processor(
            audio=batch,
            sampling_rate=target_sr,
            return_tensors="pt",
            padding=True,
        )
        with torch.no_grad():
            outputs = model.get_audio_features(**inputs)

        if hasattr(outputs, "pooler_output"):
            outputs = outputs.pooler_output

        # outputs shape: (batch, 512) — split into individual vectors
        batch_embs = outputs.cpu().numpy()
        for j in range(batch_embs.shape[0]):
            embeddings.append(batch_embs[j])

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
        duration = len(y) / sr
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

        # --- 5. CLAP embedding --------------------------------------------
        embedding = cls._compute_embedding(y, sr)

        return cls(
            file_path=str(path),
            filename=path.name,
            bpm=round(bpm, 2),
            key=key,
            embedding=embedding,
            beat_times=beat_times,
            downbeat_times=downbeat_times,
        )

    # ------------------------------------------------------------------
    # Embedding helper
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_embedding(y: np.ndarray, sr: int) -> np.ndarray:
        """
        Generate a fixed-size semantic embedding for the audio signal
        using the CLAP (Contrastive Language–Audio Pretraining) model.

        The CLAP model expects audio at 48 kHz, so we resample if needed.
        Returns a 1-D numpy array (typically 512 dimensions).
        """
        import torch

        model, processor = _get_clap_model()

        # CLAP expects 48 kHz input
        target_sr = 48_000
        if sr != target_sr:
            y = _resample(y, orig_sr=sr, target_sr=target_sr)

        # Prepare inputs and run inference (no gradients needed)
        inputs = processor(audio=y, sampling_rate=target_sr, return_tensors="pt")
        with torch.no_grad():
            outputs = model.get_audio_features(**inputs)

        # Newer transformers return a BaseModelOutputWithPooling instead of
        # a raw tensor.  Extract the projected embedding in that case.
        if hasattr(outputs, "pooler_output"):
            outputs = outputs.pooler_output

        # Flatten to a plain 1-D numpy vector
        return outputs.squeeze().cpu().numpy()

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
