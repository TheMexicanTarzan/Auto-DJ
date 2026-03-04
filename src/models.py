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
    """Intermediate result from audio analysis (before CLAP embedding)."""
    file_path: str
    filename: str
    bpm: float
    key: str
    audio: np.ndarray  # mono waveform at 22050 Hz
    sr: int

# ---------------------------------------------------------------------------
# CLAP model singleton — loaded once, shared across all Song instances.
# We use laion/clap-htsat-unfused for general-purpose audio embeddings.
#
# Heavy libraries (librosa, torch, transformers) are imported lazily inside
# the functions that need them so that cached-graph startups stay fast.
# ---------------------------------------------------------------------------

_clap_model = None
_clap_processor = None


def _get_clap_model():
    """Lazy-load the CLAP model and processor on first use."""
    global _clap_model, _clap_processor

    if _clap_model is None:
        from transformers import ClapModel, ClapProcessor

        model_name = "laion/clap-htsat-unfused"
        logger.info("Loading CLAP model '%s' (this only happens once)...", model_name)
        _clap_processor = ClapProcessor.from_pretrained(model_name)
        _clap_model = ClapModel.from_pretrained(model_name)
        _clap_model.eval()  # inference mode — no gradient tracking needed

    return _clap_model, _clap_processor


# ---------------------------------------------------------------------------
# Key detection helper
# ---------------------------------------------------------------------------

# Mapping from Librosa's pitch-class indices to musical key names.
# Index 0 = C, 1 = C#, ... 11 = B.  We append 'major' or 'minor'
# based on whichever profile correlates more strongly.
_PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _estimate_key(y: np.ndarray, sr: int) -> str:
    """
    Estimate the musical key of an audio signal using chroma features
    and the Krumhansl-Schmuckler key-finding algorithm.

    Returns a string like 'C major' or 'A minor'.
    """
    import librosa

    # Compute the chromagram and average across time to get a 12-bin profile
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_profile = np.mean(chroma, axis=1)

    # Krumhansl-Kessler key profiles (how strongly each pitch class
    # correlates with a given key in major vs minor tonality)
    major_profile = np.array(
        [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    )
    minor_profile = np.array(
        [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    )

    # Vectorized: build all 12 shifted chroma profiles at once (12×12 matrix)
    # Each row is np.roll(chroma_profile, -shift) for shift in 0..11
    indices = np.arange(12)
    shifted_chromas = np.array([chroma_profile[(indices + s) % 12] for s in range(12)])

    # Standardise each row (zero mean, unit variance) for Pearson correlation
    def _row_correlations(matrix: np.ndarray, ref: np.ndarray) -> np.ndarray:
        """Pearson correlation of each row in *matrix* against *ref*."""
        m_centered = matrix - matrix.mean(axis=1, keepdims=True)
        r_centered = ref - ref.mean()
        numer = m_centered @ r_centered
        denom = np.sqrt((m_centered ** 2).sum(axis=1)) * np.sqrt((r_centered ** 2).sum())
        # Guard against zero denominator (constant profile — extremely unlikely)
        return np.where(denom > 0, numer / denom, 0.0)

    major_corrs = _row_correlations(shifted_chromas, major_profile)  # shape (12,)
    minor_corrs = _row_correlations(shifted_chromas, minor_profile)  # shape (12,)

    # Find the best among all 24 candidates
    all_corrs = np.concatenate([major_corrs, minor_corrs])  # shape (24,)
    best_idx = int(np.argmax(all_corrs))
    shift = best_idx % 12
    mode = "major" if best_idx < 12 else "minor"

    return f"{_PITCH_CLASSES[shift]} {mode}"


def analyse_audio(path: str | Path) -> AudioAnalysis:
    """
    Load an audio file and compute BPM + key (no CLAP embedding).

    This is a top-level function so it can be pickled for use with
    multiprocessing.Pool.  The CLAP embedding step is deliberately
    excluded — it runs in the main process via batch inference.
    """
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    import librosa

    logger.info("Analysing '%s'...", path.name)

    y, sr = librosa.load(str(path), sr=22050, mono=True)

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(np.atleast_1d(tempo)[0])

    key = _estimate_key(y, sr)

    return AudioAnalysis(
        file_path=str(path),
        filename=path.name,
        bpm=round(bpm, 2),
        key=key,
        audio=y,
        sr=sr,
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

    import librosa
    import torch

    model, processor = _get_clap_model()
    target_sr = 48_000

    # Resample all audio to 48 kHz (CLAP's expected rate)
    resampled = []
    for y, sr in audio_list:
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        resampled.append(y)

    embeddings: list[np.ndarray] = []

    for i in range(0, len(resampled), batch_size):
        batch = resampled[i : i + batch_size]
        inputs = processor(
            audios=batch,
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
        file_path:  Absolute path to the audio file on disk.
        filename:   Just the file name (e.g. 'track.mp3').
        bpm:        Estimated tempo in beats per minute.
        key:        Estimated musical key (e.g. 'A minor').
        embedding:  512-dim CLAP embedding capturing the track's sonic character.
    """

    file_path: str
    filename: str
    bpm: float = 0.0
    key: str = ""
    embedding: np.ndarray = field(default_factory=lambda: np.array([]))

    # ------------------------------------------------------------------
    # Factory method — the primary way to create a Song from a file
    # ------------------------------------------------------------------

    @classmethod
    def from_file(cls, path: str | Path) -> Song:
        """
        Analyse an audio file and return a fully-populated Song instance.

        Steps:
            1. Load the audio with librosa (mono, 22050 Hz).
            2. Estimate BPM via librosa's beat tracker.
            3. Estimate the musical key via chroma analysis.
            4. Extract a CLAP embedding for semantic similarity.
        """
        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")

        logger.info("Analysing '%s'...", path.name)

        import librosa

        # --- 1. Load audio ------------------------------------------------
        y, sr = librosa.load(str(path), sr=22050, mono=True)

        # --- 2. Tempo estimation ------------------------------------------
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        # librosa may return an ndarray; ensure we get a plain float
        bpm = float(np.atleast_1d(tempo)[0])

        # --- 3. Key estimation --------------------------------------------
        key = _estimate_key(y, sr)

        # --- 4. CLAP embedding --------------------------------------------
        embedding = cls._compute_embedding(y, sr)

        return cls(
            file_path=str(path),
            filename=path.name,
            bpm=round(bpm, 2),
            key=key,
            embedding=embedding,
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
        import librosa
        import torch

        model, processor = _get_clap_model()

        # CLAP expects 48 kHz input
        target_sr = 48_000
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

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
