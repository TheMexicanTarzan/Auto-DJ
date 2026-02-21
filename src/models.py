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

import librosa
import numpy as np
import torch
from transformers import ClapModel, ClapProcessor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CLAP model singleton — loaded once, shared across all Song instances.
# We use laion/clap-htsat-unfused for general-purpose audio embeddings.
# ---------------------------------------------------------------------------

_clap_model: ClapModel | None = None
_clap_processor: ClapProcessor | None = None


def _get_clap_model() -> tuple[ClapModel, ClapProcessor]:
    """Lazy-load the CLAP model and processor on first use."""
    global _clap_model, _clap_processor

    if _clap_model is None:
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

    best_corr = -2.0
    best_key = "C major"

    # Try every possible root note (0–11) for both major and minor
    for shift in range(12):
        shifted_chroma = np.roll(chroma_profile, -shift)

        major_corr = float(np.corrcoef(shifted_chroma, major_profile)[0, 1])
        if major_corr > best_corr:
            best_corr = major_corr
            best_key = f"{_PITCH_CLASSES[shift]} major"

        minor_corr = float(np.corrcoef(shifted_chroma, minor_profile)[0, 1])
        if minor_corr > best_corr:
            best_corr = minor_corr
            best_key = f"{_PITCH_CLASSES[shift]} minor"

    return best_key


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
            1. Load the audio with librosa (mono, native sample rate).
            2. Estimate BPM via librosa's beat tracker.
            3. Estimate the musical key via chroma analysis.
            4. Extract a CLAP embedding for semantic similarity.
        """
        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")

        logger.info("Analysing '%s'...", path.name)

        # --- 1. Load audio ------------------------------------------------
        y, sr = librosa.load(str(path), sr=None, mono=True)

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
        model, processor = _get_clap_model()

        # CLAP expects 48 kHz input
        target_sr = 48_000
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

        # Prepare inputs and run inference (no gradients needed)
        inputs = processor(audio=y, sampling_rate=target_sr, return_tensors="pt")
        with torch.no_grad():
            outputs = model.get_audio_features(**inputs)

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
