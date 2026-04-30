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

import torch._dynamo
torch._dynamo.config.suppress_errors = True


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
    duration_sec: float = 0.0   # total audio duration in seconds
    intro_is_rhythmic: bool = True  # False → genuinely beatless intro (drop)

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
_clap_embed_dtype = None  # set after model load; drives input casting

# Legacy fixed-stride chunk length (seconds).  Kept for Song._compute_embedding
# fallback and _probe_batch_size worst-case sizing.
_EMBED_CHUNK_SEC = 30

# Conservative default batch size for CPU / MPS or when probing is skipped.
_EMBED_BATCH_SIZE = 4

# Probed CUDA batch size (cached per session; None = not yet determined).
_PROBED_BATCH_SIZE: int | None = None


def _select_embed_dtype(device: "torch.device") -> "torch.dtype":
    """Select the best floating-point dtype for MERT inference.

    - bfloat16 → CUDA Ampere+ (sm ≥ 8.0) or Apple MPS
    - float16  → CUDA Volta / Turing (sm 7.0–7.9)
    - float32  → older CUDA, CPU, or unknown device
    """
    import torch

    if device.type == "cuda":
        major, _ = torch.cuda.get_device_capability(device)
        if major >= 8:
            return torch.bfloat16
        if major >= 7:
            return torch.float16
        return torch.float32
    if device.type == "mps":
        return torch.bfloat16
    return torch.float32


def _get_clap_model():
    """Lazy-load the MERT model and feature extractor on first use.

    When a CUDA-capable GPU is available the model is moved to it
    automatically, giving a significant speed-up for embedding inference.
    SDPA (PyTorch native Flash Attention) and torch.compile are enabled
    where supported, with graceful fallbacks for older hardware/software.
    """
    global _clap_model, _clap_processor, _clap_device, _clap_embed_dtype

    if _clap_model is None:
        import torch
        from transformers import AutoFeatureExtractor, AutoModel

        local_path = str(Path(__file__).resolve().parent / "mert")
        logger.info("Loading MERT model from '%s' (this only happens once)...", local_path)
        _clap_processor = AutoFeatureExtractor.from_pretrained(
            local_path, trust_remote_code=True
        )

        # Attempt SDPA (PyTorch native Flash / mem-efficient attention).
        # Falls back silently if the model's custom code doesn't support it
        # or if the transformers version predates the attn_implementation arg.
        try:
            _clap_model = AutoModel.from_pretrained(
                local_path, trust_remote_code=True, attn_implementation="sdpa"
            )
            logger.info("MERT loaded with SDPA attention (Flash / mem-efficient).")
        except Exception:
            _clap_model = AutoModel.from_pretrained(local_path, trust_remote_code=True)
            logger.info("MERT loaded with default attention (SDPA not supported).")

        _clap_model.eval()  # inference mode — no gradient tracking needed

        _clap_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _clap_model.to(_clap_device)

        # Select the best precision for this device and cast model weights.
        _clap_embed_dtype = _select_embed_dtype(_clap_device)
        if _clap_embed_dtype != torch.float32:
            _clap_model.to(_clap_embed_dtype)
            logger.info("MERT weights cast to %s.", _clap_embed_dtype)

        # torch.compile — JIT-compiles MERT for faster inference.
        # dynamic=True handles varying batch/chunk sizes without recompilation.
        # The very first forward pass triggers compilation (~30–60 s); all
        # subsequent calls run faster.  Requires PyTorch 2.0+.
        if hasattr(torch, "compile"):
            try:
                _clap_model = torch.compile(_clap_model, dynamic=True)
                logger.info(
                    "MERT compiled with torch.compile "
                    "(first inference call will be slow — compiling)."
                )
            except Exception:
                logger.warning(
                    "torch.compile failed; running in eager mode.",
                    exc_info=True,
                )
        else:
            logger.info("torch.compile not available (requires PyTorch 2.0+).")

        logger.info("MERT model ready on device: %s", _clap_device)

    return _clap_model, _clap_processor


# ---------------------------------------------------------------------------
# Batch-size probing  (run once per session on CUDA; cached result)
# ---------------------------------------------------------------------------

def _probe_batch_size() -> int:
    """Probe the largest safe MERT batch size via exponential search.

    Doubles the batch size until a CUDA OOM error occurs, then backs off to
    75 % of the last OOM-free size.  Result is cached for the session.

    Returns the conservative default (_EMBED_BATCH_SIZE) on CPU and MPS
    where memory is not the limiting factor or is not easily probed.
    """
    global _PROBED_BATCH_SIZE
    if _PROBED_BATCH_SIZE is not None:
        return _PROBED_BATCH_SIZE

    import torch

    if _clap_device is None or _clap_device.type != "cuda":
        _PROBED_BATCH_SIZE = _EMBED_BATCH_SIZE
        return _PROBED_BATCH_SIZE

    model, processor = _get_clap_model()
    target_sr = 24_000
    chunk_samples = _EMBED_CHUNK_SEC * target_sr  # worst-case chunk length

    batch_size = 1
    last_good = 1
    while batch_size <= 64:
        try:
            dummy = [np.zeros(chunk_samples, dtype=np.float32)] * batch_size
            inputs = processor(
                dummy, sampling_rate=target_sr, return_tensors="pt", padding=True,
            )
            inputs = {
                k: (v.to(_clap_device).to(_clap_embed_dtype)
                    if v.is_floating_point() and _clap_embed_dtype is not None
                    else v.to(_clap_device))
                for k, v in inputs.items()
            }
            with torch.no_grad():
                model(**inputs)
            torch.cuda.synchronize()
            last_good = batch_size
            batch_size *= 2
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
                break
            raise

    _PROBED_BATCH_SIZE = max(1, int(last_good * 0.75))
    logger.info(
        "MERT batch size probed: %d (last OOM-free: %d)",
        _PROBED_BATCH_SIZE, last_good,
    )
    return _PROBED_BATCH_SIZE


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
# Intro classification  (rhythmic vs genuinely beatless)
# ---------------------------------------------------------------------------

def _detect_intro_type(
    y: np.ndarray,
    sr: int,
    beat_times: list[float],
    bpm: float,
) -> bool:
    """Return True if a long pre-beat intro appears rhythmic (Essentia false
    negative), or False if it is genuinely beatless.

    Only meaningful when ``beat_times[0] > 5 s``.  For songs with short or
    absent intros, or when BPM is 0, returns True (keep) unconditionally.

    Algorithm:
        1. Extrapolate expected beat positions backward from ``beat_times[0]``
           using the global BPM, up to 30 s before the first detected beat.
        2. Compute per-expected-beat on-beat RMS (±25 ms window) and
           off-beat RMS (midpoint to the next expected beat).
        3. If the mean on-beat / off-beat RMS ratio exceeds 2.0, the intro
           has a detectable rhythmic accent pattern → Essentia false negative.
    """
    if not beat_times or beat_times[0] <= 5.0 or bpm <= 0.0:
        return True

    beat_interval = 60.0 / bpm
    analysis_start = max(0.0, beat_times[0] - 30.0)
    window = max(1, int(0.025 * sr))  # ±25 ms in samples

    # Extrapolate expected beat positions backward from the first real beat.
    expected: list[float] = []
    t = beat_times[0] - beat_interval
    while t >= analysis_start:
        expected.append(t)
        t -= beat_interval

    if not expected:
        return True

    on_rms: list[float] = []
    off_rms: list[float] = []
    for bt in expected:
        mid = bt + beat_interval / 2.0  # off-beat midpoint

        for center_sec, bucket in ((bt, on_rms), (mid, off_rms)):
            center = int(center_sec * sr)
            s = max(0, center - window)
            e = min(len(y), center + window)
            if e > s:
                bucket.append(float(np.sqrt(np.mean(y[s:e] ** 2))))

    if not on_rms or not off_rms:
        return True

    mean_on = float(np.mean(on_rms))
    mean_off = float(np.mean(off_rms))
    ratio = mean_on / mean_off if mean_off > 1e-9 else 2.0
    return ratio > 2.0


# ---------------------------------------------------------------------------
# Phrase-aligned chunking
# ---------------------------------------------------------------------------

def _phrase_chunks(
    audio: np.ndarray,
    sr: int,
    beat_times: list[float],
    bpm: float,
    intro_is_rhythmic: bool,
) -> list[np.ndarray]:
    """Split *audio* into phrase-aligned segments using the beat grid.

    Split points are placed every 32 beats (= 8 bars in 4/4 time) so each
    chunk roughly corresponds to a musical phrase.

    Fallback:
        When fewer than 32 beats are available, falls back to a fixed
        _EMBED_CHUNK_SEC-second stride (same as the legacy behaviour).

    Min / max clamps:
        Chunks shorter than 12 s are merged with their neighbour.
        Chunks longer than 30 s are split with a fixed stride.

    Intro handling:
        - ``intro_is_rhythmic=True``, intro ≥ 12 s  → own chunk.
        - ``intro_is_rhythmic=True``, intro < 12 s   → prepended to phrase 1.
        - ``intro_is_rhythmic=False``                → dropped entirely.

    Args:
        audio:             Mono waveform (already at MERT's target_sr).
        sr:                Sample rate of *audio*.
        beat_times:        Beat timestamps in seconds from Essentia analysis.
        bpm:               Global tempo (used only for the fallback condition).
        intro_is_rhythmic: Whether the pre-beat intro carries rhythmic content.

    Returns:
        List of non-overlapping audio chunks, each ≥ 1 s and ≤ 30 s.
    """
    from src.config import CHUNK_MIN_SEC, CHUNK_MAX_SEC

    min_samples = CHUNK_MIN_SEC * sr
    max_samples = CHUNK_MAX_SEC * sr
    min_valid   = sr  # 1 s — minimum length MERT can handle

    def _fixed_stride(y: np.ndarray) -> list[np.ndarray]:
        parts = [y[s:s + max_samples] for s in range(0, len(y), max_samples)]
        valid = [p for p in parts if len(p) >= min_valid]
        return valid if valid else [y]

    def _merge_undersized(segs: list[np.ndarray]) -> list[np.ndarray]:
        """Merge consecutive chunks until each is at least min_samples long."""
        if not segs:
            return segs
        merged: list[np.ndarray] = []
        acc = segs[0]
        for seg in segs[1:]:
            if len(acc) < min_samples:
                acc = np.concatenate([acc, seg])
            else:
                merged.append(acc)
                acc = seg
        # Final accumulated segment
        if merged and len(acc) < min_samples:
            merged[-1] = np.concatenate([merged[-1], acc])
        elif len(acc) >= min_valid:
            merged.append(acc)
        return merged or [audio]

    def _split_oversized(segs: list[np.ndarray]) -> list[np.ndarray]:
        """Split chunks longer than max_samples with a fixed stride."""
        result: list[np.ndarray] = []
        for seg in segs:
            if len(seg) > max_samples:
                result.extend(_fixed_stride(seg))
            elif len(seg) >= min_valid:
                result.append(seg)
        return result

    # --- Fallback: fixed stride when not enough beats for phrase detection ---
    if len(beat_times) < 32:
        return _fixed_stride(audio)

    # --- Phrase boundary sample indices (every 32nd beat) ---
    phrase_times = beat_times[::32]
    phrase_samples = [int(t * sr) for t in phrase_times] + [len(audio)]

    # --- Pre-beat intro region ---
    pre_end = phrase_samples[0]
    pending_intro: np.ndarray | None = None
    raw_segments: list[np.ndarray] = []

    if pre_end > 0:
        intro_seg = audio[:pre_end]
        if intro_is_rhythmic:
            if len(intro_seg) >= min_samples:
                raw_segments.append(intro_seg)   # own chunk
            else:
                pending_intro = intro_seg         # prepend to phrase 1
        # else: genuinely beatless → drop

    # --- Phrase segments ---
    for i in range(len(phrase_samples) - 1):
        seg = audio[phrase_samples[i]:phrase_samples[i + 1]]
        if len(seg) == 0:
            continue
        if pending_intro is not None:
            seg = np.concatenate([pending_intro, seg])
            pending_intro = None
        raw_segments.append(seg)

    if not raw_segments:
        return _fixed_stride(audio)

    # --- Apply min / max clamps ---
    merged = _merge_undersized(raw_segments)
    final  = _split_oversized(merged)
    return final if final else _fixed_stride(audio)


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


def analyse_audio(path: str | Path, enable_fingerprint: bool = False) -> AudioAnalysis:
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

    # Classify any long pre-beat intro as rhythmic or genuinely beatless.
    intro_is_rhythmic = _detect_intro_type(y, sr, beat_times, bpm)

    # Key estimation (Essentia)
    key = _estimate_key(y, sr)

    # Audio fingerprint (Chromaprint)
    fingerprint = _compute_fingerprint(y, sr) if enable_fingerprint else ""

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
        intro_is_rhythmic=intro_is_rhythmic,
    )


def compute_embeddings_batch(
    audio_list: list[tuple[np.ndarray, int, list[float], float, bool]],
) -> list[np.ndarray]:
    """Compute MERT embeddings for a batch of audio signals.

    Each song is split into phrase-aligned chunks (see ``_phrase_chunks``).
    Chunks are sorted into three length buckets so that items batched
    together have similar lengths and padding waste is minimised:

    - short  (≤ 17 s)
    - medium (17 – 24 s)
    - long   (24 – 30 s)

    Batch size is determined once per session by ``_probe_batch_size()``.
    Per-song embeddings are reassembled via a length-weighted average.

    Args:
        audio_list: List of ``(waveform, sr, beat_times, bpm,
                    intro_is_rhythmic)`` tuples.

    Returns:
        List of 1-D numpy arrays (one per input, shape = (hidden_dim,)).
    """
    if not audio_list:
        return []

    import torch

    model, processor = _get_clap_model()
    target_sr = 24_000  # MERT was trained at 24 kHz
    batch_size = _probe_batch_size()

    # Resample all audio to 24 kHz in parallel.
    def _resample_one(args: tuple) -> np.ndarray:
        y, sr, *_ = args
        return _resample(y, orig_sr=sr, target_sr=target_sr) if sr != target_sr else y

    with ThreadPoolExecutor() as pool:
        resampled = list(pool.map(_resample_one, audio_list))

    # Bucket thresholds in samples (at 24 kHz)
    _SHORT_MAX  = 17 * target_sr   # ≤ 17 s
    _MEDIUM_MAX = 24 * target_sr   # 17 – 24 s  (> 24 s → long bucket)

    # Three queues: (song_idx, chunk_len_samples, chunk_array)
    short_q:  list[tuple[int, int, np.ndarray]] = []
    medium_q: list[tuple[int, int, np.ndarray]] = []
    long_q:   list[tuple[int, int, np.ndarray]] = []

    for song_idx, (entry, y_24k) in enumerate(zip(audio_list, resampled)):
        _, _, beat_times, bpm, intro_is_rhythmic = entry
        chunks = _phrase_chunks(y_24k, target_sr, beat_times, bpm, intro_is_rhythmic)
        for chunk in chunks:
            item = (song_idx, len(chunk), chunk)
            if len(chunk) <= _SHORT_MAX:
                short_q.append(item)
            elif len(chunk) <= _MEDIUM_MAX:
                medium_q.append(item)
            else:
                long_q.append(item)

    # ------------------------------------------------------------------
    # Inner helper: one MERT forward pass for a list of chunks
    # ------------------------------------------------------------------
    def _forward(items: list[tuple[int, int, np.ndarray]]) -> np.ndarray:
        """Run MERT on *items*; return shape ``(len(items), hidden_dim)``."""
        arrays = [arr for _, _, arr in items]
        inputs = processor(
            arrays, sampling_rate=target_sr, return_tensors="pt", padding=True,
        )
        inputs = {k: v.to(_clap_device) for k, v in inputs.items()}
        if _clap_embed_dtype is not None and _clap_embed_dtype != torch.float32:
            inputs = {
                k: v.to(_clap_embed_dtype) if v.is_floating_point() else v
                for k, v in inputs.items()
            }
        with torch.no_grad():
            out = model(**inputs)
        return out.last_hidden_state.mean(dim=1).cpu().float().numpy()

    # ------------------------------------------------------------------
    # Run inference: iterate each bucket in mini-batches
    # ------------------------------------------------------------------
    results: list[tuple[int, int, np.ndarray]] = []

    for queue in (short_q, medium_q, long_q):
        for i in range(0, len(queue), batch_size):
            batch = queue[i : i + batch_size]
            embs = _forward(batch)               # (batch, hidden_dim)
            for j, (si, cl, _) in enumerate(batch):
                results.append((si, cl, embs[j]))

    # ------------------------------------------------------------------
    # Reassemble per-song embeddings via length-weighted average
    # ------------------------------------------------------------------
    song_chunks: list[list[tuple[int, np.ndarray]]] = [[] for _ in resampled]
    for si, cl, emb in results:
        song_chunks[si].append((cl, emb))

    embeddings: list[np.ndarray] = []
    for per_song in song_chunks:
        if not per_song:
            # Fallback: return a zero vector of standard MERT hidden_dim
            embeddings.append(np.zeros(768, dtype=np.float32))
            continue
        lengths = np.array([cl for cl, _ in per_song], dtype=np.float64)
        weights = lengths / lengths.sum()
        embs = [e for _, e in per_song]
        embeddings.append(np.average(embs, axis=0, weights=weights))

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

        # --- 3a. Classify any long pre-beat intro -------------------------
        intro_is_rhythmic = _detect_intro_type(y, sr, beat_times, bpm)

        # --- 4. Key estimation (Essentia) ---------------------------------
        key = _estimate_key(y, sr)

        # --- 5. Audio fingerprint (Chromaprint) ---------------------------
        fingerprint = _compute_fingerprint(y, sr)

        # --- 6. CLAP embedding (phrase-aware) -----------------------------
        embedding = cls._compute_embedding(y, sr, beat_times, bpm, intro_is_rhythmic)

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
    def _compute_embedding(
        y: np.ndarray,
        sr: int,
        beat_times: list[float] | None = None,
        bpm: float = 0.0,
        intro_is_rhythmic: bool = True,
    ) -> np.ndarray:
        """Generate a fixed-size semantic embedding using MERT with
        phrase-aware chunking.

        MERT expects audio at 24 kHz; resamples if needed.
        Returns a 1-D numpy array (hidden_dim dimensions, typically 768).
        """
        import torch

        model, processor = _get_clap_model()
        target_sr = 24_000
        if sr != target_sr:
            y = _resample(y, orig_sr=sr, target_sr=target_sr)

        chunks = _phrase_chunks(
            y, target_sr,
            beat_times if beat_times is not None else [],
            bpm,
            intro_is_rhythmic,
        )

        chunk_embs: list[np.ndarray] = []
        chunk_lengths: list[int] = []
        for chunk in chunks:
            inputs = processor(chunk, sampling_rate=target_sr, return_tensors="pt")
            inputs = {k: v.to(_clap_device) for k, v in inputs.items()}
            if _clap_embed_dtype is not None and _clap_embed_dtype != torch.float32:
                inputs = {
                    k: v.to(_clap_embed_dtype) if v.is_floating_point() else v
                    for k, v in inputs.items()
                }
            with torch.no_grad():
                out = model(**inputs)
            chunk_embs.append(
                out.last_hidden_state.mean(dim=1).squeeze().cpu().float().numpy()
            )
            chunk_lengths.append(len(chunk))

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
