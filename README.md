# Auto-DJ — DJ Mixing Pathfinding System

A Python system that analyses a library of audio files and models them as nodes in a graph, enabling intelligent mix-path discovery between tracks based on BPM, musical key, and sonic similarity.

## Project Structure

```
Auto-DJ/
├── src/
│   ├── __init__.py
│   ├── models.py      # Song data model (metadata + CLAP embeddings)
│   └── utils.py       # Directory scanner with deduplication
├── tests/
│   ├── __init__.py
│   └── test_models.py  # Unit tests
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```python
from src.models import Song
from src.utils import scan_directory

# Analyse a single track
song = Song.from_file("path/to/track.mp3")
print(song.bpm, song.key, song.embedding.shape)

# Scan an entire library
songs = scan_directory("path/to/music/")
for s in songs:
    print(s)
```

## How It Works

1. **Audio Loading** — Files are loaded via `librosa` (mono, native sample rate).
2. **BPM Detection** — Tempo is estimated with librosa's beat tracker.
3. **Key Detection** — Musical key is derived from chroma features using the Krumhansl-Schmuckler algorithm.
4. **Embedding** — Each track is passed through the `laion/clap-htsat-unfused` CLAP model to produce a 512-dimensional vector representing its sonic character.
5. **Deduplication** — The directory scanner hashes file contents (SHA-256) to skip byte-identical duplicates.
