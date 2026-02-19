# Auto-DJ — DJ Mixing Pathfinding System

A Python system that analyses a library of audio files and models them as nodes in a weighted directed graph, enabling intelligent mix-path discovery between tracks based on BPM, musical key, and sonic similarity.

## Project Structure

```
Auto-DJ/
├── src/
│   ├── __init__.py
│   ├── models.py       # Song data model (metadata + CLAP embeddings)
│   ├── metrics.py      # Transition cost functions (harmonic, tempo, semantic)
│   ├── graph.py        # DJGraph class with Dijkstra pathfinding
│   └── utils.py        # Directory scanner with deduplication
├── tests/
│   ├── __init__.py
│   ├── test_models.py  # Song model & scanner tests
│   ├── test_metrics.py # Transition cost tests
│   └── test_graph.py   # Graph construction & pathfinding tests
├── main.py             # CLI entry point
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### CLI

```bash
python main.py ./my_music "opening_track.mp3" "closer.mp3"
```

### Python API

```python
from src.models import Song
from src.utils import scan_directory
from src.graph import DJGraph

# Scan and analyse an entire library
songs = scan_directory("path/to/music/")

# Build the mixing graph
graph = DJGraph.build(songs)

# Find the smoothest path between two tracks
path, cost = graph.get_shortest_path("track_a.mp3", "track_z.mp3")

for song in path:
    print(f"  {song.filename}  (BPM: {song.bpm}, Key: {song.key})")
print(f"Total cost: {cost:.4f}")
```

## How It Works

1. **Audio Loading** — Files are loaded via `librosa` (mono, native sample rate).
2. **BPM Detection** — Tempo is estimated with librosa's beat tracker.
3. **Key Detection** — Musical key is derived from chroma features using the Krumhansl-Schmuckler algorithm.
4. **Embedding** — Each track is passed through the `laion/clap-htsat-unfused` CLAP model to produce a 512-dimensional vector representing its sonic character.
5. **Deduplication** — The directory scanner hashes file contents (SHA-256) to skip byte-identical duplicates.
6. **Transition Cost** — Each ordered pair of songs is scored by combining harmonic distance (Circle of Fifths), tempo penalty (8% threshold), and cosine distance between embeddings.
7. **Graph Construction** — Songs become nodes in a NetworkX DiGraph; finite-cost pairs become weighted directed edges.
8. **Pathfinding** — Dijkstra's algorithm finds the lowest-cost sequence of transitions from any source track to any target track.
