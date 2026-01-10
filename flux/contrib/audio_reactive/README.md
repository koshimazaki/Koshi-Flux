# Audio Reactive Animation Module

Generate audio-driven animation schedules for Deforum and Parseq.

## Installation

```bash
# Core (no audio extraction, can use pre-extracted features)
pip install numpy

# Full (with audio extraction)
pip install numpy librosa audioread soundfile
```

## Quick Start

```python
from audio_reactive import generate_schedule

# One-liner: audio file → Parseq schedule
schedule = generate_schedule("music.mp3", mapping="bass_pulse")
schedule.save("schedule.json")
```

## Architecture

```
audio_reactive/
├── __init__.py      # Public API exports
├── types.py         # Core enums: FeatureType, ParameterType, CurveType
├── features.py      # AudioFeatures dataclass
├── extractor.py     # AudioFeatureExtractor (librosa-based)
├── curves.py        # Easing curves and smoothing
├── mapping.py       # FeatureMapping, MappingConfig
├── presets.py       # 7 built-in presets
├── schedule.py      # ParseqKeyframe, ParseqSchedule
├── generator.py     # ScheduleGenerator
├── api.py           # Convenience functions
└── cli.py           # Command-line interface
```

## Presets

| Preset | Description |
|--------|-------------|
| `ambient` | Subtle, flowing movement |
| `bass_pulse` | Zoom pulses on bass hits |
| `beat_rotation` | Rotation synced to beats |
| `spectrum` | Bass/mid/high → different axes |
| `immersive_3d` | Full 3D rotation + depth |
| `cinematic` | Slow, smooth movements |
| `intense` | Aggressive high-energy |

## API Reference

### High-Level Functions

```python
from audio_reactive import extract_features, generate_schedule, create_mapping

# Extract features (requires librosa)
features = extract_features("music.mp3", fps=24, duration=60)

# Generate schedule
schedule = generate_schedule(features, mapping="bass_pulse")
schedule = generate_schedule("music.mp3", mapping="spectrum")

# Custom mapping
config = create_mapping(
    "My Style",
    bass_to_zoom=(1.0, 1.3),
    beat_strength_to_angle=(-10, 10, "ease_out"),
)
schedule = generate_schedule(features, mapping=config)
```

### Classes

```python
from audio_reactive import (
    AudioFeatures,        # Feature container
    AudioFeatureExtractor, # Librosa-based extraction
    MappingConfig,        # Mapping configuration
    FeatureMapping,       # Single feature→param mapping
    ScheduleGenerator,    # Schedule generation
    ParseqSchedule,       # Output schedule
)
```

### Presets

```python
from audio_reactive import PRESETS, get_preset, list_presets

# List available presets
print(list_presets())  # ['ambient', 'bass_pulse', ...]

# Get a preset
config = get_preset("bass_pulse")
```

## CLI Usage

```bash
# Generate schedule
python -m audio_reactive music.mp3 -o schedule.json

# Different preset
python -m audio_reactive music.mp3 -m spectrum -o schedule.json

# Custom FPS/duration
python -m audio_reactive music.mp3 --fps 30 --duration 120 -o out.json

# Show Deforum strings
python -m audio_reactive music.mp3 --format deforum

# List presets
python -m audio_reactive --list-presets
```

## Output Formats

### Parseq JSON
```json
{
  "meta": {"name": "...", "fps": 24, "bpm": 120},
  "keyframes": [
    {"frame": 0, "zoom": 1.0, "angle": 0},
    {"frame": 12, "zoom": 1.05, "angle": 2.5, "info": "beat"},
    ...
  ]
}
```

### Deforum Strings
```python
strings = schedule.to_deforum_strings()
# {"zoom": "0:(1.0), 12:(1.05), ...", "angle": "0:(0), 12:(2.5), ..."}
```

## Running Tests

```bash
cd flux/contrib
python -m pytest audio_reactive/tests -v
```
