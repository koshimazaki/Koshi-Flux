# ComfyUI LTX-Audio Nodes

Custom ComfyUI nodes for audio-reactive video generation with LTX-Video. This package enables voice-driven storytelling, music-synced visuals, and deep audio-video integration.

## Features

- **Voice-Driven Generation**: Transcribe speech and generate videos where spoken content controls visuals
- **Music Parameter Mapping**: Map audio features (beats, energy, frequency bands) to generation parameters
- **Multiple Integration Methods**: Audio Adapter, Audio LoRA, Audio ControlNet
- **Deforum Integration**: Export audio-reactive schedules for Deforum workflows
- **Real-time Audio Analysis**: Beat detection, onset detection, spectral analysis

## Installation

### Method 1: ComfyUI Manager (Recommended)

1. Open ComfyUI Manager
2. Search for "LTX Audio"
3. Click Install

### Method 2: Manual Installation

```bash
# Navigate to ComfyUI custom nodes directory
cd ComfyUI/custom_nodes

# Clone the repository
git clone https://github.com/your-repo/Deforum2026.git

# Create symlink to the ComfyUI nodes
ln -s Deforum2026/comfyui_ltx_audio comfyui_ltx_audio

# Install dependencies
pip install -r Deforum2026/comfyui_ltx_audio/requirements.txt
```

### Method 3: Direct Copy

1. Copy the `comfyui_ltx_audio` folder to `ComfyUI/custom_nodes/`
2. Copy the `ltx_audio_injection` folder to `ComfyUI/custom_nodes/comfyui_ltx_audio/` or ensure it's in your Python path
3. Install dependencies

## Requirements

```txt
torch>=2.0.0
torchaudio>=2.0.0
numpy>=1.21.0
einops>=0.6.0
transformers>=4.30.0  # For CLAP/Whisper
librosa>=0.10.0       # For audio analysis
matplotlib>=3.5.0     # For audio preview
```

### Optional Dependencies

```txt
openai-whisper>=20230314  # For speech transcription
laion-clap>=1.0.0         # For semantic audio encoding
```

## Node Categories

### LTX-Audio/Load
Audio loading and preprocessing nodes.

| Node | Description |
|------|-------------|
| **Load Audio** | Load audio files (mp3, wav, flac, ogg, m4a) |
| **Audio Preview** | Visualize waveform and spectrogram |

### LTX-Audio/Encode
Audio encoding and feature extraction.

| Node | Description |
|------|-------------|
| **Audio Encoder** | Encode audio to latent embeddings (spectrogram/CLAP/wav2vec2) |
| **Extract Audio Features** | Extract energy, beats, onsets, frequency bands |

### LTX-Audio/Voice
Voice-driven generation nodes.

| Node | Description |
|------|-------------|
| **Transcribe Audio** | Speech-to-text with word timing (Whisper/wav2vec2) |
| **Speech to Prompts** | Convert transcribed speech to timed prompts |
| **Temporal Prompt Scheduler** | Schedule prompts over video frames with crossfades |
| **Voice-Driven Generator** | Complete voice-to-video pipeline |
| **Create Timed Prompt** | Manually create a timed prompt |
| **Combine Timed Prompts** | Combine multiple timed prompt lists |

### LTX-Audio/Music
Music-driven parameter mapping.

| Node | Description |
|------|-------------|
| **Audio Parameter Mapper** | Map audio features to generation parameters |
| **Audio Reactive Preset** | Pre-configured audio reactive behaviors |
| **Beat Detector** | Detect beats and rhythm |
| **Audio to Deforum Schedule** | Export schedules for Deforum |

### LTX-Audio/Integration
Core LTX-Video integration nodes.

| Node | Description |
|------|-------------|
| **LTX Audio Conditioner** | Prepare audio embeddings for injection |
| **LTX Audio Adapter Loader** | Load IP-Adapter style audio adapter |
| **LTX Audio LoRA Loader** | Load and inject audio-modulated LoRA |
| **LTX Audio ControlNet Loader** | Load audio ControlNet |
| **Apply Audio Adapter** | Apply adapter to embeddings |
| **Apply Audio ControlNet** | Generate control signals |
| **Combine Audio + Video** | Execute audio-conditioned generation |
| **Audio Features to Conditioning** | Convert features to conditioning |

## Example Workflows

### Voice-Driven Storytelling

Create videos where your voice narration controls what appears on screen:

```
Load Audio -> Transcribe Audio -> Speech to Prompts -> Temporal Prompt Scheduler -> Voice-Driven Generator
```

**Use Case**: "Character painting their world" - narrator describes scenes and they appear in the video.

### Music-Synced Visuals

Generate videos that react to music beats and energy:

```
Load Audio -> Extract Audio Features -> Audio Parameter Mapper -> Audio to Deforum Schedule
```

**Features mapped**:
- Bass -> Camera zoom
- Energy -> Motion intensity
- Beats -> Scene transitions
- High frequencies -> Color saturation

### Beat-Synced Prompt Changes

Change prompts on every beat:

```
Load Audio -> Beat Detector -> Temporal Prompt Scheduler (with timed prompts)
```

### Audio Adapter Integration

Use trained audio adapter for deep audio conditioning:

```
Load Audio -> Audio Encoder -> LTX Audio Adapter Loader -> Apply Audio Adapter -> [Your Generation Pipeline]
```

## Audio Reactive Presets

Built-in presets for common use cases:

| Preset | Description |
|--------|-------------|
| **cinematic** | Smooth, film-like movements |
| **energetic** | High-energy music videos |
| **ambient** | Subtle, atmospheric reactions |
| **beat_sync** | Tight synchronization to beats |
| **voice_reactive** | Optimized for speech |

## Training Custom Models

### Audio Adapter Training

```python
from ltx_audio_injection.models.audio_adapter import train_audio_adapter

train_audio_adapter(
    adapter=adapter,
    transformer=base_model,
    train_dataloader=dataloader,
    num_epochs=100,
    learning_rate=1e-4,
)
```

### Audio LoRA Training

```python
from ltx_audio_injection.models.audio_lora import inject_audio_lora, get_audio_lora_parameters

# Inject LoRA layers
lora_layers = inject_audio_lora(model, config)

# Get trainable parameters
lora_params = get_audio_lora_parameters(model)
optimizer = torch.optim.AdamW(lora_params, lr=1e-4)
```

### Audio ControlNet Training

```python
from ltx_audio_injection.models.audio_controlnet import train_audio_controlnet

train_audio_controlnet(
    controlnet=controlnet,
    transformer=base_model,  # Frozen
    train_dataloader=dataloader,
    num_epochs=100,
)
```

## API Reference

### AudioEncoder Config

```python
from ltx_audio_injection import AudioEncoderConfig

config = AudioEncoderConfig(
    encoder_type="spectrogram",  # or "clap", "wav2vec2"
    hidden_dim=2048,
    use_beat_features=True,
    audio_context_frames=4,
    sample_rate=16000,
)
```

### Audio Feature Types

| Feature | Description | Range |
|---------|-------------|-------|
| `energy` | Overall loudness | 0.0 - 1.0 |
| `beat` | Beat presence | 0.0 or 1.0 |
| `onset` | Note/sound onsets | 0.0 - 1.0 |
| `bass` | Low frequency energy | 0.0 - 1.0 |
| `mid` | Mid frequency energy | 0.0 - 1.0 |
| `high` | High frequency energy | 0.0 - 1.0 |
| `spectral_centroid` | Brightness | Normalized |
| `spectral_flux` | Rate of change | 0.0 - 1.0 |

### Parameter Mapping Curves

| Curve | Description |
|-------|-------------|
| `linear` | Direct mapping |
| `exponential` | Amplify high values |
| `logarithmic` | Compress high values |
| `sigmoid` | S-curve (smooth threshold) |
| `pulse` | Binary on/off |
| `smooth` | Gaussian smoothing |
| `inverse` | Invert the curve |

## Troubleshooting

### "ltx_audio_injection not found"

Ensure the `ltx_audio_injection` package is in your Python path:

```python
import sys
sys.path.append("/path/to/Deforum2026")
```

### "torchaudio not available"

Install torchaudio:

```bash
pip install torchaudio
```

### "CLAP model not found"

For CLAP encoding, install:

```bash
pip install laion-clap transformers
```

### "Whisper not available"

For speech transcription:

```bash
pip install openai-whisper
```

## License

MIT License - See LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Credits

- LTX-Video: Lightricks
- ComfyUI: comfyanonymous
- Audio analysis: librosa, torchaudio
- Speech recognition: OpenAI Whisper
- Semantic audio: LAION CLAP
