# LTX-Video Audio Injection

Deep audio integration for LTX-Video diffusion transformer, enabling audio-reactive video generation through cross-attention mechanisms.

## Overview

This module provides comprehensive audio conditioning for LTX-Video by injecting audio latents directly into the transformer's attention layers. This enables:

- **Audio-reactive video generation**: Videos that pulse, move, and transform in sync with audio
- **Beat-synchronized effects**: Visual changes aligned with musical beats and onsets
- **Semantic audio conditioning**: Using CLAP embeddings for audio-to-visual mapping
- **Flexible injection modes**: Cross-attention, additive, or gated fusion

## Architecture

### Injection Points

The audio conditioning is injected at multiple levels in the LTX-Video transformer:

1. **Cross-Attention Layer (attn3)**: New attention layer specifically for audio-video cross-attention
2. **Adaptive Normalization**: Audio can modulate the timestep conditioning
3. **Gated Fusion**: Audio features can gate video features for dynamic blending

```
┌─────────────────────────────────────────────────────────────┐
│                  Audio-Conditioned Transformer               │
├─────────────────────────────────────────────────────────────┤
│  Input Video Latents                                         │
│         │                                                    │
│         ▼                                                    │
│  ┌─────────────┐                                            │
│  │ Self-Attn   │◄── Video self-attention                    │
│  │  (attn1)    │                                            │
│  └──────┬──────┘                                            │
│         │                                                    │
│         ▼                                                    │
│  ┌─────────────┐     ┌──────────────┐                       │
│  │ Text Cross  │◄────│ Text Embeds  │                       │
│  │   (attn2)   │     │    (T5)      │                       │
│  └──────┬──────┘     └──────────────┘                       │
│         │                                                    │
│         ▼                                                    │
│  ┌─────────────┐     ┌──────────────┐                       │
│  │Audio Cross  │◄────│Audio Encoder │◄── Audio Input        │
│  │  (attn3)   │     │(Mel/CLAP/W2V)│                       │
│  └──────┬──────┘     └──────────────┘                       │
│         │                                                    │
│         ▼                                                    │
│  ┌─────────────┐                                            │
│  │ Feed-Forward│                                            │
│  └──────┬──────┘                                            │
│         │                                                    │
│         ▼                                                    │
│  Output Video Latents                                        │
└─────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/Deforum2026.git
cd Deforum2026

# Install dependencies
pip install torch torchaudio einops diffusers transformers

# Install LTX-Video
pip install git+https://github.com/Lightricks/LTX-Video.git

# Install this module
pip install -e ltx_audio_injection
```

## Quick Start

### Basic Usage

```python
from ltx_audio_injection import (
    AudioEncoder,
    AudioEncoderConfig,
    LTXAudioVideoPipeline,
)

# Configure audio encoder
audio_config = AudioEncoderConfig(
    encoder_type="spectrogram",  # or "clap", "wav2vec2"
    use_beat_features=True,
    frames_per_second=24.0,
)

# Load pipeline from pretrained LTX-Video
pipeline = LTXAudioVideoPipeline.from_pretrained_ltx(
    "Lightricks/LTX-Video",
    audio_encoder_config=audio_config,
    audio_injection_mode="cross_attention",
    audio_scale=1.0,
)

# Generate audio-reactive video
output = pipeline(
    height=512,
    width=768,
    num_frames=121,
    frame_rate=24.0,
    prompt="A cosmic nebula pulsing with energy",
    audio="path/to/music.mp3",
    audio_scale=1.0,
    num_inference_steps=20,
)
```

### Standalone Audio Encoding

```python
from ltx_audio_injection import AudioEncoder, AudioEncoderConfig
from ltx_audio_injection.utils import load_audio

# Load audio
waveform, sr = load_audio("path/to/audio.mp3", target_sample_rate=16000)

# Create encoder
config = AudioEncoderConfig(encoder_type="spectrogram")
encoder = AudioEncoder(config)

# Encode audio for 121 video frames
audio_embeddings = encoder(waveform, num_video_frames=121)
print(f"Audio embeddings: {audio_embeddings.shape}")
# Output: Audio embeddings: torch.Size([1, 30, 2048])
```

## Configuration

### AudioEncoderConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `encoder_type` | `"spectrogram"` | Encoder backend: `"spectrogram"`, `"clap"`, `"wav2vec2"` |
| `sample_rate` | `16000` | Audio sample rate |
| `hidden_dim` | `2048` | Hidden dimension (matches LTX transformer) |
| `use_beat_features` | `True` | Enable beat/onset detection |
| `frames_per_second` | `24.0` | Video FPS for temporal alignment |
| `audio_context_frames` | `4` | Video frames per audio token |

### Audio Injection Modes

| Mode | Description |
|------|-------------|
| `"cross_attention"` | Dedicated cross-attention layer for audio (recommended) |
| `"add"` | Additive injection after self-attention |
| `"gate"` | Gated fusion based on audio energy |
| `"concat"` | Concatenate audio to text embeddings |

## API Reference

### AudioEncoder

```python
class AudioEncoder(nn.Module):
    def __init__(self, config: AudioEncoderConfig = None):
        """Initialize audio encoder with given configuration."""

    def forward(
        self,
        audio: Union[torch.Tensor, str],
        num_video_frames: int,
        sample_rate: Optional[int] = None,
        return_features: bool = False,
    ) -> torch.Tensor:
        """
        Encode audio to latent embeddings.

        Args:
            audio: Waveform tensor or path to audio file
            num_video_frames: Number of video frames to align with
            sample_rate: Sample rate of input audio
            return_features: Return intermediate features

        Returns:
            audio_embeddings: (batch, num_tokens, hidden_dim)
        """
```

### AudioConditionedTransformer3D

```python
class AudioConditionedTransformer3D(ModelMixin, ConfigMixin):
    def forward(
        self,
        hidden_states: torch.Tensor,
        indices_grid: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        audio_hidden_states: Optional[torch.Tensor] = None,  # NEW
        audio_attention_mask: Optional[torch.Tensor] = None,  # NEW
        timestep: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> AudioConditionedTransformer3DOutput:
        """Forward pass with audio conditioning."""
```

### LTXAudioVideoPipeline

```python
class LTXAudioVideoPipeline(DiffusionPipeline):
    def __call__(
        self,
        # Video parameters
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float = 24.0,
        # Text conditioning
        prompt: str = None,
        negative_prompt: str = "",
        # Audio conditioning
        audio: Union[torch.Tensor, str] = None,  # NEW
        audio_scale: float = 1.0,  # NEW
        # Generation parameters
        num_inference_steps: int = 20,
        guidance_scale: float = 4.5,
        **kwargs,
    ) -> AudioVideoOutput:
        """Generate audio-conditioned video."""
```

## Examples

### Beat-Synchronized Generation

```python
from ltx_audio_injection.utils import extract_beat_times, create_audio_mask_from_beats

# Extract beats
beats = extract_beat_times(waveform, sample_rate=16000)
print(f"Found {len(beats)} beats")

# Create beat mask for frame-level conditioning
beat_mask = create_audio_mask_from_beats(
    beats,
    num_frames=121,
    video_fps=24.0,
    beat_window=0.1,
)
```

### Custom Injection Layers

```python
# Only inject audio in deeper layers (more semantic influence)
pipeline = LTXAudioVideoPipeline.from_pretrained_ltx(
    "Lightricks/LTX-Video",
    audio_injection_layers=[8, 9, 10, 11, 12, 13, 14, 15],  # Last 8 layers
    audio_injection_mode="cross_attention",
)
```

### Using CLAP for Semantic Audio

```python
audio_config = AudioEncoderConfig(
    encoder_type="clap",
    clap_model_path="laion/clap-htsat-fused",
)

# CLAP provides semantic audio embeddings that capture
# high-level audio concepts for better audio-visual alignment
```

## Technical Details

### Temporal Alignment

Audio features are temporally aligned with video frames using adaptive pooling:

1. Audio is processed at the encoder's native frame rate
2. Features are pooled to match `num_frames / audio_context_frames` tokens
3. Each audio token spans multiple video frames for temporal consistency

### Memory Efficiency

- Gradient checkpointing supported for training
- Audio encoder can run on CPU with embeddings transferred to GPU
- Mixed precision (fp16/bf16) compatible

### Integration with LTX-Video

The module is designed to be a drop-in enhancement:

1. Load pretrained LTX-Video weights
2. Upgrade transformer blocks with audio cross-attention
3. New audio-specific layers are randomly initialized
4. Original video generation quality is preserved

## License

MIT License - See LICENSE file for details.

## Citation

```bibtex
@software{ltx_audio_injection,
  title = {LTX-Video Audio Injection},
  year = {2024},
  url = {https://github.com/your-repo/Deforum2026}
}
```
