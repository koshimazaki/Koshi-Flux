"""
Audio Loading and Encoding Nodes for ComfyUI

These nodes handle:
- Loading audio files (mp3, wav, flac, etc.)
- Audio preprocessing and resampling
- Audio encoding to latent embeddings
- Audio feature visualization
"""

import os
import torch
import numpy as np
from typing import Tuple, Dict, Any, Optional

try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False

try:
    from ltx_audio_injection import AudioEncoder, AudioEncoderConfig
    from ltx_audio_injection.utils import load_audio, compute_audio_energy
    from ltx_audio_injection.models.audio_parameter_mapper import AudioFeatureExtractor
    LTX_AUDIO_AVAILABLE = True
except ImportError:
    LTX_AUDIO_AVAILABLE = False

import folder_paths


class LoadAudio:
    """
    Load audio file from disk.

    Supports: mp3, wav, flac, ogg, m4a
    Outputs: waveform tensor and sample rate
    """

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        # Scan input directory for audio files
        audio_extensions = (".mp3", ".wav", ".flac", ".ogg", ".m4a")
        files = []
        if os.path.exists(input_dir):
            for f in os.listdir(input_dir):
                if f.lower().endswith(audio_extensions):
                    files.append(f)

        return {
            "required": {
                "audio_file": (sorted(files) if files else ["none"], {"default": "none"}),
            },
            "optional": {
                "target_sample_rate": ("INT", {
                    "default": 16000,
                    "min": 8000,
                    "max": 48000,
                    "step": 1000,
                }),
                "mono": ("BOOLEAN", {"default": True}),
                "normalize": ("BOOLEAN", {"default": True}),
                "max_duration": ("FLOAT", {
                    "default": 0,
                    "min": 0,
                    "max": 600,
                    "step": 1,
                    "tooltip": "Max duration in seconds (0 = no limit)",
                }),
            },
        }

    RETURN_TYPES = ("AUDIO", "INT", "FLOAT")
    RETURN_NAMES = ("audio", "sample_rate", "duration")
    FUNCTION = "load_audio"
    CATEGORY = "LTX-Audio/Load"

    def load_audio(
        self,
        audio_file: str,
        target_sample_rate: int = 16000,
        mono: bool = True,
        normalize: bool = True,
        max_duration: float = 0,
    ) -> Tuple[torch.Tensor, int, float]:
        if audio_file == "none":
            raise ValueError("No audio file selected")

        audio_path = folder_paths.get_annotated_filepath(audio_file)

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if not TORCHAUDIO_AVAILABLE:
            raise ImportError("torchaudio is required. Install with: pip install torchaudio")

        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)

        # Resample if needed
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
            waveform = resampler(waveform)
            sample_rate = target_sample_rate

        # Convert to mono
        if mono and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Truncate if needed
        if max_duration > 0:
            max_samples = int(max_duration * sample_rate)
            if waveform.shape[-1] > max_samples:
                waveform = waveform[..., :max_samples]

        # Normalize
        if normalize:
            waveform = waveform / (waveform.abs().max() + 1e-8)

        duration = waveform.shape[-1] / sample_rate

        return (waveform, sample_rate, duration)


class AudioEncoderNode:
    """
    Encode audio to latent embeddings for LTX-Video conditioning.

    Supports multiple encoder backends:
    - spectrogram: Fast, no external dependencies
    - clap: Semantic audio understanding (requires transformers)
    - wav2vec2: Speech-optimized (requires transformers)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "sample_rate": ("INT", {"default": 16000, "forceInput": True}),
                "num_frames": ("INT", {
                    "default": 121,
                    "min": 1,
                    "max": 1000,
                    "step": 1,
                }),
                "encoder_type": (["spectrogram", "clap", "wav2vec2"], {
                    "default": "spectrogram",
                }),
            },
            "optional": {
                "hidden_dim": ("INT", {
                    "default": 2048,
                    "min": 256,
                    "max": 4096,
                    "step": 256,
                }),
                "use_beat_features": ("BOOLEAN", {"default": True}),
                "audio_context_frames": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 16,
                }),
            },
        }

    RETURN_TYPES = ("AUDIO_EMBEDS", "AUDIO_FEATURES")
    RETURN_NAMES = ("audio_embeddings", "features")
    FUNCTION = "encode"
    CATEGORY = "LTX-Audio/Encode"

    def encode(
        self,
        audio: torch.Tensor,
        sample_rate: int,
        num_frames: int,
        encoder_type: str = "spectrogram",
        hidden_dim: int = 2048,
        use_beat_features: bool = True,
        audio_context_frames: int = 4,
    ) -> Tuple[torch.Tensor, Dict]:
        if not LTX_AUDIO_AVAILABLE:
            raise ImportError(
                "ltx_audio_injection is required. "
                "Install from the Deforum2026 repository."
            )

        # Create encoder config
        config = AudioEncoderConfig(
            encoder_type=encoder_type,
            hidden_dim=hidden_dim,
            use_beat_features=use_beat_features,
            audio_context_frames=audio_context_frames,
            sample_rate=sample_rate,
        )

        # Create encoder
        encoder = AudioEncoder(config)

        # Encode
        result = encoder(
            audio,
            num_video_frames=num_frames,
            sample_rate=sample_rate,
            return_features=True,
        )

        if isinstance(result, tuple):
            embeddings, features = result
        else:
            embeddings = result
            features = {}

        return (embeddings, features)


class AudioPreviewNode:
    """
    Preview audio waveform and features.

    Generates visualization of:
    - Waveform
    - Energy envelope
    - Spectrogram (optional)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "sample_rate": ("INT", {"default": 16000, "forceInput": True}),
            },
            "optional": {
                "width": ("INT", {"default": 800, "min": 200, "max": 2000}),
                "height": ("INT", {"default": 200, "min": 100, "max": 500}),
                "show_spectrogram": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("preview",)
    FUNCTION = "preview"
    CATEGORY = "LTX-Audio/Load"
    OUTPUT_NODE = True

    def preview(
        self,
        audio: torch.Tensor,
        sample_rate: int,
        width: int = 800,
        height: int = 200,
        show_spectrogram: bool = False,
    ) -> Tuple[torch.Tensor]:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from io import BytesIO
        from PIL import Image

        # Ensure mono
        if audio.dim() == 2:
            waveform = audio.mean(dim=0).cpu().numpy()
        else:
            waveform = audio.cpu().numpy()

        fig_height = height * 2 if show_spectrogram else height
        fig, axes = plt.subplots(
            2 if show_spectrogram else 1, 1,
            figsize=(width / 100, fig_height / 100),
            dpi=100,
        )

        if not show_spectrogram:
            axes = [axes]

        # Waveform plot
        time = np.arange(len(waveform)) / sample_rate
        axes[0].plot(time, waveform, color='#00D4AA', linewidth=0.5)
        axes[0].set_xlim(0, time[-1])
        axes[0].set_ylim(-1, 1)
        axes[0].set_ylabel('Amplitude')
        axes[0].set_title(f'Audio Waveform ({len(waveform)/sample_rate:.2f}s)')
        axes[0].set_facecolor('#1a1a2e')
        axes[0].grid(True, alpha=0.3)

        # Spectrogram (optional)
        if show_spectrogram:
            axes[1].specgram(waveform, Fs=sample_rate, cmap='magma')
            axes[1].set_ylabel('Frequency (Hz)')
            axes[1].set_xlabel('Time (s)')
            axes[1].set_title('Spectrogram')

        plt.tight_layout()

        # Convert to image tensor
        buf = BytesIO()
        plt.savefig(buf, format='png', facecolor='#1a1a2e', edgecolor='none')
        buf.seek(0)
        plt.close()

        img = Image.open(buf).convert('RGB')
        img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

        return (img_tensor,)


class ExtractAudioFeatures:
    """
    Extract detailed audio features for parameter mapping.

    Features extracted:
    - Energy/Loudness
    - Beats and onsets
    - Spectral centroid, bandwidth, flux
    - Frequency bands (bass, mid, high)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "sample_rate": ("INT", {"default": 16000, "forceInput": True}),
                "num_frames": ("INT", {
                    "default": 121,
                    "min": 1,
                    "max": 1000,
                }),
            },
            "optional": {
                "fps": ("FLOAT", {
                    "default": 24.0,
                    "min": 1.0,
                    "max": 120.0,
                }),
            },
        }

    RETURN_TYPES = (
        "FEATURE_DICT",
        "TENSOR", "TENSOR", "TENSOR",
        "TENSOR", "TENSOR", "TENSOR",
    )
    RETURN_NAMES = (
        "all_features",
        "energy", "beats", "onsets",
        "bass", "mid", "high",
    )
    FUNCTION = "extract"
    CATEGORY = "LTX-Audio/Encode"

    def extract(
        self,
        audio: torch.Tensor,
        sample_rate: int,
        num_frames: int,
        fps: float = 24.0,
    ) -> Tuple[Dict, torch.Tensor, ...]:
        if not LTX_AUDIO_AVAILABLE:
            raise ImportError("ltx_audio_injection is required.")

        extractor = AudioFeatureExtractor(sample_rate=sample_rate)
        features = extractor.extract_all(audio, num_frames, fps)

        # Import AudioFeature enum for dict access
        from ltx_audio_injection.models.audio_parameter_mapper import AudioFeature

        # Convert enum keys to strings for serialization
        feature_dict = {k.value: v for k, v in features.items()}

        # Get features using enum keys
        return (
            feature_dict,
            features.get(AudioFeature.ENERGY, torch.zeros(num_frames)),
            features.get(AudioFeature.BEAT, torch.zeros(num_frames)),
            features.get(AudioFeature.ONSET, torch.zeros(num_frames)),
            features.get(AudioFeature.BASS, torch.zeros(num_frames)),
            features.get(AudioFeature.MID, torch.zeros(num_frames)),
            features.get(AudioFeature.HIGH, torch.zeros(num_frames)),
        )
