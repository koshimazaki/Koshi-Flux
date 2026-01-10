"""Audio injection models for LTX-Video."""

from .audio_encoder import AudioEncoder, AudioEncoderConfig
from .audio_attention import (
    AudioCrossAttentionProcessor,
    AudioConditionedTransformerBlock,
)
from .audio_transformer import AudioConditionedTransformer3D

# Advanced integration methods
from .audio_adapter import (
    LTXAudioAdapter,
    AudioProjectionModel,
    AudioAdapterAttnProcessor,
)
from .audio_lora import (
    AudioLoRALinear,
    TemporalAudioLoRA,
    AudioLoRAConfig,
    inject_audio_lora,
    get_audio_lora_parameters,
    save_audio_lora,
    load_audio_lora,
)
from .audio_controlnet import (
    LTXAudioControlNet,
    AudioControlNetBlock,
    TemporalAudioEncoder,
)

# Voice and music-driven generation
from .voice_driven_generation import (
    VoiceDrivenGenerator,
    VoiceToTextEngine,
    SpeechToPromptConverter,
    TemporalPromptScheduler,
    TimedPrompt,
    NarratorCharacter,
)
from .audio_parameter_mapper import (
    AudioFeatureExtractor,
    ParameterScheduler,
    AudioReactiveConfig,
    ParameterMapping,
    AudioFeature,
    MappingCurve,
    AudioReactivePresets,
    MusicReactiveGenerator,
)

__all__ = [
    # Core
    "AudioEncoder",
    "AudioEncoderConfig",
    "AudioCrossAttentionProcessor",
    "AudioConditionedTransformerBlock",
    "AudioConditionedTransformer3D",
    # Adapter
    "LTXAudioAdapter",
    "AudioProjectionModel",
    "AudioAdapterAttnProcessor",
    # LoRA
    "AudioLoRALinear",
    "TemporalAudioLoRA",
    "AudioLoRAConfig",
    "inject_audio_lora",
    "get_audio_lora_parameters",
    "save_audio_lora",
    "load_audio_lora",
    # ControlNet
    "LTXAudioControlNet",
    "AudioControlNetBlock",
    "TemporalAudioEncoder",
    # Voice-driven
    "VoiceDrivenGenerator",
    "VoiceToTextEngine",
    "SpeechToPromptConverter",
    "TemporalPromptScheduler",
    "TimedPrompt",
    "NarratorCharacter",
    # Parameter mapping
    "AudioFeatureExtractor",
    "ParameterScheduler",
    "AudioReactiveConfig",
    "ParameterMapping",
    "AudioFeature",
    "MappingCurve",
    "AudioReactivePresets",
    "MusicReactiveGenerator",
]
