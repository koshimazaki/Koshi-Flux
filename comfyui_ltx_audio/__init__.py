"""
ComfyUI Custom Nodes for LTX-Video Audio Injection

This package provides ComfyUI nodes for audio-reactive video generation
using the LTX-Video Audio Injection module.

Node Categories:
- LTX-Audio/Load: Audio loading and preprocessing
- LTX-Audio/Encode: Audio encoding to embeddings
- LTX-Audio/Voice: Voice-to-prompt conversion
- LTX-Audio/Music: Music-driven parameter mapping
- LTX-Audio/Generate: Audio-conditioned generation
"""

import os
import sys
import importlib

# Add parent directory to path for ltx_audio_injection import
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import node classes
from .nodes.audio_nodes import (
    LoadAudio,
    AudioEncoderNode,
    AudioPreviewNode,
    ExtractAudioFeatures,
)
from .nodes.voice_nodes import (
    TranscribeAudio,
    SpeechToPrompts,
    TemporalPromptSchedulerNode,
    VoiceDrivenGeneratorNode,
    CreateTimedPrompt,
    CombineTimedPrompts,
)
from .nodes.music_nodes import (
    AudioParameterMapper,
    AudioReactivePresetNode,
    BeatDetectorNode,
    AudioToDeforumSchedule,
)
from .nodes.integration_nodes import (
    LTXAudioConditioner,
    LTXAudioAdapterLoader,
    LTXAudioLoRALoader,
    LTXAudioControlNetLoader,
    ApplyAudioAdapter,
    ApplyAudioControlNet,
    CombineAudioVideo,
    AudioFeaturesToConditioning,
)

# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    # Audio Loading & Processing
    "LoadAudio": LoadAudio,
    "AudioEncoder": AudioEncoderNode,
    "AudioPreview": AudioPreviewNode,
    "ExtractAudioFeatures": ExtractAudioFeatures,

    # Voice-Driven Generation
    "TranscribeAudio": TranscribeAudio,
    "SpeechToPrompts": SpeechToPrompts,
    "TemporalPromptScheduler": TemporalPromptSchedulerNode,
    "VoiceDrivenGenerator": VoiceDrivenGeneratorNode,
    "CreateTimedPrompt": CreateTimedPrompt,
    "CombineTimedPrompts": CombineTimedPrompts,

    # Music Parameter Mapping
    "AudioParameterMapper": AudioParameterMapper,
    "AudioReactivePreset": AudioReactivePresetNode,
    "BeatDetector": BeatDetectorNode,
    "AudioToDeforumSchedule": AudioToDeforumSchedule,

    # LTX Integration
    "LTXAudioConditioner": LTXAudioConditioner,
    "LTXAudioAdapterLoader": LTXAudioAdapterLoader,
    "LTXAudioLoRALoader": LTXAudioLoRALoader,
    "LTXAudioControlNetLoader": LTXAudioControlNetLoader,
    "ApplyAudioAdapter": ApplyAudioAdapter,
    "ApplyAudioControlNet": ApplyAudioControlNet,
    "CombineAudioVideo": CombineAudioVideo,
    "AudioFeaturesToConditioning": AudioFeaturesToConditioning,
}

# Display names for nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    # Audio Loading & Processing
    "LoadAudio": "🎵 Load Audio",
    "AudioEncoder": "🎧 Audio Encoder",
    "AudioPreview": "👁️ Audio Preview",
    "ExtractAudioFeatures": "📊 Extract Audio Features",

    # Voice-Driven Generation
    "TranscribeAudio": "🎤 Transcribe Audio",
    "SpeechToPrompts": "💬 Speech to Prompts",
    "TemporalPromptScheduler": "📅 Temporal Prompt Scheduler",
    "VoiceDrivenGenerator": "🗣️ Voice-Driven Generator",
    "CreateTimedPrompt": "⏱️ Create Timed Prompt",
    "CombineTimedPrompts": "🔗 Combine Timed Prompts",

    # Music Parameter Mapping
    "AudioParameterMapper": "🎹 Audio Parameter Mapper",
    "AudioReactivePreset": "🎛️ Audio Reactive Preset",
    "BeatDetector": "🥁 Beat Detector",
    "AudioToDeforumSchedule": "📈 Audio to Deforum Schedule",

    # LTX Integration
    "LTXAudioConditioner": "🎬 LTX Audio Conditioner",
    "LTXAudioAdapterLoader": "🔌 LTX Audio Adapter Loader",
    "LTXAudioLoRALoader": "🎚️ LTX Audio LoRA Loader",
    "LTXAudioControlNetLoader": "🕹️ LTX Audio ControlNet Loader",
    "ApplyAudioAdapter": "🔊 Apply Audio Adapter",
    "ApplyAudioControlNet": "🎮 Apply Audio ControlNet",
    "CombineAudioVideo": "🎞️ Combine Audio + Video",
    "AudioFeaturesToConditioning": "📉 Audio Features to Conditioning",
}

# Web directory for custom JavaScript
WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
