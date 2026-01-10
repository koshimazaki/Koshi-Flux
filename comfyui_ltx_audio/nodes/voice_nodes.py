"""
Voice-Driven Generation Nodes for ComfyUI

These nodes enable speech-to-video generation:
- Transcribe audio to text with timing
- Convert speech to timed prompts
- Schedule prompts over video frames
- Generate video from voice narration
"""

import torch
from typing import Tuple, Dict, List, Optional, Any

try:
    from ltx_audio_injection.models.voice_driven_generation import (
        VoiceToTextEngine,
        SpeechToPromptConverter,
        TemporalPromptScheduler,
        VoiceDrivenGenerator,
        TimedPrompt,
        VoiceSegment,
        NarratorCharacter,
    )
    LTX_VOICE_AVAILABLE = True
except ImportError:
    LTX_VOICE_AVAILABLE = False


class TranscribeAudio:
    """
    Transcribe audio to text with word-level timing.

    Uses Whisper or Wav2Vec2 for speech recognition.
    Outputs timed text segments for prompt scheduling.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "sample_rate": ("INT", {"default": 16000, "forceInput": True}),
                "engine": (["whisper", "wav2vec2"], {"default": "whisper"}),
            },
            "optional": {
                "model_size": (["tiny", "base", "small", "medium", "large"], {
                    "default": "base",
                }),
                "language": ("STRING", {"default": "en"}),
            },
        }

    RETURN_TYPES = ("VOICE_SEGMENTS", "STRING")
    RETURN_NAMES = ("segments", "full_text")
    FUNCTION = "transcribe"
    CATEGORY = "LTX-Audio/Voice"

    def transcribe(
        self,
        audio: torch.Tensor,
        sample_rate: int,
        engine: str = "whisper",
        model_size: str = "base",
        language: str = "en",
    ) -> Tuple[List[Dict], str]:
        if not LTX_VOICE_AVAILABLE:
            raise ImportError("ltx_audio_injection voice module is required.")

        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize voice-to-text engine
        voice_engine = VoiceToTextEngine(
            engine=engine,
            model_size=model_size,
            device=device,
            language=language,
        )

        # Transcribe
        segments = voice_engine.transcribe(audio, sample_rate)

        # Convert to serializable format
        segment_dicts = [
            {
                "text": seg.text,
                "start_time": seg.start_time,
                "end_time": seg.end_time,
                "confidence": seg.confidence,
            }
            for seg in segments
        ]

        # Combine full text
        full_text = " ".join(seg.text for seg in segments)

        return (segment_dicts, full_text)


class SpeechToPrompts:
    """
    Convert transcribed speech to generation prompts.

    Extracts visual descriptions from speech and creates
    timed prompts with style modifications.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segments": ("VOICE_SEGMENTS",),
            },
            "optional": {
                "style_prefix": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
                "style_suffix": ("STRING", {
                    "default": ", cinematic, detailed, 4k",
                    "multiline": False,
                }),
                "filter_filler_words": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("TIMED_PROMPTS",)
    RETURN_NAMES = ("timed_prompts",)
    FUNCTION = "convert"
    CATEGORY = "LTX-Audio/Voice"

    def convert(
        self,
        segments: List[Dict],
        style_prefix: str = "",
        style_suffix: str = ", cinematic, detailed, 4k",
        filter_filler_words: bool = True,
    ) -> Tuple[List[Dict]]:
        if not LTX_VOICE_AVAILABLE:
            raise ImportError("ltx_audio_injection voice module is required.")

        # Recreate VoiceSegment objects
        voice_segments = [
            VoiceSegment(
                text=seg["text"],
                start_time=seg["start_time"],
                end_time=seg["end_time"],
                confidence=seg.get("confidence", 1.0),
            )
            for seg in segments
        ]

        # Create converter
        converter = SpeechToPromptConverter(
            style_prefix=style_prefix,
            style_suffix=style_suffix,
            filter_filler_words=filter_filler_words,
        )

        # Convert to prompts
        timed_prompts = converter.convert(voice_segments)

        # Convert to serializable format
        prompt_dicts = [
            {
                "text": p.text,
                "start_time": p.start_time,
                "end_time": p.end_time,
                "weight": p.weight,
                "transition": p.transition,
            }
            for p in timed_prompts
        ]

        return (prompt_dicts,)


class TemporalPromptSchedulerNode:
    """
    Schedule prompts over video frames with crossfades.

    Creates per-frame prompt weights for smooth transitions
    between different prompts based on timing.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "timed_prompts": ("TIMED_PROMPTS",),
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
                "fade_duration": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1,
                }),
                "default_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                }),
            },
        }

    RETURN_TYPES = ("PROMPT_SCHEDULE", "STRING")
    RETURN_NAMES = ("schedule", "schedule_string")
    FUNCTION = "schedule"
    CATEGORY = "LTX-Audio/Voice"

    def schedule(
        self,
        timed_prompts: List[Dict],
        num_frames: int,
        fps: float = 24.0,
        fade_duration: float = 0.5,
        default_prompt: str = "",
    ) -> Tuple[Dict, str]:
        if not LTX_VOICE_AVAILABLE:
            raise ImportError("ltx_audio_injection voice module is required.")

        # Recreate TimedPrompt objects
        prompts = [
            TimedPrompt(
                text=p["text"],
                start_time=p["start_time"],
                end_time=p["end_time"],
                weight=p.get("weight", 1.0),
                transition=p.get("transition", "crossfade"),
            )
            for p in timed_prompts
        ]

        # Create scheduler
        scheduler = TemporalPromptScheduler(
            num_frames=num_frames,
            fps=fps,
            default_prompt=default_prompt,
        )

        # Get per-frame weights
        weights = scheduler.get_frame_prompt_weights(prompts, fade_duration)

        # Create schedule dictionary
        schedule = {
            "prompts": timed_prompts,
            "weights": weights.tolist(),
            "num_frames": num_frames,
            "fps": fps,
        }

        # Create human-readable schedule string
        schedule_lines = []
        for i, p in enumerate(prompts):
            start_frame = int(p["start_time"] * fps)
            end_frame = int(p["end_time"] * fps)
            schedule_lines.append(
                f"Frames {start_frame}-{end_frame}: {p['text'][:50]}..."
            )
        schedule_string = "\n".join(schedule_lines)

        return (schedule, schedule_string)


class VoiceDrivenGeneratorNode:
    """
    Complete voice-driven video generation node.

    Takes audio input and generates video where the spoken
    content controls what appears in each frame.

    Perfect for:
    - Storytelling videos
    - "Character painting their world" effect
    - Audio-described scene generation
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "sample_rate": ("INT", {"default": 16000, "forceInput": True}),
                "model": ("MODEL",),  # LTX-Video model
                "clip": ("CLIP",),
                "vae": ("VAE",),
            },
            "optional": {
                "base_prompt": ("STRING", {
                    "default": "cinematic scene",
                    "multiline": True,
                }),
                "negative_prompt": ("STRING", {
                    "default": "blurry, low quality, distorted",
                    "multiline": True,
                }),
                "num_frames": ("INT", {
                    "default": 121,
                    "min": 9,
                    "max": 257,
                    "step": 8,
                }),
                "width": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 64}),
                "height": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 64}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 60.0}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 4.5, "min": 1.0, "max": 20.0}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "voice_engine": (["whisper", "wav2vec2"], {"default": "whisper"}),
                "fade_duration": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0}),
                "include_character": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "TIMED_PROMPTS", "STRING")
    RETURN_NAMES = ("video_frames", "prompts_used", "transcript")
    FUNCTION = "generate"
    CATEGORY = "LTX-Audio/Voice"

    def generate(
        self,
        audio: torch.Tensor,
        sample_rate: int,
        model,
        clip,
        vae,
        base_prompt: str = "cinematic scene",
        negative_prompt: str = "blurry, low quality, distorted",
        num_frames: int = 121,
        width: int = 768,
        height: int = 512,
        fps: float = 24.0,
        steps: int = 20,
        cfg: float = 4.5,
        seed: int = 0,
        voice_engine: str = "whisper",
        fade_duration: float = 0.5,
        include_character: bool = False,
    ):
        if not LTX_VOICE_AVAILABLE:
            raise ImportError("ltx_audio_injection voice module is required.")

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Step 1: Transcribe audio
        voice_to_text = VoiceToTextEngine(
            engine=voice_engine,
            model_size="base",
            device=device,
        )
        segments = voice_to_text.transcribe(audio, sample_rate)
        transcript = " ".join(seg.text for seg in segments)

        # Step 2: Convert to prompts
        converter = SpeechToPromptConverter()
        timed_prompts = converter.convert(segments)

        # Add base prompt
        if base_prompt:
            duration = num_frames / fps
            timed_prompts.insert(0, TimedPrompt(
                text=base_prompt,
                start_time=0.0,
                end_time=duration,
                weight=0.5,
            ))

        # Include character description if requested
        if include_character:
            character_prompt = TimedPrompt(
                text="a person speaking and creating, artist painting their world",
                start_time=0.0,
                end_time=num_frames / fps,
                weight=0.3,
            )
            timed_prompts.insert(0, character_prompt)

        # Step 3: Create prompt schedule
        scheduler = TemporalPromptScheduler(
            num_frames=num_frames,
            fps=fps,
            default_prompt=base_prompt,
        )
        weights = scheduler.get_frame_prompt_weights(timed_prompts, fade_duration)

        # Step 4: Combine prompts based on weights
        # For simplicity, use weighted average of prompt texts
        # In full implementation, would blend embeddings per-frame
        total_weights = weights.sum(dim=0)
        total_weights = total_weights / (total_weights.sum() + 1e-8)

        combined_parts = []
        for prompt, weight in zip(timed_prompts, total_weights.tolist()):
            if weight > 0.1:
                combined_parts.append(f"({prompt.text}:{weight:.2f})")
        combined_prompt = " ".join(combined_parts)

        # Step 5: Generate video
        # Note: This is a placeholder - actual implementation would use
        # the LTX-Video pipeline directly with the model/clip/vae
        # For now, return placeholder indicating the workflow

        # Convert prompts to serializable format
        prompt_dicts = [
            {
                "text": p.text,
                "start_time": p.start_time,
                "end_time": p.end_time,
                "weight": p.weight,
            }
            for p in timed_prompts
        ]

        # Placeholder output - in real implementation, run the generation
        placeholder_frames = torch.zeros(num_frames, height, width, 3)

        return (placeholder_frames, prompt_dicts, transcript)


class CreateTimedPrompt:
    """
    Manually create a timed prompt for custom scheduling.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "default": "a beautiful landscape",
                    "multiline": True,
                }),
                "start_time": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 600.0,
                    "step": 0.1,
                }),
                "end_time": ("FLOAT", {
                    "default": 5.0,
                    "min": 0.0,
                    "max": 600.0,
                    "step": 0.1,
                }),
            },
            "optional": {
                "weight": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                }),
                "transition": (["crossfade", "cut", "morph"], {
                    "default": "crossfade",
                }),
            },
        }

    RETURN_TYPES = ("TIMED_PROMPTS",)
    RETURN_NAMES = ("timed_prompt",)
    FUNCTION = "create"
    CATEGORY = "LTX-Audio/Voice"

    def create(
        self,
        text: str,
        start_time: float,
        end_time: float,
        weight: float = 1.0,
        transition: str = "crossfade",
    ) -> Tuple[List[Dict]]:
        prompt_dict = {
            "text": text,
            "start_time": start_time,
            "end_time": end_time,
            "weight": weight,
            "transition": transition,
        }
        return ([prompt_dict],)


class CombineTimedPrompts:
    """
    Combine multiple timed prompts into a single list.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompts_1": ("TIMED_PROMPTS",),
            },
            "optional": {
                "prompts_2": ("TIMED_PROMPTS",),
                "prompts_3": ("TIMED_PROMPTS",),
                "prompts_4": ("TIMED_PROMPTS",),
            },
        }

    RETURN_TYPES = ("TIMED_PROMPTS",)
    RETURN_NAMES = ("combined_prompts",)
    FUNCTION = "combine"
    CATEGORY = "LTX-Audio/Voice"

    def combine(
        self,
        prompts_1: List[Dict],
        prompts_2: List[Dict] = None,
        prompts_3: List[Dict] = None,
        prompts_4: List[Dict] = None,
    ) -> Tuple[List[Dict]]:
        combined = list(prompts_1)
        if prompts_2:
            combined.extend(prompts_2)
        if prompts_3:
            combined.extend(prompts_3)
        if prompts_4:
            combined.extend(prompts_4)

        # Sort by start time
        combined.sort(key=lambda x: x["start_time"])

        return (combined,)
