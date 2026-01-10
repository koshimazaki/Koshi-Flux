"""
Voice-Driven Video Generation for LTX-Video

This module enables dynamic, speech-controlled video generation where:
1. Voice narration drives prompt changes over time
2. Audio timing controls scene transitions
3. A "character painting their world" - describing scenes as they appear
4. Real-time speech-to-prompt with temporal alignment

Use cases:
- Storytelling: Narrator describes scenes that materialize
- Audio-reactive art: Music/speech drives visual evolution
- Interactive generation: Voice controls what appears
- Self-referential loops: Character describes what they're creating

Architecture:
    Voice → Speech Recognition → Timed Prompts → Frame-aligned Generation
           ↓
    Audio Features → Temporal Conditioning → Visual Style/Motion
"""

import math
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any, Union, Callable
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


@dataclass
class TimedPrompt:
    """A prompt with temporal bounds."""
    text: str
    start_time: float  # seconds
    end_time: float    # seconds
    weight: float = 1.0
    transition: str = "crossfade"  # "crossfade", "cut", "morph"

    def overlaps(self, time: float) -> bool:
        return self.start_time <= time < self.end_time

    def get_weight_at(self, time: float, fade_duration: float = 0.5) -> float:
        """Get interpolated weight with fade in/out."""
        if time < self.start_time or time >= self.end_time:
            return 0.0

        # Fade in
        if time < self.start_time + fade_duration:
            return self.weight * (time - self.start_time) / fade_duration

        # Fade out
        if time > self.end_time - fade_duration:
            return self.weight * (self.end_time - time) / fade_duration

        return self.weight


@dataclass
class VoiceSegment:
    """A segment of transcribed speech with timing."""
    text: str
    start_time: float
    end_time: float
    confidence: float = 1.0
    speaker_id: Optional[str] = None


class SpeechToPromptConverter:
    """
    Converts speech transcriptions to generation prompts.

    Handles:
    - Extracting visual descriptions from speech
    - Mapping spoken concepts to prompt keywords
    - Handling narrative vs. descriptive speech
    - Scene change detection from speech patterns
    """

    def __init__(
        self,
        style_prefix: str = "",
        style_suffix: str = ", cinematic, detailed, 4k",
        filter_filler_words: bool = True,
        enhance_descriptions: bool = True,
    ):
        self.style_prefix = style_prefix
        self.style_suffix = style_suffix
        self.filter_filler_words = filter_filler_words
        self.enhance_descriptions = enhance_descriptions

        # Words that indicate visual descriptions
        self.visual_keywords = {
            "see", "look", "appears", "showing", "imagine", "picture",
            "visualize", "there is", "there are", "in front", "behind",
            "above", "below", "surrounded by", "filled with", "covered in",
        }

        # Filler words to remove
        self.filler_words = {
            "um", "uh", "like", "you know", "basically", "actually",
            "literally", "so", "well", "i mean", "kind of", "sort of",
        }

        # Scene transition words
        self.transition_words = {
            "now": "crossfade",
            "suddenly": "cut",
            "then": "crossfade",
            "next": "crossfade",
            "transforms": "morph",
            "becomes": "morph",
            "changes to": "morph",
            "dissolves": "crossfade",
            "cuts to": "cut",
        }

    def convert(self, segments: List[VoiceSegment]) -> List[TimedPrompt]:
        """Convert voice segments to timed prompts."""
        prompts = []

        for segment in segments:
            text = segment.text.strip()

            # Filter filler words
            if self.filter_filler_words:
                text = self._remove_fillers(text)

            # Skip empty or too short
            if len(text.split()) < 2:
                continue

            # Extract visual description
            description = self._extract_visual(text)

            # Detect transition type
            transition = self._detect_transition(text)

            # Add style
            full_prompt = self.style_prefix + description + self.style_suffix

            prompts.append(TimedPrompt(
                text=full_prompt,
                start_time=segment.start_time,
                end_time=segment.end_time,
                weight=segment.confidence,
                transition=transition,
            ))

        return prompts

    def _remove_fillers(self, text: str) -> str:
        """Remove filler words from text."""
        text_lower = text.lower()
        for filler in self.filler_words:
            text_lower = text_lower.replace(filler, " ")
        # Clean up whitespace
        return " ".join(text_lower.split())

    def _extract_visual(self, text: str) -> str:
        """Extract visual description from speech."""
        # For narrative speech, try to extract the visual elements
        text_lower = text.lower()

        # Check for explicit visual descriptions
        for keyword in self.visual_keywords:
            if keyword in text_lower:
                # Extract the part after the keyword
                idx = text_lower.find(keyword)
                visual_part = text[idx + len(keyword):].strip()
                if visual_part:
                    return visual_part

        # Otherwise use the whole text as description
        return text

    def _detect_transition(self, text: str) -> str:
        """Detect transition type from speech."""
        text_lower = text.lower()
        for word, transition in self.transition_words.items():
            if word in text_lower:
                return transition
        return "crossfade"


class VoiceToTextEngine:
    """
    Wrapper for speech recognition engines.

    Supports multiple backends:
    - whisper: OpenAI Whisper (local)
    - whisper_api: OpenAI Whisper API
    - wav2vec2: Facebook Wav2Vec2
    """

    def __init__(
        self,
        engine: str = "whisper",
        model_size: str = "base",
        device: str = "cuda",
        language: str = "en",
    ):
        self.engine = engine
        self.device = device

        if engine == "whisper":
            self._init_whisper(model_size)
        elif engine == "wav2vec2":
            self._init_wav2vec2(model_size)
        else:
            raise ValueError(f"Unknown engine: {engine}")

    def _init_whisper(self, model_size: str):
        """Initialize Whisper model."""
        try:
            import whisper
            self.model = whisper.load_model(model_size, device=self.device)
        except ImportError:
            raise ImportError("Install whisper: pip install openai-whisper")

    def _init_wav2vec2(self, model_size: str):
        """Initialize Wav2Vec2 model."""
        try:
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
            model_name = f"facebook/wav2vec2-{model_size}-960h"
            self.processor = Wav2Vec2Processor.from_pretrained(model_name)
            self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(self.device)
        except ImportError:
            raise ImportError("Install transformers: pip install transformers")

    def transcribe(
        self,
        audio: torch.Tensor,
        sample_rate: int = 16000,
    ) -> List[VoiceSegment]:
        """
        Transcribe audio to timed segments.

        Args:
            audio: Waveform tensor (samples,) or (channels, samples)
            sample_rate: Audio sample rate

        Returns:
            List of VoiceSegment with timing information
        """
        if self.engine == "whisper":
            return self._transcribe_whisper(audio, sample_rate)
        elif self.engine == "wav2vec2":
            return self._transcribe_wav2vec2(audio, sample_rate)

    def _transcribe_whisper(
        self,
        audio: torch.Tensor,
        sample_rate: int,
    ) -> List[VoiceSegment]:
        """Transcribe using Whisper with word-level timing."""
        import whisper

        # Convert to numpy
        if audio.dim() == 2:
            audio = audio.mean(dim=0)
        audio_np = audio.cpu().numpy()

        # Transcribe with word timestamps
        result = self.model.transcribe(
            audio_np,
            word_timestamps=True,
            language="en",
        )

        segments = []
        for segment in result.get("segments", []):
            segments.append(VoiceSegment(
                text=segment["text"].strip(),
                start_time=segment["start"],
                end_time=segment["end"],
                confidence=segment.get("confidence", 1.0),
            ))

        return segments

    def _transcribe_wav2vec2(
        self,
        audio: torch.Tensor,
        sample_rate: int,
    ) -> List[VoiceSegment]:
        """Transcribe using Wav2Vec2."""
        if audio.dim() == 2:
            audio = audio.mean(dim=0)

        # Process
        inputs = self.processor(
            audio.cpu().numpy(),
            sampling_rate=sample_rate,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits

        # Decode
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]

        # Simple segmentation (Wav2Vec2 doesn't provide word timing easily)
        # Split by pauses (we'd need more sophisticated VAD for real timing)
        duration = len(audio) / sample_rate
        segments = [VoiceSegment(
            text=transcription,
            start_time=0.0,
            end_time=duration,
            confidence=1.0,
        )]

        return segments


class TemporalPromptScheduler:
    """
    Schedules prompts over time for video generation.

    Handles:
    - Blending multiple overlapping prompts
    - Smooth transitions between scenes
    - Frame-level prompt interpolation
    """

    def __init__(
        self,
        num_frames: int,
        fps: float = 24.0,
        default_prompt: str = "",
        blend_mode: str = "lerp",  # "lerp", "slerp", "attention"
    ):
        self.num_frames = num_frames
        self.fps = fps
        self.duration = num_frames / fps
        self.default_prompt = default_prompt
        self.blend_mode = blend_mode

    def schedule(
        self,
        timed_prompts: List[TimedPrompt],
        text_encoder: Callable,
        fade_duration: float = 0.5,
    ) -> torch.Tensor:
        """
        Generate per-frame prompt embeddings.

        Args:
            timed_prompts: List of timed prompts
            text_encoder: Function to encode text to embeddings
            fade_duration: Duration of crossfades in seconds

        Returns:
            embeddings: (num_frames, seq_len, hidden_dim)
        """
        # Encode all prompts
        prompt_texts = [p.text for p in timed_prompts]
        if self.default_prompt:
            prompt_texts.append(self.default_prompt)

        # Encode all unique prompts
        all_embeddings = {}
        for text in set(prompt_texts):
            all_embeddings[text] = text_encoder(text)

        # Get embedding shape from first
        first_embed = next(iter(all_embeddings.values()))
        seq_len, hidden_dim = first_embed.shape[-2:]

        # Build per-frame embeddings
        frame_embeddings = []

        for frame_idx in range(self.num_frames):
            time = frame_idx / self.fps

            # Find active prompts at this time
            active_prompts = []
            for prompt in timed_prompts:
                weight = prompt.get_weight_at(time, fade_duration)
                if weight > 0:
                    active_prompts.append((prompt, weight))

            if not active_prompts:
                # Use default prompt
                if self.default_prompt:
                    frame_embeddings.append(all_embeddings[self.default_prompt])
                else:
                    frame_embeddings.append(torch.zeros(1, seq_len, hidden_dim))
            else:
                # Blend active prompts
                total_weight = sum(w for _, w in active_prompts)
                blended = torch.zeros(1, seq_len, hidden_dim)

                for prompt, weight in active_prompts:
                    embed = all_embeddings[prompt.text]
                    blended += (weight / total_weight) * embed

                frame_embeddings.append(blended)

        return torch.cat(frame_embeddings, dim=0)  # (num_frames, seq_len, hidden)

    def get_frame_prompt_weights(
        self,
        timed_prompts: List[TimedPrompt],
        fade_duration: float = 0.5,
    ) -> torch.Tensor:
        """Get per-frame weights for each prompt."""
        weights = torch.zeros(self.num_frames, len(timed_prompts))

        for frame_idx in range(self.num_frames):
            time = frame_idx / self.fps
            for prompt_idx, prompt in enumerate(timed_prompts):
                weights[frame_idx, prompt_idx] = prompt.get_weight_at(
                    time, fade_duration
                )

        # Normalize
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        return weights


class VoiceDrivenGenerator:
    """
    Complete voice-driven video generation system.

    This is the main interface for generating videos from voice narration,
    where the spoken words control what appears in each frame.

    Example - "Character painting their world":
        audio: "I see a vast ocean... now mountains rise... the sky turns purple..."
        → Video shows ocean, then mountains emerging, then purple sky

    Usage:
        generator = VoiceDrivenGenerator(pipeline)
        video = generator.generate(
            audio="narration.mp3",
            base_prompt="fantasy landscape",
        )
    """

    def __init__(
        self,
        pipeline,  # LTXVideoPipeline or LTXAudioVideoPipeline
        voice_engine: str = "whisper",
        voice_model_size: str = "base",
        device: str = "cuda",
    ):
        self.pipeline = pipeline
        self.device = device

        # Initialize voice-to-text
        self.voice_to_text = VoiceToTextEngine(
            engine=voice_engine,
            model_size=voice_model_size,
            device=device,
        )

        # Initialize prompt converter
        self.prompt_converter = SpeechToPromptConverter()

    def generate(
        self,
        audio: Union[str, torch.Tensor],
        base_prompt: str = "",
        negative_prompt: str = "blurry, low quality",
        num_frames: int = 121,
        fps: float = 24.0,
        height: int = 512,
        width: int = 768,
        num_inference_steps: int = 20,
        guidance_scale: float = 4.5,
        audio_scale: float = 0.5,  # Also use audio features for style
        fade_duration: float = 0.5,
        seed: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate video from voice narration.

        Args:
            audio: Path to audio file or waveform tensor
            base_prompt: Base prompt to combine with speech
            negative_prompt: Negative prompt
            num_frames: Number of video frames
            fps: Frame rate
            height, width: Video dimensions
            audio_scale: How much to use audio features for style (vs just prompts)
            fade_duration: Duration of prompt crossfades
            seed: Random seed

        Returns:
            Dictionary with video frames and metadata
        """
        import torchaudio

        # Load audio
        if isinstance(audio, str):
            waveform, sample_rate = torchaudio.load(audio)
        else:
            waveform = audio
            sample_rate = 16000

        # Transcribe voice to text segments
        print("Transcribing audio...")
        segments = self.voice_to_text.transcribe(waveform, sample_rate)

        print(f"Found {len(segments)} speech segments:")
        for seg in segments:
            print(f"  [{seg.start_time:.1f}s - {seg.end_time:.1f}s]: {seg.text[:50]}...")

        # Convert to timed prompts
        timed_prompts = self.prompt_converter.convert(segments)

        # Add base prompt as background
        if base_prompt:
            duration = num_frames / fps
            timed_prompts.insert(0, TimedPrompt(
                text=base_prompt,
                start_time=0.0,
                end_time=duration,
                weight=0.5,  # Lower weight so speech prompts dominate
            ))

        print(f"Generated {len(timed_prompts)} timed prompts")

        # Create prompt scheduler
        scheduler = TemporalPromptScheduler(
            num_frames=num_frames,
            fps=fps,
            default_prompt=base_prompt,
        )

        # Get per-frame prompt weights
        prompt_weights = scheduler.get_frame_prompt_weights(
            timed_prompts, fade_duration
        )

        # Generate with temporal prompts
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # For now, use the dominant prompt per frame
        # More sophisticated: use per-frame embeddings
        output = self._generate_with_temporal_prompts(
            timed_prompts=timed_prompts,
            prompt_weights=prompt_weights,
            audio_waveform=waveform,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            fps=fps,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            audio_scale=audio_scale,
            generator=generator,
            **kwargs,
        )

        return {
            "video": output.images if hasattr(output, 'images') else output,
            "segments": segments,
            "timed_prompts": timed_prompts,
            "prompt_weights": prompt_weights,
        }

    def _generate_with_temporal_prompts(
        self,
        timed_prompts: List[TimedPrompt],
        prompt_weights: torch.Tensor,
        audio_waveform: torch.Tensor,
        **kwargs,
    ):
        """Generate video using temporal prompt scheduling."""
        # Simple approach: use weighted combination of all prompts
        # More advanced: modify the denoising loop to use per-step prompts

        # Combine prompts weighted by their total presence
        total_weights = prompt_weights.sum(dim=0)
        total_weights = total_weights / total_weights.sum()

        # Create weighted prompt
        weighted_parts = []
        for prompt, weight in zip(timed_prompts, total_weights.tolist()):
            if weight > 0.1:
                weighted_parts.append(f"({prompt.text}:{weight:.2f})")

        combined_prompt = " ".join(weighted_parts)

        # Check if pipeline supports audio
        if hasattr(self.pipeline, 'audio_encoder'):
            return self.pipeline(
                prompt=combined_prompt,
                audio=audio_waveform,
                **kwargs,
            )
        else:
            return self.pipeline(
                prompt=combined_prompt,
                **kwargs,
            )


class NarratorCharacter:
    """
    Creates a "character painting their world" effect.

    The character describes what they see, and the scene evolves
    based on their narration - a self-referential generation loop.

    Example:
        "I open my eyes and see... darkness. Then, slowly,
        stars appear... millions of them... and I realize
        I'm floating in space..."

        → Video shows darkness → stars appearing → space scene
    """

    def __init__(
        self,
        generator: VoiceDrivenGenerator,
        character_voice_prompt: str = "a person speaking and looking around",
        blend_character: bool = True,
    ):
        self.generator = generator
        self.character_voice_prompt = character_voice_prompt
        self.blend_character = blend_character

    def generate_narrated_world(
        self,
        narration_audio: Union[str, torch.Tensor],
        include_character: bool = True,
        character_visibility: float = 0.3,  # How visible is the narrator
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate a world being narrated into existence.

        The narration controls what appears, creating a dreamlike
        effect where speech manifests as visual reality.
        """
        # Modify base prompt based on whether to include character
        base_prompt = kwargs.pop("base_prompt", "")
        if include_character and self.blend_character:
            base_prompt = f"{self.character_voice_prompt}, {base_prompt}"

        return self.generator.generate(
            audio=narration_audio,
            base_prompt=base_prompt,
            **kwargs,
        )


# Convenience function
def create_voice_driven_pipeline(
    model_path: str = "Lightricks/LTX-Video",
    voice_engine: str = "whisper",
    device: str = "cuda",
    **kwargs,
) -> VoiceDrivenGenerator:
    """
    Create a complete voice-driven generation pipeline.

    Args:
        model_path: Path to LTX-Video model
        voice_engine: Speech recognition engine ("whisper" or "wav2vec2")
        device: Device to use

    Returns:
        VoiceDrivenGenerator ready for use
    """
    try:
        from ..pipelines.audio_video_pipeline import LTXAudioVideoPipeline
        pipeline = LTXAudioVideoPipeline.from_pretrained_ltx(
            model_path,
            torch_dtype=torch.float16,
            **kwargs,
        ).to(device)
    except Exception as e:
        print(f"Note: Using base pipeline (audio features disabled): {e}")
        from ltx_video.pipelines.pipeline_ltx_video import LTXVideoPipeline
        pipeline = LTXVideoPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
        ).to(device)

    return VoiceDrivenGenerator(
        pipeline=pipeline,
        voice_engine=voice_engine,
        device=device,
    )
