"""
Advanced Usage Example - Keyframe Prompts and Complex Motion

Demonstrates:
- Multiple keyframed prompts for scene transitions
- Complex motion schedules
- Custom callbacks for progress tracking
"""

from pathlib import Path
import torch
from deforum_flux import create_pipeline, FluxVersion, configure_logging
import logging


def progress_callback(frame_idx: int, total_frames: int, latent: torch.Tensor):
    """Track generation progress."""
    percent = (frame_idx + 1) / total_frames * 100
    print(f"Frame {frame_idx + 1}/{total_frames} ({percent:.1f}%)")


def main():
    # Enable debug logging
    configure_logging(level=logging.DEBUG)
    
    # Create pipeline
    pipe = create_pipeline(
        version=FluxVersion.FLUX_1_DEV,
        enable_cpu_offload=True,
    )
    
    # Complex animation with scene transitions
    video_path = pipe.generate_animation(
        # Keyframed prompts for scene evolution
        prompts={
            0: "a peaceful meadow at dawn, soft golden light, wildflowers",
            30: "a peaceful meadow at noon, bright sunlight, butterflies",
            60: "a peaceful meadow at sunset, warm orange glow, fireflies",
            90: "a peaceful meadow at night, moonlight, stars, bioluminescence",
        },
        
        # Complex motion schedule
        motion_params={
            # Breathing zoom effect
            "zoom": "0:(1.0), 15:(1.02), 30:(1.0), 45:(1.02), 60:(1.0), "
                   "75:(1.02), 90:(1.0), 105:(1.02), 120:(1.0)",
            
            # Slow rotation
            "angle": "0:(0), 120:(15)",
            
            # Subtle drift
            "translation_x": "0:(0), 60:(10), 120:(0)",
            "translation_y": "0:(0), 30:(-5), 90:(5), 120:(0)",
            
            # Depth pulsing (affects channel groups)
            "translation_z": "0:(0), 20:(10), 40:(0), 60:(-10), 80:(0), "
                           "100:(10), 120:(0)",
            
            # Strength schedule (lower = more motion, higher = more prompt adherence)
            "strength_schedule": "0:(0.65), 30:(0.55), 60:(0.65), 90:(0.55), 120:(0.65)",
        },
        
        num_frames=120,
        width=1024,
        height=576,  # 16:9 aspect ratio
        num_inference_steps=28,
        guidance_scale=3.5,
        fps=24,
        output_path="meadow_day_cycle.mp4",
        callback=progress_callback,
        seed=12345,
    )
    
    print(f"\nAnimation saved to: {video_path}")
    print(f"Duration: {120/24:.1f} seconds at 24fps")


if __name__ == "__main__":
    main()
