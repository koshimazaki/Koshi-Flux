"""
Basic Usage Example for Koshi FLUX Pipeline

Demonstrates simple zoom animation with FLUX.1-dev.
"""

from koshi_flux import create_pipeline, FluxVersion


def main():
    # Create pipeline for FLUX.1
    pipe = create_pipeline(
        version=FluxVersion.FLUX_1_DEV,
        device="cuda",
        enable_cpu_offload=True,  # Save VRAM
    )
    
    # Simple zoom animation
    video_path = pipe.generate_animation(
        # Single prompt or keyframed prompts
        prompts={
            0: "a mystical forest with ancient trees, morning mist, volumetric lighting",
        },
        
        # Deforum-style motion parameters
        motion_params={
            "zoom": "0:(1.0), 60:(1.08)",  # Slow zoom in
            "angle": "0:(0)",               # No rotation
            "translation_x": "0:(0)",
            "translation_y": "0:(0)",
            "translation_z": "0:(0), 30:(5), 60:(0)",  # Subtle depth pulse
        },
        
        # Generation settings
        num_frames=60,
        width=1024,
        height=1024,
        num_inference_steps=28,
        guidance_scale=3.5,
        strength=0.65,
        
        # Output settings
        fps=24,
        output_path="forest_zoom.mp4",
        seed=42,
    )
    
    print(f"Animation saved to: {video_path}")


if __name__ == "__main__":
    main()
