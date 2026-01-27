"""
RunPod Serverless Handler - GPU inference endpoint for Cloudflare Workers.
"""

import runpod
import sys
from pathlib import Path

# Add flux to path
sys.path.insert(0, "/workspace/flux/src")


def handler(event):
    """
    RunPod serverless handler for video generation.

    Input:
        {
            "input": {
                "model": "flux-schnell",
                "prompt": "cosmic nebula",
                "num_frames": 48,
                "width": 512,
                "height": 512,
                "fps": 12,
                "seed": 42,
                "strength": 0.1,
                "motion_params": {"zoom": "0:(1.0), 48:(1.05)"},
                "feedback_mode": true,
                "noise_amount": 0.015,
                "sharpen_amount": 0.15
            }
        }

    Output:
        {
            "output": {
                "video_url": "https://...",
                "frames": 48,
                "duration": 4.0
            }
        }
    """
    try:
        input_data = event.get("input", {})

        model = input_data.get("model", "flux-schnell")
        prompt = input_data.get("prompt", "beautiful landscape")
        num_frames = input_data.get("num_frames", 48)
        width = input_data.get("width", 512)
        height = input_data.get("height", 512)
        fps = input_data.get("fps", 12)
        seed = input_data.get("seed")
        strength = input_data.get("strength", 0.1)
        motion_params = input_data.get("motion_params", {"zoom": f"0:(1.0), {num_frames}:(1.05)"})
        feedback_mode = input_data.get("feedback_mode", True)
        noise_amount = input_data.get("noise_amount", 0.015)
        sharpen_amount = input_data.get("sharpen_amount", 0.15)

        # Import and run
        from flux_motion import Flux1Pipeline, FeedbackConfig

        pipe = Flux1Pipeline(model_name=model, offload=True)

        steps = 4 if "schnell" in model else 20
        cfg = 1.0 if "schnell" in model else 3.5

        output_path = f"/tmp/output_{seed or 'video'}.mp4"

        pipe.generate_animation(
            prompts={0: prompt},
            motion_params=motion_params,
            num_frames=num_frames,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=cfg,
            strength=strength,
            fps=fps,
            seed=seed,
            feedback_mode=feedback_mode,
            feedback_config=FeedbackConfig(
                noise_amount=noise_amount,
                sharpen_amount=sharpen_amount,
            ),
            output_path=output_path,
        )

        # Upload to R2 or return base64
        # For now, return local path (configure R2 upload in production)
        return {
            "output": {
                "video_path": output_path,
                "frames": num_frames,
                "duration": num_frames / fps,
                "model": model,
            }
        }

    except Exception as e:
        return {"error": str(e)}


# RunPod serverless entry point
runpod.serverless.start({"handler": handler})
