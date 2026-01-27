"""
Koshi FLUX Adapter - Wraps the FLUX pipeline for the unified API.
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Callable, Any

# Add flux module to path
FLUX_PATH = Path(__file__).parent.parent.parent.parent / "flux" / "src"
sys.path.insert(0, str(FLUX_PATH))


class FluxAdapter:
    """Adapter for Koshi FLUX pipeline."""

    def __init__(self, model_name: str = "flux-schnell", device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self._pipeline = None

    def _load_pipeline(self):
        """Lazy load the pipeline."""
        if self._pipeline is None:
            from flux_motion import Flux1Pipeline
            self._pipeline = Flux1Pipeline(
                model_name=self.model_name,
                device=self.device,
                offload=True,
            )
        return self._pipeline

    def generate(
        self,
        prompt: str,
        motion_params: Dict[str, str],
        num_frames: int = 48,
        width: int = 512,
        height: int = 512,
        fps: int = 12,
        seed: Optional[int] = None,
        strength: float = 0.1,
        feedback_mode: bool = True,
        noise_amount: float = 0.015,
        sharpen_amount: float = 0.15,
        callback: Optional[Callable] = None,
        output_dir: str = "./outputs",
    ) -> Path:
        """Generate video using Koshi FLUX."""
        from flux_motion import FeedbackConfig

        pipe = self._load_pipeline()

        # Determine steps based on model
        steps = 4 if "schnell" in self.model_name else 20
        cfg = 1.0 if "schnell" in self.model_name else 3.5

        output_path = Path(output_dir) / f"flux_{seed or 'random'}.mp4"
        output_path.parent.mkdir(parents=True, exist_ok=True)

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
            callback=callback,
            output_path=str(output_path),
        )

        return output_path

    def get_info(self) -> Dict[str, Any]:
        """Get adapter info."""
        return {
            "adapter": "flux",
            "model": self.model_name,
            "device": self.device,
            "loaded": self._pipeline is not None,
        }
