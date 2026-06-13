#!/usr/bin/env python3
"""Shared native-SDK pipeline for presets/native/.

Pure BFL/PyTorch path - no diffusers anywhere:
- AutoEncoder via flux2.util.load_ae() (proper BatchNorm stats)
- DiT via flux2.util.load_flow_model()
- Rectified-flow denoise loop via flux2.sampling

This is the counterpart to klein_utils.get_pipeline() (Flux2Pipeline =
diffusers AutoencoderKL VAE + native BFL DiT denoise). The presets in this
folder mirror their hybrid-v2v/ twins with identical CLIs and defaults so
outputs can be A/B compared - the only variable is the VAE/latent path.
"""
import os
import sys
from pathlib import Path

# klein_utils bootstraps sys.path for the flux2 SDK (repo-local flux2-main/src,
# /workspace layouts) - import it before flux2.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import klein_utils  # noqa: F401,E402  (side effect: sys.path bootstrap)

import torch  # noqa: E402
import torchvision.transforms as T  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402

from flux2.util import load_ae, load_flow_model, load_text_encoder  # noqa: E402
from flux2.sampling import (  # noqa: E402
    prc_img, prc_txt, denoise, get_schedule, scatter_ids, default_prep
)


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL Image to tensor [-1, 1]."""
    t = T.ToTensor()(img)
    return (2 * t - 1).unsqueeze(0)  # (1, 3, H, W)


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """Tensor [-1, 1] to PIL Image."""
    t = (t.clamp(-1, 1) + 1) / 2  # [0, 1]
    t = t.squeeze(0).cpu()
    return T.ToPILImage()(t)


class NativePipeline:
    """Pure BFL/PyTorch pipeline (no diffusers)."""

    def __init__(self, model_name: str = "flux.2-klein-4b", device: str = "cuda"):
        self.device = device
        self.model_name = model_name
        self._motion_engine = None  # Lazy - only loaded when motion_params are used

        tqdm.write(f"Loading BFL native components: {model_name}")

        # Load BFL VAE with proper BatchNorm stats
        self.ae = self._load_native_ae(model_name, device)
        self.ae.eval()

        # Load BFL DiT
        self.model = load_flow_model(model_name, device=device)
        self.model.eval()

        # Load text encoder
        self.text_enc = load_text_encoder(model_name, device=device)

        tqdm.write("Native pipeline ready")

    @staticmethod
    def _load_native_ae(model_name: str, device: str):
        """Load the native BFL AutoEncoder, handling Klein repos' missing AE file.

        VERIFIED 2026-06-13: the Klein HF repos ship ONLY the diffusers-format
        VAE (vae/ subfolder) - there is no root ae.safetensors, so plain
        flux2.util.load_ae fails for Klein (this is why the hybrid lane uses
        the diffusers VAE). Resolution order:
          1. load_ae as-is (honors AE_MODEL_PATH env var; works if BFL ships it)
          2. fall back to the FLUX.2 family AutoEncoder from
             black-forest-labs/FLUX.2-dev (ae.safetensors, ~336MB; the SDK uses
             one shared AutoEncoderParams across the family). Requires accepted
             access to the gated FLUX.2-dev repo + huggingface-cli login.
        """
        try:
            return load_ae(model_name, device=device)
        except Exception as first_err:
            if os.environ.get("AE_MODEL_PATH"):
                raise  # an explicit local path failed - don't mask that error
            try:
                import huggingface_hub
                path = huggingface_hub.hf_hub_download(
                    repo_id="black-forest-labs/FLUX.2-dev",
                    filename="ae.safetensors",
                )
            except Exception as dl_err:
                raise RuntimeError(
                    f"Native AutoEncoder unavailable for {model_name}: the Klein "
                    f"repo ships no ae.safetensors ({first_err}); downloading the "
                    f"FLUX.2-dev family AE also failed ({dl_err}). Fix: set "
                    "AE_MODEL_PATH=/path/to/ae.safetensors, or accept access to "
                    "black-forest-labs/FLUX.2-dev and `huggingface-cli login`."
                ) from dl_err
            tqdm.write(
                "[native_utils] Klein repo has no native ae.safetensors - using "
                "the FLUX.2-dev family AutoEncoder (~336MB). A/B caveat: verify "
                "output quality vs the hybrid diffusers VAE."
            )
            os.environ["AE_MODEL_PATH"] = path
            try:
                return load_ae(model_name, device=device)
            finally:
                os.environ.pop("AE_MODEL_PATH", None)

    @property
    def motion_engine(self):
        """Lazy Flux2MotionEngine - geometric motion (zoom/angle/translation)
        applied directly in the 128-channel latent space. Same engine the
        hybrid Flux2Pipeline uses, so motion behavior matches across lanes."""
        if self._motion_engine is None:
            from flux_motion.flux2.motion_engine import Flux2MotionEngine
            self._motion_engine = Flux2MotionEngine(device=self.device)
        return self._motion_engine

    @torch.no_grad()
    def encode(self, img: Image.Image) -> torch.Tensor:
        """Encode image to latent with BFL VAE (deterministic - mean, no sampling).

        NOTE: default_prep center-crops to a multiple of 16 and downscales above
        limit_pixels, so the decoded output size can differ from the input size.
        Presets that pixel-blend/warp prev_gen with raw input frames must resize
        prev_gen to frame.size first. limit_pixels matches the hybrid pipeline
        (2024**2) so native/hybrid A/B runs compare the same resolution.
        """
        img_tensor = default_prep(img, limit_pixels=2024**2, ensure_multiple=16)
        if isinstance(img_tensor, list):
            img_tensor = img_tensor[0]
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        # Encode with proper BatchNorm
        z = self.ae.encode(img_tensor)
        return z

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> Image.Image:
        """Decode latent to image with BFL VAE.

        Mirrors the official CLI: ``ae.decode(x).float()`` - the .float() cast
        is required because denoised latents are bfloat16 and torchvision's
        ToPILImage does not support bfloat16.
        """
        img_tensor = self.ae.decode(z).float()
        img_tensor = img_tensor.clamp(-1, 1)
        return tensor_to_pil(img_tensor)

    @torch.no_grad()
    def encode_prompt(self, prompt: str) -> tuple:
        """Encode text prompt.

        The BFL embedders (Qwen3Embedder for Klein) are nn.Modules called as
        ``text_encoder([prompt])`` returning (b, l, d) - there is no .encode().
        """
        txt_emb = self.text_enc([prompt])
        txt_tokens, txt_ids = prc_txt(txt_emb[0])
        return txt_tokens.unsqueeze(0).to(self.device), txt_ids.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def generate_from_latent(
        self,
        z: torch.Tensor,
        prompt: str,
        strength: float = 0.3,
        num_steps: int = 4,
        guidance: float = 1.0,
        seed: int = 42,
        motion_params: dict = None,
    ) -> tuple:
        """Denoise from a spatial latent (1, 128, H/16, W/16).

        Enables latent-space workflows (reference blending, latent color
        matching, Deforum-style geometric motion) on the native path.

        Args:
            motion_params: Optional dict (zoom/angle/translation_x/y/z) applied
                to the spatial latent by Flux2MotionEngine BEFORE token packing
                - same semantics as the hybrid pipeline's motion schedules.

        Returns:
            (PIL image, output latent) - the latent can be reused as a color
            anchor or for further latent ops / feedback loops.
        """
        torch.manual_seed(seed)

        # Deforum-style geometric motion in latent space
        if motion_params:
            z = self.motion_engine.apply_motion(z, motion_params)

        # Prep image tokens with position IDs
        img_tokens, img_ids = prc_img(z[0])
        img_tokens = img_tokens.unsqueeze(0).to(self.device, dtype=torch.bfloat16)
        img_ids = img_ids.unsqueeze(0).to(self.device)

        # Encode prompt
        txt_tokens, txt_ids = self.encode_prompt(prompt)
        txt_tokens = txt_tokens.to(dtype=torch.bfloat16)

        # Get schedule (mu-shifted for resolution)
        seq_len = img_tokens.shape[1]
        full_timesteps = get_schedule(num_steps, seq_len)

        # Calculate start step based on strength
        # strength=1.0 -> start from pure noise (step 0)
        # strength=0.0 -> no change (skip all)
        start_step = int(num_steps * (1.0 - strength))
        timesteps = full_timesteps[start_step:]

        if len(timesteps) <= 1:
            # No denoising needed
            return self.decode(z), z

        # Add noise based on starting timestep
        t_start = timesteps[0]
        noise = torch.randn_like(img_tokens)

        # Rectified flow: x_t = (1-t)*img + t*noise
        noised = (1 - t_start) * img_tokens + t_start * noise

        # Denoise
        out_tokens = denoise(
            model=self.model,
            img=noised,
            img_ids=img_ids,
            txt=txt_tokens,
            txt_ids=txt_ids,
            timesteps=timesteps,
            guidance=guidance,
        )

        # Scatter back to spatial
        out_list = scatter_ids(out_tokens, img_ids)
        out_z = out_list[0].squeeze(2)  # Remove time dim -> (1, 128, H, W)

        return self.decode(out_z), out_z

    @torch.no_grad()
    def generate(
        self,
        img: Image.Image,
        prompt: str,
        strength: float = 0.3,
        num_steps: int = 4,
        guidance: float = 1.0,
        seed: int = 42,
    ) -> Image.Image:
        """img2img: encode then denoise. Returns the generated image.

        encode() is deterministic (mean, no sampling), so seeding inside
        generate_from_latent() reproduces the original single-seed behavior.
        """
        z = self.encode(img)
        out_img, _ = self.generate_from_latent(
            z, prompt, strength=strength, num_steps=num_steps,
            guidance=guidance, seed=seed,
        )
        return out_img


__all__ = ["NativePipeline", "pil_to_tensor", "tensor_to_pil"]
