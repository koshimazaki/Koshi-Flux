# Implement FluxDeforumPipeline
---
allowed-tools: all
description: Implement the core generation pipeline for Deforum Flux
argument-hint: Optional specific component to implement
examples: |
  /implement-pipeline
  /implement-pipeline "VAE encode/decode"
  /implement-pipeline "generation loop"
---

## Mission

Implement the **FluxDeforumPipeline** - the missing ~200 lines that connects FLUX model inference to the Deforum motion system.

## Current State (80% Complete)

**Done:**
- Motion engine (Flux16ChannelMotionEngine)
- Parameter engine (keyframe parsing)
- File utilities (frame saving, video creation)
- Bridge structure (FluxDeforumBridge)
- Config system (unified Config dataclass)
- Exception hierarchy

**Missing:**
- FluxDeforumPipeline class
- VAE encode/decode integration
- Generation loop (Frame N → motion → denoise → Frame N+1)
- Video output encoding

## Target Architecture

```python
# flux/src/deforum_flux/pipeline/flux_deforum_pipeline.py

class FluxDeforumPipeline:
    """Main generation orchestrator for Deforum Flux animations."""

    def __init__(self, model_id: str, device: str, config: Config):
        self.pipe = FluxPipeline.from_pretrained(model_id)
        self.vae = self.pipe.vae
        self.motion_engine = Flux16ChannelMotionEngine(device=device)
        self.param_engine = ParameterEngine()

    def encode_to_latent(self, image: torch.Tensor) -> torch.Tensor:
        """Encode RGB image to 16-channel FLUX latent."""
        ...

    def decode_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode 16-channel latent to RGB image."""
        ...

    def generate_animation(
        self,
        prompts: Dict[int, str],
        motion_params: Dict[str, str],
        num_frames: int,
        **kwargs
    ) -> List[Image.Image]:
        """Generate Deforum-style animation."""
        # Frame 0: Generate from text
        # Loop: Apply motion → Denoise → Next frame
        ...
```

## Implementation Steps

1. **Create pipeline module** at `flux/src/deforum_flux/pipeline/`
2. **Implement VAE wrappers** - encode/decode with proper scaling
3. **Implement generation loop** - integrate motion engine
4. **Connect to bridge** - Update FluxDeforumBridge to use pipeline
5. **Add video encoding** - ffmpeg integration via FileUtils
6. **Test end-to-end** - Verify with test mode first

## Code Standards

- **Channel-agnostic**: Design for 16 AND 128 channels
- **Memory-efficient**: Clear tensors, use autocast
- **Modular**: Each method does one thing
- **Tested**: Include test mode fallback

## Completion Criteria

- [ ] FluxDeforumPipeline class implemented
- [ ] VAE encode/decode working
- [ ] Generation loop produces frames
- [ ] Motion transforms applied correctly
- [ ] Video output created
- [ ] Test mode works without GPU
- [ ] Ruff passes
- [ ] No hardcoded paths

## Task

$ARGUMENTS

If no specific component given, implement the full pipeline step by step. Test each component before moving to the next. Store progress in memory.
