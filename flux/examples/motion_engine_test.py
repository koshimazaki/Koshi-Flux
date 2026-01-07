"""
Motion Engine Testing Example

Test motion transforms without loading the full FLUX model.
Useful for development and debugging motion behavior.
"""

import torch
import numpy as np
from deforum_flux.motion import Flux1MotionEngine, Flux2MotionEngine


def test_flux1_motion():
    """Test FLUX.1 16-channel motion engine."""
    print("=" * 60)
    print("Testing FLUX.1 Motion Engine (16 channels)")
    print("=" * 60)
    
    # Create engine
    engine = Flux1MotionEngine(device="cpu")
    print(f"Engine info: {engine.get_engine_info()}")
    
    # Create dummy latent (batch=1, channels=16, height=64, width=64)
    latent = torch.randn(1, 16, 64, 64)
    print(f"\nInput latent shape: {latent.shape}")
    print(f"Input stats: mean={latent.mean():.4f}, std={latent.std():.4f}")
    
    # Test zoom
    print("\n--- Testing Zoom (1.1x) ---")
    motion = {"zoom": 1.1, "angle": 0, "translation_x": 0, "translation_y": 0, "translation_z": 0}
    result = engine.apply_motion(latent, motion)
    print(f"Output shape: {result.shape}")
    print(f"Output stats: mean={result.mean():.4f}, std={result.std():.4f}")
    
    # Test rotation
    print("\n--- Testing Rotation (15°) ---")
    motion = {"zoom": 1.0, "angle": 15, "translation_x": 0, "translation_y": 0, "translation_z": 0}
    result = engine.apply_motion(latent, motion)
    print(f"Output shape: {result.shape}")
    print(f"Output stats: mean={result.mean():.4f}, std={result.std():.4f}")
    
    # Test depth (channel-aware transform)
    print("\n--- Testing Depth (translation_z=50) ---")
    motion = {"zoom": 1.0, "angle": 0, "translation_x": 0, "translation_y": 0, "translation_z": 50}
    result = engine.apply_motion(latent, motion)
    print(f"Output shape: {result.shape}")
    
    # Show per-channel-group effects
    stats = engine.get_motion_statistics(result)
    print("Channel group stats after depth transform:")
    for gs in stats["channel_groups"]:
        print(f"  Group {gs['group']} ({gs['channels']}): mean={gs['mean']:.4f}, std={gs['std']:.4f}")
    
    # Test combined motion
    print("\n--- Testing Combined Motion ---")
    motion = {
        "zoom": 1.05,
        "angle": 5,
        "translation_x": 10,
        "translation_y": -5,
        "translation_z": 20
    }
    result = engine.apply_motion(latent, motion)
    print(f"Output shape: {result.shape}")
    print(f"Output stats: mean={result.mean():.4f}, std={result.std():.4f}")
    
    return True


def test_flux2_motion():
    """Test FLUX.2 128-channel motion engine."""
    print("\n" + "=" * 60)
    print("Testing FLUX.2 Motion Engine (128 channels)")
    print("=" * 60)
    
    # Create engine
    engine = Flux2MotionEngine(device="cpu")
    print(f"Engine info: {engine.get_engine_info()}")
    
    # Create dummy latent (batch=1, channels=128, height=64, width=64)
    latent = torch.randn(1, 128, 64, 64)
    print(f"\nInput latent shape: {latent.shape}")
    
    # Test depth with 8 channel groups
    print("\n--- Testing Depth (8 channel groups) ---")
    motion = {"zoom": 1.0, "angle": 0, "translation_x": 0, "translation_y": 0, "translation_z": 50}
    result = engine.apply_motion(latent, motion)
    
    stats = engine.get_motion_statistics(result)
    print("Channel group stats after depth transform:")
    for gs in stats["channel_groups"]:
        print(f"  Group {gs['group']} ({gs['channels']}): mean={gs['mean']:.4f}, std={gs['std']:.4f}")
    
    # Test semantic motion (FLUX.2 specific)
    print("\n--- Testing Semantic Motion Control ---")
    result = engine.apply_semantic_motion(
        latent,
        motion_params={"zoom": 1.05, "translation_z": 30},
        semantic_weights={
            "structure": 1.0,   # Full motion on structure
            "color": 0.5,       # Half motion on color
            "texture": 0.8,     # Most motion on texture
            "context": 0.3,     # Minimal on context
        }
    )
    print(f"Semantic motion applied successfully")
    
    return True


def test_interpolation():
    """Test latent interpolation."""
    print("\n" + "=" * 60)
    print("Testing Latent Interpolation")
    print("=" * 60)
    
    engine = Flux1MotionEngine(device="cpu")
    
    # Create two random latents
    latent_a = torch.randn(1, 16, 64, 64)
    latent_b = torch.randn(1, 16, 64, 64)
    
    print(f"Latent A mean: {latent_a.mean():.4f}")
    print(f"Latent B mean: {latent_b.mean():.4f}")
    
    # Linear interpolation
    print("\n--- Linear Interpolation (5 steps) ---")
    interp_linear = engine.interpolate_latents(latent_a, latent_b, num_steps=5, mode="linear")
    for i, interp in enumerate(interp_linear):
        print(f"  Step {i}: mean={interp.mean():.4f}")
    
    # Spherical interpolation
    print("\n--- Spherical Interpolation (5 steps) ---")
    interp_slerp = engine.interpolate_latents(latent_a, latent_b, num_steps=5, mode="slerp")
    for i, interp in enumerate(interp_slerp):
        print(f"  Step {i}: mean={interp.mean():.4f}")
    
    return True


def test_sequence_processing():
    """Test batch/sequence processing."""
    print("\n" + "=" * 60)
    print("Testing Sequence Processing")
    print("=" * 60)
    
    engine = Flux1MotionEngine(device="cpu")
    
    # Create sequence of latents (batch=1, time=10, channels=16, h=32, w=32)
    sequence = torch.randn(1, 10, 16, 32, 32)
    print(f"Input sequence shape: {sequence.shape}")
    
    motion = {"zoom": 1.02, "angle": 2, "translation_z": 10}
    result = engine.apply_motion(sequence, motion)
    
    print(f"Output sequence shape: {result.shape}")
    print("Motion applied to all frames in sequence")
    
    return True


def main():
    """Run all motion engine tests."""
    print("FLUX Deforum Motion Engine Tests")
    print("================================\n")
    
    tests = [
        ("FLUX.1 Motion", test_flux1_motion),
        ("FLUX.2 Motion", test_flux2_motion),
        ("Interpolation", test_interpolation),
        ("Sequence Processing", test_sequence_processing),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, "PASS" if success else "FAIL"))
        except Exception as e:
            print(f"\nERROR in {name}: {e}")
            results.append((name, "ERROR"))
    
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    for name, status in results:
        print(f"  {name}: {status}")


if __name__ == "__main__":
    main()
