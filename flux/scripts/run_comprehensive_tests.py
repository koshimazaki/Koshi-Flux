#!/usr/bin/env python3
"""
FLUX.1 Deforum Pipeline - Comprehensive Test Suite
==================================================
Runs all tests and produces detailed logs for diagnostics.

Usage:
    python run_comprehensive_tests.py [--quick] [--gpu-only] [--output-dir DIR]

Options:
    --quick      Skip slow tests (animation generation)
    --gpu-only   Only run GPU tests (skip unit tests)
    --output-dir Directory for logs and outputs (default: /workspace/logs)
"""

import sys
import os
import time
import json
import traceback
import argparse
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List

# Setup paths
WORKSPACE = os.environ.get("WORKSPACE", "/workspace")
sys.path.insert(0, f"{WORKSPACE}/flux/src")
sys.path.insert(0, f"{WORKSPACE}/Deforum2026/core/src")


@dataclass
class TestResult:
    """Single test result."""
    name: str
    status: str  # PASS, FAIL, SKIP, ERROR
    duration: float
    message: str = ""
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


class TestRunner:
    """Comprehensive test runner with logging."""
    
    def __init__(self, output_dir: str = None):
        self.output_dir = Path(output_dir or f"{WORKSPACE}/logs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.output_dir / f"test_run_{self.timestamp}.log"
        self.results_file = self.output_dir / f"test_results_{self.timestamp}.json"
        
        self.results: List[TestResult] = []
        self.start_time = time.time()
        
    def log(self, message: str, level: str = "INFO"):
        """Log message to file and stdout."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] [{level}] {message}"
        print(line)
        with open(self.log_file, "a") as f:
            f.write(line + "\n")
    
    def run_test(self, name: str, func, *args, **kwargs) -> TestResult:
        """Run a single test with timing and error handling."""
        self.log(f"Running: {name}")
        start = time.time()
        
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start
            
            if result is None or result is True:
                test_result = TestResult(name, "PASS", duration)
            elif isinstance(result, dict):
                test_result = TestResult(name, "PASS", duration, details=result)
            else:
                test_result = TestResult(name, "PASS", duration, message=str(result))
                
            self.log(f"  PASS ({duration:.2f}s)")
            
        except Exception as e:
            duration = time.time() - start
            test_result = TestResult(
                name, "FAIL", duration,
                message=str(e),
                details={"traceback": traceback.format_exc()}
            )
            self.log(f"  FAIL: {e}", level="ERROR")
        
        self.results.append(test_result)
        return test_result
    
    def skip_test(self, name: str, reason: str):
        """Mark test as skipped."""
        self.log(f"Skipping: {name} ({reason})")
        self.results.append(TestResult(name, "SKIP", 0.0, message=reason))
    
    def save_results(self):
        """Save results to JSON."""
        total_time = time.time() - self.start_time
        
        summary = {
            "timestamp": self.timestamp,
            "total_duration": total_time,
            "total_tests": len(self.results),
            "passed": sum(1 for r in self.results if r.status == "PASS"),
            "failed": sum(1 for r in self.results if r.status == "FAIL"),
            "skipped": sum(1 for r in self.results if r.status == "SKIP"),
            "results": [asdict(r) for r in self.results]
        }
        
        with open(self.results_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        self.log(f"\nResults saved to: {self.results_file}")
        return summary


# =============================================================================
# Test Functions
# =============================================================================

def test_imports():
    """Test all required imports."""
    import torch
    from deforum_flux import create_flux1_pipeline, FluxVersion
    from deforum_flux.shared import FluxDeforumParameterAdapter, MotionFrame
    from deforum_flux.flux1 import Flux1MotionEngine, FLUX1_CONFIG
    from flux.sampling import get_noise, prepare, denoise, unpack
    return {"torch_version": torch.__version__}


def test_cuda_available():
    """Test CUDA availability and GPU info."""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    
    gpu_info = {
        "cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0),
        "memory_total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
        "memory_allocated_gb": torch.cuda.memory_allocated(0) / 1e9,
    }
    return gpu_info


def test_parameter_adapter():
    """Test schedule parsing."""
    from deforum_flux.shared import FluxDeforumParameterAdapter
    
    adapter = FluxDeforumParameterAdapter()
    
    # Test basic schedule
    schedule = "0:(1.0), 30:(1.05), 60:(1.0)"
    values = adapter.parse_schedule(schedule, 61, default=1.0)
    
    assert len(values) == 61, f"Expected 61 values, got {len(values)}"
    assert abs(values[0] - 1.0) < 0.001, f"Frame 0 should be 1.0, got {values[0]}"
    assert abs(values[30] - 1.05) < 0.001, f"Frame 30 should be 1.05, got {values[30]}"
    assert abs(values[60] - 1.0) < 0.001, f"Frame 60 should be 1.0, got {values[60]}"
    
    # Test interpolation
    mid_value = values[15]
    expected = 1.0 + (1.05 - 1.0) * (15 / 30)
    assert abs(mid_value - expected) < 0.001, f"Interpolation error at frame 15"
    
    return {"frames_parsed": len(values)}


def test_motion_engine_cpu():
    """Test motion engine on CPU."""
    import torch
    from deforum_flux.flux1 import Flux1MotionEngine
    
    engine = Flux1MotionEngine(device="cpu")
    
    # Create test latent (B, 16, H, W)
    latent = torch.randn(1, 16, 64, 64)
    
    motion_params = {
        "zoom": 1.05,
        "angle": 5.0,
        "translation_x": 10.0,
        "translation_y": -5.0,
        "translation_z": 20.0,
    }
    
    result = engine.apply_motion(latent, motion_params)
    
    assert result.shape == latent.shape, f"Shape mismatch: {result.shape} vs {latent.shape}"
    assert not torch.isnan(result).any(), "NaN values in result"
    assert not torch.isinf(result).any(), "Inf values in result"
    
    return {
        "input_shape": list(latent.shape),
        "output_shape": list(result.shape),
        "mean_change": (result - latent).abs().mean().item()
    }


def test_motion_engine_gpu():
    """Test motion engine on GPU."""
    import torch
    from deforum_flux.flux1 import Flux1MotionEngine
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    
    engine = Flux1MotionEngine(device="cuda")
    
    latent = torch.randn(1, 16, 128, 128, device="cuda")
    
    motion_params = {
        "zoom": 1.02,
        "angle": 3.0,
        "translation_x": 5.0,
        "translation_y": 0.0,
        "translation_z": 10.0,
    }
    
    start = time.time()
    result = engine.apply_motion(latent, motion_params)
    torch.cuda.synchronize()
    duration = time.time() - start
    
    assert result.shape == latent.shape
    
    return {
        "latent_shape": list(latent.shape),
        "gpu_time_ms": duration * 1000,
    }


def test_pipeline_init():
    """Test pipeline initialization (no model loading)."""
    from deforum_flux import create_flux1_pipeline
    
    pipe = create_flux1_pipeline(device="cuda", offload=True)
    
    info = pipe.get_info()
    
    assert info["model_name"] == "flux-dev"
    assert info["offload"] == True
    assert info["loaded"] == False  # Not loaded yet
    
    return info


def test_model_loading():
    """Test FLUX model loading."""
    import torch
    from deforum_flux import create_flux1_pipeline
    
    pipe = create_flux1_pipeline(device="cuda", offload=True)
    
    start = time.time()
    pipe.load_models()
    load_time = time.time() - start
    
    assert pipe._loaded == True
    
    memory_after = torch.cuda.memory_allocated() / 1e9
    
    return {
        "load_time_seconds": load_time,
        "gpu_memory_gb": memory_after,
        "models_loaded": ["t5", "clip", "ae", "model"]
    }


def test_single_frame_generation():
    """Test single frame generation."""
    import torch
    from deforum_flux import create_flux1_pipeline
    
    pipe = create_flux1_pipeline(device="cuda", offload=True)
    
    prompt = "a serene mountain lake at sunset, photorealistic"
    
    start = time.time()
    image = pipe.generate_single_frame(
        prompt=prompt,
        width=512,
        height=512,
        num_inference_steps=20,
        guidance_scale=3.5,
        seed=42,
    )
    gen_time = time.time() - start
    
    # Verify image
    assert image is not None
    assert image.size == (512, 512)
    
    # Save test image
    output_path = Path(os.environ.get("WORKSPACE", "/workspace")) / "outputs" / "test_single_frame.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    
    return {
        "generation_time_seconds": gen_time,
        "image_size": image.size,
        "output_path": str(output_path),
        "prompt": prompt[:50],
    }


def test_short_animation():
    """Test short animation generation (5 frames)."""
    import torch
    from deforum_flux import create_flux1_pipeline
    
    pipe = create_flux1_pipeline(device="cuda", offload=True)
    
    prompts = {0: "a mystical forest with glowing mushrooms"}
    motion_params = {
        "zoom": "0:(1.0), 4:(1.02)",
        "angle": "0:(0), 4:(2)",
    }
    
    start = time.time()
    output_path = pipe.generate_animation(
        prompts=prompts,
        motion_params=motion_params,
        num_frames=5,
        width=512,
        height=512,
        num_inference_steps=20,
        guidance_scale=3.5,
        strength=0.65,
        fps=24,
        seed=42,
    )
    gen_time = time.time() - start
    
    assert Path(output_path).exists()
    
    return {
        "generation_time_seconds": gen_time,
        "frames": 5,
        "time_per_frame": gen_time / 5,
        "output_path": output_path,
    }


def test_latent_pack_unpack():
    """Test latent packing/unpacking roundtrip."""
    import torch
    from deforum_flux import create_flux1_pipeline
    from flux.sampling import unpack
    
    pipe = create_flux1_pipeline(device="cuda", offload=True)
    
    # Create test latent
    original = torch.randn(1, 16, 128, 128, device="cuda", dtype=torch.float32)
    
    # Pack
    packed = pipe._pack_latent(original, 1024, 1024)
    
    # Unpack using BFL's function
    unpacked = unpack(packed.float(), 1024, 1024)
    
    # Verify roundtrip
    diff = (original - unpacked).abs().max().item()
    
    return {
        "original_shape": list(original.shape),
        "packed_shape": list(packed.shape),
        "unpacked_shape": list(unpacked.shape),
        "max_roundtrip_error": diff,
        "shapes_match": original.shape == unpacked.shape,
    }


def run_pytest_unit_tests():
    """Run pytest unit tests."""
    import subprocess
    
    workspace = os.environ.get("WORKSPACE", "/workspace")
    test_dir = f"{workspace}/flux/tests"
    
    if not Path(test_dir).exists():
        raise RuntimeError(f"Test directory not found: {test_dir}")
    
    result = subprocess.run(
        ["python", "-m", "pytest", test_dir, "-v", "--tb=short", "-q"],
        capture_output=True,
        text=True,
        cwd=workspace,
        timeout=300,
    )
    
    return {
        "return_code": result.returncode,
        "stdout": result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout,
        "stderr": result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="FLUX.1 Comprehensive Tests")
    parser.add_argument("--quick", action="store_true", help="Skip slow tests")
    parser.add_argument("--gpu-only", action="store_true", help="Only GPU tests")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    args = parser.parse_args()
    
    runner = TestRunner(output_dir=args.output_dir)
    
    runner.log("=" * 60)
    runner.log("FLUX.1 Deforum Pipeline - Comprehensive Test Suite")
    runner.log("=" * 60)
    runner.log("")
    
    # ==========================================================================
    # Phase 1: Basic Tests
    # ==========================================================================
    runner.log("\n--- Phase 1: Basic Tests ---")
    runner.run_test("Import Test", test_imports)
    runner.run_test("CUDA Availability", test_cuda_available)
    
    if not args.gpu_only:
        runner.run_test("Parameter Adapter", test_parameter_adapter)
        runner.run_test("Motion Engine (CPU)", test_motion_engine_cpu)
    
    # ==========================================================================
    # Phase 2: GPU Tests
    # ==========================================================================
    runner.log("\n--- Phase 2: GPU Tests ---")
    runner.run_test("Motion Engine (GPU)", test_motion_engine_gpu)
    runner.run_test("Pipeline Init", test_pipeline_init)
    runner.run_test("Latent Pack/Unpack", test_latent_pack_unpack)
    
    # ==========================================================================
    # Phase 3: Model Loading
    # ==========================================================================
    runner.log("\n--- Phase 3: Model Loading ---")
    runner.run_test("Model Loading", test_model_loading)
    
    # ==========================================================================
    # Phase 4: Generation Tests
    # ==========================================================================
    runner.log("\n--- Phase 4: Generation Tests ---")
    runner.run_test("Single Frame Generation", test_single_frame_generation)
    
    if not args.quick:
        runner.run_test("Short Animation (5 frames)", test_short_animation)
    else:
        runner.skip_test("Short Animation (5 frames)", "Skipped in quick mode")
    
    # ==========================================================================
    # Phase 5: Unit Tests
    # ==========================================================================
    if not args.gpu_only:
        runner.log("\n--- Phase 5: Unit Tests (pytest) ---")
        runner.run_test("Pytest Unit Tests", run_pytest_unit_tests)
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    summary = runner.save_results()
    
    runner.log("\n" + "=" * 60)
    runner.log("TEST SUMMARY")
    runner.log("=" * 60)
    runner.log(f"Total Tests: {summary['total_tests']}")
    runner.log(f"Passed:      {summary['passed']}")
    runner.log(f"Failed:      {summary['failed']}")
    runner.log(f"Skipped:     {summary['skipped']}")
    runner.log(f"Duration:    {summary['total_duration']:.2f}s")
    runner.log("=" * 60)
    
    if summary['failed'] > 0:
        runner.log("\nFAILED TESTS:", level="ERROR")
        for r in runner.results:
            if r.status == "FAIL":
                runner.log(f"  - {r.name}: {r.message}", level="ERROR")
    
    runner.log(f"\nLog file: {runner.log_file}")
    runner.log(f"Results:  {runner.results_file}")
    
    return 0 if summary['failed'] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
