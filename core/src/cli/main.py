#!/usr/bin/env python3
"""
Main CLI for Koshi Flux

"""

import argparse
import sys
import time
from pathlib import Path
from typing import List

from koshi.config.settings import Config, get_preset
from koshi_flux.bridge import FluxBridge
from koshi_flux.shared import FluxParameterAdapter
from koshi.core.logging_config import setup_logging
from koshi.core.exceptions import KoshiException
from koshi.utils.file_utils import FileUtils
from koshi.utils.validation import InputValidator


class FluxCLI:
    """Main CLI class for Koshi-Flux integration."""
    
    def __init__(self):
        """Initialize the CLI."""
        self.adapter = FluxParameterAdapter()
        self.validator = InputValidator()
        self.file_utils = FileUtils()
        
    def create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser."""
        parser = argparse.ArgumentParser(
            description="Koshi Flux Animation CLI - Production Ready",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Basic animation
  %(prog)s "a serene mountain landscape" --frames 10
  
  # Custom motion
  %(prog)s "cosmic nebula" --frames 20 --zoom "0:(1.0), 10:(1.3), 20:(1.0)" --angle "0:(0), 20:(15)"
  
  # Use configuration file
  %(prog)s --config config.json
  
  # Test mode (no Flux required)
  %(prog)s --test --frames 5
            """
        )
        
        # Basic parameters
        parser.add_argument(
            "prompt",
            nargs="?",
            default="a beautiful landscape with gentle motion",
            help="Text prompt for generation"
        )
        
        parser.add_argument(
            "--config",
            type=str,
            help="Configuration file path"
        )
        
        parser.add_argument(
            "--preset",
            choices=["fast", "quality", "balanced", "production"],
            default="balanced",
            help="Configuration preset"
        )
        
        # Generation parameters
        parser.add_argument(
            "--frames",
            type=int,
            default=10,
            help="Number of frames to generate"
        )
        
        parser.add_argument(
            "--width",
            type=int,
            default=1024,
            help="Image width"
        )
        
        parser.add_argument(
            "--height", 
            type=int,
            default=1024,
            help="Image height"
        )
        
        parser.add_argument(
            "--steps",
            type=int,
            help="Generation steps (overrides preset)"
        )
        
        parser.add_argument(
            "--guidance",
            type=float,
            help="Guidance scale (overrides preset)"
        )
        
        parser.add_argument(
            "--seed",
            type=int,
            help="Random seed"
        )
        
        # Motion parameters
        parser.add_argument(
            "--zoom",
            default="0:(1.0)",
            help="Zoom schedule (e.g., '0:(1.0), 10:(1.2)')"
        )
        
        parser.add_argument(
            "--angle",
            default="0:(0)",
            help="Rotation angle schedule"
        )
        
        parser.add_argument(
            "--translation-x",
            default="0:(0)",
            help="X translation schedule"
        )
        
        parser.add_argument(
            "--translation-y",
            default="0:(0)", 
            help="Y translation schedule"
        )
        
        # Output options
        parser.add_argument(
            "--output",
            default="./outputs",
            help="Output directory"
        )
        
        parser.add_argument(
            "--prefix",
            default="frame",
            help="Output filename prefix"
        )
        
        parser.add_argument(
            "--format",
            choices=["png", "jpg", "jpeg"],
            default="png",
            help="Output image format"
        )
        
        parser.add_argument(
            "--video",
            action="store_true",
            help="Create video from frames"
        )
        
        parser.add_argument(
            "--fps",
            type=int,
            default=24,
            help="Video FPS"
        )
        
        # System options
        parser.add_argument(
            "--device",
            choices=["cuda", "cpu", "mps"],
            default="cuda",
            help="Device to use"
        )
        
        parser.add_argument(
            "--model",
            choices=["flux-schnell", "flux-dev"],
            help="Flux model to use (overrides preset)"
        )
        
        parser.add_argument(
            "--test",
            action="store_true",
            help="Test mode (generate without Flux)"
        )
        
        parser.add_argument(
            "--verbose",
            action="store_true",
            help="Verbose logging"
        )
        
        parser.add_argument(
            "--log-file",
            help="Log file path"
        )
        
        return parser
    
    def validate_args(self, args: argparse.Namespace) -> None:
        """Validate command line arguments."""
        # Validate prompt
        if not args.test:
            self.validator.validate_prompt(args.prompt)
        
        # Validate dimensions
        self.validator.validate_dimensions(args.width, args.height)
        
        # Validate generation parameters
        steps = args.steps or 20
        guidance = args.guidance or 7.5
        self.validator.validate_generation_params(steps, guidance, args.seed)
        
        # Validate device
        self.validator.validate_device_string(args.device)
        
        # Validate frame count
        if args.frames <= 0 or args.frames > 1000:
            raise ValueError(f"Frame count must be between 1 and 1000, got {args.frames}")
    
    def create_config_from_args(self, args: argparse.Namespace) -> Config:
        """Create configuration from command line arguments."""
        # Start with preset
        if args.config:
            config = Config.from_file(args.config)
        else:
            config = get_preset(args.preset)
        
        # Override with command line arguments
        overrides = {}
        
        if args.model:
            overrides["model_name"] = args.model
        if args.device:
            overrides["device"] = args.device
        if args.width:
            overrides["width"] = args.width
        if args.height:
            overrides["height"] = args.height
        if args.steps:
            overrides["steps"] = args.steps
        if args.guidance:
            overrides["guidance_scale"] = args.guidance
        if args.frames:
            overrides["max_frames"] = args.frames
        if args.output:
            overrides["output_dir"] = args.output
        if args.log_file:
            overrides["log_file"] = args.log_file
        if args.verbose:
            overrides["log_level"] = "DEBUG"
        
        return config.update(**overrides) if overrides else config
    
    def create_animation_config(self, args: argparse.Namespace) -> Config:
        """Create animation configuration from arguments."""
        return Config(
            max_frames=args.frames,
            zoom=args.zoom,
            angle=args.angle,
            translation_x=args.translation_x,
            translation_y=args.translation_y,
            positive_prompts={"0": args.prompt}
        )
    
    def generate_test_animation(self, args: argparse.Namespace) -> List[str]:
        """Generate test animation without Flux."""
        print("🧪 Running in test mode (no Flux required)")
        
        import numpy as np
        from PIL import Image
        
        output_dir = Path(args.output)
        self.file_utils.ensure_directory(output_dir)
        
        frames = []
        
        for i in range(args.frames):
            # Create test image with motion
            width, height = args.width, args.height
            
            # Generate colorful test pattern
            x = np.linspace(0, 4*np.pi, width)
            y = np.linspace(0, 4*np.pi, height)
            X, Y = np.meshgrid(x, y)
            
            # Add time-based animation
            t = i / args.frames * 2 * np.pi
            pattern = np.sin(X + t) * np.cos(Y + t/2) + np.sin(X/2 + t/3) * np.cos(Y/3)
            
            # Normalize and convert to RGB
            pattern = (pattern + 2) / 4  # Normalize to [0, 1]
            rgb = np.stack([
                pattern,
                np.roll(pattern, width//4, axis=1),
                np.roll(pattern, -width//4, axis=1)
            ], axis=2)
            
            rgb = (rgb * 255).astype(np.uint8)
            
            # Save frame
            image = Image.fromarray(rgb)
            filename = f"{args.prefix}_{i:04d}.{args.format}"
            filepath = output_dir / filename
            image.save(filepath)
            frames.append(str(filepath))
            
            print(f"Generated test frame {i+1}/{args.frames}")
        
        return frames
    
    def run(self, args: List[str] = None) -> int:
        """Run the CLI."""
        parser = self.create_parser()
        args = parser.parse_args(args)
        
        try:
            # Setup logging
            setup_logging(
                level="DEBUG" if args.verbose else "INFO",
                console_output=True,
                log_file=args.log_file,
                structured_logging=False
            )
            
            print("🎬 Koshi Flux CLI - Production Ready")
            print("=" * 50)
            
            # Validate arguments
            self.validate_args(args)
            
            # Handle test mode
            if args.test:
                frames = self.generate_test_animation(args)
                print("\n ++[√]++ Test animation completed!")
                print(f"Generated {len(frames)} frames in {args.output}")
                
                if args.video:
                    try:
                        video_path = Path(args.output) / "animation.mp4"
                        self.file_utils.create_video_from_frames(
                            args.output, video_path, args.fps
                        )
                        print(f"Video saved: {video_path}")
                    except Exception as e:
                        print(f"⚠️  Video creation failed: {e}")
                
                return 0
            
            # Create configuration
            config = self.create_config_from_args(args)
            motion_config = self.create_animation_config(args)

            print("Configuration:")
            print(f"  Model: {config.model_name}")
            print(f"  Device: {config.device}")
            print(f"  Resolution: {config.width}x{config.height}")
            print(f"  Steps: {config.steps}")
            print(f"  Frames: {config.max_frames}")

            # Initialize bridge
            print("\n🔧 Initializing Koshi-Flux Bridge...")
            bridge = FluxBridge(config)

            # Create motion schedule from motion config
            motion_schedule = motion_config.to_motion_schedule()

            animation_params = {
                "prompt": args.prompt,
                "max_frames": args.frames,
                "width": config.width,
                "height": config.height,
                "steps": config.steps,
                "guidance_scale": config.guidance_scale,
                "motion_schedule": motion_schedule,
                "seed": args.seed
            }
            
            print("\n🎥 Generating animation...")
            print(f"  Prompt: {args.prompt}")
            print(f"  Motion: zoom={args.zoom}, angle={args.angle}")
            
            # Generate animation
            start_time = time.time()
            frames = bridge.generate_animation(animation_params)
            generation_time = time.time() - start_time
            
            # Save frames
            print("\n💾 Saving frames...")
            self.file_utils.save_animation_frames(
                frames, args.output, args.prefix, args.format
            )
            
            print("\n ++[√]++ Animation completed!")
            print(f"  Generated {len(frames)} frames in {generation_time:.2f}s")
            print(f"  Average time per frame: {generation_time/len(frames):.2f}s")
            print(f"  Output directory: {args.output}")
            
            # Create video if requested
            if args.video:
                try:
                    video_path = Path(args.output) / "animation.mp4"
                    self.file_utils.create_video_from_frames(
                        args.output, video_path, args.fps
                    )
                    print(f"  Video saved: {video_path}")
                except Exception as e:
                    print(f"⚠️  Video creation failed: {e}")
            
            # Save configuration for reference
            config_path = Path(args.output) / "config.json"
            config_data = {
                "config": config.to_dict(),
                "animation_params": animation_params,
                "generation_stats": bridge.get_stats()
            }
            self.file_utils.save_config(config_data, config_path)
            print(f"  Configuration saved: {config_path}")
            
            # Cleanup
            bridge.cleanup()
            
            return 0
            
        except KoshiException as e:
            print(f"\n==[X]== Koshi error: {e}")
            if hasattr(e, 'details') and e.details:
                print(f"Details: {e.details}")
            return 1
            
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")
            return 130
            
        except Exception as e:
            print(f"\n==[X]== Unexpected error: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1


def main():
    """Main entry point."""
    cli = FluxCLI()
    return cli.run()


if __name__ == "__main__":
    sys.exit(main()) 