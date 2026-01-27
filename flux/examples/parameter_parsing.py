"""
Parameter Parsing Example

Demonstrates how Deforum-style string schedules are converted
to per-frame motion parameters.
"""

from flux_motion.adapters import FluxParameterAdapter, MotionFrame


def main():
    adapter = FluxParameterAdapter()
    
    print("Koshi FLUX Parameter Adapter Demo")
    print("=" * 50)
    
    # Example 1: Parse simple schedule
    print("\n--- Example 1: Simple Zoom Schedule ---")
    schedule = "0:(1.0), 30:(1.05), 60:(1.0)"
    values = adapter.parse_schedule(schedule, num_frames=61, default=1.0)
    
    print(f"Schedule: {schedule}")
    print(f"Frames: 61")
    print(f"Sample values:")
    for i in [0, 15, 30, 45, 60]:
        print(f"  Frame {i}: {values[i]:.4f}")
    
    # Example 2: Parse complex rotation
    print("\n--- Example 2: Oscillating Rotation ---")
    schedule = "0:(0), 15:(10), 30:(0), 45:(-10), 60:(0)"
    values = adapter.parse_schedule(schedule, num_frames=61, default=0.0)
    
    print(f"Schedule: {schedule}")
    print(f"Sample values:")
    for i in [0, 7, 15, 22, 30, 37, 45, 52, 60]:
        print(f"  Frame {i}: {values[i]:.2f}°")
    
    # Example 3: Full Deforum parameter conversion
    print("\n--- Example 3: Full Parameter Conversion ---")
    deforum_params = {
        "zoom": "0:(1.0), 60:(1.1)",
        "angle": "0:(0), 30:(5), 60:(0)",
        "translation_x": "0:(0), 60:(20)",
        "translation_y": 0,  # Constant value
        "translation_z": "0:(0), 30:(15), 60:(0)",
        "strength_schedule": "0:(0.65), 30:(0.55), 60:(0.65)",
        "prompts": {
            0: "a beautiful sunrise over mountains",
            30: "a beautiful day over mountains, clouds",
            60: "a beautiful sunset over mountains",
        }
    }
    
    frames = adapter.convert_deforum_params(deforum_params, num_frames=61)
    
    print(f"Converted {len(frames)} frames")
    print("\nSample frames:")
    for i in [0, 30, 60]:
        f = frames[i]
        print(f"\nFrame {i}:")
        print(f"  zoom: {f.zoom:.4f}")
        print(f"  angle: {f.angle:.2f}°")
        print(f"  translation_x: {f.translation_x:.2f}")
        print(f"  translation_y: {f.translation_y:.2f}")
        print(f"  translation_z: {f.translation_z:.2f}")
        print(f"  strength: {f.strength:.2f}")
        print(f"  prompt: {f.prompt[:40]}..." if f.prompt else "  prompt: (inherited)")
    
    # Example 4: Generate motion schedule dict
    print("\n--- Example 4: Motion Schedule Dictionary ---")
    motion_schedule = adapter.generate_motion_schedule(frames)
    
    print(f"Generated schedule for {len(motion_schedule)} frames")
    print(f"Frame 0 motion: {motion_schedule[0]}")
    print(f"Frame 30 motion: {motion_schedule[30]}")
    
    # Example 5: Simple animation helper
    print("\n--- Example 5: Simple Animation Helper ---")
    simple_frames = adapter.create_simple_animation(
        num_frames=30,
        zoom_start=1.0,
        zoom_end=1.1,
        rotation=0.5,  # Per-frame rotation
        prompt="a cosmic nebula, stars, space"
    )
    
    print(f"Created {len(simple_frames)} frames")
    print(f"First frame: zoom={simple_frames[0].zoom:.2f}, angle={simple_frames[0].angle:.1f}°")
    print(f"Last frame: zoom={simple_frames[-1].zoom:.2f}, angle={simple_frames[-1].angle:.1f}°")


if __name__ == "__main__":
    main()
