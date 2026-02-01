"""Klein Multi-Reference with Temporal Smoothing + JSON metadata"""
from diffusers import Flux2KleinPipeline
import torch
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import json
from datetime import datetime
import os

# === CONFIG ===
INPUT_VIDEO = "/workspace/Candy-Grey/1024/1_1024.mp4"
REFS = ["/workspace/input/native1.png", "/workspace/input/native2.png", "/workspace/input/native3.png"]
OUTPUT_NAME = "multi_smooth"
MAX_FRAMES = 60

# Temporal smoothing (like warp85)
EMA_ALPHA = 0.85      # High = smooth, low = responsive
WARP_BLEND = 0.85     # High = follow previous output, low = follow video

PROMPT = "vibrant, abstract creature composed of swirling colors and dynamic shapes. The creature pulsates with energy, as glowing orbs and tendrils undulate rhythmically around it. The background is a deep black, enhancing the vividness of the colors. Soft, diffused lighting creates a mystical atmosphere, while the overall style is psychedelic and surreal."

SIZE = 1024
STEPS = 4
SEED = 42

# === FUNCTIONS ===
def warp_frame(prev, curr, blend):
    """Optical flow warp for temporal consistency."""
    prev = prev.astype(np.uint8)
    curr = curr.astype(np.uint8)
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
    curr_gray = cv2.cvtColor(curr, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 21, 3, 7, 1.5, 0)
    h, w = prev.shape[:2]
    flow_map = np.stack(np.meshgrid(np.arange(w), np.arange(h)), -1).astype(np.float32) + flow
    warped = cv2.remap(prev, flow_map[:,:,0], flow_map[:,:,1], cv2.INTER_LINEAR)
    return cv2.addWeighted(warped, blend, curr, 1-blend, 0)

def save_json(video_path, **kwargs):
    """Save generation metadata."""
    meta = {
        "timestamp": datetime.now().isoformat(),
        "version": "1.2.0",
        "preset": "multi_smooth",
        "video": video_path,
        "input_video": INPUT_VIDEO,
        "references": REFS,
        "prompt": PROMPT,
        "model": "flux.2-klein-4b",
        "size": SIZE,
        "steps": STEPS,
        "seed": SEED,
        "ema_alpha": EMA_ALPHA,
        "warp_blend": WARP_BLEND,
        **kwargs
    }
    json_path = video_path.replace(".mp4", ".json")
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata: {json_path}")

# === SETUP ===
print("Loading Klein 4B...")
pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4b", torch_dtype=torch.bfloat16).to("cuda")

ref_images = [Image.open(r).resize((SIZE, SIZE)) for r in REFS]
print(f"Loaded {len(ref_images)} reference images")

cap = cv2.VideoCapture(INPUT_VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS) or 60
frames = []
while len(frames) < MAX_FRAMES:
    ret, f = cap.read()
    if not ret: break
    frames.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (SIZE, SIZE)))
cap.release()
print(f"Video: {len(frames)} frames @ {fps:.1f}fps")

# === GENERATION ===
gen = torch.Generator(device="cuda").manual_seed(SEED)
out_frames = []
ema_frame = None

for i, inp in enumerate(tqdm(frames, desc=OUTPUT_NAME)):
    # Warp previous output to follow video motion
    if ema_frame is not None:
        ema_uint8 = np.clip(ema_frame, 0, 255).astype(np.uint8)
        init_frame = warp_frame(ema_uint8, inp, WARP_BLEND)
    else:
        init_frame = inp
    
    # Pass warped frame + all references to Klein
    images = [Image.fromarray(init_frame)] + ref_images
    
    out = pipe(
        prompt=PROMPT,
        image=images,
        num_inference_steps=STEPS,
        height=SIZE, width=SIZE,
        generator=gen
    ).images[0]
    
    out_arr = np.array(out)
    
    # EMA smoothing
    if ema_frame is None:
        ema_frame = out_arr.astype(np.float32)
    else:
        ema_frame = EMA_ALPHA * out_arr.astype(np.float32) + (1 - EMA_ALPHA) * ema_frame
    
    out_frames.append(np.clip(ema_frame, 0, 255).astype(np.uint8))
    
    if i % 50 == 0:
        torch.cuda.empty_cache()

# === SAVE ===
os.makedirs("/workspace/outputs/tests", exist_ok=True)
tmp = f"/workspace/outputs/tests/{OUTPUT_NAME}_raw.mp4"
out_path = f"/workspace/outputs/tests/{OUTPUT_NAME}.mp4"

writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (SIZE, SIZE))
for f in out_frames:
    writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
writer.release()

os.system(f"ffmpeg -y -i {tmp} -c:v libx264 -pix_fmt yuv420p -crf 18 {out_path} 2>/dev/null && rm {tmp}")
save_json(out_path, frames=len(frames), fps=fps)
print(f"DONE: {out_path}")
