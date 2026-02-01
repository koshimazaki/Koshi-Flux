"""Klein Ultra Smooth - Maximum temporal consistency"""
from diffusers import Flux2KleinPipeline
import torch
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import json
from datetime import datetime
import os

INPUT_VIDEO = "/workspace/Candy-Grey/1024/1_1024.mp4"
OUTPUT_NAME = "ultra_smooth"
MAX_FRAMES = 60

# HIGHER values for maximum smoothness
EMA_ALPHA = 0.92      # Was 0.85 - more smoothing
WARP_BLEND = 0.90     # Was 0.85 - stronger warp
POST_LAB = 0.95       # Was 0.9 - stronger color lock

PROMPT = "vibrant, abstract creature composed of swirling colors and dynamic shapes. The creature pulsates with energy, as glowing orbs and tendrils undulate rhythmically around it. The background is a deep black, enhancing the vividness of the colors. Soft, diffused lighting creates a mystical atmosphere, while the overall style is psychedelic and surreal."

SIZE = 1024
STEPS = 4
SEED = 42

def match_lab(source, reference):
    src_lab = cv2.cvtColor(source, cv2.COLOR_RGB2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(reference, cv2.COLOR_RGB2LAB).astype(np.float32)
    for c in range(3):
        sm, ss = src_lab[:,:,c].mean(), src_lab[:,:,c].std() + 1e-6
        rm, rs = ref_lab[:,:,c].mean(), ref_lab[:,:,c].std() + 1e-6
        src_lab[:,:,c] = (src_lab[:,:,c] - sm) * (rs / ss) + rm
    return cv2.cvtColor(np.clip(src_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)

def warp_frame(prev, curr, blend):
    prev = np.clip(prev, 0, 255).astype(np.uint8)
    flow = cv2.calcOpticalFlowFarneback(
        cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY),
        cv2.cvtColor(curr, cv2.COLOR_RGB2GRAY),
        None, 0.5, 3, 21, 3, 7, 1.5, 0
    )
    h, w = prev.shape[:2]
    fm = np.stack(np.meshgrid(np.arange(w), np.arange(h)), -1).astype(np.float32) + flow
    warped = cv2.remap(prev, fm[:,:,0], fm[:,:,1], cv2.INTER_LINEAR)
    return cv2.addWeighted(warped, blend, curr, 1-blend, 0)

print("Loading Klein 4B...")
pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4b", torch_dtype=torch.bfloat16).to("cuda")

cap = cv2.VideoCapture(INPUT_VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS) or 60
frames = []
while len(frames) < MAX_FRAMES:
    ret, f = cap.read()
    if not ret: break
    frames.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (SIZE, SIZE)))
cap.release()
print(f"Video: {len(frames)} frames, EMA={EMA_ALPHA}, WARP={WARP_BLEND}")

gen = torch.Generator(device="cuda").manual_seed(SEED)
out_frames, ema_frame, anchor = [], None, None

for i, inp in enumerate(tqdm(frames, desc=OUTPUT_NAME)):
    init_frame = warp_frame(ema_frame, inp, WARP_BLEND) if ema_frame is not None else inp
    result = pipe(prompt=PROMPT, image=Image.fromarray(init_frame.astype(np.uint8)), num_inference_steps=STEPS, height=SIZE, width=SIZE, generator=gen).images[0]
    out = np.array(result)
    if anchor is None: anchor = out.copy()
    ema_frame = out.astype(np.float32) if ema_frame is None else EMA_ALPHA * out.astype(np.float32) + (1 - EMA_ALPHA) * ema_frame
    out_final = cv2.addWeighted(match_lab(out, anchor), POST_LAB, out, 1-POST_LAB, 0)
    out_frames.append(out_final)
    if i % 50 == 0: torch.cuda.empty_cache()

os.makedirs("/workspace/outputs/tests", exist_ok=True)
tmp, out_path = f"/workspace/outputs/tests/{OUTPUT_NAME}_raw.mp4", f"/workspace/outputs/tests/{OUTPUT_NAME}.mp4"
writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (SIZE, SIZE))
for f in out_frames: writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
writer.release()
os.system(f"ffmpeg -y -i {tmp} -c:v libx264 -pix_fmt yuv420p -crf 18 {out_path} 2>/dev/null && rm {tmp}")

with open(out_path.replace(".mp4", ".json"), "w") as f:
    json.dump({"timestamp": datetime.now().isoformat(), "preset": "ultra_smooth", "input_video": INPUT_VIDEO, "prompt": PROMPT, "ema_alpha": EMA_ALPHA, "warp_blend": WARP_BLEND, "post_lab": POST_LAB, "frames": len(frames), "fps": fps, "seed": SEED}, f, indent=2)
print(f"DONE: {out_path}")
