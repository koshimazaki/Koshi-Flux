"""Klein Ultimate V2V - Multi-ref + EMA + Warp + LAB Color Anchor

Same workflow as warp85 batch that worked, but with multi-reference support.
"""
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
INIT_IMAGE = "/workspace/input/native1.png"  # Color anchor
REFS = ["/workspace/input/native1.png", "/workspace/input/native2.png", "/workspace/input/native3.png"]
OUTPUT_NAME = "ultimate"
MAX_FRAMES = 60

# Temporal settings (warp85 that worked)
EMA_ALPHA = 0.85
WARP_BLEND = 0.85
POST_LAB = 0.9        # LAB color match to init image

PROMPT = "vibrant, abstract creature composed of swirling colors and dynamic shapes. The creature pulsates with energy, as glowing orbs and tendrils undulate rhythmically around it. The background is a deep black, enhancing the vividness of the colors. Soft, diffused lighting creates a mystical atmosphere, while the overall style is psychedelic and surreal."

SIZE = 1024
STEPS = 4
SEED = 42

# === FUNCTIONS ===
def match_lab(source, reference):
    """Match source colors to reference using LAB space."""
    src = source.astype(np.uint8)
    ref = reference.astype(np.uint8)
    src_lab = cv2.cvtColor(src, cv2.COLOR_RGB2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(ref, cv2.COLOR_RGB2LAB).astype(np.float32)
    for c in range(3):
        src_mean, src_std = src_lab[:,:,c].mean(), src_lab[:,:,c].std() + 1e-6
        ref_mean, ref_std = ref_lab[:,:,c].mean(), ref_lab[:,:,c].std() + 1e-6
        src_lab[:,:,c] = (src_lab[:,:,c] - src_mean) * (ref_std / src_std) + ref_mean
    return cv2.cvtColor(np.clip(src_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)

def warp_frame(prev, curr, blend):
    """Optical flow warp for temporal consistency."""
    prev = np.clip(prev, 0, 255).astype(np.uint8)
    curr = curr.astype(np.uint8)
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
    curr_gray = cv2.cvtColor(curr, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 21, 3, 7, 1.5, 0)
    h, w = prev.shape[:2]
    flow_map = np.stack(np.meshgrid(np.arange(w), np.arange(h)), -1).astype(np.float32) + flow
    warped = cv2.remap(prev, flow_map[:,:,0], flow_map[:,:,1], cv2.INTER_LINEAR)
    return cv2.addWeighted(warped, blend, curr, 1-blend, 0)

def save_json(video_path, **kwargs):
    meta = {
        "timestamp": datetime.now().isoformat(),
        "version": "1.3.0",
        "preset": "ultimate",
        **kwargs
    }
    with open(video_path.replace(".mp4", ".json"), "w") as f:
        json.dump(meta, f, indent=2)

# === SETUP ===
print("Loading Klein 4B...")
pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4b", torch_dtype=torch.bfloat16).to("cuda")

# Load color anchor (init image)
anchor = cv2.resize(cv2.cvtColor(cv2.imread(INIT_IMAGE), cv2.COLOR_BGR2RGB), (SIZE, SIZE))
print(f"Color anchor: {INIT_IMAGE}")

# Load references for Klein multi-image input
ref_images = [Image.open(r).resize((SIZE, SIZE)) for r in REFS]
print(f"Loaded {len(ref_images)} reference images")

# Load video
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
    # 1. Warp previous EMA output to follow video motion
    if ema_frame is not None:
        init_frame = warp_frame(ema_frame, inp, WARP_BLEND)
    else:
        init_frame = inp
    
    # 2. Generate with Klein (video frame + references)
    images = [Image.fromarray(init_frame.astype(np.uint8))] + ref_images
    
    result = pipe(
        prompt=PROMPT,
        image=images,
        num_inference_steps=STEPS,
        height=SIZE, width=SIZE,
        generator=gen
    ).images[0]
    
    out = np.array(result)
    
    # 3. Update EMA for temporal smoothing
    if ema_frame is None:
        ema_frame = out.astype(np.float32)
    else:
        ema_frame = EMA_ALPHA * out.astype(np.float32) + (1 - EMA_ALPHA) * ema_frame
    
    # 4. LAB color match to anchor for color coherence
    out_matched = match_lab(out, anchor)
    out_final = cv2.addWeighted(out_matched, POST_LAB, out, 1-POST_LAB, 0)
    
    out_frames.append(out_final)
    
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

save_json(out_path,
    video=out_path,
    input_video=INPUT_VIDEO,
    init_image=INIT_IMAGE,
    references=REFS,
    prompt=PROMPT,
    model="flux.2-klein-4b",
    size=SIZE,
    steps=STEPS,
    seed=SEED,
    ema_alpha=EMA_ALPHA,
    warp_blend=WARP_BLEND,
    post_lab=POST_LAB,
    frames=len(frames),
    fps=fps
)
print(f"DONE: {out_path}")
