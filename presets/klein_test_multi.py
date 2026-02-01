"""Quick Klein test with multiple native references"""
from diffusers import Flux2KleinPipeline
import torch
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import os

# === SETTINGS ===
INPUT_VIDEO = "/workspace/Candy-Grey/1024/1_1024.mp4"
REFS = ["/workspace/input/native1.png", "/workspace/input/native2.png", "/workspace/input/native3.png"]
OUTPUT_NAME = "test_multi"
MAX_FRAMES = 60

VIDEO_BLEND = 0.0       # Pure prompt
WARP_BLEND = 0.5        
EMA_ALPHA = 0.7         
COLOR_MATCH = 0.85      # How much to match ref colors

PROMPT = "vibrant, abstract creature composed of swirling colors and dynamic shapes. The creature pulsates with energy, as glowing orbs and tendrils undulate rhythmically around it. The background is a deep black, enhancing the vividness of the colors. Soft, diffused lighting creates a mystical atmosphere, while the overall style is psychedelic and surreal."

SIZE = 1024
STEPS = 4
SEED = 42

# === SETUP ===
pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4b", torch_dtype=torch.bfloat16).to("cuda")

# Load and blend all references into one color palette
refs = [cv2.resize(cv2.cvtColor(cv2.imread(r), cv2.COLOR_BGR2RGB), (SIZE, SIZE)) for r in REFS]
ref_blend = np.mean(refs, axis=0).astype(np.uint8)
print(f"Blended {len(refs)} reference images for color guidance")

cap = cv2.VideoCapture(INPUT_VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS) or 60
frames = []
while len(frames) < MAX_FRAMES:
    ret, f = cap.read()
    if not ret: break
    frames.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (SIZE, SIZE)))
cap.release()

def match_lab(src, ref):
    src_lab = cv2.cvtColor(src, cv2.COLOR_RGB2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(ref, cv2.COLOR_RGB2LAB).astype(np.float32)
    for c in range(3):
        sm, ss = src_lab[:,:,c].mean(), src_lab[:,:,c].std() + 1e-6
        rm, rs = ref_lab[:,:,c].mean(), ref_lab[:,:,c].std() + 1e-6
        src_lab[:,:,c] = (src_lab[:,:,c] - sm) * (rs / ss) + rm
    return cv2.cvtColor(np.clip(src_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)

def warp_frame(prev, curr, blend):
    p, c = prev.astype(np.uint8), curr.astype(np.uint8)
    flow = cv2.calcOpticalFlowFarneback(cv2.cvtColor(p, cv2.COLOR_RGB2GRAY), cv2.cvtColor(c, cv2.COLOR_RGB2GRAY), None, 0.5, 3, 21, 3, 7, 1.5, 0)
    h, w = p.shape[:2]
    fm = np.stack(np.meshgrid(np.arange(w), np.arange(h)), -1).astype(np.float32) + flow
    return cv2.addWeighted(cv2.remap(p, fm[:,:,0], fm[:,:,1], cv2.INTER_LINEAR), blend, c, 1-blend, 0)

gen = torch.Generator(device="cuda").manual_seed(SEED)
out_frames, ema = [], None

for i, inp in enumerate(tqdm(frames, desc=OUTPUT_NAME)):
    init = warp_frame(np.clip(ema, 0, 255).astype(np.uint8), inp, WARP_BLEND) if ema is not None else inp
    out = np.array(pipe(prompt=PROMPT, image=Image.fromarray(init), num_inference_steps=STEPS, height=SIZE, width=SIZE, generator=gen).images[0])
    ema = out.astype(np.float32) if ema is None else EMA_ALPHA * out.astype(np.float32) + (1 - EMA_ALPHA) * ema
    
    # Apply color from blended references
    out_matched = match_lab(out, ref_blend)
    out = cv2.addWeighted(out_matched, COLOR_MATCH, out, 1-COLOR_MATCH, 0)
    
    if VIDEO_BLEND > 0:
        out = cv2.addWeighted(out, 1-VIDEO_BLEND, inp, VIDEO_BLEND, 0)
    out_frames.append(out)

os.makedirs("/workspace/outputs/tests", exist_ok=True)
tmp = f"/workspace/outputs/tests/{OUTPUT_NAME}_raw.mp4"
out_path = f"/workspace/outputs/tests/{OUTPUT_NAME}.mp4"
writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (SIZE, SIZE))
for f in out_frames: writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
writer.release()
os.system(f"ffmpeg -y -i {tmp} -c:v libx264 -pix_fmt yuv420p -crf 18 {out_path} 2>/dev/null && rm {tmp}")
print(f"DONE: {out_path}")
