"""Klein V2V with Init Image Color Reference - Parallel Run"""
from diffusers import Flux2KleinPipeline
import torch
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import json
from datetime import datetime
import os

INIT_IMAGE = "/workspace/input/native3.png"
INPUT_VIDEO = "/workspace/Candy-Grey/1024/1_1024.mp4"
OUTPUT_VIDEO = "/workspace/outputs/candy/1_initcolor_native3_1024.mp4"

PROMPT = "vibrant, abstract creature composed of swirling colors and dynamic shapes. The creature pulsates with energy, as glowing orbs and tendrils undulate rhythmically around it. The background is a deep black, enhancing the vividness of the colors. Soft, diffused lighting creates a mystical atmosphere, while the overall style is psychedelic and surreal. The pacing is steady, allowing viewers to immerse themselves in the intricate details of the design."

SIZE = 1024
STEPS = 4
SEED = 42
EMA_ALPHA = 0.85
WARP_BLEND = 0.85
COLOR_MATCH = 0.9
VIDEO_BLEND = 0.3

pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4b", torch_dtype=torch.bfloat16).to("cuda")

init_img = cv2.resize(cv2.cvtColor(cv2.imread(INIT_IMAGE), cv2.COLOR_BGR2RGB), (SIZE, SIZE))
cap = cv2.VideoCapture(INPUT_VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS) or 60
input_frames = []
while True:
    ret, frame = cap.read()
    if not ret: break
    input_frames.append(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (SIZE, SIZE)))
cap.release()

def match_lab(src, ref):
    src_lab = cv2.cvtColor(src, cv2.COLOR_RGB2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(ref, cv2.COLOR_RGB2LAB).astype(np.float32)
    for c in range(3):
        sm, ss = src_lab[:,:,c].mean(), src_lab[:,:,c].std() + 1e-6
        rm, rs = ref_lab[:,:,c].mean(), ref_lab[:,:,c].std() + 1e-6
        src_lab[:,:,c] = (src_lab[:,:,c] - sm) * (rs / ss) + rm
    return cv2.cvtColor(np.clip(src_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)

def warp(prev, curr, blend):
    prev, curr = prev.astype(np.uint8), curr.astype(np.uint8)
    flow = cv2.calcOpticalFlowFarneback(cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY), cv2.cvtColor(curr, cv2.COLOR_RGB2GRAY), None, 0.5, 3, 21, 3, 7, 1.5, 0)
    h, w = prev.shape[:2]
    fm = np.stack(np.meshgrid(np.arange(w), np.arange(h)), -1).astype(np.float32) + flow
    return cv2.addWeighted(cv2.remap(prev, fm[:,:,0], fm[:,:,1], cv2.INTER_LINEAR), blend, curr, 1-blend, 0)

generator = torch.Generator(device="cuda").manual_seed(SEED)
output_frames, ema_frame = [], None

for i, inp in enumerate(tqdm(input_frames, desc="InitColor_v2")):
    init = warp(np.clip(ema_frame, 0, 255).astype(np.uint8), inp, WARP_BLEND) if ema_frame is not None else inp
    out = np.array(pipe(prompt=PROMPT, image=Image.fromarray(init), num_inference_steps=STEPS, height=SIZE, width=SIZE, generator=generator).images[0])
    ema_frame = out.astype(np.float32) if ema_frame is None else EMA_ALPHA * out.astype(np.float32) + (1 - EMA_ALPHA) * ema_frame
    out = cv2.addWeighted(match_lab(out, init_img), COLOR_MATCH, out, 1-COLOR_MATCH, 0)
    if VIDEO_BLEND > 0: out = cv2.addWeighted(out, 1-VIDEO_BLEND, inp, VIDEO_BLEND, 0)
    output_frames.append(out)
    if i % 50 == 0: torch.cuda.empty_cache()

os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)
writer = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), fps, (SIZE, SIZE))
for f in output_frames: writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
writer.release()

with open(OUTPUT_VIDEO.replace(".mp4", ".json"), "w") as f:
    json.dump({"timestamp": datetime.now().isoformat(), "preset": "init_color_v2", "init_image": INIT_IMAGE, "input_video": INPUT_VIDEO, "prompt": PROMPT, "size": SIZE, "steps": STEPS, "seed": SEED, "ema_alpha": EMA_ALPHA, "warp_blend": WARP_BLEND, "color_match": COLOR_MATCH, "video_blend": VIDEO_BLEND, "frames": len(input_frames), "fps": fps}, f, indent=2)
print(f"DONE: {OUTPUT_VIDEO}")
