"""Quick Klein test - 60 frames, adjustable settings"""
from diffusers import Flux2KleinPipeline
import torch
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import os

# === ADJUST THESE ===
INPUT_VIDEO = "/workspace/Candy-Grey/1024/1_1024.mp4"
OUTPUT_NAME = "test_v1"  # Will save as test_v1.mp4
MAX_FRAMES = 60

# Lower = more prompt, less video
VIDEO_BLEND = 0.0       # 0 = pure generation, 0.3 = 30% video shows
WARP_BLEND = 0.5        # Lower = less temporal carry-over
EMA_ALPHA = 0.7         # Lower = more responsive to new frames

PROMPT = "vibrant, abstract creature composed of swirling colors and dynamic shapes. The creature pulsates with energy, as glowing orbs and tendrils undulate rhythmically around it. The background is a deep black, enhancing the vividness of the colors. Soft, diffused lighting creates a mystical atmosphere, while the overall style is psychedelic and surreal."

SIZE = 1024
STEPS = 4
SEED = 42

# === RUN ===
print(f"Settings: video_blend={VIDEO_BLEND}, warp={WARP_BLEND}, ema={EMA_ALPHA}")
pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4b", torch_dtype=torch.bfloat16).to("cuda")

cap = cv2.VideoCapture(INPUT_VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS) or 60
frames = []
while len(frames) < MAX_FRAMES:
    ret, f = cap.read()
    if not ret: break
    frames.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (SIZE, SIZE)))
cap.release()

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
    if VIDEO_BLEND > 0:
        out = cv2.addWeighted(out, 1-VIDEO_BLEND, inp, VIDEO_BLEND, 0)
    out_frames.append(out)

# Save with ffmpeg for proper playback
os.makedirs("/workspace/outputs/tests", exist_ok=True)
tmp = f"/workspace/outputs/tests/{OUTPUT_NAME}_raw.mp4"
out = f"/workspace/outputs/tests/{OUTPUT_NAME}.mp4"
writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (SIZE, SIZE))
for f in out_frames: writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
writer.release()
os.system(f"ffmpeg -y -i {tmp} -c:v libx264 -pix_fmt yuv420p -crf 18 {out} 2>/dev/null && rm {tmp}")
print(f"DONE: {out}")
