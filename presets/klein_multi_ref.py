"""Klein with multiple reference images"""
from diffusers import Flux2KleinPipeline
import torch
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import os

INPUT_VIDEO = "/workspace/Candy-Grey/1024/1_1024.mp4"
REFS = ["/workspace/input/native1.png", "/workspace/input/native2.png", "/workspace/input/native3.png"]
OUTPUT_NAME = "multi_ref"
MAX_FRAMES = 60

PROMPT = "vibrant, abstract creature composed of swirling colors and dynamic shapes. The creature pulsates with energy, as glowing orbs and tendrils undulate rhythmically around it. The background is a deep black, enhancing the vividness of the colors. Soft, diffused lighting creates a mystical atmosphere, while the overall style is psychedelic and surreal."

SIZE = 1024
STEPS = 4
SEED = 42

pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4b", torch_dtype=torch.bfloat16).to("cuda")

# Load references
ref_images = [Image.open(r).resize((SIZE, SIZE)) for r in REFS]
print(f"Loaded {len(ref_images)} reference images")

cap = cv2.VideoCapture(INPUT_VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS) or 60
frames = []
while len(frames) < MAX_FRAMES:
    ret, f = cap.read()
    if not ret: break
    frames.append(Image.fromarray(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (SIZE, SIZE))))
cap.release()

gen = torch.Generator(device="cuda").manual_seed(SEED)
out_frames = []

for i, inp in enumerate(tqdm(frames, desc=OUTPUT_NAME)):
    # Pass video frame + all references as list
    images = [inp] + ref_images
    
    out = pipe(
        prompt=PROMPT,
        image=images,
        num_inference_steps=STEPS,
        height=SIZE, width=SIZE,
        generator=gen
    ).images[0]
    
    out_frames.append(np.array(out))

os.makedirs("/workspace/outputs/tests", exist_ok=True)
tmp = f"/workspace/outputs/tests/{OUTPUT_NAME}_raw.mp4"
out_path = f"/workspace/outputs/tests/{OUTPUT_NAME}.mp4"
writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (SIZE, SIZE))
for f in out_frames: writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
writer.release()
os.system(f"ffmpeg -y -i {tmp} -c:v libx264 -pix_fmt yuv420p -crf 18 {out_path} 2>/dev/null && rm {tmp}")
print(f"DONE: {out_path}")
