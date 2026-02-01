"""Klein edge-guided - video edges as structure, prompt as fill"""
from diffusers import Flux2KleinPipeline
import torch
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import os

INPUT_VIDEO = "/workspace/Candy-Grey/1024/1_1024.mp4"
OUTPUT_NAME = "edge_guided"
MAX_FRAMES = 60

# Edge extraction settings
EDGE_BLEND = 0.4    # How much edge structure influences init
STRENGTH = 0.75     # Higher = more prompt influence

PROMPT = "vibrant, abstract creature composed of swirling colors and dynamic shapes. The creature pulsates with energy, as glowing orbs and tendrils undulate rhythmically around it. The background is a deep black, enhancing the vividness of the colors. Soft, diffused lighting creates a mystical atmosphere, while the overall style is psychedelic and surreal."

SIZE = 1024
STEPS = 4
SEED = 42

pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4b", torch_dtype=torch.bfloat16).to("cuda")

cap = cv2.VideoCapture(INPUT_VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS) or 60
frames = []
while len(frames) < MAX_FRAMES:
    ret, f = cap.read()
    if not ret: break
    frames.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (SIZE, SIZE)))
cap.release()

def extract_edges(img):
    """Extract edges and create edge-highlighted version"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    # Blend edges with darkened original to preserve structure
    dark = (img * 0.3).astype(np.uint8)
    return cv2.addWeighted(dark, 1-EDGE_BLEND, edges_rgb, EDGE_BLEND, 0)

gen = torch.Generator(device="cuda").manual_seed(SEED)
out_frames = []
prev_out = None

for i, inp in enumerate(tqdm(frames, desc=OUTPUT_NAME)):
    # Create edge-guided init image
    edge_init = extract_edges(inp)
    
    # If we have previous output, blend for temporal consistency
    if prev_out is not None:
        edge_init = cv2.addWeighted(edge_init, 0.6, prev_out, 0.4, 0)
    
    out = pipe(
        prompt=PROMPT,
        image=Image.fromarray(edge_init),
        strength=STRENGTH,
        num_inference_steps=STEPS,
        height=SIZE, width=SIZE,
        generator=gen
    ).images[0]
    
    prev_out = np.array(out)
    out_frames.append(prev_out)

os.makedirs("/workspace/outputs/tests", exist_ok=True)
tmp = f"/workspace/outputs/tests/{OUTPUT_NAME}_raw.mp4"
out_path = f"/workspace/outputs/tests/{OUTPUT_NAME}.mp4"
writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (SIZE, SIZE))
for f in out_frames: writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
writer.release()
os.system(f"ffmpeg -y -i {tmp} -c:v libx264 -pix_fmt yuv420p -crf 18 {out_path} 2>/dev/null && rm {tmp}")
print(f"DONE: {out_path}")
