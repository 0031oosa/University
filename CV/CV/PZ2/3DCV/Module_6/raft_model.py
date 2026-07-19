import os

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from torchvision.models.optical_flow import Raft_Small_Weights, raft_small
from torchvision.utils import flow_to_image

file_name = "trm.169.007.avi"
video_path = os.path.join("..", "data", "city", file_name)

weights = Raft_Small_Weights.DEFAULT
transforms = weights.transforms()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using {device}")

model = raft_small(weights=weights, progress=False).to(device)
model = model.eval()


def preprocess(img1_batch, img2_batch):
    img1_batch = F.resize(img1_batch, size=[520, 960], antialias=False)
    img2_batch = F.resize(img2_batch, size=[520, 960], antialias=False)
    return transforms(img1_batch, img2_batch)


def bgr_uint8_to_chw_torch(bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)


cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    raise SystemExit(f"Cannot open video: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps is None or fps <= 1e-3:
    fps = 2.0

ret, bgr_prev = cap.read()
if not ret:
    raise SystemExit("Empty video")

out_dir = os.path.join(os.path.dirname(__file__), "../data/network_video")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "RAFT.avi")

img1 = bgr_uint8_to_chw_torch(bgr_prev)
result_frames = []
pair_idx = 0
total_pairs = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
if total_pairs < 1:
    total_pairs = None

while True:
    ret, bgr = cap.read()
    if not ret:
        break
    img2 = bgr_uint8_to_chw_torch(bgr)
    img1_p, img2_p = preprocess(img1, img2)
    img1_p = img1_p[None, :, :, :]
    img2_p = img2_p[None, :, :, :]
    with torch.no_grad():
        list_of_flows = model(img1_p.to(device), img2_p.to(device))
    predicted_flow = list_of_flows[-1][0]
    flow_img = flow_to_image(predicted_flow).to("cpu")
    pair_idx += 1
    if total_pairs is not None:
        print(f"Done {pair_idx}/{total_pairs}")
    else:
        print(f"Done frame pair {pair_idx}")
    result_frames.append(flow_img.permute(1, 2, 0).numpy())
    img1 = img2

cap.release()

if not result_frames:
    raise SystemExit("No frame pairs processed")

h, w = result_frames[0].shape[:2]
fourcc = cv2.VideoWriter_fourcc(*"XVID")
writer = cv2.VideoWriter(out_path, fourcc, float(fps), (w, h))
if not writer.isOpened():
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = os.path.join(out_dir, "RAFT.mp4")
    writer = cv2.VideoWriter(out_path, fourcc, float(fps), (w, h))
if not writer.isOpened():
    raise SystemExit("Could not open VideoWriter; try another fourcc or path")

for rgb in result_frames:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    writer.write(bgr)
writer.release()
print(f"done -> {out_path}")
