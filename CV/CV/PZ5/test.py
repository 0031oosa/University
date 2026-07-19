import torch
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

weights = Raft_Small_Weights.DEFAULT
model = raft_small(weights=weights).eval()

# Два синтетических кадра — просто чтобы посмотреть на output
img1 = torch.randn(1, 3, 520, 960)  # (B, C, H, W), H и W кратны 8
img2 = torch.randn(1, 3, 520, 960)

with torch.no_grad():
    result = model(img1, img2)

print(type(result))
print(len(result) if isinstance(result, list) else "not a list")
print(result[-1].shape if isinstance(result, list) else result.shape)
print(res)