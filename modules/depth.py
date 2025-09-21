import torch
import cv2
import numpy as np

def load_midas_model():
    midas = torch.hub.load('intel-isl/MiDaS', 'DPT_Hybrid')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    midas.to(device)
    midas.eval()
    transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
    transform = transforms.dpt_transform
    return midas, transform, device

def run_midas_depth(img, midas, transform, device):
    input_batch = transform(img).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth_map = prediction.cpu().numpy()
    return depth_map

def fuse_depth(lidar_depth, mono_depth, mask):
    fused = mono_depth.copy()
    fused[mask] = lidar_depth[mask]
    return fused
