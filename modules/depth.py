# depth.py
import torch
import cv2
import numpy as np
from typing import Tuple, Any
#----Depth_Anything(V2)------

def load_depth_anything_v2(model_id: str = "depth-anything/Depth-Anything-V2-Small"):
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation


    # model = AutoModelForDepthEstimation.from_pretrained(model_id)

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device).eval()
    return model, processor, device


def run_depth_anything_v2(img: np.ndarray, model, processor, device: str) -> np.ndarray:
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    inputs = processor(images=rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        pred = outputs.predicted_depth  # [B, H', W']

        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False
        ).squeeze(1)

    depth_map = pred.squeeze().cpu().numpy()
    return depth_map

# ---------- MiDaS ----------
def load_midas_model():
    midas = torch.hub.load('intel-isl/MiDaS', 'DPT_Hybrid')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    midas.to(device).eval()
    transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
    transform = transforms.dpt_transform
    return midas, transform, device

def run_midas_depth(img: np.ndarray, midas: Any, transform: Any, device: str) -> np.ndarray:
    input_batch = transform(img).to(device)
    with torch.no_grad():
        pred = midas(input_batch)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1), size=img.shape[:2], mode="bicubic", align_corners=False
        ).squeeze(1)
    depth_map = pred.squeeze().cpu().numpy()
    return depth_map

# ---------- ZoeDepth (Hugging Face Transformers) ----------
# Model card: Intel/zoedepth-nyu-kitti; classes: ZoeDepthForDepthEstimation + AutoImageProcessor
# (works on CPU; .to(device) for GPU)  [HF docs confirm API]
def load_zoe_model(model_id: str = "Intel/zoedepth-nyu-kitti"):
    from transformers import AutoImageProcessor, ZoeDepthForDepthEstimation  # pip install transformers>=4.40
    processor = AutoImageProcessor.from_pretrained(model_id,use_fast=True)
    model = ZoeDepthForDepthEstimation.from_pretrained(model_id)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device).eval()
    return model, processor, device

def run_zoe_depth(img: np.ndarray, model: Any, processor: Any, device: str) -> np.ndarray:
    # img: BGR np.ndarray (as from cv2). Convert to RGB for HF processors.
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    inputs = processor(images=rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        # outputs.predicted_depth: [B, H', W'] tensor
        pred = outputs.predicted_depth  # (B, H', W')
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False
        ).squeeze(1)
    depth_map = pred.squeeze().cpu().numpy()
    return depth_map

# ---------- Simple fusion (unchanged) ----------
def fuse_depth(lidar_depth: np.ndarray, mono_depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    fused = mono_depth.copy()
    fused[mask] = lidar_depth[mask]
    return fused


# --- add to modules/depth.py ---
import onnxruntime as ort

def load_fastdepth_onnx(model_path="weights/fastdepth.onnx"):
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    # Heuristics; many FastDepth ONNX ports use these names/shapes:
    in_name  = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    input_shape = sess.get_inputs()[0].shape  # e.g. [1,3,224,224] or [1,224,224,3]
    nchw = (len(input_shape)==4 and input_shape[1] in (1,3))  # NCHW if channels at dim=1
    target_hw = (input_shape[2], input_shape[3]) if nchw else (input_shape[1], input_shape[2])
    return sess, in_name, out_name, nchw, target_hw

def run_fastdepth_onnx(img_bgr, ort_sess, in_name, out_name, nchw, target_hw):
    import cv2, numpy as np
    h, w = target_hw
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    if nchw:
        inp = resized.transpose(2,0,1)[None, ...]           # [1,3,H,W]
    else:
        inp = resized[None, ...]                             # [1,H,W,3]
    pred = ort_sess.run([out_name], {in_name: inp})[0]       # [1,1,H,W] or [1,H,W,1]
    pred = pred.squeeze()
    if pred.ndim == 3:  # [1,H,W] -> [H,W]
        pred = pred[0]
    return pred.astype(np.float32)


# ---------- Optional: small factory so main.py can choose model ----------
def load_depth_backend(backend: str = "zoe"):
    backend = backend.lower()
    if backend == "zoe":
        model, proc, device = load_zoe_model()
        runner = lambda img: run_zoe_depth(img, model, proc, device)
        return runner, device, "zoe"
    elif backend == "midas":
        model, trans, device = load_midas_model()
        runner = lambda img: run_midas_depth(img, model, trans, device)
        return runner, device, "midas"
    elif backend == "fastdepth":
        sess, in_name, out_name, nchw, target_hw = load_fastdepth_onnx()
        runner = lambda img: run_fastdepth_onnx(img, sess, in_name, out_name, nchw, target_hw)
        device = "cpu"
        return runner, device, "fastdepth"
    elif backend == ""    
    else:
        raise ValueError(f"Unknown backend '{backend}'. Use 'zoe' or 'midas'.")
