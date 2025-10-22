# depth.py
import torch
import cv2
import numpy as np
from typing import Tuple, Any
from PIL import Image
from pathlib import Path

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


# ---------- Depth Anything v2 (HuggingFace) ----------
def load_depth_anything_v2(encoder: str = "vits",
                                 ckpt_rel_path: str = "third_party/Depth-Anything-V2/checkpoints",
                                 input_size: int = 518):
    """
    Lightweight Depth Anything V2 loader (Small model = vits).
    Returns a callable that maps BGR→float32 depth (HxW).
    """
    import os, sys, cv2, torch, numpy as np
    repo_root = Path(__file__).resolve().parents[1]
    depth_anything_dir = repo_root / "third_party" / "Depth-Anything-V2"
    sys.path.append(str(depth_anything_dir))
    from depth_anything_v2.dpt import DepthAnythingV2

    DEVICE = ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")

    model_cfgs = {
        'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
    }

    ckpt_dir = depth_anything_dir / "checkpoints"
    ckpt_path = ckpt_dir / f"depth_anything_v2_{encoder}.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    model = DepthAnythingV2(**model_cfgs[encoder])
    model.load_state_dict(torch.load(str(ckpt_path), map_location="cpu"))
    model = model.to(DEVICE).eval()

    def runner(img_bgr: np.ndarray) -> np.ndarray:
        depth = model.infer_image(img_bgr, input_size=input_size)
        return depth.astype(np.float32)

    return runner, DEVICE, f"depthanythingv2-{encoder}"



def run_depth_anything_v2(img: np.ndarray, model: Any, processor: Any, device: str) -> np.ndarray:
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    inputs = processor(images=rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        pred = outputs.predicted_depth
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False
        ).squeeze(1)
    return pred.squeeze().cpu().numpy()


# ---------- MonoDepth2 (PyTorch) ----------
def load_monodepth2_gluoncv(model_id: str = "monodepth2_resnet18_kitti_stereo_640x192"):
    import mxnet as mx
    from mxnet.gluon.data.vision import transforms
    import gluoncv

    ctx = mx.cpu(0)
    transform = transforms.ToTensor()
    net = gluoncv.model_zoo.get_model(model_id, pretrained_base=False, ctx=ctx, pretrained=True)

    feed_width = 640
    feed_height = 192

    def runner(img_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        orig_width, orig_height = pil_img.size
        resized = pil_img.resize((feed_width, feed_height), Image.LANCZOS)
        img_nd = mx.nd.array(np.asarray(resized))
        input_tensor = transform(img_nd).expand_dims(0).as_in_context(ctx)

        outputs = net.predict(input_tensor)
        disp = outputs[("disp", 0)]
        disp_resized = mx.nd.contrib.BilinearResize2D(disp, height=orig_height, width=orig_width)
        return disp_resized.squeeze().as_in_context(mx.cpu()).asnumpy().astype(np.float32)

    return runner, "cpu", "monodepth2"


# ---------- MonoDepth2 (Local PyTorch, no Hub) ----------
def load_monodepth2_local(model_dir="models/mono+stereo_640x192"):
    import torch, torchvision.transforms as T
    import torch.nn.functional as F
    import cv2, numpy as np
    from PIL import Image
    import sys, os

    sys.path.append("third_party/monodepth2")   # only if repo lives there
    from networks.resnet_encoder import ResnetEncoder
    from networks.depth_decoder import DepthDecoder

    encoder_path = os.path.join(model_dir, "encoder.pth")
    depth_path   = os.path.join(model_dir, "depth.pth")

    device = (
        "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    # ---- Load encoder ----
    enc = ResnetEncoder(18, False)
    loaded_enc = torch.load(encoder_path, map_location=device)
    feed_h, feed_w = loaded_enc["height"], loaded_enc["width"]
    enc.load_state_dict({k: v for k, v in loaded_enc.items() if k in enc.state_dict()})
    enc.to(device).eval()

    # ---- Load decoder ----
    dec = DepthDecoder(num_ch_enc=enc.num_ch_enc, scales=range(4))
    dec.load_state_dict(torch.load(depth_path, map_location=device))
    dec.to(device).eval()

    transform = T.Compose([
        T.Resize((feed_h, feed_w), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
    ])

    def runner(img_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        H, W = pil.height, pil.width
        inp = transform(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            feats = enc(inp)
            disp = dec(feats)[("disp", 0)]
            disp = F.interpolate(disp, size=(H, W), mode="bilinear", align_corners=False)
        return disp.squeeze().cpu().numpy().astype(np.float32)

    return runner, device, "monodepth2"


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
    elif backend in ("depthanythingv2", "depth-anything-v2", "dav2", "dav2small"):
        runner, device, name = load_depth_anything_v2(
            encoder="vits",  # Small version
            input_size=518
        )
        return runner, device, name
    elif backend in ("monodepth2", "mono2"):
        runner, device, name = load_monodepth2_local()
        return runner, device, name
    else:
        raise ValueError(f"Unknown backend '{backend}'. Use one of 'zoe', 'midas', 'fastdepth', 'depthanythingv2', 'monodepth2'.")
