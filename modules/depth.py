# depth.py
import torch
import cv2
import numpy as np
from typing import Tuple, Any
# ---------- Monodepth2 ----------
def load_monodepth2(model_dir="weights/monodepth2/mono+stereo_640x192"):
    import torch
    from monodepth2.layers import disp_to_depth
    from monodepth2.networks import ResnetEncoder, DepthDecoder

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- Load encoder ---
    encoder = ResnetEncoder(18, False)
    encoder_path = f"{model_dir}/encoder.pth"
    loaded_dict_enc = torch.load(encoder_path, map_location=device)
    # extract height and width from encoder
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    # --- Load depth decoder ---
    depth_decoder = DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))
    depth_decoder_path = f"{model_dir}/depth.pth"
    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)
    depth_decoder.to(device)
    depth_decoder.eval()

    return encoder, depth_decoder, device, feed_height, feed_width
def run_monodepth2(img: np.ndarray, encoder, depth_decoder, device: str, feed_height: int, feed_width: int) -> np.ndarray:
    import cv2, torch
    import numpy as np

    # Resize to training resolution
    input_image = cv2.resize(img, (feed_width, feed_height))
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB) / 255.0
    input_image = torch.from_numpy(input_image).float().permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        features = encoder(input_image)
        outputs = depth_decoder(features)
        disp = outputs[("disp", 0)]
        disp_resized = torch.nn.functional.interpolate(
            disp, (img.shape[0], img.shape[1]), mode="bilinear", align_corners=False
        )

    depth_map = disp_resized.squeeze().cpu().numpy()
    return depth_map    
#----Depth_Anything(V2)------

def load_depth_anything_v2(model_id: str = "depth-anything/Depth-Anything-V2-Base"):
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
    elif backend == "depth-anything-v2":
        model, proc, device = load_depth_anything_v2()
        runner = lambda img: run_depth_anything_v2(img, model, proc, device)
        return runner, device, "depth-anything-v2" 
    elif backend == "monodepth2":
        encoder, depth_decoder, device, H, W = load_monodepth2()
        runner = lambda img: run_monodepth2(img, encoder, depth_decoder, device, H, W)
        return runner, device, "monodepth2"
    else:
        raise ValueError(f"Unknown backend '{backend}'. Use 'zoe' or 'midas'.")
