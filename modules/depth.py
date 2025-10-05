# depth.py
import torch
import cv2
import numpy as np
from typing import Tuple, Any
import sys
sys.path.append("/kaggle/working/new_model")

# ---------- Monodepth2 ----------
# ---------- Monodepth2 ----------

from modules.monodepth2 import network

# ---------- Loader ----------
def load_monodepth2(model_dir="/kaggle/working/lidar_monocular_depth/weights/monodepth2"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load encoder
    encoder = network.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(f"{model_dir}/encoder.pth", map_location=device)
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device).eval()

    # Load depth decoder
    depth_decoder = network.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))
    loaded_dict = torch.load(f"{model_dir}/depth.pth", map_location=device)
    depth_decoder.load_state_dict(loaded_dict)
    depth_decoder.to(device).eval()

    # Dummy processor for interface consistency
    processor = None

    return (encoder, depth_decoder, processor, device)


# ---------- Runner ----------
def run_monodepth2(img: np.ndarray, encoder, depth_decoder, processor, device) -> np.ndarray:
    """
    img: HxWx3 BGR image
    returns: HxW depth map
    """
    h_orig, w_orig = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (1024, 320))  # model training resolution

    input_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    input_tensor = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        features = encoder(input_tensor)
        outputs = depth_decoder(features)
        disp = outputs[("disp", 0)]
        # Resize back to original image size
        disp_resized = torch.nn.functional.interpolate(
            disp, size=(h_orig, w_orig), mode="bilinear", align_corners=False
        )

    depth_map = disp_resized.squeeze().cpu().numpy()
    return depth_map

  
#----Depth_Anything(V2)------

# def load_depth_anything_v2(model_id: str = "depth-anything/Depth-Anything-V2-Base"):
#     from transformers import AutoImageProcessor, AutoModelForDepthEstimation ,AutoFeatureExtractor

#     model_id = "depth-anything/Depth-Anything-V2-Base"
#     model = AutoFeatureExtractor.from_pretrained(model_id)

#     # model = AutoModelForDepthEstimation.from_pretrained(model_id)

#     processor = AutoImageProcessor.from_pretrained(model_id)
#     # model = AutoModelForDepthEstimation.from_pretrained(model_id)
    
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     model.to(device).eval()
#     return model, processor, device


# def run_depth_anything_v2(img: np.ndarray, model, processor, device: str) -> np.ndarray:
#     h, w = img.shape[:2]
#     rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     inputs = processor(images=rgb, return_tensors="pt").to(device)
#     with torch.no_grad():
#         outputs = model(**inputs)
#         pred = outputs.predicted_depth  # [B, H', W']

#         pred = torch.nn.functional.interpolate(
#             pred.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False
#         ).squeeze(1)

#     depth_map = pred.squeeze().cpu().numpy()
                                     #     return depth_map
from modules.depth_anything_v2.dpt import DepthAnythingV2
from transformers import DPTImageProcessor
import math

# ---------- Helper: resize/pad to multiple of patch size ----------
def resize_to_multiple(img: np.ndarray, multiple: int = 14):
    h, w = img.shape[:2]
    new_h = math.ceil(h / multiple) * multiple
    new_w = math.ceil(w / multiple) * multiple
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return img_resized

# ---------- Load model ----------
def load_depth_anything_v2(
    checkpoint_path: str = "/kaggle/working/lidar_monocular_depth/checkpoints/depth_anything_v2_vitb.pth"
):
    # Initialize model
    model = DepthAnythingV2(
        encoder='vitb',
        features=128,
        out_channels=[96, 192, 384, 768]
    )
    # Load weights
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # Processor (DPT-style)
    processor = DPTImageProcessor(
        do_resize=True,
        size={"height": 518, "width": 518},
        keep_aspect_ratio=True,
        do_rescale=True,
        rescale_factor=1/255.0,
        do_normalize=True,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
        ensure_multiple_of=14
    )

    return model, processor, device

# ---------- Run inference ----------

def run_depth_anything_v2(img: np.ndarray, model, processor, device: str) -> np.ndarray:
    h_orig, w_orig = img.shape[:2]
    
    # Convert BGR->RGB
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Make height and width multiples of 14 (for DepthAnythingV2 patching)
    patch_size = 14
    new_h = (h_orig // patch_size) * patch_size
    new_w = (w_orig // patch_size) * patch_size
    rgb_resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Convert to tensor: [B, C, H, W]
    tensor_input = processor(images=rgb_resized, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        pred = model(tensor_input)  # DepthAnythingV2 expects [B, C, H, W]
        
        # If output is [B, 1, H, W] or [B, H, W], ensure 4D for interpolate
        if pred.ndim == 3:  # [B, H, W]
            pred = pred.unsqueeze(1)  # -> [B, 1, H, W]
        
        # Resize back to original image
        pred = torch.nn.functional.interpolate(
            pred, size=(h_orig, w_orig), mode="bilinear", align_corners=False
        )
    
    # Return as 2D numpy array
    depth_map = pred[0,0].cpu().numpy()  # take first batch and channel
    depth_map = np.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0)
    depth_map = np.clip(depth_map, 0, np.percentile(depth_map, 99))  # remove extreme outliers

    return depth_map


# def run_depth_anything_v2(img: np.ndarray, model, processor, device: str) -> np.ndarray:
#     h_orig, w_orig = img.shape[:2]
    
#     # Convert to RGB
#     rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     # Make height and width multiples of 14
#     patch_size = 14
#     new_h = (h_orig // patch_size) * patch_size
#     new_w = (w_orig // patch_size) * patch_size
#     rgb_resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

#     # Preprocess using processor -> tensor
#     tensor_input = processor(images=rgb_resized, return_tensors="pt").pixel_values.to(device)

#     # Forward pass
#     with torch.no_grad():
#         pred = model(tensor_input)  # Pass tensor directly
#         # Upsample to resized image
#         pred = torch.nn.functional.interpolate(
#             pred.unsqueeze(1), size=(new_h, new_w), mode="bilinear", align_corners=False
#         ).squeeze(1)

#         # Resize back to original image size
#         pred = torch.nn.functional.interpolate(
#             pred.unsqueeze(0), size=(h_orig, w_orig), mode="bilinear", align_corners=False
#         ).squeeze(0)

#     depth_map = pred.cpu().numpy()  # final shape [H,W]
#     return depth_map


# def load_depth_anything_v2(
#     checkpoint_path: str = "/kaggle/working/lidar_monocular_depth/checkpoints/depth_anything_v2_vitb.pth"
# ):
#     # Initialize model
#     model = DepthAnythingV2(
#         encoder='vitb',
#         features=128,
#         out_channels=[96, 192, 384, 768]
#     )
#     # Load weights
#     state_dict = torch.load(checkpoint_path, map_location='cpu')
#     model.load_state_dict(state_dict)
#     model.eval()
    
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     model.to(device)

#     # Processor (DPT-style preprocessing)
#     processor = DPTImageProcessor(
#         do_resize=True,
#         size={"height": 518, "width": 518},
#         keep_aspect_ratio=True,
#         do_rescale=True,
#         rescale_factor=1/255.0,
#         do_normalize=True,
#         image_mean=[0.485, 0.456, 0.406],
#         image_std=[0.229, 0.224, 0.225]
#     )

#     return model, processor, device



# def run_depth_anything_v2(img: np.ndarray, model, processor, device: str) -> np.ndarray:
#     h, w = img.shape[:2]
#     rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     # Get tensor from processor
#     inputs = processor(images=rgb, return_tensors="pt")
#     tensor_input = inputs["pixel_values"].to(device)

#     with torch.no_grad():
#         pred = model(tensor_input)  # pass tensor directly
#         pred = torch.nn.functional.interpolate(
#             pred.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False
#         ).squeeze(1)

#     depth_map = pred.squeeze().cpu().numpy()
#     return depth_map

# # ---------- MiDaS ----------

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
        encoder, depth_decoder, proc, device = load_monodepth2()
        runner = lambda img: run_monodepth2(img, encoder, depth_decoder, proc, device)
        return runner, device, "monodepth2"
    else:
        raise ValueError(f"Unknown backend '{backend}'. Use 'zoe' or 'midas'.")
