"""
Baseline models for comparison with our fusion approach.
Includes implementations/wrappers for:
- MonoDepth2
- DORN
- Pure LiDAR processing
- SOTA Fusion (DeepLiDAR)
- VehicleDistanceBaseline (YOLO-based distance)
"""

import os
import time
import torch
import numpy as np
from PIL import Image

class BaselineModel:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_model(self):
        raise NotImplementedError

    def preprocess(self, img):
        raise NotImplementedError

    def postprocess(self, output):
        raise NotImplementedError

    def measure_inference_time(self, img):
        times = []
        for _ in range(10):
            start = time.perf_counter()
            with torch.no_grad():
                _ = self.predict_depth(img)
            times.append(time.perf_counter() - start)
        return np.mean(times) * 1000

    def measure_memory_usage(self):
        import psutil
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)

class MonoDepth2Baseline(BaselineModel):
    def __init__(self):
        super().__init__()
        self.model = None

    def load_model(self):
        try:
            import monodepth2
            model_path = "pretrained/mono_1024x320"
            self.model = monodepth2.load_model(model_path)
            self.model.to(self.device)
            self.model.eval()
            return True
        except ImportError:
            print("MonoDepth2 not installed. Install from: https://github.com/nianticlabs/monodepth2")
            return False

    def predict_depth(self, img):
        if self.model is None:
            if not self.load_model():
                return None
        input_img = self.preprocess(img)
        with torch.no_grad():
            output = self.model(input_img)
        depth = self.postprocess(output)
        return depth

class DORNBaseline(BaselineModel):
    def __init__(self):
        super().__init__()
        self.model = None

    def load_model(self):
        try:
            return True
        except:
            print("DORN model loading failed")
            return False

class PureLiDARBaseline(BaselineModel):
    def __init__(self):
        super().__init__()

    def process_lidar(self, lidar_points, img_shape):
        pass

class DeepLiDARBaseline(BaselineModel):
    def __init__(self):
        super().__init__()
        self.model = None

    def load_model(self):
        try:
            return True
        except:
            print("DeepLiDAR model loading failed")
            return False

class VehicleDistanceBaseline(BaselineModel):
    """Baseline model for Vehicle Distance Measurement using YOLO and geometry."""
    def __init__(self):
        super().__init__()
        from ultralytics import YOLO
        self.model = YOLO("yolo12x.pt")
        self.license_plate_model = YOLO("license-plate.pt")

        self.FOCAL_LENGTH = 500
        self.REAL_VEHICLE_HEIGHTS = {2: 1.55, 3: 1.2, 5: 3.0, 7: 2.5}
        self.TARGET_CLASSES = [2, 3, 5, 7]
        self.CONFIDENCE_THRESHOLD = 0.7

    def load_model(self):
        return True

    def preprocess(self, img):
        import cv2
        if isinstance(img, np.ndarray):
            return img
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    def predict_distance(self, img):
        frame = self.preprocess(img)
        results = self.model(frame, classes=self.TARGET_CLASSES, verbose=False)
        distances = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if conf < self.CONFIDENCE_THRESHOLD or cls not in self.REAL_VEHICLE_HEIGHTS:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                bbox_height = y2 - y1
                if bbox_height <= 0:
                    continue

                distance = (self.REAL_VEHICLE_HEIGHTS[cls] * self.FOCAL_LENGTH) / bbox_height
                distances.append(distance)

        return np.mean(distances) if distances else None

def run_baseline_comparison(img_path, lidar_path):
    results = {}
    img = Image.open(img_path).convert('RGB')

    baselines = [
        ('MonoDepth2', MonoDepth2Baseline()),
        ('DORN', DORNBaseline()),
        ('PureLiDAR', PureLiDARBaseline()),
        ('DeepLiDAR', DeepLiDARBaseline()),
        ('VehicleDistance', VehicleDistanceBaseline())
    ]

    for name, model in baselines:
        try:
            inf_time = model.measure_inference_time(img)
            mem_usage = model.measure_memory_usage()
            results[name] = {
                'inference_time_ms': inf_time,
                'memory_usage_mb': mem_usage
            }
        except Exception as e:
            print(f"Error running {name}: {str(e)}")
            results[name] = {
                'inference_time_ms': float('nan'),
                'memory_usage_mb': float('nan')
            }

    return results
