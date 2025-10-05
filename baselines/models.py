"""
Baseline models for comparison with our fusion approach.
Includes implementations/wrappers for:
- MonoDepth2
- DORN
- Pure LiDAR processing
- SOTA Fusion (DeepLiDAR)
"""

import os
import time
import torch
import numpy as np
from PIL import Image
import sys 
sys.path.append("\kaggle\working\new_model")
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
        for _ in range(10):  # Run 10 times for average
            start = time.perf_counter()
            with torch.no_grad():
                _ = self.predict_depth(img)
            times.append(time.perf_counter() - start)
        return np.mean(times) * 1000  # Convert to ms
        
    def measure_memory_usage(self):
        import psutil
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)  # Convert to MB

class MonoDepth2Baseline(BaselineModel):
    def __init__(self):
        super().__init__()
        self.model = None
        
    def load_model(self):
        """Load MonoDepth2 model"""
        try:
            import monodepth2
            model_path = "pretrained/mono_1024x320"  # You'll need to download this
            self.model = monodepth2.load_model(model_path)
            self.model.to(self.device)
            self.model.eval()
            return True
        except ImportError:
            print("MonoDepth2 not installed. Install from: https://github.com/nianticlabs/monodepth2")
            return False

    def predict_depth(self, img):
        """Predict depth using MonoDepth2"""
        if self.model is None:
            if not self.load_model():
                return None
                
        # Preprocess
        input_img = self.preprocess(img)
        
        # Inference
        with torch.no_grad():
            output = self.model(input_img)
            
        # Postprocess
        depth = self.postprocess(output)
        return depth

class DORNBaseline(BaselineModel):
    def __init__(self):
        super().__init__()
        self.model = None
        
    def load_model(self):
        """Load DORN model"""
        try:
            # Implementation for loading DORN
            # You'll need to add proper model loading code
            return True
        except:
            print("DORN model loading failed")
            return False

class PureLiDARBaseline(BaselineModel):
    def __init__(self):
        super().__init__()
        
    def process_lidar(self, lidar_points, img_shape):
        """Process LiDAR points to create depth map"""
        # Implement LiDAR-only processing
        # This should match your LiDAR processing pipeline
        pass

class DeepLiDARBaseline(BaselineModel):
    def __init__(self):
        super().__init__()
        self.model = None
        
    def load_model(self):
        """Load DeepLiDAR fusion model"""
        try:
            # Implementation for loading DeepLiDAR
            # You'll need to add proper model loading code
            return True
        except:
            print("DeepLiDAR model loading failed")
            return False

class VehicleDistanceBaseline(BaselineModel):
    def __init__(self):
        super().__init__()
        self.model = None

    def load_model(self):
        """
        Load YOLO model used in Vehicle Distance Measurement System
        (https://github.com/kemalkilicaslan/Vehicle-Distance-Measurement-System)
        """
        try:
            from ultralytics import YOLO  # Uses YOLOv5/YOLOv8
            self.model = YOLO("kaggle/working/weights/yolov5s.pt")  # or path to your trained weights
            self.model.to(self.device)
            self.model.eval()
            return True
        except Exception as e:
            print(f"Failed to load VehicleDistance model: {e}")
            return False

    def predict_depth(self, img):
        """
        Adapts Vehicle Distance Measurement approach to fit our BaselineModel API.
        Returns a 'distance map' or list of distances to detected cars.
        """
        if self.model is None:
            if not self.load_model():
                return None

        # Perform detection
        results = self.model.predict(img, verbose=False)

        # Simple distance estimation using width of bounding box
        distances = []
        for r in results:
            boxes = r.boxes.xywh.cpu().numpy()  # (x_center, y_center, w, h)
            for box in boxes:
                pixel_width = box[2]
                KNOWN_WIDTH = 1.8  # average car width in meters
                FOCAL_LENGTH = 700  # adjust based on your calibration
                distance = (KNOWN_WIDTH * FOCAL_LENGTH) / pixel_width
                distances.append(distance)

        # Return distances as a NumPy array (so the rest of your pipeline can handle it)
        return np.array(distances)

def run_baseline_comparison(img_path, lidar_path):
    """
    Run comparison with all baseline methods
    Returns: Dictionary of results including inference times and memory usage
    """
    results = {}
    
    # Load input
    img = Image.open(img_path).convert('RGB')
    
    # Test each baseline
    baselines = [
        ('MonoDepth2', MonoDepth2Baseline()),
        ('DORN', DORNBaseline()),
        ('PureLiDAR', PureLiDARBaseline()),
        ('DeepLiDAR', DeepLiDARBaseline()),
        ('VehicleDistance', VehicleDistanceBaseline())
    ]
    
    for name, model in baselines:
        try:
            # Measure inference time
            inf_time = model.measure_inference_time(img)
            
            # Measure memory usage
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
