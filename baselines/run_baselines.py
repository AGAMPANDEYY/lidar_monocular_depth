"""
Run baseline models and collect actual performance metrics.
"""
import os
import time
import torch
import numpy as np
from PIL import Image
import sys
# sys.path.append('baselines/monodepth')  # Add MonoDepth2 to path
sys.path.append('r/kaggle/working/baselines')  # Add MonoDepth2 to path

# class BaselineRunner:
#     def __init__(self):
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.models = {}
        
#     def load_monodepth2(self):
#         """Load MonoDepth2 model"""
#         try:
#             import monodepth2.networks as networks
#             model_path = "baselines/weights/monodepth2/mono_1024x320/encoder.pth"
            
#             encoder = networks.ResnetEncoder(18, False)
#             loaded_dict_enc = torch.load(model_path, map_location=self.device)
#             encoder.load_state_dict(loaded_dict_enc)
#             encoder.to(self.device)
#             encoder.eval()
            
#             self.models['monodepth2'] = encoder
#             return True
#         except Exception as e:
#             print(f"Error loading MonoDepth2: {str(e)}")
#             return False
            
#     def run_monodepth2(self, img):
#         """Run MonoDepth2 inference"""
#         if 'monodepth2' not in self.models:
#             if not self.load_monodepth2():
#                 return None
                
#         model = self.models['monodepth2']
        
#         # Preprocess
#         input_img = self.preprocess_monodepth2(img)
        
#         # Time the inference
#         start_time = time.perf_counter()
#         with torch.no_grad():
#             output = model(input_img)
#         end_time = time.perf_counter()
        
#         # Get memory usage
#         if torch.cuda.is_available():
#             memory_mb = torch.cuda.max_memory_allocated() / (1024*1024)
#         else:
#             memory_mb = 0
            
#         return {
#             'depth': output['disp', 0].cpu().numpy(),
#             'inference_time_ms': (end_time - start_time) * 1000,
#             'memory_usage_mb': memory_mb
#         }
    
#     def preprocess_monodepth2(self, img):
#         """Preprocess image for MonoDepth2"""
#         # Resize to model input size
#         img = img.resize((1024, 320))
#         # Convert to tensor and normalize
#         img = torch.from_numpy(np.array(img)).float() / 255.0
#         img = img.to(self.device)
#         return img

# def run_comparison(img_path):
#     """Run all baselines on a single image"""
#     runner = BaselineRunner()
#     img = Image.open(img_path).convert('RGB')
    
#     results = {}
    
#     # Run MonoDepth2
#     print("Running MonoDepth2...")
#     mono_result = runner.run_monodepth2(img)
#     if mono_result is not None:
#         results['MonoDepth2'] = mono_result
    
#     return results

# if __name__ == '__main__':
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--img_path', required=True, help='Path to input image')
#     args = parser.parse_args()
    
#     results = run_comparison(args.img_path)
    
#     print("\nResults:")
#     for model, metrics in results.items():
#         print(f"\n{model}:")
#         print(f"  Inference time: {metrics['inference_time_ms']:.2f} ms")
#         print(f"  Memory usage:   {metrics['memory_usage_mb']:.1f} MB")

# import torch
# import numpy as np
# from PIL import Image
# from ultralytics import YOLO
# import time
# import cv2

# class BaselineRunner:
#     def __init__(self):
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.models = {}

#     def load_vehicle_distance_model(self, model_path="r/kaggle/working/baselines/weights/yolo8s.pt"):
#         """Load YOLO model for Vehicle Distance Measurement"""
#         try:
#             self.models['VehicleDistance'] = YOLO(model_path)
#             return True
#         except Exception as e:
#             print(f"Failed to load VehicleDistance model: {e}")
#             return False

#     def run_vehicle_distance(self, img):
#         """Run VehicleDistance inference on a single image"""
#         if 'VehicleDistance' not in self.models:
#             if not self.load_vehicle_distance_model():
#                 return None

#         model = self.models['VehicleDistance']

#         # Convert PIL image to OpenCV format
#         img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

#         # Start inference timing
#         start_time = time.perf_counter()
#         results = model(img_cv, classes=[2, 3, 5, 7], verbose=False)  # Car, Motorcycle, Bus, Truck
#         end_time = time.perf_counter()

#         # Calculate distances
#         distances = []
#         FOCAL_LENGTH = 500
#         REAL_VEHICLE_HEIGHTS = {2: 1.55, 3: 1.2, 5: 3.0, 7: 2.5}

#         for r in results:
#             if r.boxes is not None:
#                 for box in r.boxes:
#                     bbox = box.xyxy[0].cpu().numpy()
#                     class_id = int(box.cls[0])
#                     x1, y1, x2, y2 = bbox
#                     bbox_height = max(y2 - y1, 1e-3)
#                     distance = (REAL_VEHICLE_HEIGHTS[class_id] * FOCAL_LENGTH) / bbox_height
#                     distances.append(distance)

#         # Memory usage
#         memory_mb = torch.cuda.max_memory_allocated() / (1024*1024) if torch.cuda.is_available() else 0

#         return {
#             'num_vehicles': len(distances),
#             'distances_m': distances,
#             'inference_time_ms': (end_time - start_time) * 1000,
#             'memory_usage_mb': memory_mb
#         }


# def run_comparison(img_path):
#     """Run VehicleDistance baseline on a single image"""
#     runner = BaselineRunner()
#     img = Image.open(img_path).convert('RGB')

#     results = {}

#     print("Running VehicleDistance baseline...")
#     vehicle_result = runner.run_vehicle_distance(img)
#     if vehicle_result is not None:
#         results['VehicleDistance'] = vehicle_result

#     return results


# if __name__ == '__main__':
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--img_path', required=True, help='Path to input image')
#     parser.add_argument('--yolo_weights', default="baselines/weights/yolo12x.pt", help='Path to YOLO weights')
#     args = parser.parse_args()

#     results = run_comparison(args.img_path)

#     print("\nResults:")
#     for model, metrics in results.items():
#         print(f"\n{model}:")
#         print(f"  Number of vehicles: {metrics['num_vehicles']}")
#         print(f"  Distances (m): {metrics['distances_m']}")
#         print(f"  Inference time (ms): {metrics['inference_time_ms']:.2f}")
#         print(f"  Memory usage (MB): {metrics['memory_usage_mb']:.2f}")
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import torch
import time

class BaselineRunner:
    def __init__(self, yolo_weights="yolo12x.pt", license_plate_weights="license-plate.pt"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.yolo_weights = yolo_weights
        self.license_plate_weights = license_plate_weights
        self.load_vehicle_distance_model()

        # Configs
        self.FOCAL_LENGTH = 500
        self.TARGET_CLASSES = [2, 3, 5, 7]
        self.REAL_VEHICLE_HEIGHTS = {2: 1.55, 3: 1.2, 5: 3.0, 7: 2.5}
        self.CONFIDENCE_THRESHOLD = 0.7
        self.MAX_DISPLAY_DISTANCE = 15
        self.WARNING_DISTANCES = {'LEFT': 1, 'MAIN': 2, 'RIGHT': 1}
        self.ROI_ZONES = {
            'LEFT': np.array([[240, 600], [925, 550], [312, 1100], [100, 1100]], dtype=np.int32),
            'MAIN': np.array([[925, 550], [1025, 550], [1712, 1100], [312, 1100]], dtype=np.int32),
            'RIGHT': np.array([[1025, 550], [1802, 600], [1942, 1100], [1712, 1100]], dtype=np.int32)
        }
        self.OPTICAL_CENTERS = {'LEFT': (156, 1050), 'MAIN': (1000, 1020), 'RIGHT': (1868, 1050)}

    def load_vehicle_distance_model(self):
        try:
            self.models['VehicleDistance'] = YOLO(self.yolo_weights)
            self.models['LicensePlate'] = YOLO(self.license_plate_weights)
            return True
        except Exception as e:
            print(f"Failed to load models: {e}")
            return False

    # ROI helper
    def is_point_in_roi(self, point, roi_coordinates):
        return cv2.pointPolygonTest(roi_coordinates, point, False) >= 0

    def get_vehicle_roi_zone(self, center_point):
        for zone_name, roi_coords in self.ROI_ZONES.items():
            if self.is_point_in_roi(center_point, roi_coords):
                return zone_name
        return None

    def calculate_distance(self, bbox, class_id, zone_name):
        if class_id not in self.REAL_VEHICLE_HEIGHTS or zone_name not in self.OPTICAL_CENTERS:
            return 0
        x1, y1, x2, y2 = bbox
        bbox_height = max(y2 - y1, 1e-3)
        vehicle_center_x, vehicle_center_y = (x1 + x2) / 2, (y1 + y2) / 2
        optical_center_x, optical_center_y = self.OPTICAL_CENTERS[zone_name]
        displacement = np.sqrt((vehicle_center_x - optical_center_x) ** 2 + (vehicle_center_y - optical_center_y) ** 2)
        distance = (self.REAL_VEHICLE_HEIGHTS[class_id] * self.FOCAL_LENGTH) / bbox_height
        return distance * (1.0 + displacement * 0.0001)

    def run_vehicle_distance(self, img):
        """Run Vehicle Distance inference on a single image"""
        if 'VehicleDistance' not in self.models:
            if not self.load_vehicle_distance_model():
                return None

        model = self.models['VehicleDistance']

        # Convert PIL image to OpenCV
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        start_time = time.perf_counter()
        results = model(img_cv, classes=self.TARGET_CLASSES, verbose=False)
        end_time = time.perf_counter()

        distances = []
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    bbox = box.xyxy[0].cpu().numpy()
                    class_id = int(box.cls[0])
                    center_point = ((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2)
                    zone_name = self.get_vehicle_roi_zone(center_point)
                    if zone_name is None:
                        continue
                    distance = self.calculate_distance(bbox, class_id, zone_name)
                    distances.append(distance)

        memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0

        return {
            'num_vehicles': len(distances),
            'distances_m': distances,
            'inference_time_ms': (end_time - start_time) * 1000,
            'memory_usage_mb': memory_mb
        }

# Example usage
if __name__ == "__main__":
    import argparse
    runner = BaselineRunner()
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', required=True)
    args = parser.parse_args()
    img = Image.open(args.img_path).convert('RGB')
    results = runner.run_vehicle_distance(img)
    print(results)
