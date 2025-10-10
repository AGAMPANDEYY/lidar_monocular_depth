"""
Run baseline models and collect actual performance metrics.
Includes added support for VehicleDistanceBaseline.
"""
import os
import time
import torch
import numpy as np
from PIL import Image
import sys
sys.path.append('baselines/monodepth2')  # Add MonoDepth2 to path

from models import VehicleDistanceBaseline  # ✅ Added import

class BaselineRunner:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}

    # ===================== MonoDepth2 ===================== #
    def load_monodepth2(self):
        try:
            import monodepth2.networks as networks
            model_path = "baselines/weights/monodepth2/mono_1024x320/encoder.pth"

            encoder = networks.ResnetEncoder(18, False)
            loaded_dict_enc = torch.load(model_path, map_location=self.device)
            encoder.load_state_dict(loaded_dict_enc)
            encoder.to(self.device)
            encoder.eval()

            self.models['monodepth2'] = encoder
            return True
        except Exception as e:
            print(f"Error loading MonoDepth2: {str(e)}")
            return False

    def run_monodepth2(self, img):
        if 'monodepth2' not in self.models:
            if not self.load_monodepth2():
                return None

        model = self.models['monodepth2']
        input_img = self.preprocess_monodepth2(img)

        start_time = time.perf_counter()
        with torch.no_grad():
            output = model(input_img)
        end_time = time.perf_counter()

        if torch.cuda.is_available():
            memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        else:
            memory_mb = 0

        return {
            'depth': output['disp', 0].cpu().numpy(),
            'inference_time_ms': (end_time - start_time) * 1000,
            'memory_usage_mb': memory_mb
        }

    def preprocess_monodepth2(self, img):
        img = img.resize((1024, 320))
        img = torch.from_numpy(np.array(img)).float() / 255.0
        img = img.to(self.device)
        return img

    # ===================== Vehicle Distance ===================== #
    def load_vehicle_distance(self):
        try:
            self.models['vehicle_distance'] = VehicleDistanceBaseline()
            return True
        except Exception as e:
            print(f"Error loading VehicleDistanceBaseline: {str(e)}")
            return False

    def run_vehicle_distance(self, img):
        if 'vehicle_distance' not in self.models:
            if not self.load_vehicle_distance():
                return None

        model = self.models['vehicle_distance']

        start_time = time.perf_counter()
        distance = model.predict_distance(img)
        end_time = time.perf_counter()

        if torch.cuda.is_available():
            memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        else:
            memory_mb = 0

        return {
            'avg_distance_m': distance if distance is not None else -1,
            'inference_time_ms': (end_time - start_time) * 1000,
            'memory_usage_mb': memory_mb
        }

# ===================== Comparison Runner ===================== #
def run_comparison(img_path):
    runner = BaselineRunner()
    img = Image.open(img_path).convert('RGB')

    results = {}

    print("Running MonoDepth2...")
    mono_result = runner.run_monodepth2(img)
    if mono_result is not None:
        results['MonoDepth2'] = mono_result

    print("Running Vehicle Distance Baseline...")
    vd_result = runner.run_vehicle_distance(img)
    if vd_result is not None:
        results['VehicleDistance'] = vd_result

    return results

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', required=True, help='Path to input image')
    args = parser.parse_args()

    results = run_comparison(args.img_path)

    print("\nResults:")
    for model, metrics in results.items():
        print(f"\n{model}:")
        for key, val in metrics.items():
            if isinstance(val, (int, float)):
                print(f"  {key}: {val:.2f}")
            else:
                print(f"  {key}: {val}")
