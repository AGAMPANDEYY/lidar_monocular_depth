# main_project/baselines/vehicle_distance_baseline.py
import numpy as np
from baselines.baseline_framework import BaselineModel
from vehicle_distance.vehicle_distance import VehicleDistanceBaseline as VehicleDistanceModel
from PIL import Image

class VehicleDistanceBaseline(BaselineModel):
    def __init__(self):
        super().__init__()
        self.model = VehicleDistanceModel()  # YOLO-based distance estimator

    def load_model(self):
        return True

    def predict_depth(self, img):
        img_np = np.array(img)
        distance = self.model.estimate_distance(img_np)
        if distance is not None:
            return np.full((img_np.shape[0], img_np.shape[1]), distance, dtype=np.float32)
        else:
            return np.zeros((img_np.shape[0], img_np.shape[1]), dtype=np.float32)
