from ultralytics import YOLO
import cv2
import numpy as np

CLASSES = ['Auto', 'Car', 'HV', 'LCV', 'MTW', 'Others']  # Original classes from best-e150 model

def load_yolo_model(weights_path):
    return YOLO(weights_path)

def run_obstacle_detection(img, model, conf_thresh=0.2):  # Lowered threshold
    img = np.array(img)
    results = model(img)
    pred_bboxes = []
    
    print("\nYOLO Detection Results:")
    for r in results:
        print(f"Found {len(r.boxes)} potential objects")
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]
            cls = int(box.cls[0])
            print(f"- Class: {CLASSES[cls]}, Confidence: {conf:.2f}")
            if conf > conf_thresh:
                pred_bboxes.append([x1, y1, x2, y2, cls, conf])
                print(f"  Added to results (above threshold {conf_thresh})")
            else:
                print(f"  Skipped (below threshold {conf_thresh})")
    
    print(f"Final detection count: {len(pred_bboxes)}")
    return np.array(pred_bboxes)
