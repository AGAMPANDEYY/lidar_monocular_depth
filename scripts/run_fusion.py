import os
import numpy as np
import cv2

# --- User must provide these modules ---
from monocular_depth_model import estimate_depth  # User's mono depth model

from ultralytics import YOLO

def detect_objects_yolov8(image, model):
    results = model(image)
    bboxes = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            bboxes.append([x1, y1, x2, y2])
    import os
    import numpy as np
    import cv2
    from PIL import Image
    import torch
    import pandas as pd
    import matplotlib.pyplot as plt
    from ultralytics import YOLO

    # --- Load YOLOv8 model ---
    YOLO_WEIGHTS = 'detection_beest_e150.pt'
    CLASSES = ['Auto', 'Car', 'HV', 'LCV', 'MTW', 'Others']
    model = YOLO(YOLO_WEIGHTS)

    # --- Load MiDaS model ---
    midas = torch.hub.load('intel-isl/MiDaS', 'DPT_Hybrid')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    midas.to(device)
    midas.eval()
    transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
    transform = transforms.dpt_transform

    def run_obstacle_detection(img):
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = model(img)
        pred_bboxes = []
        result_img = img.copy()
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = box.conf[0]
                cls = int(box.cls[0])
                if conf > 0.4:
                    pred_bboxes.append([x1, y1, x2, y2, cls, conf])
                    cv2.rectangle(result_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(result_img, f"{CLASSES[cls]} {conf:.2f}", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return result_img, np.array(pred_bboxes)

    def run_midas_depth(img):
        input_batch = transform(img).to(device)
        with torch.no_grad():
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        depth_map = prediction.cpu().numpy()
        return depth_map

    def fuse_depth(lidar_depth, mono_depth, mask):
        fused = mono_depth.copy()
        fused[mask] = lidar_depth[mask]
        return fused

    def main():
        frame_dir = 'data/frames'
        lidar_dir = 'data/processed_lidar'
        out_dir = 'data/fused_output'
        os.makedirs(out_dir, exist_ok=True)
        csv_rows = []
        for fname in sorted(os.listdir(frame_dir)):
            if not fname.endswith('.png'):
                continue
            base = os.path.splitext(fname)[0]
            frame_path = os.path.join(frame_dir, fname)
            lidar_path = os.path.join(lidar_dir, base + '.npz')
            if not os.path.exists(lidar_path):
                print(f'Missing LiDAR for {base}, skipping.')
                continue
            # Load image
            img = Image.open(frame_path)
            # Object detection
            annotated_image, pred_bboxes = run_obstacle_detection(img)
            # Monocular depth
            depth_map = run_midas_depth(annotated_image)
            # Load LiDAR projection
            lidar_data = np.load(lidar_path)
            Dlidar = lidar_data['Dlidar']
            Mlidar = lidar_data['Mlidar']
            # Resize depth map to match image size
            if Dlidar.shape != depth_map.shape:
                depth_map = cv2.resize(depth_map, (Dlidar.shape[1], Dlidar.shape[0]))
            # Fuse depth
            fused_depth = fuse_depth(Dlidar, depth_map, Mlidar)
            # Overlay and metrics
            img_np = np.array(img)
            depth_map_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_map_color = cv2.applyColorMap(depth_map_norm, cv2.COLORMAP_JET)
            if img_np.shape[:2] != depth_map_color.shape[:2]:
                img_np = cv2.resize(img_np, (depth_map_color.shape[1], depth_map_color.shape[0]))
            alpha = 0.6
            overlaid_img = cv2.addWeighted(depth_map_color, alpha, img_np, 1 - alpha, 0)
            final_img = overlaid_img.copy()
            # LiDAR points overlay (optional, if you have u,v,lidar_distance)
            # TODO: Add LiDAR point overlay if needed
            # Process each YOLO box
            for box in pred_bboxes:
                x1, y1, x2, y2, cls, conf = map(int, box[:6])
                mono_patch = depth_map[y1:y2, x1:x2]
                mono_depths = mono_patch.flatten()
                avg_midas = np.mean(mono_depths) if len(mono_depths) > 0 else None
                lidar_patch = Dlidar[y1:y2, x1:x2]
                lidar_depths = lidar_patch.flatten()
                avg_lidar = np.mean(lidar_depths) if len(lidar_depths) > 0 else None
                label = f"{CLASSES[cls]} ({conf:.2f})"
                if avg_lidar is not None:
                    label += f" | LiDAR: {avg_lidar:.2f}m"
                else:
                    label += " | LiDAR: N/A"
                if avg_midas is not None:
                    label += f" | MiDaS: {avg_midas:.2f}m"
                else:
                    label += " | MiDaS: N/A"
                cv2.rectangle(final_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(final_img, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                # Add to CSV
                csv_rows.append({
                    "frame": base,
                    "class": CLASSES[cls],
                    "confidence": conf,
                    "bbox": (x1, y1, x2, y2),
                    "midas_metric_depth": avg_midas,
                    "lidar_distance": avg_lidar
                })
            # Save final image
            out_img_path = os.path.join(out_dir, base + '_overlay.png')
            cv2.imwrite(out_img_path, final_img)
            print(f'Saved overlay for {base}')
        # Save CSV
        df = pd.DataFrame(csv_rows)
        df.to_csv(os.path.join(out_dir, 'object_depth_metrics.csv'), index=False)
        print('CSV saved as object_depth_metrics.csv')

    if __name__ == '__main__':
        main()
