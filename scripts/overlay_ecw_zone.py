import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
image_path = os.path.join(project_root, 'data', 'figures', 'frame_27435_enhanced_ttc.png')
ecw_json_path = os.path.join(project_root, 'data', 'ecw_annotations', 'ecw_frame_27435.json')
output_path = os.path.join(project_root, 'data', 'figures', 'frame_27435_enhanced_ttc_ecw.png')

# Load image
img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_h, img_w = img_rgb.shape[:2]

# Load ECW annotation
with open(ecw_json_path, 'r') as f:
    ecw_data = json.load(f)

# Extract ECW points (assume ecw_data['points'] is a list of [x, y] pairs)
ecw_points = np.array(ecw_data['points'], dtype=np.int32)
print(f"[DEBUG] Image shape: {img_w}x{img_h}")
print(f"[DEBUG] ECW points: {ecw_points}")
for idx, (x, y) in enumerate(ecw_points):
    if not (0 <= x < img_w and 0 <= y < img_h):
        print(f"[WARNING] ECW point {idx} ({x},{y}) is out of image bounds!")

# Plot image and ECW region

# Main overlay with correct axis orientation
plt.figure(figsize=(12, 8))
plt.imshow(img_rgb, origin='upper')
plt.xlim([0, img_w])
plt.ylim([img_h, 0])  # Invert y-axis so origin is top-left
plt.fill(ecw_points[:, 0], ecw_points[:, 1], color='#2a1a8a', alpha=0.55, label='ECW Zone')
plt.plot(np.append(ecw_points[:, 0], ecw_points[0, 0]), np.append(ecw_points[:, 1], ecw_points[0, 1]), color='#1a1a1a', linewidth=4)
plt.scatter(ecw_points[:, 0], ecw_points[:, 1], c='#ff00ff', s=80, edgecolor='black', zorder=10)
center_x = int(np.mean(ecw_points[:, 0]))
center_y = int(np.mean(ecw_points[:, 1]))
plt.text(center_x, center_y, 'ECW Zone', color='white', fontsize=18, fontweight='bold', ha='center', va='center', bbox=dict(facecolor='#2a1a8a', alpha=0.8, edgecolor='white'))
plt.title('Enhanced TTC + ECW Bubble Overlay')
plt.axis('off')
plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
print(f'Overlay saved: {output_path}')

# Debug overlay: show ECW points and indices
debug_path = output_path.replace('.png', '_debug.png')
plt.figure(figsize=(12, 8))
plt.imshow(img_rgb, origin='upper')
plt.xlim([0, img_w])
plt.ylim([img_h, 0])
plt.plot(ecw_points[:, 0], ecw_points[:, 1], 'ro-', label='ECW Points')
for i, pt in enumerate(ecw_points):
    plt.text(pt[0], pt[1], f'{i}', color='white', fontsize=12, ha='center', va='center', bbox=dict(facecolor='red', alpha=0.5, edgecolor='none'))
plt.axis('off')
plt.tight_layout()
plt.savefig(debug_path, dpi=300, bbox_inches='tight')
plt.close()
print(f'Debug overlay saved: {debug_path}')
