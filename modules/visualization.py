import cv2
import numpy as np

def overlay_results(image, bboxes, classes, confidences, lidar_depths, mono_depths, ttc_list, ecw_bubbles):
    overlay = image.copy()
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = map(int, bbox)
        # Shorter label format
        label = f"{classes[i]} ({confidences[i]:.2f})"
        metrics = f"L:{lidar_depths[i]:.1f}m M:{mono_depths[i]:.1f}m T:{ttc_list[i]:.1f}s"
        
        # Draw green bbox
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0,255,0), 2)
        
        # Draw text with dark background for better visibility
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1
        
        # Get text sizes for positioning
        (label_w, label_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
        (metrics_w, metrics_h), _ = cv2.getTextSize(metrics, font, font_scale, thickness)
        
        # Draw background rectangles
        pad = 2
        cv2.rectangle(overlay, (x1, y1-label_h-2*pad), (x1+label_w+2*pad, y1), (0,0,0), -1)
        cv2.rectangle(overlay, (x1, y1), (x1+metrics_w+2*pad, y1+metrics_h+2*pad), (0,0,0), -1)
        
        # Draw text
        cv2.putText(overlay, label, (x1+pad, y1-pad), font, font_scale, (255,255,255), thickness)
        cv2.putText(overlay, metrics, (x1+pad, y1+metrics_h+pad), font, font_scale, (255,255,255), thickness)
        
        # Add ECW indicator if present
        if ecw_bubbles[i]:
            ecw_text = "ECW!"
            (ecw_w, ecw_h), _ = cv2.getTextSize(ecw_text, font, font_scale, thickness)
            cv2.rectangle(overlay, (x2-ecw_w-2*pad, y2), (x2, y2+ecw_h+2*pad), (0,0,0), -1)
            cv2.putText(overlay, ecw_text, (x2-ecw_w-pad, y2+ecw_h+pad), font, font_scale, (0,0,255), thickness)
    
    return overlay
