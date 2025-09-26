#!/usr/bin/env python3
import os
import cv2
import numpy as np
import json
from PIL import Image
import argparse

class BBoxAnnotator:
    def __init__(self, window_name="Manual Annotator"):
        self.window_name = window_name
        self.current_box = []
        self.boxes = []  # List of [x1, y1, x2, y2, class_id, conf]
        self.drawing = False
        self.current_class = 0  # 0: car, 1: person, 2: bus, etc.
        # Colors match the image: car (magenta), bus (orange), motorcycle/bicycle (cyan), person (green)
        self.colors = {
            0: (255, 0, 255),   # Car - Magenta
            1: (0, 255, 0),     # Person - Green
            2: (255, 165, 0),   # Bus - Orange
            3: (0, 255, 255),   # Motorcycle/Bicycle - Cyan
            4: (255, 255, 0),   # Truck - Yellow
        }
        self.class_names = {
            0: "car",          # Magenta
            1: "person",       # Green
            2: "bus",          # Orange
            3: "motorcycle",   # Cyan
            4: "truck"         # Yellow
        }
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.current_box = [x, y]
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            img_copy = self.image.copy()
            cv2.rectangle(img_copy, 
                         (self.current_box[0], self.current_box[1]), 
                         (x, y), 
                         self.colors[self.current_class], 2)
            cv2.imshow(self.window_name, img_copy)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            x1, y1 = self.current_box
            x2, y2 = x, y
            # Ensure x1,y1 is top-left and x2,y2 is bottom-right
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            # Add box with class_id and confidence=1.0
            self.boxes.append([x1, y1, x2, y2, self.current_class, 1.0])
            self.draw_boxes()

    def draw_boxes(self):
        img_copy = self.image.copy()
        for box in self.boxes:
            x1, y1, x2, y2, class_id, conf = box
            color = self.colors[int(class_id)]
            cv2.rectangle(img_copy, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            label = f"{self.class_names[class_id]} ({conf:.2f})"
            cv2.putText(img_copy, label, (int(x1), int(y1)-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.imshow(self.window_name, img_copy)

    def annotate(self, image):
        self.image = image
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        print("\nControls:")
        print("- Left click and drag to draw box")
        print("- Number keys (0-4) to change class:")
        for k, v in self.class_names.items():
            print(f"  {k}: {v}")
        print("- 'c' to clear all boxes")
        print("- 'd' to delete last box")
        print("- 's' to save and exit")
        print("- 'q' to quit without saving")
        
        self.draw_boxes()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            # Number keys for class selection
            if key >= ord('0') and key <= ord('4'):
                self.current_class = key - ord('0')
                print(f"Selected class: {self.class_names[self.current_class]}")
            
            # Clear all boxes
            elif key == ord('c'):
                self.boxes = []
                self.draw_boxes()
            
            # Delete last box
            elif key == ord('d'):
                if self.boxes:
                    self.boxes.pop()
                    self.draw_boxes()
            
            # Save and exit
            elif key == ord('s'):
                cv2.destroyAllWindows()
                return self.boxes
            
            # Quit without saving
            elif key == ord('q'):
                cv2.destroyAllWindows()
                return None

def save_annotations(boxes, output_path):
    """Save annotations in YOLO-compatible format"""
    data = {
        'boxes': boxes,
        'format': 'x1,y1,x2,y2,class_id,confidence'
    }
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved annotations to {output_path}")

def load_annotations(filepath):
    """Load saved annotations"""
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data['boxes']
    return None

def main():
    parser = argparse.ArgumentParser(description='Manual BBox Annotator')
    parser.add_argument('--frame', type=int, required=True,
                      help='Frame number to annotate')
    parser.add_argument('--video', type=str, default='data/input.avi',
                      help='Path to input video')
    args = parser.parse_args()
    
    # Extract frame
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    video_path = os.path.join(project_root, args.video)
    
    # Use extract_frame from generate_paper_figures
    from generate_paper_figures import extract_frame
    frame_rgb = extract_frame(video_path, args.frame, None)
    
    # Setup annotator
    annotator = BBoxAnnotator()
    
    # Check for existing annotations
    annotations_dir = os.path.join(project_root, 'data', 'manual_annotations')
    os.makedirs(annotations_dir, exist_ok=True)
    annotation_file = os.path.join(annotations_dir, f'frame_{args.frame}.json')
    
    existing_boxes = load_annotations(annotation_file)
    if existing_boxes:
        print(f"Found existing annotations for frame {args.frame}")
        annotator.boxes = existing_boxes
    
    # Run annotation
    boxes = annotator.annotate(frame_rgb)
    
    if boxes is not None:
        save_annotations(boxes, annotation_file)
        print("Annotations saved successfully!")
    else:
        print("Annotation cancelled.")

if __name__ == '__main__':
    main()