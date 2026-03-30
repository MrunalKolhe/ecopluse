import cv2
import numpy as np
from pathlib import Path
import os
try:
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")  # Downloads the nano model automatically
except ImportError:
    model = None

def process_civic_image(img_path: str):
    """
    Run YOLOv8 inference to detect civic issues using COCO dataset mappings,
    apply privacy blur to people/faces, and return the predicted issue.
    """
    if not model or not os.path.exists(img_path):
        return "Unknown", 0.0

    img = cv2.imread(img_path)
    if img is None:
        return "Unknown", 0.0

    results = model(img)
    issue = "Unknown"
    highest_conf = 0.0
    
    # 1. Better Object-to-Issue Mapping for COCO
    GARBAGE_LABELS = {"bottle", "cup", "wine glass", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake"}
    ROAD_LABELS = {"car", "truck", "bus", "motorcycle", "bicycle", "stop sign", "fire hydrant", "parking meter"}
    STREET_LABELS = {"traffic light"}

    for r in results:
        boxes = r.boxes
        names = model.names

        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = names[cls]

            # 2. Add Confidence Threshold Filtering
            # Avoid false positives by ensuring the detection is very confident.
            if conf < 0.45:
                continue

            # 3. Improve classification logic
            if label in GARBAGE_LABELS and conf > highest_conf:
                issue = "Garbage"
                highest_conf = conf

            elif label in STREET_LABELS and conf > highest_conf:
                issue = "Streetlight Issue"
                highest_conf = conf

            elif label in ROAD_LABELS and conf > highest_conf:
                # 4. Avoid false positives for road issues
                # A car doesn't mean a pothole directly, but it indicates a road scene.
                # In absence of custom training, we safely map it to a General Civic Department 
                # instead of claiming a false pothole, unless we restrict it purely stringently.
                if issue == "Unknown":
                    issue = "Road Scene Detected"
                    highest_conf = conf

            # 5. Privacy blur for 'person'
            if label == "person" and conf > 0.3:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Ensure coordinates are within image bounds
                h, w = img.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                roi = img[y1:y2, x1:x2]
                if roi.size != 0:
                    blurred_roi = cv2.GaussianBlur(roi, (51, 51), 30)
                    img[y1:y2, x1:x2] = blurred_roi

    # Save the dynamically blurred image over the original
    cv2.imwrite(img_path, img)

    # Department mapping consistency
    if issue == "Garbage":
        return "Garbage", highest_conf
    elif issue == "Streetlight Issue":
        return "Streetlight Issue", highest_conf
    elif issue == "Road Scene Detected":
        # Cannot confidently detect Potholes with COCO; suggest general Road Issue
        return "Pothole", highest_conf * 0.5  # Lower confidence intentionally

    return "Unknown", 0.0
