from ultralytics import YOLO
import numpy as np

# Load YOLOv8 model (can be 'yolov8n.pt', 'yolov8s.pt', etc.)
# You can replace with your custom trained model if needed.
model = YOLO("yolov8n.pt")  # Or yolov8s.pt, yolov8m.pt, etc.

def detect_vehicles(frame):
    """
    Runs YOLOv8 inference on a frame and returns detections.

    Returns:
        List of tuples: [(x1, y1, x2, y2, confidence, class_id), ...]
    """
    results = model(frame)[0]

    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf[0])
        class_id = int(box.cls[0])
        detections.append((x1, y1, x2, y2, conf, class_id))

    return detections
