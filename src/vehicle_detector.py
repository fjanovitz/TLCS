from ultralytics import YOLO
import numpy as np

model = YOLO("yolov8n.pt")

def detect_objects(frame):
    results = model(frame)[0]
    
    cars = []
    traffic_lights = []
    
    for box in results.boxes:
        coords = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf[0])
        class_id = int(box.cls[0])
        
        detection = (coords[0], coords[1], coords[2], coords[3], conf, class_id)

        if class_id == 2:
            cars.append(detection)
        elif class_id == 9:
            traffic_lights.append(detection)
    
    return cars, traffic_lights
