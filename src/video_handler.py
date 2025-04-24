import cv2
import numpy as np
from sort import Sort

def open_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video: {path}")
    return cap

def get_video_info(cap):
    return {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    }

def display_video_with_tracking(path, yolo_detector):
    cap = open_video(path)
    info = get_video_info(cap)
    print(f"Video Info: {info}")

    tracker = Sort()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # yolo_detector should return [(x1, y1, x2, y2, confidence, class_id), ...]
        detections = yolo_detector(frame)

        # Filter only "car" class (e.g., class_id == 2 in COCO)
        car_detections = [det for det in detections if det[5] == 2]

        dets_np = np.array([
            [x1, y1, x2, y2, conf] for (x1, y1, x2, y2, conf, class_id) in car_detections
        ])

        if len(dets_np) == 0:
            dets_np = np.empty((0, 5))

        tracks = tracker.update(dets_np)

        for trk in tracks:
            x1, y1, x2, y2, track_id = map(int, trk)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Vehicle Tracking", frame)
        if cv2.waitKey(int(1000 / info["fps"])) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
