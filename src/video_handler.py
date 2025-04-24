import cv2
import numpy as np
from sort import Sort
from src.vehicle_detector import detect_vehicles
from src.counter import VehicleCounter
from src.traffic_light import TrafficLightDetector
from src.reporter import ReportManager

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

def process_video(path):
    cap = open_video(path)
    info = get_video_info(cap)
    print(f"Video Info: {info}")

    tracker = Sort()
    counter = VehicleCounter(line_start=(300, 200), line_end=(600, 200), direction="horizontal")
    light_detector = TrafficLightDetector(roi_coords=(50, 50, 40, 80))  # adjust as needed
    reporter = ReportManager()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        timestamp = frame_index / info["fps"]

        # 1. Get traffic light state
        light_state = light_detector.get_light_state(frame)
        light_detector.draw_roi(frame, light_state)

        # 2. Detect vehicles
        detections = detect_vehicles(frame)
        car_detections = [d for d in detections if d[5] == 2]  # class_id 2 = car

        dets_np = np.array([[x1, y1, x2, y2, conf] for (x1, y1, x2, y2, conf, cls) in car_detections])
        if len(dets_np) == 0:
            dets_np = np.empty((0, 5))

        # 3. Track vehicles
        tracks = tracker.update(dets_np)

        # 4. Extract centroids
        centroids = {
            int(trk[4]): ((trk[0]+trk[2])//2, (trk[1]+trk[3])//2) for trk in tracks
        }

        # 5. Count vehicles if green light
        previous_count = counter.vehicle_count
        count = counter.update(centroids, light_state)

        # 6. Log events
        for trk in tracks:
            track_id = int(trk[4])
            if track_id in counter.counted_ids and track_id not in reporter.records:
                reporter.log_event(
                    vehicle_id=track_id,
                    frame_idx=frame_index,
                    timestamp=timestamp,
                    traffic_light_state=light_state
                )

        # 7. Draw results
        for trk in tracks:
            x1, y1, x2, y2, track_id = map(int, trk)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        counter.draw_line(frame)

        # 8. Show frame
        cv2.imshow("Vehicle Counter", frame)
        if cv2.waitKey(int(1000 / info["fps"])) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Final report
    df = reporter.save_csv()
    reporter.generate_report(df)
