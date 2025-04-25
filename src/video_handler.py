import cv2
import numpy as np
from src.sort import Sort
from src.vehicle_detector import detect_objects
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
    print(f"[INFO] Video Info: {info}")

    tracker = Sort()
    counter = None
    light_detector = TrafficLightDetector()
    reporter = ReportManager()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        timestamp = frame_index / info["fps"]

        if counter is None:
            frame_height, frame_width = frame.shape[:2]
            y_line = int(frame_height * 0.7)
            counter = VehicleCounter(
                line_start=(0, y_line),
                line_end=(frame_width, y_line),
                direction="horizontal"
            )

        # --- DETECTION ---
        car_detections, light_detections = detect_objects(frame)

        if light_detections:
            best_light = max(light_detections, key=lambda d: d[4])
            light_box = best_light[:4]
            light_state = light_detector.get_light_state_from_box(frame, light_box)

            x1, y1, x2, y2 = map(int, light_box)
            color = (0, 255, 0) if light_state == "green" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"Light: {light_state}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            light_state = "unknown"

        # --- TRACKING ---
        dets_np = np.array([
            [x1, y1, x2, y2, conf] for (x1, y1, x2, y2, conf, cls) in car_detections
        ]) if car_detections else np.empty((0, 5))

        tracks = tracker.update(dets_np)

        # --- COUNTING ---
        centroids = {
            int(trk[4]): ((trk[0]+trk[2])//2, (trk[1]+trk[3])//2) for trk in tracks
        }

        previous_count = counter.vehicle_count
        count = counter.update(centroids, light_state)

        # --- LOGGING ---
        for trk in tracks:
            track_id = int(trk[4])
            if track_id in counter.counted_ids:
                already_logged = {rec["vehicle_id"] for rec in reporter.records}
                if track_id not in already_logged:
                    reporter.log_event(
                        vehicle_id=track_id,
                        frame_idx=frame_index,
                        timestamp=timestamp,
                        traffic_light_state=light_state
                    )
                    print(f"[INFO] Vehicle ID {track_id} at frame {frame_index} | State: {light_state}")

        # --- DRAWING ---
        for trk in tracks:
            x1, y1, x2, y2, track_id = map(int, trk)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        counter.draw_line(frame)
        cv2.imshow("Vehicle Counter", frame)

        if cv2.waitKey(int(1000 / info["fps"])) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    # --- FINAL REPORT ---
    df = reporter.save_csv()
    reporter.generate_report(df)
    print(f"[INFO] Total vehicles counted: {counter.vehicle_count}")
