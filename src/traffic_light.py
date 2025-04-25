import cv2
import numpy as np

class TrafficLightDetector:
    def __init__(self):
        pass

    def get_light_state_from_box(self, frame, box):
        
        x1, y1, x2, y2 = box

        h, w = frame.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return "unknown"

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        green_mask = cv2.inRange(hsv, np.array([45, 100, 50]), np.array([90, 255, 255]))

        red_mask1 = cv2.inRange(hsv, np.array([0, 100, 50]), np.array([10, 255, 255]))
        red_mask2 = cv2.inRange(hsv, np.array([170, 100, 50]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        green_pixels = cv2.countNonZero(green_mask)
        red_pixels = cv2.countNonZero(red_mask)

        if green_pixels > red_pixels and green_pixels > 100:
            return "green"
        elif red_pixels > green_pixels and red_pixels > 100:
            return "red"
        else:
            return "unknown"
