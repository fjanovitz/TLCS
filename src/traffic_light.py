import cv2
import numpy as np

class TrafficLightDetector:
    def __init__(self, roi_coords):
        """
        Args:
            roi_coords: (x, y, w, h) rectangle containing the traffic light
        """
        self.x, self.y, self.w, self.h = roi_coords

    def get_light_state(self, frame):
        """
        Returns:
            'green', 'red', or 'unknown'
        """
        roi = frame[self.y:self.y+self.h, self.x:self.x+self.w]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Mask for green
        green_lower = np.array([45, 100, 50])
        green_upper = np.array([90, 255, 255])
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        green_pixels = cv2.countNonZero(green_mask)

        # Mask for red (2 ranges)
        red_lower1 = np.array([0, 100, 50])
        red_upper1 = np.array([10, 255, 255])
        red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)

        red_lower2 = np.array([170, 100, 50])
        red_upper2 = np.array([180, 255, 255])
        red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)

        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        red_pixels = cv2.countNonZero(red_mask)

        if green_pixels > red_pixels and green_pixels > 100:
            return "green"
        elif red_pixels > green_pixels and red_pixels > 100:
            return "red"
        else:
            return "unknown"

    def draw_roi(self, frame, state):
        color = (0, 255, 0) if state == "green" else (0, 0, 255) if state == "red" else (255, 255, 0)
        cv2.rectangle(frame, (self.x, self.y), (self.x+self.w, self.y+self.h), color, 2)
        cv2.putText(frame, f"Light: {state}", (self.x, self.y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
