import numpy as np
import cv2

class VehicleCounter:
    def __init__(self, line_start, line_end, direction="vertical"):
        self.line_start = line_start
        self.line_end = line_end
        self.direction = direction
        self.counted_ids = set()
        self.vehicle_count = 0
        self.previous_centroids = {}

    def update(self, tracked_objects, traffic_light_state):
        if traffic_light_state != "green":
            return self.vehicle_count

        for track_id, (cx, cy) in tracked_objects.items():
            if track_id in self.counted_ids:
                continue

            prev = self.previous_centroids.get(track_id)

            if prev is not None:
                px, py = prev
                if self._crossed_line(px, py, cx, cy):
                    self.counted_ids.add(track_id)
                    self.vehicle_count += 1

            self.previous_centroids[track_id] = (cx, cy)

        return self.vehicle_count

    def _crossed_line(self, px, py, cx, cy):
        if self.direction == "vertical":
            line_x = self.line_start[0]
            return (px < line_x and cx >= line_x) or (px > line_x and cx <= line_x)
        else:
            line_y = self.line_start[1]
            return (py < line_y and cy >= line_y) or (py > line_y and cy <= line_y)

    def draw_line(self, frame):
        cv2.line(frame, self.line_start, self.line_end, (255, 0, 0), 2)
        cv2.putText(frame, f"Count: {self.vehicle_count}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
