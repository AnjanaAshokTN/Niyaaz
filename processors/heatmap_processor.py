import cv2
import threading
import time
import logging
import numpy as np
from collections import defaultdict

# --- HEATMAP CONFIGURATION ---
HEATMAP_GRID_SIZE = 30
HEATMAP_TIME_THRESHOLD_SEC = 10.0
HEATMAP_PEOPLE_THRESHOLD = 5
HEATMAP_RADIUS = 20
HEATMAP_BLUR_KERNEL = (91, 91)

class HeatmapProcessor(threading.Thread):
    def __init__(self, rtsp_url, channel_id, channel_name, model):
        super().__init__(name=f"Heatmap-{channel_name}")
        self.rtsp_url = rtsp_url
        self.channel_id = channel_id
        self.channel_name = channel_name
        self.model = model
        self.is_running = True

        self.lock = threading.Lock()
        self.latest_frame = None
        self.heatmap_data = defaultdict(lambda: {'timestamps': []})
        self.hotspots = []
        self.last_logic_update = 0

    def stop(self):
        self.is_running = False

    def shutdown(self):
        logging.info(f"Shutting down Heatmap for {self.channel_name}")
        self.is_running = False

    def _update_heatmap_logic(self):
        current_time = time.time()
        with self.lock:
            new_hotspots = []
            for key in list(self.heatmap_data.keys()):
                cell = self.heatmap_data[key]
                cell['timestamps'] = [ts for ts in cell['timestamps'] if current_time - ts <= HEATMAP_TIME_THRESHOLD_SEC]
                
                person_count = len(cell['timestamps'])
                if person_count >= HEATMAP_PEOPLE_THRESHOLD:
                    heat_level = (person_count // HEATMAP_PEOPLE_THRESHOLD)
                    col, row = map(int, key.split(','))
                    new_hotspots.append({'col': col, 'row': row, 'heatLevel': heat_level})
                elif person_count == 0:
                    del self.heatmap_data[key]
            self.hotspots = new_hotspots

    def get_snapshot_frame(self):
        with self.lock:
            if self.latest_frame is None: return None
            frame_copy = self.latest_frame.copy()
            hotspots_copy = self.hotspots[:]
        return self._apply_heatmap_overlay(frame_copy, hotspots_copy)

    def _apply_heatmap_overlay(self, frame, hotspots):
        if not hotspots: return frame
        heatmap_canvas = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        for spot in hotspots:
            col, row, heat_level = spot['col'], spot['row'], spot['heatLevel']
            center_x = int((col + 0.5) * HEATMAP_GRID_SIZE)
            center_y = int((row + 0.5) * HEATMAP_GRID_SIZE)
            intensity = min(heat_level * 50, 255)
            cv2.circle(heatmap_canvas, (center_x, center_y), HEATMAP_RADIUS, int(intensity), -1)
        
        blurred_heatmap = cv2.GaussianBlur(heatmap_canvas, HEATMAP_BLUR_KERNEL, 0)
        colored_heatmap = cv2.applyColorMap(blurred_heatmap, cv2.COLORMAP_JET)
        return cv2.addWeighted(frame, 0.6, colored_heatmap, 0.4, 0)

    def get_frame(self):
        snapshot = self.get_snapshot_frame()
        if snapshot is None:
            placeholder = np.full((480, 640, 3), (22, 27, 34), dtype=np.uint8)
            cv2.putText(placeholder, 'Connecting...', (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (201, 209, 217), 2)
            _, jpeg = cv2.imencode('.jpg', placeholder)
            return jpeg.tobytes()
        
        success, jpeg = cv2.imencode('.jpg', snapshot)
        return jpeg.tobytes() if success else b''

    def run(self):
        cap = cv2.VideoCapture(self.rtsp_url)
        if not cap.isOpened():
            logging.error(f"Could not open Heatmap stream for {self.channel_name}")
            return

        is_file = any(self.rtsp_url.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mov'])

        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                if is_file:
                    logging.info(f"Restarting video file for Heatmap {self.channel_name}...")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    logging.warning(f"Reconnecting to Heatmap stream {self.channel_name}...")
                    time.sleep(5)
                    cap.release()
                    cap = cv2.VideoCapture(self.rtsp_url)
                    continue

            with self.lock:
                self.latest_frame = frame.copy()

            results = self.model(frame, classes=[0], verbose=False, conf=0.6)
            if results and results[0].boxes.xyxy is not None:
                with self.lock:
                    for box in results[0].boxes.xyxy.cpu().numpy():
                        x_center, y_bottom = (box[0] + box[2]) / 2, box[3]
                        col = int(x_center // HEATMAP_GRID_SIZE)
                        row = int(y_bottom // HEATMAP_GRID_SIZE)
                        self.heatmap_data[f"{col},{row}"]['timestamps'].append(time.time())

            current_time = time.time()
            if current_time - self.last_logic_update > 1.0:
                self._update_heatmap_logic()
                self.last_logic_update = current_time
        cap.release()

