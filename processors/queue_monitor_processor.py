import cv2
import threading
import time
import logging
import json
import numpy as np
from collections import defaultdict
from shapely.geometry import Point, Polygon
import pytz
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.orm import declarative_base

# --- Basic Configuration ---
IST = pytz.timezone('Asia/Kolkata')
Base = declarative_base()

# --- Configuration ---
QUEUE_DWELL_TIME_SEC = 3.0
QUEUE_ALERT_THRESHOLD = 2
QUEUE_ALERT_COOLDOWN_SEC = 180

# --- Database Table Definition ---
class QueueLog(Base):
    __tablename__ = "queue_logs"
    id = Column(Integer, primary_key=True, index=True)
    channel_id = Column(String, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(IST), index=True)
    queue_count = Column(Integer)

class QueueMonitorProcessor(threading.Thread):
    def __init__(self, rtsp_url, channel_id, channel_name, model, socketio, SessionLocal, handle_detection_callback, send_telegram_callback):
        super().__init__(name=f"QueueMonitor-{channel_name}")
        self.rtsp_url, self.channel_id, self.channel_name, self.model = rtsp_url, channel_id, channel_name, model
        self.socketio, self.SessionLocal = socketio, SessionLocal
        self.handle_detection = handle_detection_callback
        self.send_telegram = send_telegram_callback

        self.is_running = True
        self.lock = threading.Lock()
        self.latest_frame = None
        self.frame_dimensions = None

        self.queue_tracker = defaultdict(lambda: {'entry_time': 0})
        self.current_queue_count = 0
        self.last_alert_time = 0

        self.normalized_main_roi, self.normalized_secondary_roi = [], []
        self.roi_poly, self.secondary_roi_poly = Polygon(), Polygon()

    @staticmethod
    def initialize_tables(engine):
        try:
            Base.metadata.create_all(bind=engine)
            logging.info("Table 'queue_logs' checked/created.")
        except Exception as e:
            logging.error(f"Could not create QueueMonitor table: {e}")

    def update_roi(self, new_roi_points):
        with self.lock:
            try:
                self.normalized_main_roi = new_roi_points.get("main", [])
                self.normalized_secondary_roi = new_roi_points.get("secondary", [])
                logging.info(f"QueueMonitor {self.channel_name} received new ROI config.")
                self._recalculate_polygons()
            except Exception as e:
                logging.error(f"Error updating ROI for {self.channel_name}: {e}")

    def _recalculate_polygons(self):
        if self.frame_dimensions:
            h, w = self.frame_dimensions
            if self.normalized_main_roi:
                self.roi_poly = Polygon([(int(p[0]*w), int(p[1]*h)) for p in self.normalized_main_roi])
            if self.normalized_secondary_roi:
                self.secondary_roi_poly = Polygon([(int(p[0]*w), int(p[1]*h)) for p in self.normalized_secondary_roi])

    def shutdown(self):
        logging.info(f"Shutting down QueueMonitor for {self.channel_name}.")
        self.is_running = False

    def get_frame(self):
        with self.lock:
            if self.latest_frame is None:
                placeholder = np.full((480, 640, 3), (22, 27, 34), dtype=np.uint8)
                cv2.putText(placeholder, 'Connecting...', (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (201, 209, 217), 2)
                _, jpeg = cv2.imencode('.jpg', placeholder)
                return jpeg.tobytes()
            success, jpeg = cv2.imencode('.jpg', self.latest_frame)
            return jpeg.tobytes() if success else b''

    def run(self):
        # Check for test mode
        import os
        use_placeholder = os.environ.get('USE_PLACEHOLDER_FEED', 'false').lower() == 'true'
        
        if not use_placeholder:
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|timeout;5000000'
            cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
            
            if not cap.isOpened():
                logging.warning(f"Could not open QueueMonitor stream for {self.channel_name}, using placeholder")
                use_placeholder = True
            else:
                is_file = any(self.rtsp_url.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mov'])
        
        if use_placeholder:
            logging.info(f"Using placeholder feed for QueueMonitor {self.channel_name}")
            frame_counter = 0
            while self.is_running:
                frame = np.full((480, 640, 3), (22, 27, 34), dtype=np.uint8)
                cv2.putText(frame, f'{self.channel_name}', (180, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (201, 209, 217), 2)
                cv2.putText(frame, f'Camera Offline - Test Mode', (120, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 150, 255), 2)
                cv2.putText(frame, f'Queue: {self.current_queue_count}', (230, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                with self.lock:
                    self.latest_frame = frame
                frame_counter += 1
                time.sleep(0.1)
            return

        is_file = any(self.rtsp_url.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mov'])

        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                if is_file:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    logging.warning(f"Reconnecting to QueueMonitor stream {self.channel_name}...")
                    time.sleep(5)
                    cap.release()
                    cap = cv2.VideoCapture(self.rtsp_url)
                    continue
            
            if self.frame_dimensions is None:
                h, w, _ = frame.shape
                self.frame_dimensions = (h, w)
                self._recalculate_polygons()
            
            self.process_frame(frame)
        cap.release()

    def process_frame(self, frame):
        current_time = time.time()
        results = self.model.track(frame, persist=True, classes=[0], verbose=False, conf=0.4)
        annotated_frame = frame.copy()
        
        current_tracks_in_main_roi, people_in_secondary_roi = set(), 0
        if results and results[0].boxes.id is not None:
            for box, track_id in zip(results[0].boxes.xyxy.cpu(), results[0].boxes.id.int().cpu().tolist()):
                x1, y1, x2, y2 = map(int, box)
                person_point = Point(int((x1 + x2) / 2), y2) # Bottom center of box
                
                is_in_main = self.roi_poly.is_valid and not self.roi_poly.is_empty and self.roi_poly.contains(person_point)
                is_in_secondary = self.secondary_roi_poly.is_valid and not self.secondary_roi_poly.is_empty and self.secondary_roi_poly.contains(person_point)

                if is_in_main:
                    current_tracks_in_main_roi.add(track_id)
                    tracker = self.queue_tracker[track_id]
                    if tracker['entry_time'] == 0: tracker['entry_time'] = current_time
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 255, 0), 2) # Yellow for main ROI
                
                if is_in_secondary:
                    people_in_secondary_roi += 1
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 2) # Cyan for secondary ROI

        valid_queue_count = 0
        for track_id in list(self.queue_tracker.keys()):
            if track_id not in current_tracks_in_main_roi:
                del self.queue_tracker[track_id]
            elif (current_time - self.queue_tracker[track_id]['entry_time']) >= QUEUE_DWELL_TIME_SEC:
                valid_queue_count += 1
        
        if self.current_queue_count != valid_queue_count:
            self.current_queue_count = valid_queue_count
            self.socketio.emit('queue_update', {'channel_id': self.channel_id, 'count': self.current_queue_count})

        if valid_queue_count > QUEUE_ALERT_THRESHOLD and people_in_secondary_roi <= 1 and (current_time - self.last_alert_time) > QUEUE_ALERT_COOLDOWN_SEC:
            self.last_alert_time = current_time
            alert_message = f"Queue is full ({valid_queue_count} people), but the counter is free."
            logging.warning(f"QUEUE ALERT on {self.channel_name}: {alert_message}")
            self.send_telegram(f"🚨 Queue Alert: {self.channel_name}\n{alert_message}")
            self.handle_detection('QueueMonitor', self.channel_id, [frame], alert_message, is_gif=False)

        if self.roi_poly.is_valid and not self.roi_poly.is_empty: cv2.polylines(annotated_frame, [np.array(self.roi_poly.exterior.coords, dtype=np.int32)], True, (255, 255, 0), 2)
        if self.secondary_roi_poly.is_valid and not self.secondary_roi_poly.is_empty: cv2.polylines(annotated_frame, [np.array(self.secondary_roi_poly.exterior.coords, dtype=np.int32)], True, (0, 255, 255), 2)

        cv2.putText(annotated_frame, f"Queue: {self.current_queue_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(annotated_frame, f"Counter Area: {people_in_secondary_roi}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        with self.lock:
            self.latest_frame = annotated_frame.copy()

