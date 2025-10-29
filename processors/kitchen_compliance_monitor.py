import cv2
import torch
from ultralytics import YOLO
import threading
import time
from datetime import datetime
from collections import defaultdict
import os
import logging
import pytz
import numpy as np
from sqlalchemy import Column, Integer, String, DateTime, Text, UniqueConstraint
from sqlalchemy.orm import declarative_base

# --- Basic Configuration ---
IST = pytz.timezone('Asia/Kolkata')
Base = declarative_base()

# --- Model Paths (Standardized) ---
MODELS_FOLDER = 'models'
APRON_CAP_MODEL_PATH = os.path.join(MODELS_FOLDER, 'apron-cap.pt')
GLOVES_MODEL_PATH = os.path.join(MODELS_FOLDER, 'gloves.pt')
GENERAL_MODEL_PATH = os.path.join(MODELS_FOLDER, 'yolov8n.pt')

# --- Detection Configuration ---
CONFIDENCE_THRESHOLD = 0.50
FRAME_SKIP_RATE = 5
PHONE_PERSISTENCE_SECONDS = 3
ALERT_COOLDOWN_SECONDS = 60

# --- Uniform Color Ranges (HSV) ---
YELLOW_LOWER = np.array([18, 80, 80])
YELLOW_UPPER = np.array([35, 255, 255])
BLACK_LOWER = np.array([0, 0, 0])
BLACK_UPPER = np.array([180, 255, 50])

# --- Database Table Definition ---
class KitchenViolation(Base):
    __tablename__ = "kitchen_violations"
    id = Column(Integer, primary_key=True, index=True)
    channel_id = Column(String, index=True)
    channel_name = Column(String)
    timestamp = Column(DateTime, default=lambda: datetime.now(IST))
    violation_type = Column(String)
    details = Column(String)
    media_path = Column(String)
    __table_args__ = (UniqueConstraint('media_path', name='_kitchen_media_path_uc'),)

class KitchenComplianceProcessor(threading.Thread):
    def __init__(self, rtsp_url, channel_id, channel_name, SessionLocal, socketio, telegram_sender, detection_callback):
        super().__init__(name=f"Kitchen-{channel_name}")
        self.rtsp_url = rtsp_url
        self.channel_id = channel_id
        self.channel_name = channel_name
        self.is_running = True
        self.error_message = None
        self.latest_frame = None
        self.lock = threading.Lock()

        self.SessionLocal = SessionLocal
        self.socketio = socketio
        self.send_telegram_notification = telegram_sender
        self.handle_main_detection = detection_callback

        try:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logging.info(f"Using device: {self.device} for Kitchen channel {self.channel_name}")
            for model_path in [APRON_CAP_MODEL_PATH, GLOVES_MODEL_PATH, GENERAL_MODEL_PATH]:
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Missing model file: {model_path}")
            
            self.apron_cap_model = YOLO(APRON_CAP_MODEL_PATH)
            self.gloves_model = YOLO(GLOVES_MODEL_PATH)
            self.general_model = YOLO(GENERAL_MODEL_PATH)
            self.apron_cap_model.to(self.device)
            self.gloves_model.to(self.device)
            self.general_model.to(self.device)
            logging.info(f"Successfully loaded Kitchen Compliance models for {self.channel_name}")
        except Exception as e:
            self.error_message = f"Model Error: {e}"
            logging.error(f"FATAL: Failed to initialize Kitchen models for {self.channel_name}. Error: {e}")

        self.person_violation_tracker = defaultdict(lambda: defaultdict(float))
        self.phone_tracker = {}
        self.last_apron_cap_results = []
        self.last_gloves_results = []
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    @staticmethod
    def initialize_tables(engine):
        try:
            Base.metadata.create_all(bind=engine)
            logging.info("Table 'kitchen_violations' checked/created.")
        except Exception as e:
            logging.error(f"Could not create 'kitchen_violations' table: {e}")

    def stop(self):
        self.is_running = False

    def shutdown(self):
        logging.info(f"Shutting down Kitchen Compliance processor for {self.channel_name}.")
        self.is_running = False

    def get_frame(self):
        with self.lock:
            if self.error_message:
                placeholder = np.full((480, 640, 3), (22, 27, 34), dtype=np.uint8)
                cv2.putText(placeholder, f'Error: {self.error_message}', (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                _, jpeg = cv2.imencode('.jpg', placeholder)
                return jpeg.tobytes()
            
            if self.latest_frame is not None:
                success, jpeg = cv2.imencode('.jpg', self.latest_frame)
                return jpeg.tobytes() if success else b''
            else:
                placeholder = np.full((480, 640, 3), (22, 27, 34), dtype=np.uint8)
                cv2.putText(placeholder, 'Connecting...', (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (201, 209, 217), 2)
                _, jpeg = cv2.imencode('.jpg', placeholder)
                return jpeg.tobytes()

    def _save_violation_to_db(self, violation_type, details, media_path):
        with self.SessionLocal() as db:
            try:
                violation = KitchenViolation(
                    channel_id=self.channel_id, channel_name=self.channel_name,
                    violation_type=violation_type, details=details, media_path=media_path
                )
                db.add(violation)
                db.commit()
            except Exception as e:
                logging.error(f"Failed to save kitchen violation to DB: {e}")
                db.rollback()

    def _trigger_alert(self, frame, violation_type, details):
        logging.warning(f"ALERT on {self.channel_name}: {details}")
        telegram_message = f"🚨 Kitchen Alert: {self.channel_name}\nViolation: {violation_type}\nDetails: {details}"
        self.send_telegram_notification(telegram_message)
        media_path = self.handle_main_detection(
            'KitchenCompliance', self.channel_id, [frame], details, is_gif=False
        )
        if media_path:
            self._save_violation_to_db(violation_type, details, media_path)

    def run(self):
        if self.error_message: return
        
        # Check for test mode
        use_placeholder = os.environ.get('USE_PLACEHOLDER_FEED', 'false').lower() == 'true'
        
        if not use_placeholder:
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|timeout;5000000'
            cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
            
            if not cap.isOpened():
                logging.warning(f"Could not open Kitchen stream for {self.channel_name}, using placeholder")
                use_placeholder = True
            else:
                is_file = any(self.rtsp_url.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mov'])
        
        if use_placeholder:
            logging.info(f"Using placeholder feed for Kitchen {self.channel_name}")
            frame_counter = 0
            while self.is_running:
                frame = np.full((480, 640, 3), (22, 27, 34), dtype=np.uint8)
                cv2.putText(frame, f'{self.channel_name}', (180, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (201, 209, 217), 2)
                cv2.putText(frame, f'Camera Offline - Test Mode', (120, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 150, 255), 2)
                cv2.putText(frame, f'Frame: {frame_counter}', (230, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
                
                with self.lock:
                    self.latest_frame = frame
                frame_counter += 1
                time.sleep(0.1)
            return

        is_file = any(self.rtsp_url.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mov'])
        frame_count = 0
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        phone_persistence_frames = int(PHONE_PERSISTENCE_SECONDS * video_fps)


        while self.is_running:
            success, frame = cap.read()
            if not success:
                if is_file:
                    logging.info(f"Restarting video file for Kitchen {self.channel_name}...")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    logging.warning(f"Reconnecting to Kitchen stream {self.channel_name}...")
                    time.sleep(5)
                    cap.release()
                    cap = cv2.VideoCapture(self.rtsp_url)
                    continue

            frame_count += 1
            current_time = time.time()
            annotated_frame = frame.copy()
            
            with self.lock:
                self.latest_frame = annotated_frame.copy()

            # --- Run Inferences ---
            person_results = self.general_model.track(frame, persist=True, classes=[0], conf=CONFIDENCE_THRESHOLD, verbose=False)
            phone_results = self.general_model(frame, classes=[67], conf=CONFIDENCE_THRESHOLD, verbose=False)

            if frame_count % FRAME_SKIP_RATE == 0:
                self.last_apron_cap_results = self.apron_cap_model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
                self.last_gloves_results = self.gloves_model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

            # --- Process Each Person ---
            if person_results and person_results[0].boxes.id is not None:
                track_ids = person_results[0].boxes.id.int().cpu().tolist()
                person_boxes = person_results[0].boxes.xyxy.cpu()

                detected_gloves_boxes = [box.xyxy[0] for r in self.last_gloves_results for box in r.boxes if self.gloves_model.names[int(box.cls[0])] == 'surgical-gloves']

                for person_box, track_id in zip(person_boxes, track_ids):
                    px1, py1, px2, py2 = map(int, person_box)
                    
                    # 1. Check for Apron/Cap Violations
                    for r in self.last_apron_cap_results:
                        for box in r.boxes:
                            violation_class = self.apron_cap_model.names[int(box.cls[0])]
                            if violation_class in ['Without-apron', 'Without-cap']:
                                if current_time - self.person_violation_tracker[track_id][violation_class] > ALERT_COOLDOWN_SECONDS:
                                    self.person_violation_tracker[track_id][violation_class] = current_time
                                    details = f"Person ID {track_id} detected with '{violation_class}'."
                                    self._trigger_alert(annotated_frame, violation_class, details)

                    # 2. Check for Gloves Violation
                    has_gloves = any(g_box[0] > px1 and g_box[2] < px2 and g_box[1] > py1 and g_box[3] < py2 for g_box in detected_gloves_boxes)
                    if not has_gloves:
                        if current_time - self.person_violation_tracker[track_id]['No-Gloves'] > ALERT_COOLDOWN_SECONDS:
                            self.person_violation_tracker[track_id]['No-Gloves'] = current_time
                            details = f"Person ID {track_id} has no gloves."
                            self._trigger_alert(annotated_frame, "No-Gloves", details)
                    
                    # 3. Check for Uniform Color Violation
                    torso_crop = frame[py1 + int((py2-py1)*0.1):py1 + int((py2-py1)*0.7), px1:px2]
                    if torso_crop.size > 0:
                        lab_torso = cv2.cvtColor(torso_crop, cv2.COLOR_BGR2LAB)
                        l, a, b = cv2.split(lab_torso)
                        equalized_l = self.clahe.apply(l)
                        merged_lab = cv2.merge((equalized_l, a, b))
                        equalized_torso = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
                        hsv_torso = cv2.cvtColor(equalized_torso, cv2.COLOR_BGR2HSV)
                        
                        mask_yellow = cv2.inRange(hsv_torso, YELLOW_LOWER, YELLOW_UPPER)
                        mask_black = cv2.inRange(hsv_torso, BLACK_LOWER, BLACK_UPPER)
                        compliant_mask = cv2.bitwise_or(mask_yellow, mask_black)
                        
                        total_pixels = torso_crop.shape[0] * torso_crop.shape[1]
                        compliant_ratio = np.count_nonzero(compliant_mask) / total_pixels if total_pixels > 0 else 0

                        if compliant_ratio < 0.30: # If less than 30% of torso is compliant color
                            if current_time - self.person_violation_tracker[track_id]['Uniform-Violation'] > ALERT_COOLDOWN_SECONDS:
                                self.person_violation_tracker[track_id]['Uniform-Violation'] = current_time
                                details = f"Person ID {track_id} has a uniform color violation."
                                self._trigger_alert(annotated_frame, "Uniform-Violation", details)

            # --- 4. Detect and Track Mobile Phones ---
            current_phones = [box.xyxy[0] for r in phone_results for box in r.boxes]
            new_phone_tracker = {}

            for phone_box in current_phones:
                cx, cy = int((phone_box[0] + phone_box[2]) / 2), int((phone_box[1] + phone_box[3]) / 2)
                found_match = False
                for phone_id, data in self.phone_tracker.items():
                    dist = np.sqrt((cx - data['center'][0])**2 + (cy - data['center'][1])**2)
                    if dist < 50:
                        new_phone_tracker[phone_id] = {'box': phone_box, 'frames': data['frames'] + 1, 'center': (cx, cy), 'alerted': data.get('alerted', False)}
                        found_match = True
                        break
                if not found_match:
                    new_id = max(self.phone_tracker.keys(), default=0) + 1
                    new_phone_tracker[new_id] = {'box': phone_box, 'frames': 1, 'center': (cx, cy), 'alerted': False}

            self.phone_tracker = new_phone_tracker

            for phone_id, data in self.phone_tracker.items():
                if data['frames'] > phone_persistence_frames and not data['alerted']:
                    data['alerted'] = True # Mark as alerted to prevent spamming
                    details = f"Mobile phone detected in restricted area for {PHONE_PERSISTENCE_SECONDS} seconds."
                    self._trigger_alert(annotated_frame, "Mobile-Phone", details)
        
        cap.release()

