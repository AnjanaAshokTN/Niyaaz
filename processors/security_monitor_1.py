# import cv2
# import torch
# from ultralytics import YOLO
# import threading
# import time
# from collections import defaultdict
# import os
# import logging
# import numpy as np

# # --- Configuration (Standardized) ---
# MODELS_FOLDER = 'models'
# SECURITY_MODEL_PATH = os.path.join(MODELS_FOLDER, 'security.pt')
# PERSON_MODEL_PATH = os.path.join(MODELS_FOLDER, 'yolov8n.pt')
# INTERACTION_TIME_THRESHOLD_SEC = 3.0
# CONFIDENCE_PERSON = 0.5
# CONFIDENCE_SECURITY = 0.5

# def create_error_frame(title, line1, line2=""):
#     frame = np.full((480, 640, 3), (50, 20, 20), dtype=np.uint8)
#     cv2.putText(frame, title, (250, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
#     cv2.putText(frame, line1, (40, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
#     if line2:
#         cv2.putText(frame, line2, (40, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
#     _, jpeg = cv2.imencode('.jpg', frame)
#     return jpeg.tobytes()

# class SecurityProcessor(threading.Thread):
#     def __init__(self, rtsp_url, channel_id, channel_name, SessionLocal, socketio, SecurityViolationModel):
#         super().__init__(name=f"Security-{channel_name}")
#         self.rtsp_url = rtsp_url
#         self.channel_id = channel_id
#         self.channel_name = channel_name
#         self.is_running = True
#         self.lock = threading.Lock()
#         self.latest_frame = None
#         self.error_message = None

#         self.SessionLocal = SessionLocal
#         self.socketio = socketio
#         self.SecurityViolationModel = SecurityViolationModel

#         try:
#             self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#             logging.info(f"Using device: {self.device} for Security channel {self.channel_name}")
            
#             for model_path in [SECURITY_MODEL_PATH, PERSON_MODEL_PATH]:
#                 if not os.path.exists(model_path):
#                     raise FileNotFoundError(f"Missing model file: {model_path}")
            
#             self.person_model = YOLO(PERSON_MODEL_PATH)
#             self.security_model = YOLO(SECURITY_MODEL_PATH)
#             self.person_model.to(self.device)
#             self.security_model.to(self.device)
#             logging.info(f"Successfully loaded Security models for {self.channel_name}")
#         except Exception as e:
#             self.error_message = f"Model Error: {e}"
#             logging.error(f"FATAL: Failed to initialize Security models for {self.channel_name}. Error: {e}")

#         self.person_tracker = defaultdict(lambda: {
#             'history': [], 'is_outgoing': False, 'interaction_start': None,
#             'total_interaction_time': 0, 'last_seen': 0
#         })
#         self.last_cleanup_time = time.time()

#     def stop(self):
#         self.is_running = False

#     def shutdown(self):
#         logging.info(f"Shutting down Security processor for {self.channel_name}.")
#         self.is_running = False

#     def get_frame(self):
#         with self.lock:
#             if self.error_message:
#                 return create_error_frame('ERROR', self.error_message)
#             if self.latest_frame is None:
#                 placeholder = np.full((480, 640, 3), (20, 20, 20), dtype=np.uint8)
#                 cv2.putText(placeholder, 'Connecting...', (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#                 _, jpeg = cv2.imencode('.jpg', placeholder)
#                 return jpeg.tobytes()
#             success, encoded_image = cv2.imencode('.jpg', self.latest_frame)
#             return encoded_image.tobytes() if success else create_error_frame('Encoding Error', "Frame encode failed.")

#     def save_violation_to_db(self, message, details):
#         with self.SessionLocal() as db:
#             try:
#                 violation = self.SecurityViolationModel(
#                     channel_id=self.channel_id, channel_name=self.channel_name,
#                     message=message, details=details
#                 )
#                 db.add(violation)
#                 db.commit()
#                 db.refresh(violation)
#                 logging.info(f"Saved security violation for {self.channel_name}: {message}")

#                 self.socketio.emit('security_violation', {
#                     'channel_id': self.channel_id,
#                     'channel_name': self.channel_name,
#                     'timestamp': violation.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
#                     'message': message,
#                     'details': details
#                 })
#             except Exception as e:
#                 logging.error(f"Failed to save security violation to DB: {e}")
#                 db.rollback()

#     def check_overlap(self, box1, box2):
#         x1_min, y1_min, x1_max, y1_max = box1
#         x2_min, y2_min, x2_max, y2_max = box2
#         return not (x1_max < x2_min or x1_min > x2_max or y1_max < y2_min or y1_min > y2_max)

#     def run(self):
#         if self.error_message: return
#         cap = cv2.VideoCapture(self.rtsp_url)
#         if not cap.isOpened():
#             self.error_message = f"Could not open RTSP stream"
#             logging.error(self.error_message + f" for {self.channel_name}")
#             return

#         is_file = any(self.rtsp_url.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mov'])

#         while self.is_running:
#             ret, frame = cap.read()
#             if not ret:
#                 if is_file:
#                     logging.info(f"Restarting video file for Security {self.channel_name}...")
#                     cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#                     continue
#                 else:
#                     logging.warning(f"Reconnecting to Security stream {self.channel_name}...")
#                     time.sleep(5)
#                     cap.release()
#                     cap = cv2.VideoCapture(self.rtsp_url)
#                     continue

#             current_time = time.time()
#             person_results = self.person_model.track(frame, persist=True, classes=[0], conf=CONFIDENCE_PERSON, verbose=False)
#             security_results = self.security_model(frame, classes=[0], conf=CONFIDENCE_SECURITY, verbose=False)
#             annotated_frame = frame.copy()
#             line_x = int(frame.shape[1] * 0.40)
#             cv2.line(annotated_frame, (line_x, 0), (line_x, frame.shape[0]), (0, 255, 255), 2)
#             security_boxes = security_results[0].boxes.xyxy.cpu().numpy()

#             if len(security_boxes) == 0:
#                 cv2.putText(annotated_frame, "SYSTEM PAUSED: No Security Zone", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
#             else:
#                 for s_box in security_boxes:
#                     x1, y1, x2, y2 = map(int, s_box)
#                     cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

#                 if person_results and person_results[0].boxes.id is not None:
#                     person_boxes = person_results[0].boxes.xyxy.cpu().numpy()
#                     track_ids = person_results[0].boxes.id.int().cpu().tolist()

#                     for p_box, track_id in zip(person_boxes, track_ids):
#                         is_security_personnel = any(self.check_overlap(p_box, s_box) for s_box in security_boxes)
#                         x1, y1, x2, y2 = map(int, p_box)
#                         if is_security_personnel:
#                             cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
#                             cv2.putText(annotated_frame, "Security", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
#                             continue

#                         tracker = self.person_tracker[track_id]
#                         tracker['last_seen'] = current_time
#                         center_x = int((p_box[0] + p_box[2]) / 2)
#                         tracker['history'].append(center_x)
#                         if len(tracker['history']) > 2: tracker['history'].pop(0)
#                         if len(tracker['history']) == 2 and not tracker['is_outgoing']:
#                             if tracker['history'][0] < line_x and tracker['history'][1] >= line_x:
#                                 tracker['is_outgoing'] = True

#                         is_interacting = any(self.check_overlap(p_box, s_box) for s_box in security_boxes)
#                         if is_interacting and tracker['interaction_start'] is None:
#                             tracker['interaction_start'] = current_time
#                         elif not is_interacting and tracker['interaction_start'] is not None:
#                             interaction_duration = current_time - tracker['interaction_start']
#                             tracker['total_interaction_time'] += interaction_duration
#                             tracker['interaction_start'] = None

#                         color = (0, 255, 0) if not tracker['is_outgoing'] else (255, 165, 0)
#                         cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
#                         label = f"ID:{track_id}"
#                         if tracker['is_outgoing']:
#                             duration = tracker['total_interaction_time'] + (current_time - tracker['interaction_start'] if tracker['interaction_start'] else 0)
#                             label += f" Out | Sec: {duration:.1f}s"
#                         cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#             if current_time - self.last_cleanup_time > 10.0:
#                 self.last_cleanup_time = current_time
#                 stale_tracks = [tid for tid, data in self.person_tracker.items() if current_time - data['last_seen'] > 10.0]
#                 for track_id in stale_tracks:
#                     tracker = self.person_tracker[track_id]
#                     if tracker['interaction_start']:
#                         tracker['total_interaction_time'] += current_time - tracker['interaction_start']
#                     if tracker['is_outgoing'] and tracker['total_interaction_time'] < INTERACTION_TIME_THRESHOLD_SEC:
#                         msg = "Security Interaction Violation"
#                         details = f"Person {track_id} outgoing, spent only {tracker['total_interaction_time']:.2f}s with security."
#                         self.save_violation_to_db(msg, details)
#                     del self.person_tracker[track_id]

#             with self.lock:
#                 self.latest_frame = annotated_frame.copy()

#         cap.release()

import cv2
import torch
from ultralytics import YOLO
import threading
import time
from collections import defaultdict
import os
import logging
import numpy as np

# --- Configuration (Standardized) ---
MODELS_FOLDER = 'models'
SECURITY_MODEL_PATH = os.path.join(MODELS_FOLDER, 'security.pt')
PERSON_MODEL_PATH = os.path.join(MODELS_FOLDER, 'yolov8n.pt')
INTERACTION_TIME_THRESHOLD_SEC = 3.0
CONFIDENCE_PERSON = 0.5
CONFIDENCE_SECURITY = 0.5

def create_error_frame(title, line1, line2=""):
    frame = np.full((480, 640, 3), (50, 20, 20), dtype=np.uint8)
    cv2.putText(frame, title, (250, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    cv2.putText(frame, line1, (40, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    if line2:
        cv2.putText(frame, line2, (40, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    _, jpeg = cv2.imencode('.jpg', frame)
    return jpeg.tobytes()

class SecurityProcessor(threading.Thread):
    def __init__(self, rtsp_url, channel_id, channel_name, SessionLocal, socketio, SecurityViolationModel):
        super().__init__(name=f"Security-{channel_name}")
        self.rtsp_url = rtsp_url
        self.channel_id = channel_id
        self.channel_name = channel_name
        self.is_running = True
        self.lock = threading.Lock()
        self.latest_frame = None
        self.error_message = None

        self.SessionLocal = SessionLocal
        self.socketio = socketio
        self.SecurityViolationModel = SecurityViolationModel

        try:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logging.info(f"Using device: {self.device} for Security channel {self.channel_name}")
            
            for model_path in [SECURITY_MODEL_PATH, PERSON_MODEL_PATH]:
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Missing model file: {model_path}")
            
            self.person_model = YOLO(PERSON_MODEL_PATH)
            self.security_model = YOLO(SECURITY_MODEL_PATH)
            self.person_model.to(self.device)
            self.security_model.to(self.device)
            logging.info(f"Successfully loaded Security models for {self.channel_name}")
        except Exception as e:
            self.error_message = f"Model Error: {e}"
            logging.error(f"FATAL: Failed to initialize Security models for {self.channel_name}. Error: {e}")

        self.person_tracker = defaultdict(lambda: {
            'history': [], 'is_outgoing': False, 'interaction_start': None,
            'total_interaction_time': 0, 'last_seen': 0
        })
        self.last_cleanup_time = time.time()

    def stop(self):
        self.is_running = False

    def shutdown(self):
        logging.info(f"Shutting down Security processor for {self.channel_name}.")
        self.is_running = False

    def get_frame(self):
        with self.lock:
            if self.error_message:
                return create_error_frame('ERROR', self.error_message)
            if self.latest_frame is None:
                placeholder = np.full((480, 640, 3), (20, 20, 20), dtype=np.uint8)
                cv2.putText(placeholder, 'Connecting...', (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                _, jpeg = cv2.imencode('.jpg', placeholder)
                return jpeg.tobytes()
            success, encoded_image = cv2.imencode('.jpg', self.latest_frame)
            return encoded_image.tobytes() if success else create_error_frame('Encoding Error', "Frame encode failed.")

    def save_violation_to_db(self, message, details):
        with self.SessionLocal() as db:
            try:
                violation = self.SecurityViolationModel(
                    channel_id=self.channel_id, channel_name=self.channel_name,
                    message=message, details=details
                )
                db.add(violation)
                db.commit()
                db.refresh(violation)
                logging.info(f"Saved security violation for {self.channel_name}: {message}")

                self.socketio.emit('security_violation', {
                    'channel_id': self.channel_id,
                    'channel_name': self.channel_name,
                    'timestamp': violation.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    'message': message,
                    'details': details
                })
            except Exception as e:
                logging.error(f"Failed to save security violation to DB: {e}")
                db.rollback()

    def check_overlap(self, box1, box2):
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        return not (x1_max < x2_min or x1_min > x2_max or y1_max < y2_min or y1_min > y2_max)

    def run(self):
        if self.error_message: return
        cap = cv2.VideoCapture(self.rtsp_url)
        if not cap.isOpened():
            self.error_message = f"Could not open RTSP stream"
            logging.error(self.error_message + f" for {self.channel_name}")
            return

        is_file = any(self.rtsp_url.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mov'])

        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                if is_file:
                    logging.info(f"Restarting video file for Security {self.channel_name}...")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    logging.warning(f"Reconnecting to Security stream {self.channel_name}...")
                    time.sleep(5)
                    cap.release()
                    cap = cv2.VideoCapture(self.rtsp_url)
                    continue

            current_time = time.time()
            person_results = self.person_model.track(frame, persist=True, classes=[0], conf=CONFIDENCE_PERSON, verbose=False)
            security_results = self.security_model(frame, classes=[0], conf=CONFIDENCE_SECURITY, verbose=False)
            annotated_frame = frame.copy()
            line_x = int(frame.shape[1] * 0.40)
            cv2.line(annotated_frame, (line_x, 0), (line_x, frame.shape[0]), (0, 255, 255), 2)
            security_boxes = security_results[0].boxes.xyxy.cpu().numpy()

            if len(security_boxes) == 0:
                cv2.putText(annotated_frame, "SYSTEM PAUSED: No Security Zone", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            else:
                for s_box in security_boxes:
                    x1, y1, x2, y2 = map(int, s_box)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                if person_results and person_results[0].boxes.id is not None:
                    person_boxes = person_results[0].boxes.xyxy.cpu().numpy()
                    track_ids = person_results[0].boxes.id.int().cpu().tolist()

                    for p_box, track_id in zip(person_boxes, track_ids):
                        is_security_personnel = any(self.check_overlap(p_box, s_box) for s_box in security_boxes)
                        x1, y1, x2, y2 = map(int, p_box)
                        if is_security_personnel:
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(annotated_frame, "Security", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            continue

                        tracker = self.person_tracker[track_id]
                        tracker['last_seen'] = current_time
                        center_x = int((p_box[0] + p_box[2]) / 2)
                        tracker['history'].append(center_x)
                        if len(tracker['history']) > 2: tracker['history'].pop(0)
                        if len(tracker['history']) == 2 and not tracker['is_outgoing']:
                            if tracker['history'][0] < line_x and tracker['history'][1] >= line_x:
                                tracker['is_outgoing'] = True

                        is_interacting = any(self.check_overlap(p_box, s_box) for s_box in security_boxes)
                        if is_interacting and tracker['interaction_start'] is None:
                            tracker['interaction_start'] = current_time
                        elif not is_interacting and tracker['interaction_start'] is not None:
                            interaction_duration = current_time - tracker['interaction_start']
                            tracker['total_interaction_time'] += interaction_duration
                            tracker['interaction_start'] = None

                        color = (0, 255, 0) if not tracker['is_outgoing'] else (255, 165, 0)
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        label = f"ID:{track_id}"
                        if tracker['is_outgoing']:
                            duration = tracker['total_interaction_time'] + (current_time - tracker['interaction_start'] if tracker['interaction_start'] else 0)
                            label += f" Out | Sec: {duration:.1f}s"
                        cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if current_time - self.last_cleanup_time > 10.0:
                self.last_cleanup_time = current_time
                stale_tracks = [tid for tid, data in self.person_tracker.items() if current_time - data['last_seen'] > 10.0]
                for track_id in stale_tracks:
                    tracker = self.person_tracker.pop(track_id, None)
                    if tracker:
                        if tracker['interaction_start']:
                            tracker['total_interaction_time'] += current_time - tracker['interaction_start']
                        if tracker['is_outgoing'] and tracker['total_interaction_time'] < INTERACTION_TIME_THRESHOLD_SEC:
                            msg = "Security Interaction Violation"
                            details = f"Person {track_id} outgoing, spent only {tracker['total_interaction_time']:.2f}s with security."
                            self.save_violation_to_db(msg, details)

            with self.lock:
                self.latest_frame = annotated_frame.copy()

        cap.release()
