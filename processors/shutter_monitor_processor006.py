# import cv2
# import time as pytime
# import threading
# import logging
# from datetime import datetime, date, time as dt_time, timedelta
# import pytz
# from sqlalchemy import text
# from collections import deque
# import imageio
# import os
# import numpy as np

# IST = pytz.timezone('Asia/Kolkata')

# class ShutterMonitorProcessor(threading.Thread):
#     def __init__(self, rtsp_url, channel_id, channel_name, model, socketio, telegram_sender, SessionLocal):
#         super().__init__(name=f"ShutterMonitor-{channel_name}")
#         self.rtsp_url = rtsp_url
#         self.channel_id = channel_id
#         self.channel_name = channel_name
#         self.model = model
#         self.socketio = socketio
#         self.send_telegram_notification = telegram_sender
#         self.SessionLocal = SessionLocal
#         self.is_running = True
#         self.lock = threading.Lock()
#         self.latest_frame = None

#         self.tracking_date = date.min
#         self._reset_cycle_stats()
#         self._load_state_from_db()

#         self.last_telegram_alert_status = None
#         self.last_db_save_time = pytime.time()
#         self.close_detection_start_time = dt_time(20, 30) # 8:30 PM

#         self.fps = 10
#         self.buffer_size = self.fps * 10
#         self.frame_buffer = deque(maxlen=self.buffer_size)
#         self.last_detection_time = 0
#         self.detection_interval = 2 # Process frame every 2 seconds
#         self.static_folder = 'static'
#         self.detections_subfolder = 'detections'
        
#         logging.info(f"ShutterMonitor for {self.channel_name} initialized.")

#     def _reset_cycle_stats(self):
#         with self.lock:
#             self.tracking_date = datetime.now(IST).date()
#             self.current_status = 'Unknown'
#             self.last_status_time = datetime.now(IST)
#             self.first_open_time_today = None
#             self.first_close_time_today = None
#             self.total_open_duration = timedelta(0)
#             logging.info(f"Cycle stats reset for ShutterMonitor on {self.channel_name}")

#     def _load_state_from_db(self):
#         with self.lock, self.SessionLocal() as db:
#             today = datetime.now(IST).date()
#             try:
#                 result = db.execute(text("""
#                     SELECT first_open_time, first_close_time, total_open_duration_seconds
#                     FROM shutter_logs WHERE channel_id = :cid AND report_date = :rdate
#                 """), {'cid': self.channel_id, 'rdate': today}).first()

#                 if result:
#                     self.first_open_time_today = result.first_open_time
#                     self.first_close_time_today = result.first_close_time
#                     self.total_open_duration = timedelta(seconds=result.total_open_duration_seconds or 0)
#                     logging.info(f"Loaded state for {self.channel_name}: Open={self.first_open_time_today}, Close={self.first_close_time_today}")
#             except Exception as e:
#                 logging.error(f"DB Error loading state for {self.channel_name}: {e}")

#     def _save_cycle_to_db(self):
#         with self.lock, self.SessionLocal() as db:
#             if self.tracking_date == date.min: return
#             self._update_durations(force_update=True)
            
#             open_seconds = int(self.total_open_duration.total_seconds())
            
#             try:
#                 stmt = text("""
#                     INSERT INTO shutter_logs (channel_id, report_date, first_open_time, first_close_time, total_open_duration_seconds)
#                     VALUES (:cid, :rdate, :fot, :fct, :tods)
#                     ON CONFLICT (channel_id, report_date) 
#                     DO UPDATE SET
#                         first_open_time = COALESCE(shutter_logs.first_open_time, EXCLUDED.first_open_time),
#                         first_close_time = COALESCE(shutter_logs.first_close_time, EXCLUDED.first_close_time),
#                         total_open_duration_seconds = EXCLUDED.total_open_duration_seconds;
#                 """)
#                 db.execute(stmt, {'cid': self.channel_id, 'rdate': self.tracking_date, 'fot': self.first_open_time_today, 'fct': self.first_close_time_today, 'tods': open_seconds})
#                 db.commit()
#             except Exception as e:
#                 logging.error(f"Failed to save shutter log to DB for {self.channel_name}: {e}")
#                 db.rollback()

#     def _emit_update(self):
#         with self.lock:
#             open_seconds = int(self.total_open_duration.total_seconds())
#             closed_seconds = 0
#             if self.first_open_time_today:
#                 day_start_time = datetime.combine(self.tracking_date, dt_time.min).astimezone(IST)
#                 now_clamped = max(datetime.now(IST), day_start_time)
#                 total_duration_today = (now_clamped - day_start_time).total_seconds()
#                 closed_seconds = max(0, int(total_duration_today - open_seconds))

#             payload = {
#                 'channel_id': self.channel_id, 
#                 'last_status': self.current_status, 
#                 'last_status_time': self.last_status_time.isoformat(),
#                 'first_open_time': self.first_open_time_today.isoformat() if self.first_open_time_today else None,
#                 'first_close_time': self.first_close_time_today.isoformat() if self.first_close_time_today else None,
#                 'total_open_duration_seconds': open_seconds,
#                 'total_closed_duration_seconds': closed_seconds
#             }
#         self.socketio.emit('shutter_update', payload)

#     def _handle_telegram_alert(self, status_to_alert):
#         # (Telegram logic can be refined as per specific requirements)
#         pass

#     def _update_durations(self, force_update=False):
#         now = datetime.now(IST)
#         if self.current_status == 'open':
#             duration = now - self.last_status_time
#             self.total_open_duration += duration
#         self.last_status_time = now

#     def shutdown(self):
#         logging.info(f"Shutting down ShutterMonitor for {self.channel_name}. Saving final state...")
#         self.is_running = False
#         pytime.sleep(1.5) # allow last operations to finish
#         self._save_cycle_to_db()

#     def get_frame(self):
#         with self.lock:
#             if self.latest_frame is not None:
#                 success, jpeg = cv2.imencode('.jpg', self.latest_frame)
#                 return jpeg.tobytes() if success else b''
#             else:
#                 placeholder = np.full((480, 640, 3), (22, 27, 34), dtype=np.uint8)
#                 cv2.putText(placeholder, 'Connecting...', (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (201, 209, 217), 2)
#                 _, jpeg = cv2.imencode('.jpg', placeholder)
#                 return jpeg.tobytes()

#     def _save_video_and_update_db(self, frames_to_save, event_type):
#         now = datetime.now(IST)
#         ts_string = now.strftime("%Y%m%d_%H%M%S")
#         filename = f"Shutter_{self.channel_id}_{event_type}_{ts_string}.mp4"
#         video_subfolder = os.path.join(self.detections_subfolder, 'shutter_videos')
#         video_path_relative = os.path.join(video_subfolder, filename).replace('\\', '/')
#         video_path_full = os.path.join(self.static_folder, video_path_relative)

#         try:
#             rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_to_save]
#             imageio.mimwrite(video_path_full, rgb_frames, fps=self.fps, codec='libx264')
#         except Exception as e:
#             logging.error(f"Failed to save shutter video: {e}")
#             return

#         with self.SessionLocal() as db:
#             try:
#                 update_column = 'first_open_video_path' if event_type == 'open' else 'first_close_video_path'
#                 stmt = text(f"""
#                     UPDATE shutter_logs SET {update_column} = :vpath
#                     WHERE channel_id = :cid AND report_date = :rdate
#                 """)
#                 db.execute(stmt, {'vpath': video_path_relative, 'cid': self.channel_id, 'rdate': self.tracking_date})
#                 db.commit()
#             except Exception as e:
#                 logging.error(f"Failed to update DB with video path: {e}")
#                 db.rollback()

#     def run(self):
#         cap = cv2.VideoCapture(self.rtsp_url)
#         if not cap.isOpened():
#             logging.error(f"Could not open ShutterMonitor stream for {self.channel_name}")
#             return

#         is_file = any(self.rtsp_url.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mov'])

#         while self.is_running:
#             ret, frame = cap.read()
#             if not ret:
#                 if is_file:
#                     cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#                     continue
#                 else:
#                     logging.warning(f"Reconnecting to ShutterMonitor stream {self.channel_name}...")
#                     pytime.sleep(5)
#                     cap.release()
#                     cap = cv2.VideoCapture(self.rtsp_url)
#                     continue
            
#             self.frame_buffer.append(frame)
#             now_ist = datetime.now(IST)

#             if now_ist.date() > self.tracking_date and self.tracking_date != date.min:
#                 self._save_cycle_to_db()
#                 self._reset_cycle_stats()

#             if pytime.time() - self.last_detection_time > self.detection_interval:
#                 self.last_detection_time = pytime.time()
#                 latest_frame = self.frame_buffer[-1]
#                 results = self.model(latest_frame, conf=0.75, verbose=False)
                
#                 detected_status = None
#                 if results and len(results[0].boxes) > 0:
#                     best_detection = max(results[0].boxes, key=lambda x: x.conf)
#                     detected_status = 'open' if int(best_detection.cls) == 1 else 'close'

#                 if detected_status and detected_status != self.current_status:
#                     now = datetime.now(IST)
#                     self._update_durations()
                    
#                     logging.info(f"Status change for {self.channel_name}: {self.current_status} -> {detected_status}")
#                     self.current_status = detected_status
#                     self.last_status_time = now
                    
#                     should_record = False
#                     if self.current_status == 'open' and self.first_open_time_today is None:
#                         self.first_open_time_today = now
#                         self.tracking_date = now.date()
#                         should_record = True
#                     elif self.current_status == 'close' and self.first_open_time_today is not None and self.first_close_time_today is None and now.time() >= self.close_detection_start_time:
#                         self.first_close_time_today = now
#                         should_record = True

#                     if should_record:
#                         self._save_cycle_to_db() # Ensure record exists
#                         threading.Thread(target=self._save_video_and_update_db, args=(list(self.frame_buffer), self.current_status)).start()

#                     self._emit_update()
#                     self._save_cycle_to_db()

#             with self.lock:
#                  self.latest_frame = self.frame_buffer[-1].copy()
#                  cv2.putText(self.latest_frame, f"Status: {self.current_status.upper()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

#             if pytime.time() - self.last_db_save_time > 300: # Every 5 minutes
#                 self._save_cycle_to_db()
#                 self.last_db_save_time = pytime.time()
        
#         cap.release()

import cv2
import time as pytime
import threading
import logging
from datetime import datetime, date, time as dt_time, timedelta
import pytz
from sqlalchemy import text
from collections import deque
import imageio
import os
import numpy as np

IST = pytz.timezone('Asia/Kolkata')

class ShutterMonitorProcessor(threading.Thread):
    def __init__(self, rtsp_url, channel_id, channel_name, model, socketio, telegram_sender, SessionLocal):
        super().__init__(name=f"ShutterMonitor-{channel_name}")
        self.rtsp_url = rtsp_url
        self.channel_id = channel_id
        self.channel_name = channel_name
        self.model = model
        self.socketio = socketio
        self.send_telegram_notification = telegram_sender
        self.SessionLocal = SessionLocal
        self.is_running = True
        self.lock = threading.Lock()
        self.latest_frame = None

        self.tracking_date = date.min
        self._reset_cycle_stats()
        self._load_state_from_db()

        self.last_telegram_alert_status = None
        self.last_db_save_time = pytime.time()
        self.close_detection_start_time = dt_time(20, 30) # 8:30 PM

        self.fps = 10
        self.buffer_size = self.fps * 10
        self.frame_buffer = deque(maxlen=self.buffer_size)
        self.last_detection_time = 0
        self.detection_interval = 2 # Process frame every 2 seconds
        self.static_folder = 'static'
        self.detections_subfolder = 'detections'
        
        logging.info(f"ShutterMonitor for {self.channel_name} initialized.")

    def _reset_cycle_stats(self):
        with self.lock:
            self.tracking_date = datetime.now(IST).date()
            self.current_status = 'Unknown'
            self.last_status_time = datetime.now(IST)
            self.first_open_time_today = None
            self.first_close_time_today = None
            self.total_open_duration = timedelta(0)
            logging.info(f"Cycle stats reset for ShutterMonitor on {self.channel_name}")

    def _load_state_from_db(self):
        with self.lock, self.SessionLocal() as db:
            today = datetime.now(IST).date()
            try:
                result = db.execute(text("""
                    SELECT first_open_time, first_close_time, total_open_duration_seconds
                    FROM shutter_logs WHERE channel_id = :cid AND report_date = :rdate
                """), {'cid': self.channel_id, 'rdate': today}).first()

                if result:
                    self.first_open_time_today = result.first_open_time
                    self.first_close_time_today = result.first_close_time
                    self.total_open_duration = timedelta(seconds=result.total_open_duration_seconds or 0)
                    logging.info(f"Loaded state for {self.channel_name}: Open={self.first_open_time_today}, Close={self.first_close_time_today}")
            except Exception as e:
                logging.error(f"DB Error loading state for {self.channel_name}: {e}")

    def _save_cycle_to_db(self):
        with self.lock, self.SessionLocal() as db:
            if self.tracking_date == date.min: return
            self._update_durations(force_update=True)
            
            open_seconds = int(self.total_open_duration.total_seconds())
            
            try:
                stmt = text("""
                    INSERT INTO shutter_logs (channel_id, report_date, first_open_time, first_close_time, total_open_duration_seconds)
                    VALUES (:cid, :rdate, :fot, :fct, :tods)
                    ON CONFLICT (channel_id, report_date) 
                    DO UPDATE SET
                        first_open_time = COALESCE(shutter_logs.first_open_time, EXCLUDED.first_open_time),
                        first_close_time = COALESCE(shutter_logs.first_close_time, EXCLUDED.first_close_time),
                        total_open_duration_seconds = EXCLUDED.total_open_duration_seconds;
                """)
                db.execute(stmt, {'cid': self.channel_id, 'rdate': self.tracking_date, 'fot': self.first_open_time_today, 'fct': self.first_close_time_today, 'tods': open_seconds})
                db.commit()
            except Exception as e:
                logging.error(f"Failed to save shutter log to DB for {self.channel_name}: {e}")
                db.rollback()

    def _emit_update(self):
        with self.lock:
            open_seconds = int(self.total_open_duration.total_seconds())
            
            payload = {
                'channel_id': self.channel_id, 
                'last_status': self.current_status, 
                'last_status_time': self.last_status_time.isoformat(),
                'first_open_time': self.first_open_time_today.isoformat() if self.first_open_time_today else None,
                'first_close_time': self.first_close_time_today.isoformat() if self.first_close_time_today else None,
                'total_open_duration_seconds': open_seconds,
            }
        self.socketio.emit('shutter_update', payload)

    def _handle_telegram_alert(self, status_to_alert):
        if status_to_alert == self.last_telegram_alert_status: return
        self.last_telegram_alert_status = status_to_alert
        message = f"🚨 Shutter Alert: {self.channel_name}\nStatus changed to: **{status_to_alert.upper()}**"
        self.send_telegram_notification(message)


    def _update_durations(self, force_update=False):
        now = datetime.now(IST)
        if self.current_status == 'open':
            duration = now - self.last_status_time
            self.total_open_duration += duration
        self.last_status_time = now

    def shutdown(self):
        logging.info(f"Shutting down ShutterMonitor for {self.channel_name}. Saving final state...")
        self.is_running = False
        pytime.sleep(1.5)
        self._save_cycle_to_db()

    def get_frame(self):
        with self.lock:
            if self.latest_frame is not None:
                success, jpeg = cv2.imencode('.jpg', self.latest_frame)
                return jpeg.tobytes() if success else b''
            else:
                placeholder = np.full((480, 640, 3), (22, 27, 34), dtype=np.uint8)
                cv2.putText(placeholder, 'Connecting...', (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (201, 209, 217), 2)
                _, jpeg = cv2.imencode('.jpg', placeholder)
                return jpeg.tobytes()

    def _save_video_and_update_db(self, frames_to_save, event_type):
        now = datetime.now(IST)
        ts_string = now.strftime("%Y%m%d_%H%M%S")
        filename = f"Shutter_{self.channel_id}_{event_type}_{ts_string}.mp4"
        video_subfolder = os.path.join(self.detections_subfolder, 'shutter_videos')
        video_path_relative = os.path.join(video_subfolder, filename).replace('\\', '/')
        video_path_full = os.path.join(self.static_folder, video_path_relative)

        try:
            rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_to_save]
            imageio.mimwrite(video_path_full, rgb_frames, fps=self.fps, codec='libx264')
        except Exception as e:
            logging.error(f"Failed to save shutter video: {e}")
            return

        with self.SessionLocal() as db:
            try:
                update_column = 'first_open_video_path' if event_type == 'open' else 'first_close_video_path'
                stmt = text(f"""
                    UPDATE shutter_logs SET {update_column} = :vpath
                    WHERE channel_id = :cid AND report_date = :rdate
                """)
                db.execute(stmt, {'vpath': video_path_relative, 'cid': self.channel_id, 'rdate': self.tracking_date})
                db.commit()
            except Exception as e:
                logging.error(f"Failed to update DB with video path: {e}")
                db.rollback()

    def run(self):
        cap = cv2.VideoCapture(self.rtsp_url)
        if not cap.isOpened():
            logging.error(f"Could not open ShutterMonitor stream for {self.channel_name}")
            return

        is_file = any(self.rtsp_url.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mov'])

        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                if is_file:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    logging.warning(f"Reconnecting to ShutterMonitor stream {self.channel_name}...")
                    pytime.sleep(5)
                    cap.release()
                    cap = cv2.VideoCapture(self.rtsp_url)
                    continue
            
            self.frame_buffer.append(frame)
            now_ist = datetime.now(IST)

            if now_ist.date() > self.tracking_date and self.tracking_date != date.min:
                self._save_cycle_to_db()
                self._reset_cycle_stats()

            if pytime.time() - self.last_detection_time > self.detection_interval:
                self.last_detection_time = pytime.time()
                latest_frame = self.frame_buffer[-1]
                results = self.model(latest_frame, conf=0.75, verbose=False)
                
                detected_status = None
                if results and len(results[0].boxes) > 0:
                    best_detection = max(results[0].boxes, key=lambda x: x.conf)
                    detected_status = 'open' if int(best_detection.cls) == 1 else 'close'

                if detected_status and detected_status != self.current_status:
                    now = datetime.now(IST)
                    self._update_durations()
                    
                    logging.info(f"Status change for {self.channel_name}: {self.current_status} -> {detected_status}")
                    self.current_status = detected_status
                    self.last_status_time = now
                    
                    should_record = False
                    if self.current_status == 'open' and self.first_open_time_today is None:
                        self.first_open_time_today = now
                        self.tracking_date = now.date()
                        should_record = True
                    elif self.current_status == 'close' and self.first_open_time_today is not None and self.first_close_time_today is None and now.time() >= self.close_detection_start_time:
                        self.first_close_time_today = now
                        should_record = True

                    if should_record:
                        self._save_cycle_to_db() # Ensure record exists
                        threading.Thread(target=self._save_video_and_update_db, args=(list(self.frame_buffer), self.current_status)).start()
                    
                    self._handle_telegram_alert(self.current_status)
                    self._emit_update()
                    self._save_cycle_to_db()

            with self.lock:
                 self.latest_frame = self.frame_buffer[-1].copy()
                 cv2.putText(self.latest_frame, f"Status: {self.current_status.upper()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            if pytime.time() - self.last_db_save_time > 300: # Every 5 minutes
                self._save_cycle_to_db()
                self.last_db_save_time = pytime.time()
        
        cap.release()
