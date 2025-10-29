import cv2
import threading
import time
from datetime import datetime
from collections import defaultdict
import logging
import pytz
import numpy as np
import os
from sqlalchemy import Column, Integer, String, Date, UniqueConstraint, text, DateTime
from sqlalchemy.orm import declarative_base

# --- Basic Configuration ---
IST = pytz.timezone('Asia/Kolkata')
Base = declarative_base()

# --- Database Table Definitions ---
class DailyFootfall(Base):
    __tablename__ = "daily_footfall"
    id = Column(Integer, primary_key=True, index=True)
    channel_id = Column(String, index=True)
    report_date = Column(Date, index=True)
    in_count = Column(Integer, default=0)
    out_count = Column(Integer, default=0)
    __table_args__ = (UniqueConstraint('channel_id', 'report_date', name='_daily_footfall_uc'),)

class HourlyFootfall(Base):
    __tablename__ = "hourly_footfall"
    id = Column(Integer, primary_key=True, index=True)
    channel_id = Column(String, index=True)
    report_date = Column(Date, index=True)
    hour = Column(Integer, index=True)
    in_count = Column(Integer, default=0)
    out_count = Column(Integer, default=0)
    __table_args__ = (UniqueConstraint('channel_id', 'report_date', 'hour', name='_hourly_footfall_uc'),)


class PeopleCounterProcessor(threading.Thread):
    def __init__(self, rtsp_url, channel_id, channel_name, model, socketio, SessionLocal):
        super().__init__(name=f"PeopleCounter-{channel_name}")
        self.rtsp_url = rtsp_url
        self.channel_id = channel_id
        self.channel_name = channel_name
        self.model = model
        self.socketio = socketio
        self.SessionLocal = SessionLocal

        self.is_running = True
        self.lock = threading.Lock()
        self.latest_frame = None

        self.track_history = defaultdict(list)
        self.counts = {'in': 0, 'out': 0}
        self.last_saved_total_counts = {'in': 0, 'out': 0}
        self.last_saved_hour = datetime.now(IST).hour
        self.tracking_date = datetime.now(IST).date()
        
        self._load_initial_counts()
        self.socketio.emit('count_update', {'channel_id': self.channel_id, 'in_count': self.counts['in'], 'out_count': self.counts['out']})

    @staticmethod
    def initialize_tables(engine):
        try:
            Base.metadata.create_all(bind=engine)
            logging.info("Tables 'daily_footfall' and 'hourly_footfall' checked/created.")
        except Exception as e:
            logging.error(f"Could not create PeopleCounter tables: {e}")

    def shutdown(self):
        logging.info(f"Shutting down PeopleCounter for {self.channel_name}. Saving final counts...")
        self.is_running = False
        self._update_and_log_counts(final_save=True)

    def get_frame(self):
        with self.lock:
            if self.latest_frame is None:
                placeholder = np.full((480, 640, 3), (22, 27, 34), dtype=np.uint8)
                cv2.putText(placeholder, 'Connecting...', (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (201, 209, 217), 2)
                _, jpeg = cv2.imencode('.jpg', placeholder)
                return jpeg.tobytes()
            success, jpeg = cv2.imencode('.jpg', self.latest_frame)
            return jpeg.tobytes() if success else b''

    def _load_initial_counts(self):
        with self.SessionLocal() as db:
            today_ist = datetime.now(IST).date()
            self.tracking_date = today_ist
            record = db.query(DailyFootfall).filter_by(channel_id=self.channel_id, report_date=today_ist).first()
            if record:
                self.counts = {'in': record.in_count, 'out': record.out_count}
            else:
                self._reset_counts_for_new_day(db, today_ist)
            self.last_saved_total_counts = self.counts.copy()

    def _reset_counts_for_new_day(self, db, new_date):
        self.counts = {'in': 0, 'out': 0}
        self.tracking_date = new_date
        db.execute(text("""
            INSERT INTO daily_footfall (channel_id, report_date, in_count, out_count)
            VALUES (:cid, :rdate, 0, 0) ON CONFLICT DO NOTHING
        """), {'cid': self.channel_id, 'rdate': new_date})
        db.commit()

    def _update_and_log_counts(self, final_save=False):
        with self.SessionLocal() as db:
            try:
                # Use text() for PostgreSQL compatible ON CONFLICT
                daily_stmt = text("""
                    INSERT INTO daily_footfall (channel_id, report_date, in_count, out_count)
                    VALUES (:cid, :rdate, :inc, :outc)
                    ON CONFLICT (channel_id, report_date) 
                    DO UPDATE SET in_count = :inc, out_count = :outc;
                """)
                db.execute(daily_stmt, {'cid': self.channel_id, 'rdate': self.tracking_date, 'inc': self.counts['in'], 'outc': self.counts['out']})

                current_hour_ist = datetime.now(IST).hour
                if current_hour_ist != self.last_saved_hour or final_save:
                    hourly_in = self.counts['in'] - self.last_saved_total_counts['in']
                    hourly_out = self.counts['out'] - self.last_saved_total_counts['out']
                    
                    if hourly_in > 0 or hourly_out > 0:
                        hourly_stmt = text("""
                            INSERT INTO hourly_footfall (channel_id, report_date, hour, in_count, out_count)
                            VALUES (:cid, :rdate, :hour, :inc, :outc)
                            ON CONFLICT (channel_id, report_date, hour)
                            DO UPDATE SET in_count = hourly_footfall.in_count + :inc,
                                          out_count = hourly_footfall.out_count + :outc;
                        """)
                        db.execute(hourly_stmt, {'cid': self.channel_id, 'rdate': self.tracking_date, 'hour': self.last_saved_hour, 'inc': hourly_in, 'outc': hourly_out})
                    
                    self.last_saved_total_counts = self.counts.copy()
                    if not final_save:
                        self.last_saved_hour = current_hour_ist
                db.commit()
            except Exception as e:
                logging.error(f"Error updating counts in DB for {self.channel_name}: {e}")
                db.rollback()

    def _check_for_new_day(self):
        current_date_ist = datetime.now(IST).date()
        if current_date_ist > self.tracking_date:
            logging.info(f"New day detected for PeopleCounter on {self.channel_name}. Resetting counts.")
            self._update_and_log_counts(final_save=True)
            with self.SessionLocal() as db:
                self._reset_counts_for_new_day(db, current_date_ist)
            self.last_saved_total_counts = self.counts.copy()
            self.last_saved_hour = datetime.now(IST).hour

    def run(self):
        # Check for test mode with placeholder frames
        use_placeholder = os.environ.get('USE_PLACEHOLDER_FEED', 'false').lower() == 'true'
        
        if not use_placeholder:
            # Set OpenCV timeout for RTSP streams
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|timeout;5000000'
            cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
            
            # If can't connect after quick attempt, switch to placeholder mode
            if not cap.isOpened():
                logging.warning(f"Could not open PeopleCounter stream for {self.channel_name}, using placeholder")
                use_placeholder = True
            else:
                is_file = any(self.rtsp_url.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mov'])
        
        if use_placeholder:
            # Generate placeholder frames with test pattern
            logging.info(f"Using placeholder feed for {self.channel_name}")
            frame_counter = 0
            while self.is_running:
                frame = np.full((480, 640, 3), (22, 27, 34), dtype=np.uint8)
                cv2.putText(frame, f'{self.channel_name}', (180, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (201, 209, 217), 2)
                cv2.putText(frame, f'Camera Offline - Test Mode', (120, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 150, 255), 2)
                cv2.putText(frame, f'Frame: {frame_counter}', (230, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
                
                line_x = int(frame.shape[1] * 0.5)
                cv2.line(frame, (line_x, 0), (line_x, frame.shape[0]), (0, 255, 0), 2)
                cv2.putText(frame, f"IN: {self.counts['in']}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(frame, f"OUT: {self.counts['out']}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                
                with self.lock:
                    self.latest_frame = frame
                
                frame_counter += 1
                time.sleep(0.1)  # 10 FPS for placeholder
            return

        is_file = any(self.rtsp_url.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mov'])

        while self.is_running:
            self._check_for_new_day()
            ret, frame = cap.read()
            if not ret:
                if is_file:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
                else:
                    logging.warning(f"Reconnecting to PeopleCounter stream {self.channel_name}...")
                    time.sleep(5)
                    cap.release(); cap = cv2.VideoCapture(self.rtsp_url); continue

            results = self.model.track(frame, persist=True, classes=[0], conf=0.5, iou=0.5, verbose=False)
            annotated_frame = results[0].plot() if results and results[0].boxes.id is not None else frame.copy()
            line_x = int(frame.shape[1] * 0.5)

            if results and results[0].boxes.id is not None:
                boxes, track_ids = results[0].boxes.xywh.cpu(), results[0].boxes.id.int().cpu().tolist()
                count_changed = False
                for box, track_id in zip(boxes, track_ids):
                    center_x = int(box[0])
                    history = self.track_history[track_id]
                    history.append(center_x)
                    if len(history) > 30: history.pop(0)
                    if len(history) > 1:
                        prev_x, curr_x = history[-2], history[-1]
                        if prev_x < line_x and curr_x >= line_x:
                            self.counts['out'] += 1; count_changed = True; self.track_history.pop(track_id, None)
                        elif prev_x > line_x and curr_x <= line_x:
                            self.counts['in'] += 1; count_changed = True; self.track_history.pop(track_id, None)
                
                if count_changed:
                    self._update_and_log_counts()
                    self.socketio.emit('count_update', {'channel_id': self.channel_id, 'in_count': self.counts['in'], 'out_count': self.counts['out']})

            cv2.line(annotated_frame, (line_x, 0), (line_x, annotated_frame.shape[0]), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"IN: {self.counts['in']}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(annotated_frame, f"OUT: {self.counts['out']}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            with self.lock:
                self.latest_frame = annotated_frame
        cap.release()
