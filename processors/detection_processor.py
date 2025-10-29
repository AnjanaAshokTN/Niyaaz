import cv2
import threading
import time
import logging
import numpy as np
from collections import defaultdict

# --- Configuration ---
QPOS_TIME_THRESHOLD_SEC = 5.0
QPOS_DETECTION_THRESHOLD = 3

class DetectionProcessor(threading.Thread):
    def __init__(self, rtsp_url, channel_id, channel_name, tasks, detection_callback):
        super().__init__(name=f"Detection-{channel_name}")
        self.rtsp_url = rtsp_url
        self.channel_id = channel_id
        self.channel_name = channel_name
        self.tasks = tasks
        self.detection_callback = detection_callback
        
        self.is_running = True
        self.last_detection_times = {task['app_name']: 0 for task in self.tasks}
        self.cooldown = 30
        self.gif_duration_seconds = 3
        self.fps = 10
        self.qpos_persistence_tracker = defaultdict(list)
        self.latest_frame = None
        self.lock = threading.Lock()

    def stop(self):
        self.is_running = False

    def shutdown(self):
        logging.info(f"Shutting down DetectionProcessor for {self.channel_name}")
        self.is_running = False

    def get_frame(self):
        """Get the latest frame for video streaming"""
        with self.lock:
            if self.latest_frame is not None:
                success, jpeg = cv2.imencode('.jpg', self.latest_frame)
                return jpeg.tobytes() if success else b''
            else:
                # Return placeholder while connecting
                placeholder = np.full((480, 640, 3), (22, 27, 34), dtype=np.uint8)
                cv2.putText(placeholder, 'Connecting...', (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (201, 209, 217), 2)
                _, jpeg = cv2.imencode('.jpg', placeholder)
                return jpeg.tobytes()

    def run(self):
        cap = cv2.VideoCapture(self.rtsp_url)
        if not cap.isOpened():
            logging.error(f"Could not open stream for {self.channel_name}: {self.rtsp_url}")
            return

        is_file = any(self.rtsp_url.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mov'])
        
        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                if is_file:
                    logging.info(f"Restarting video file for {self.channel_name}...")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    logging.warning(f"Reconnecting to stream {self.channel_name}...")
                    time.sleep(5)
                    cap.release()
                    cap = cv2.VideoCapture(self.rtsp_url)
                    continue

            with self.lock:
                self.latest_frame = frame.copy()

            current_time = time.time()

            for task in self.tasks:
                app_name = task['app_name']
                if current_time - self.last_detection_times[app_name] < self.cooldown:
                    continue

                results = task['model'](frame, conf=task['confidence'], classes=task.get('target_class_id'), verbose=False)

                if app_name == 'QPOS':
                    self.qpos_persistence_tracker[self.channel_id] = [ts for ts in self.qpos_persistence_tracker[self.channel_id] if current_time - ts <= QPOS_TIME_THRESHOLD_SEC]
                    if results and len(results[0].boxes) > 0:
                        self.qpos_persistence_tracker[self.channel_id].append(current_time)
                    
                    if len(self.qpos_persistence_tracker[self.channel_id]) >= QPOS_DETECTION_THRESHOLD:
                        self.last_detection_times[app_name] = current_time
                        self.qpos_persistence_tracker[self.channel_id] = []
                        annotated_frame = results[0].plot()
                        self.detection_callback(app_name, self.channel_id, [annotated_frame], "QPOS Screen Off detected.", False)
                
                elif results and len(results[0].boxes) > 0:
                    self.last_detection_times[app_name] = current_time
                    annotated_frame = results[0].plot()
                    detection_count = len(results[0].boxes)
                    
                    if task.get('is_gif', False):
                        gif_frames = [annotated_frame]
                        # Capture subsequent frames for the GIF
                        for _ in range(self.gif_duration_seconds * self.fps - 1):
                            ret_gif, frame_gif = cap.read()
                            if not ret_gif: break
                            gif_frames.append(frame_gif.copy())
                            time.sleep(1 / self.fps)
                        self.detection_callback(app_name, self.channel_id, gif_frames, f"{app_name} detected ({detection_count} objects).", True)
                    else:
                        self.detection_callback(app_name, self.channel_id, [annotated_frame], f"{app_name} detected ({detection_count} objects).", False)
        
        cap.release()
