"""
Fall Detection Module for Sakshi.AI
- Detects person falls using YOLO and aspect ratio analysis
- Triggers alerts when a person is detected in a fallen position
- Takes snapshot and saves to database
- Emits real-time Socket.IO events
"""

import time
import cv2
import numpy as np
import logging
import os
import torch
from datetime import datetime
from ultralytics import YOLO
from pathlib import Path
from .model_manager import get_shared_model, release_shared_model

logger = logging.getLogger(__name__)


class FallDetection:
    """Person fall detection with snapshot capture and alerting"""
    
    def __init__(self, channel_id, socketio, db_manager=None, app=None):
        """
        Initialize fall detection module
        
        Args:
            channel_id: Unique identifier for this channel
            socketio: Socket.IO instance for real-time updates
            db_manager: Database manager for storing alerts
            app: Flask app instance for database context
        """
        self.channel_id = channel_id
        self.socketio = socketio
        self.db_manager = db_manager
        self.app = app
        
        # Detection configuration - Use PyTorch model (TensorRT engines cause segfault)
        self.model_weight = "models/yolov10s.pt"
        self.conf_threshold = 0.3  # Confidence threshold for person detection
        self.nms_iou = 0.45
        
        # Fall detection thresholds
        # A person is considered "fallen" when height < width (aspect ratio < 1)
        self.fall_threshold = 0  # If (height - width) < 0, person is horizontal
        
        # Alert configuration
        self.alert_cooldown = 15.0  # seconds between repeated alerts for same fall
        self.fall_duration_threshold = 0.0  # person must be fallen for 0 seconds before alert
        self.snapshot_dir = Path("static/fall_snapshots")
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize YOLO detector with shared model manager
        logger.info(f"Loading shared YOLO model for fall detection: {self.model_weight}")
        try:
            self.yolo = get_shared_model(self.model_weight, device='auto')
            logger.info("Shared fall detection model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load shared fall detection model: {e}")
            raise
        
        # State storage
        self.last_alert_time = 0.0
        self.fall_detection_count = 0
        self.total_alerts = 0
        self.current_falls = {}  # person_id -> {'first_detected': timestamp, 'bbox': [x1,y1,x2,y2]}
        
        # Frame processing
        self.frame_count = 0
        self.last_update_time = time.time()
        
        logger.info(f"FallDetection initialized for channel {channel_id}")
    
    def __del__(self):
        """Cleanup: Release shared model reference when fall detection is destroyed"""
        try:
            if hasattr(self, 'model_weight'):
                release_shared_model(self.model_weight, device='auto')
                logger.debug(f"Released shared model reference: {self.model_weight}")
        except Exception as e:
            logger.warning(f"Error releasing shared model: {e}")
    
    def process_frame(self, frame):
        """
        Process a single frame for fall detection
        
        Args:
            frame: Input video frame
            
        Returns:
            Annotated frame with detection boxes and alerts
        """
        if frame is None:
            return None
        
        self.frame_count += 1
        t_now = time.time()
        
        # Clear GPU cache periodically to prevent memory fragmentation
        if torch.cuda.is_available() and self.frame_count % 100 == 0:
            torch.cuda.empty_cache()
        
        # YOLO inference
        results = self.yolo(frame, conf=self.conf_threshold, iou=self.nms_iou, verbose=False)
        
        # Process detections
        person_detections = []
        fallen_persons = []
        
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = self.yolo.names.get(cls_id, str(cls_id))
                
                # Only process person detections
                if cls_name.lower() == 'person':
                    x1, y1, x2, y2 = map(int, xyxy.tolist())
                    
                    # Calculate dimensions
                    height = y2 - y1
                    width = x2 - x1
                    thresh = height - width  # Positive = standing, Negative = fallen
                    
                    detection = {
                        "bbox": [x1, y1, x2, y2],
                        "confidence": conf,
                        "height": height,
                        "width": width,
                        "aspect_ratio": height / width if width > 0 else 0,
                        "is_fallen": thresh < self.fall_threshold
                    }
                    
                    person_detections.append(detection)
                    
                    if detection["is_fallen"]:
                        fallen_persons.append(detection)
        
        self.fall_detection_count = len(fallen_persons)
        
        # Track fallen persons and trigger alerts
        current_time = t_now
        current_fall_ids = set()
        
        for i, fall in enumerate(fallen_persons):
            person_id = f"fall_{i}_{fall['bbox'][0]//50}_{fall['bbox'][1]//50}"
            current_fall_ids.add(person_id)
            
            if person_id not in self.current_falls:
                # New fall detected - start tracking
                self.current_falls[person_id] = {
                    'first_detected': current_time,
                    'bbox': fall['bbox'],
                    'alerted': False
                }
                logger.debug(f"New fall detected: {person_id}")
            else:
                # Existing fall - check if alert should be triggered
                fall_data = self.current_falls[person_id]
                fall_duration = current_time - fall_data['first_detected']
                
                # Trigger alert if duration threshold met and cooldown passed
                if (not fall_data['alerted'] and 
                    fall_duration >= self.fall_duration_threshold and
                    (current_time - self.last_alert_time >= self.alert_cooldown)):
                    
                    self._trigger_fall_alert(frame, fall, fall_duration, current_time)
                    self.last_alert_time = current_time
                    fall_data['alerted'] = True
        
        # Clean up falls that are no longer detected
        for person_id in list(self.current_falls.keys()):
            if person_id not in current_fall_ids:
                del self.current_falls[person_id]
        
        # Draw visualization
        vis_frame = self._draw_visualization(frame, person_detections, fallen_persons)
        
        # Send real-time updates
        if t_now - self.last_update_time >= 1.0:
            self._send_realtime_update()
            self.last_update_time = t_now
        
        # Return structured result format consistent with other modules
        return {
            'frame': vis_frame,
            'status': {
                'persons_detected': len(person_detections),
                'falls_detected': len(fallen_persons),
                'total_alerts': self.total_alerts,
                'active_falls': len(self.current_falls),
                'fall_duration_threshold': self.fall_duration_threshold
            },
            'metadata': {
                'frame_count': self.frame_count,
                'timestamp': t_now,
                'channel_id': self.channel_id,
                'person_detections': [
                    {
                        'bbox': p['bbox'],
                        'confidence': p['confidence'],
                        'height': p['height'],
                        'width': p['width'],
                        'aspect_ratio': p['aspect_ratio'],
                        'is_fallen': p['is_fallen']
                    } for p in person_detections
                ],
                'fallen_persons': [
                    {
                        'bbox': f['bbox'],
                        'confidence': f['confidence'],
                        'aspect_ratio': f['aspect_ratio']
                    } for f in fallen_persons
                ]
            }
        }
    
    def _trigger_fall_alert(self, frame, fall_detection, duration, timestamp):
        """Trigger fall detection alert with snapshot"""
        self.total_alerts += 1
        
        # Generate filename
        dt = datetime.fromtimestamp(timestamp)
        filename = f"fall_{self.channel_id}_{dt.strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = self.snapshot_dir / filename
        
        # Save snapshot
        try:
            # Draw bounding box on snapshot for better context
            snapshot = frame.copy()
            x1, y1, x2, y2 = fall_detection['bbox']
            cv2.rectangle(snapshot, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(snapshot, "PERSON FALL DETECTED!", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imwrite(str(filepath), snapshot)
            file_size = os.path.getsize(filepath)
            
            logger.warning(f"FALL DETECTED! Snapshot saved: {filename}")
            
            # Prepare alert data
            alert_message = f"Person fall detected - Duration: {duration:.1f}s"
            alert_data = {
                'bbox': fall_detection['bbox'],
                'confidence': float(fall_detection['confidence']),
                'height': fall_detection['height'],
                'width': fall_detection['width'],
                'aspect_ratio': float(fall_detection['aspect_ratio']),
                'fall_duration': round(duration, 1),
                'channel_id': self.channel_id
            }
            
            # Save to database
            if self.db_manager and self.app:
                try:
                    with self.app.app_context():
                        snapshot_id = self.db_manager.save_fall_snapshot(
                            channel_id=self.channel_id,
                            snapshot_filename=filename,
                            snapshot_path=str(filepath),
                            alert_message=alert_message,
                            alert_data=alert_data,
                            file_size=file_size,
                            fall_duration=duration
                        )
                        
                        logger.info(f"Fall snapshot saved to database: ID {snapshot_id}")
                        
                        # Emit real-time notification
                        self.socketio.emit('fall_detected', {
                            'snapshot_id': snapshot_id,
                            'channel_id': self.channel_id,
                            'snapshot_filename': filename,
                            'snapshot_url': f"/static/fall_snapshots/{filename}",
                            'alert_message': alert_message,
                            'fall_duration': round(duration, 1),
                            'timestamp': dt.isoformat()
                        })
                        
                except Exception as e:
                    logger.error(f"Error saving fall snapshot to database: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
            
            # Emit Socket.IO alert
            self.socketio.emit('fall_alert', {
                'channel_id': self.channel_id,
                'alert_message': alert_message,
                'fall_duration': round(duration, 1),
                'snapshot_filename': filename,
                'snapshot_url': f"/static/fall_snapshots/{filename}",
                'timestamp': dt.isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error saving fall snapshot: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _draw_visualization(self, frame, person_detections, fallen_persons):
        """Draw bounding boxes and detection info on frame"""
        vis = frame.copy()
        
        # Draw only fallen person detections (standing persons are hidden)
        for person in person_detections:
            x1, y1, x2, y2 = person['bbox']
            
            if person['is_fallen']:
                # Red box for fallen person
                color = (0, 0, 255)
                label = f"FALL! ({person['confidence']:.2f})"
                thickness = 3
                
                # Draw bounding box
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
                
                # Draw label with background
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(vis, (x1, y1 - label_size[1] - 10), 
                             (x1 + label_size[0], y1), color, -1)
                cv2.putText(vis, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            # else:
            #     # Green box for standing person - HIDDEN (not displayed)
            #     # Standing persons are tracked but not shown in visualization
        
        # Draw alert banner if falls detected
        if fallen_persons:
            # Red banner at top
            banner_height = 60
            cv2.rectangle(vis, (0, 0), (vis.shape[1], banner_height), (0, 0, 255), -1)
            
            text = f"FALL ALERT! {len(fallen_persons)} person(s) fallen"
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            text_x = (vis.shape[1] - text_size[0]) // 2
            
            cv2.putText(vis, text, (text_x, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        
        # Draw statistics at bottom
        stats_y = vis.shape[0] - 60
        stats_bg = (50, 50, 50)
        cv2.rectangle(vis, (0, stats_y - 10), (450, vis.shape[0]), stats_bg, -1)
        
        cv2.putText(vis, f"Persons Detected: {len(person_detections)}", (10, stats_y + 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis, f"Falls Detected: {len(fallen_persons)}", (10, stats_y + 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if fallen_persons else (0, 255, 0), 2)
        
        return vis
    
    def _send_realtime_update(self):
        """Send real-time statistics update via Socket.IO"""
        self.socketio.emit('fall_detection_update', {
            'channel_id': self.channel_id,
            'current_falls': self.fall_detection_count,
            'total_alerts': self.total_alerts,
            'frame_count': self.frame_count,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_statistics(self):
        """Get current detection statistics"""
        return {
            'current_falls': self.fall_detection_count,
            'total_alerts': self.total_alerts,
            'frame_count': self.frame_count
        }
    
    def get_current_status(self):
        """Get current module status"""
        return {
            'active': True,
            'current_falls': self.fall_detection_count,
            'total_alerts': self.total_alerts
        }
    
    def update_config(self, config):
        """Update detection configuration"""
        if 'confidence_threshold' in config:
            self.conf_threshold = float(config['confidence_threshold'])
        if 'alert_cooldown' in config:
            self.alert_cooldown = float(config['alert_cooldown'])
        if 'fall_duration_threshold' in config:
            self.fall_duration_threshold = float(config['fall_duration_threshold'])
        
        logger.info(f"FallDetection config updated for channel {self.channel_id}")
