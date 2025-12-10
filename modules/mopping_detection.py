"""
Mopping Detection Module for Sakshi.AI
- Detects mopping activity using custom YOLO model (best.pt)
- Triggers alerts when mopping is detected
- Takes snapshot and logs timestamp
- Stores detection events in database
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


class MoppingDetection:
    """Mopping detection with snapshot capture and alerting"""
    
    def __init__(self, channel_id, socketio, db_manager=None, app=None):
        """
        Initialize mopping detection module
        
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
        self.model_weight = "models/best.pt"
        self.conf_threshold = 0.5  # Confidence threshold for detections
        self.nms_iou = 0.45
        
        # Target class for detection
        self.mopping_class = "mopping"
        
        # Alert configuration
        self.alert_cooldown = 30.0  # seconds between repeated alerts for same detection
        self.detection_duration_threshold = 2.0  # seconds - mopping must be detected for this long before alert
        self.snapshot_dir = Path("static/mopping_snapshots")
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize YOLO detector with shared model manager
        logger.info(f"Loading shared YOLO model for mopping detection: {self.model_weight}")
        try:
            self.yolo = get_shared_model(self.model_weight, device='auto')
            logger.info("Shared mopping detection model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load shared mopping detection model: {e}")
            raise
        
        # State storage
        self.last_alert_time = 0.0
        self.detection_count = 0
        self.total_alerts = 0
        self.current_detections = []
        
        # Enhanced metrics for analytics
        self.total_detections = 0
        self.peak_detections = 0
        self.detection_sessions = 0  # Number of separate detection instances
        self.avg_detection_confidence = 0.0
        self.highest_confidence = 0.0
        
        # Continuous detection tracking
        self.first_detection_time = None  # When mopping was first detected
        self.consecutive_detection = False  # Whether mopping is currently being detected
        
        # Frame processing
        self.frame_count = 0
        self.last_update_time = time.time()
        
        logger.info(f"MoppingDetection initialized for channel {channel_id}")
    
    def __del__(self):
        """Cleanup: Release shared model reference when mopping detection is destroyed"""
        try:
            if hasattr(self, 'model_weight'):
                release_shared_model(self.model_weight, device='auto')
                logger.debug(f"Released shared model reference: {self.model_weight}")
        except Exception as e:
            logger.warning(f"Error releasing shared model: {e}")
    
    def process_frame(self, frame):
        """
        Process a single frame for mopping detection
        
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
        detections = []
        mopping_detected = False
        all_detections_debug = []  # For debugging
        
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = self.yolo.names.get(cls_id, str(cls_id))
                
                # Log all detections for debugging (every 100 frames)
                if self.frame_count % 100 == 0:
                    all_detections_debug.append(f"{cls_name}({conf:.2f})")
                
                # Only process mopping class
                if cls_name != self.mopping_class:
                    continue
                
                x1, y1, x2, y2 = map(int, xyxy.tolist())
                
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf,
                    "class_name": cls_name,
                    "class_id": cls_id
                })
                
                mopping_detected = True
        
        self.current_detections = detections
        self.detection_count = len(detections)
        
        # Debug logging every 100 frames
        if self.frame_count % 100 == 0:
            if all_detections_debug:
                logger.info(f"🔍 Frame {self.frame_count} - Detected: {', '.join(all_detections_debug)}")
            else:
                logger.debug(f"Frame {self.frame_count} - No detections above threshold {self.conf_threshold}")
        
        # Update enhanced metrics
        if mopping_detected:
            self.total_detections += 1
            if len(detections) > self.peak_detections:
                self.peak_detections = len(detections)
            
            # Track confidence levels
            confidences = [d['confidence'] for d in detections]
            if confidences:
                max_conf = max(confidences)
                if max_conf > self.highest_confidence:
                    self.highest_confidence = max_conf
                
                # Update average confidence (running average)
                avg_conf = sum(confidences) / len(confidences)
                if self.avg_detection_confidence == 0.0:
                    self.avg_detection_confidence = avg_conf
                else:
                    # Exponential moving average
                    self.avg_detection_confidence = 0.9 * self.avg_detection_confidence + 0.1 * avg_conf
        
        # Track continuous detection time
        if mopping_detected:
            # If this is the first detection, start the timer
            if self.first_detection_time is None:
                self.first_detection_time = t_now
                self.consecutive_detection = True
                self.detection_sessions += 1
                logger.info(f"🧹 Mopping activity detected at {datetime.fromtimestamp(t_now).strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Check if mopping has been detected continuously for required duration
            detection_duration = t_now - self.first_detection_time
            
            # Trigger alert if duration threshold met and cooldown passed
            if (detection_duration >= self.detection_duration_threshold and 
                (t_now - self.last_alert_time >= self.alert_cooldown)):
                self._trigger_mopping_alert(frame, detections, t_now)
                self.last_alert_time = t_now
                # Reset detection timer after alert
                self.first_detection_time = None
                self.consecutive_detection = False
        else:
            # Mopping not detected - reset the timer
            if self.first_detection_time is not None:
                logger.debug(f"Mopping detection interrupted after {t_now - self.first_detection_time:.2f}s")
            self.first_detection_time = None
            self.consecutive_detection = False
        
        # Draw visualization
        vis_frame = self._draw_visualization(frame, detections, mopping_detected)
        
        # Send real-time updates
        if t_now - self.last_update_time >= 1.0:
            self._send_realtime_update()
            self.last_update_time = t_now
        
        # Return structured result format consistent with other modules
        return {
            'frame': vis_frame,
            'status': {
                'mopping_detected': mopping_detected,
                'current_detections': len(detections),
                'total_alerts': self.total_alerts,
                'consecutive_detection': self.consecutive_detection,
                'detection_duration': (t_now - self.first_detection_time) if self.first_detection_time else 0
            },
            'metadata': {
                'frame_count': self.frame_count,
                'timestamp': t_now,
                'channel_id': self.channel_id,
                'detections': [
                    {
                        'bbox': d['bbox'],
                        'confidence': d['confidence'],
                        'class_name': d['class_name'],
                        'class_id': d['class_id']
                    } for d in detections
                ]
            }
        }
    
    def _trigger_mopping_alert(self, frame, detections, timestamp):
        """Trigger mopping detection alert with snapshot and timestamp"""
        self.total_alerts += 1
        
        # Generate filename with timestamp
        dt = datetime.fromtimestamp(timestamp)
        filename = f"mopping_{self.channel_id}_{dt.strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = self.snapshot_dir / filename
        
        # Save snapshot
        try:
            cv2.imwrite(str(filepath), frame)
            file_size = os.path.getsize(filepath)
            
            logger.info(f"🧹 ALERT: Mopping activity detected at {dt.strftime('%Y-%m-%d %H:%M:%S')}! Snapshot saved: {filename}")
            
            # Prepare alert data
            alert_message = f"ALERT: Mopping activity detected at {dt.strftime('%Y-%m-%d %H:%M:%S')} - {len(detections)} instance(s) found"
            alert_data = {
                'detections': [
                    {
                        'bbox': d['bbox'],
                        'confidence': float(d['confidence']),
                        'class_name': d['class_name']
                    }
                    for d in detections
                ],
                'detection_count': len(detections),
                'mopping_detected': True,
                'channel_id': self.channel_id,
                'timestamp': dt.isoformat(),
                'time_formatted': dt.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Save to database
            if self.db_manager and self.app:
                try:
                    with self.app.app_context():
                        snapshot_id = self.db_manager.save_mopping_snapshot(
                            channel_id=self.channel_id,
                            snapshot_filename=filename,
                            snapshot_path=str(filepath),
                            alert_message=alert_message,
                            alert_data=alert_data,
                            file_size=file_size,
                            detection_count=len(detections),
                            detection_time=dt
                        )
                        
                        logger.info(f"Mopping snapshot saved to database: ID {snapshot_id}")
                        
                        # Emit real-time notification
                        self.socketio.emit('mopping_detected', {
                            'snapshot_id': snapshot_id,
                            'channel_id': self.channel_id,
                            'snapshot_filename': filename,
                            'snapshot_url': f"/static/mopping_snapshots/{filename}",
                            'alert_message': alert_message,
                            'detection_count': len(detections),
                            'timestamp': dt.isoformat(),
                            'time_formatted': dt.strftime('%Y-%m-%d %H:%M:%S')
                        })
                        
                except Exception as e:
                    logger.error(f"Error saving mopping snapshot to database: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
            
            # Emit Socket.IO alert
            self.socketio.emit('mopping_alert', {
                'channel_id': self.channel_id,
                'alert_message': alert_message,
                'detection_count': len(detections),
                'snapshot_filename': filename,
                'snapshot_url': f"/static/mopping_snapshots/{filename}",
                'timestamp': dt.isoformat(),
                'time_formatted': dt.strftime('%Y-%m-%d %H:%M:%S')
            })
            
        except Exception as e:
            logger.error(f"Error saving mopping snapshot: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _draw_visualization(self, frame, detections, mopping_detected):
        """Draw bounding boxes and detection info on frame"""
        vis = frame.copy()
        
        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            cls_name = det['class_name']
            
            # Color coding: Blue for mopping
            color = (255, 165, 0)  # Orange/Blue for mopping activity
            
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)
            
            # Draw label with confidence
            label = f"{cls_name}: {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Background for label
            cv2.rectangle(vis, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Label text
            cv2.putText(vis, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw alert banner if mopping detected
        if mopping_detected:
            # Calculate detection duration
            if self.first_detection_time is not None:
                current_time = time.time()
                detection_duration = current_time - self.first_detection_time
                remaining_time = max(0, self.detection_duration_threshold - detection_duration)
                
                # Color changes from blue to orange as we approach threshold
                if detection_duration >= self.detection_duration_threshold:
                    banner_color = (0, 140, 255)  # Orange - alert triggered
                    text = f"🧹 MOPPING ACTIVITY DETECTED - ALERT!"
                else:
                    banner_color = (255, 165, 0)  # Blue - detecting
                    text = f"🧹 Mopping Detected - Alert in {remaining_time:.1f}s"
            else:
                banner_color = (255, 165, 0)  # Blue
                text = f"🧹 MOPPING ACTIVITY DETECTED"
            
            # Banner at top
            cv2.rectangle(vis, (0, 0), (vis.shape[1], 50), banner_color, -1)
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            text_x = (vis.shape[1] - text_size[0]) // 2
            cv2.putText(vis, text, (text_x, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw statistics at bottom
        stats_y = vis.shape[0] - 60
        stats_bg = (50, 50, 50)
        cv2.rectangle(vis, (0, stats_y - 10), (500, vis.shape[0]), stats_bg, -1)
        
        cv2.putText(vis, f"Mopping: {'YES' if mopping_detected else 'NO'}", 
                   (10, stats_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis, f"Current Detections: {len(detections)}", (10, stats_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis, f"Total Alerts: {self.total_alerts}", (10, stats_y + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return vis
    
    def _send_realtime_update(self):
        """Send real-time statistics update via Socket.IO"""
        self.socketio.emit('mopping_detection_update', {
            'channel_id': self.channel_id,
            'current_detections': self.detection_count,
            'total_alerts': self.total_alerts,
            'frame_count': self.frame_count,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_statistics(self):
        """Get current detection statistics"""
        return {
            'current_detections': self.detection_count,
            'total_alerts': self.total_alerts,
            'frame_count': self.frame_count,
            'total_detections': self.total_detections,
            'peak_detections': self.peak_detections,
            'detection_sessions': self.detection_sessions,
            'avg_confidence': round(self.avg_detection_confidence, 2),
            'highest_confidence': round(self.highest_confidence, 2)
        }
    
    def get_current_status(self):
        """Get current module status"""
        return {
            'active': True,
            'current_detections': self.detection_count,
            'total_alerts': self.total_alerts
        }
    
    def update_config(self, config):
        """Update detection configuration"""
        if 'confidence_threshold' in config:
            self.conf_threshold = float(config['confidence_threshold'])
        if 'alert_cooldown' in config:
            self.alert_cooldown = float(config['alert_cooldown'])
        
        logger.info(f"MoppingDetection config updated for channel {self.channel_id}")
