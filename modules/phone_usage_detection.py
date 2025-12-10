"""
Mobile Phone Usage Detection Module
Detects employees using mobile phones in restricted areas
Uses YOLO model to detect 'phone' class and triggers alerts
"""

import cv2
import numpy as np
import time
import logging
import os
import json
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class PhoneUsageDetection:
    """
    Detects mobile phone usage by employees and triggers alerts
    Uses best.pt model with 'phone' class detection
    """
    
    def __init__(self, channel_id, socketio=None, db_manager=None, app=None, config=None):
        """
        Initialize Phone Usage Detection module
        
        Args:
            channel_id: Unique identifier for the camera channel
            socketio: SocketIO instance for real-time alerts
            db_manager: Database manager for storing alerts
            app: Flask app instance for database context
            config: Configuration dictionary
        """
        self.channel_id = channel_id
        self.socketio = socketio
        self.db_manager = db_manager
        self.app = app
        
        # Configuration
        self.config = config or {}
        self.confidence_threshold = self.config.get('confidence_threshold', 0.4)
        self.alert_cooldown = self.config.get('alert_cooldown', 30)  # seconds between alerts
        self.detection_duration = self.config.get('detection_duration', 2)  # seconds to confirm detection
        
        # Model configuration
        self.model_path = self.config.get('model_path', 'models/best.pt')
        self.target_class = 'phone_use'  # Class name in best.pt model
        
        # Load shared YOLO model
        from modules.model_manager import get_shared_model
        
        logger.info(f"Loading shared YOLO model for phone detection: {self.model_path}")
        try:
            self.model = get_shared_model(self.model_path, device='auto')
            logger.info("Shared phone detection model loaded successfully")
            
            # Verify 'phone' class exists in model
            if hasattr(self.model, 'names'):
                available_classes = list(self.model.names.values())
                if self.target_class not in available_classes:
                    logger.warning(f"⚠️ '{self.target_class}' class not found in model. Available: {available_classes}")
                else:
                    logger.info(f"✓ '{self.target_class}' class found in model")
                    
        except Exception as e:
            logger.error(f"Failed to load phone detection model: {e}")
            raise
        
        # State tracking
        self.detection_start_time = None
        self.last_alert_time = 0
        self.consecutive_detections = 0
        self.detection_count = 0
        self.total_alerts = 0
        
        # Frame tracking
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()
        
        # Snapshot directory
        self.snapshot_dir = Path('static/phone_snapshots')
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"PhoneUsageDetection initialized for channel {channel_id}")
        logger.info(f"  - Confidence threshold: {self.confidence_threshold}")
        logger.info(f"  - Alert cooldown: {self.alert_cooldown}s")
        logger.info(f"  - Detection duration: {self.detection_duration}s")
    
    def __del__(self):
        """Cleanup: release shared model reference"""
        if hasattr(self, 'model') and self.model:
            try:
                from modules.model_manager import release_shared_model
                release_shared_model(self.model_path)
                logger.info(f"Released shared model: {self.model_path}")
            except Exception as e:
                logger.error(f"Error releasing model: {e}")
    
    def process_frame(self, frame):
        """
        Process frame for phone usage detection
        
        Args:
            frame: Input video frame (numpy array)
            
        Returns:
            dict: {
                'frame': annotated frame,
                'status': detection status dict,
                'metadata': additional information
            }
        """
        self.frame_count += 1
        current_time = time.time()
        
        # Calculate FPS
        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time
        
        # Run YOLO detection
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        
        # Extract phone detections
        phone_detections = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                boxes = result.boxes
                
                for i in range(len(boxes)):
                    cls = int(boxes.cls[i].cpu().numpy())
                    class_name = result.names[cls]
                    
                    # Only process 'phone' class
                    if class_name.lower() == self.target_class:
                        conf = float(boxes.conf[i].cpu().numpy())
                        box = boxes.xyxy[i].cpu().numpy()
                        x1, y1, x2, y2 = box
                        
                        phone_detections.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': conf,
                            'class_name': class_name
                        })
        
        # Update detection state
        phone_detected = len(phone_detections) > 0
        
        if phone_detected:
            self.detection_count += 1
            self.consecutive_detections += 1
            
            # Start tracking detection duration
            if self.detection_start_time is None:
                self.detection_start_time = current_time
            
            # Check if detection has persisted long enough
            detection_duration = current_time - self.detection_start_time
            
            # Trigger alert if conditions are met
            if (detection_duration >= self.detection_duration and 
                current_time - self.last_alert_time >= self.alert_cooldown):
                self._trigger_phone_alert(frame, phone_detections, detection_duration)
                self.last_alert_time = current_time
                self.detection_start_time = None  # Reset for next detection
        else:
            # Reset detection tracking
            self.consecutive_detections = 0
            self.detection_start_time = None
        
        # Draw visualization
        annotated_frame = self._draw_visualization(frame, phone_detections, phone_detected)
        
        # Prepare status
        status = {
            'phone_detected': phone_detected,
            'detection_count': len(phone_detections),
            'total_detections': self.detection_count,
            'total_alerts': self.total_alerts,
            'consecutive_detections': self.consecutive_detections,
            'fps': round(self.fps, 2)
        }
        
        # Prepare metadata
        metadata = {
            'channel_id': self.channel_id,
            'timestamp': datetime.now().isoformat(),
            'detections': phone_detections
        }
        
        return {
            'frame': annotated_frame,
            'status': status,
            'metadata': metadata
        }
    
    def _trigger_phone_alert(self, frame, detections, duration):
        """
        Trigger alert for phone usage detection
        
        Args:
            frame: Current video frame
            detections: List of phone detections
            duration: Detection duration in seconds
        """
        self.total_alerts += 1
        timestamp = datetime.now()
        
        # Generate snapshot filename
        snapshot_filename = f"phone_{self.channel_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
        snapshot_path = self.snapshot_dir / snapshot_filename
        
        # Create annotated snapshot
        snapshot = frame.copy()
        
        # Add alert header
        cv2.rectangle(snapshot, (0, 0), (snapshot.shape[1], 60), (0, 0, 255), -1)
        cv2.putText(snapshot, "MOBILE PHONE USAGE DETECTED", (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        # Draw detection boxes
        for detection in detections:
            bbox = detection['bbox']
            conf = detection['confidence']
            x1, y1, x2, y2 = bbox
            
            # Draw red box
            cv2.rectangle(snapshot, (x1, y1), (x2, y2), (0, 0, 255), 3)
            
            # Add label
            label = f"Phone {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(snapshot, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 0, 255), -1)
            cv2.putText(snapshot, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add timestamp
        timestamp_text = timestamp.strftime('%Y-%m-%d %H:%M:%S')
        cv2.putText(snapshot, timestamp_text, (10, snapshot.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Save snapshot
        cv2.imwrite(str(snapshot_path), snapshot)
        file_size = snapshot_path.stat().st_size
        
        logger.info(f"Phone usage alert triggered for {self.channel_id}")
        logger.info(f"  - Detection count: {len(detections)}")
        logger.info(f"  - Duration: {duration:.2f}s")
        logger.info(f"  - Snapshot saved: {snapshot_filename}")
        
        # Prepare alert data
        alert_data = {
            'channel_id': self.channel_id,
            'detection_count': len(detections),
            'duration': round(duration, 2),
            'timestamp': timestamp.isoformat(),
            'detections': detections
        }
        
        # Save to database
        if self.db_manager and self.app:
            try:
                with self.app.app_context():
                    snapshot_id = self.db_manager.save_phone_snapshot(
                        channel_id=self.channel_id,
                        snapshot_filename=snapshot_filename,
                        snapshot_path=str(snapshot_path),
                        alert_message=f"Phone usage detected ({len(detections)} instance{'s' if len(detections) > 1 else ''})",
                        alert_data=alert_data,
                        file_size=file_size,
                        detection_count=len(detections),
                        detection_time=timestamp
                    )
                    
                    if snapshot_id:
                        logger.info(f"Phone snapshot saved to database with ID: {snapshot_id}")
                    
            except Exception as e:
                logger.error(f"Error saving phone snapshot to database: {e}")
        
        # Send SocketIO alert
        if self.socketio:
            try:
                self.socketio.emit('phone_alert', {
                    'channel_id': self.channel_id,
                    'message': f'Phone usage detected in {self.channel_id}',
                    'snapshot': f'/static/phone_snapshots/{snapshot_filename}',
                    'detection_count': len(detections),
                    'timestamp': timestamp_text,
                    'alert_data': alert_data
                }, namespace='/')
                
                logger.info(f"SocketIO alert sent for phone detection")
                
            except Exception as e:
                logger.error(f"Error sending SocketIO alert: {e}")
    
    def _draw_visualization(self, frame, detections, phone_detected):
        """
        Draw detection visualization on frame
        
        Args:
            frame: Input frame
            detections: List of phone detections
            phone_detected: Boolean indicating if phone is detected
            
        Returns:
            annotated_frame: Frame with visualization
        """
        annotated_frame = frame.copy()
        
        # Draw detection boxes
        for detection in detections:
            bbox = detection['bbox']
            conf = detection['confidence']
            x1, y1, x2, y2 = bbox
            
            # Draw red box for phone
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Add label
            label = f"Phone {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 8), 
                         (x1 + label_size[0] + 4, y1), (0, 0, 255), -1)
            cv2.putText(annotated_frame, label, (x1 + 2, y1 - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add status overlay
        status_bg_color = (0, 0, 200) if phone_detected else (0, 100, 0)
        cv2.rectangle(annotated_frame, (0, 0), (300, 30), status_bg_color, -1)
        
        status_text = f"PHONE DETECTED!" if phone_detected else "Monitoring..."
        cv2.putText(annotated_frame, status_text, (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add detection count
        if phone_detected:
            count_text = f"Count: {len(detections)}"
            cv2.putText(annotated_frame, count_text, (10, annotated_frame.shape[0] - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Add timestamp
        timestamp_text = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cv2.putText(annotated_frame, timestamp_text, (10, annotated_frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated_frame
    
    def get_statistics(self):
        """
        Get current detection statistics
        
        Returns:
            dict: Statistics dictionary
        """
        return {
            'channel_id': self.channel_id,
            'total_detections': self.detection_count,
            'total_alerts': self.total_alerts,
            'consecutive_detections': self.consecutive_detections,
            'fps': round(self.fps, 2)
        }
    
    def get_status(self):
        """
        Get current module status
        
        Returns:
            dict: Status dictionary
        """
        return {
            'module': 'PhoneUsageDetection',
            'channel_id': self.channel_id,
            'active': True,
            'total_detections': self.detection_count,
            'total_alerts': self.total_alerts,
            'fps': round(self.fps, 2)
        }
