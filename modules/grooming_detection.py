"""
Grooming Standards Detection Module
Detects violations in employee grooming standards including:
- Missing uniform/name tag
- Prohibited items (long hair without cover)
- Required items (beard net if applicable)
"""

import cv2
import numpy as np
import time
from datetime import datetime
from ultralytics import YOLO
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GroomingDetection:
    def __init__(self, channel_id, video_source, socketio=None, db_manager=None, config=None):
        """
        Initialize Grooming Detection module
        
        Args:
            channel_id: Unique identifier for the camera channel
            video_source: Video file path or camera index
            socketio: SocketIO instance for real-time alerts
            db_manager: Database manager for storing violations
            config: Configuration dictionary with detection parameters
        """
        self.channel_id = channel_id
        self.video_source = video_source
        self.socketio = socketio
        self.db_manager = db_manager
        
        # Default configuration
        self.config = config or {}
        self.confidence_threshold = self.config.get('confidence_threshold', 0.5)
        self.alert_cooldown = self.config.get('alert_cooldown', 15)  # seconds between alerts
        
        # Detection requirements
        self.required_classes = self.config.get('required_classes', ['uniform', 'name_tag'])
        self.prohibited_classes = self.config.get('prohibited_classes', ['long_hair'])
        self.require_beard_net = self.config.get('require_beard_net', False)
        
        # Load YOLO model - Use TensorRT engine for better performance
        model_path = self.config.get('model_path', 'models/yolov8n.engine')
        try:
            self.model = YOLO(model_path)
            logger.info(f"Grooming detection model loaded: {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {e}")
            raise
        
        # State tracking
        self.last_alert_time = {}  # Track last alert per violation type
        self.frame_count = 0
        self.total_violations = 0
        self.active_violations = []
        
        # Snapshot storage
        self.snapshot_dir = 'static/grooming_snapshots'
        os.makedirs(self.snapshot_dir, exist_ok=True)
        
        logger.info(f"GroomingDetection initialized for channel {channel_id}")
    
    def process_frame(self, frame):
        """
        Process a single frame for grooming violations
        
        Args:
            frame: Input frame from video
            
        Returns:
            processed_frame: Frame with visualization
        """
        self.frame_count += 1
        
        # Run YOLO detection
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        
        # Extract detections
        detections = []
        detected_classes = []
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                class_name = self.model.names[cls_id]
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].cpu().numpy().astype(int)
                
                detections.append({
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': bbox
                })
                detected_classes.append(class_name)
        
        # Check for violations
        violations = self._check_violations(detected_classes)
        
        # Update active violations
        self.active_violations = violations
        
        # Trigger alerts if violations detected
        if violations:
            self._handle_violations(frame, violations, detections)
        
        # Draw visualization
        vis_frame = self._draw_visualization(frame, detections, violations)
        
        return vis_frame
    
    def _check_violations(self, detected_classes):
        """Check for grooming standard violations"""
        violations = []
        
        # Check required items
        for required in self.required_classes:
            if required not in detected_classes:
                violations.append({
                    'type': 'missing_required',
                    'item': required,
                    'message': f'{required.replace("_", " ").title()} not detected'
                })
        
        # Check prohibited items
        for prohibited in self.prohibited_classes:
            if prohibited in detected_classes:
                violations.append({
                    'type': 'prohibited_item',
                    'item': prohibited,
                    'message': f'{prohibited.replace("_", " ").title()} detected'
                })
        
        # Check beard net requirement
        if self.require_beard_net and 'beard' in detected_classes and 'beard_net' not in detected_classes:
            violations.append({
                'type': 'missing_required',
                'item': 'beard_net',
                'message': 'Beard detected without beard net'
            })
        
        return violations
    
    def _handle_violations(self, frame, violations, detections):
        """Handle detected violations with alerts and logging"""
        current_time = time.time()
        
        for violation in violations:
            violation_key = f"{violation['type']}_{violation['item']}"
            
            # Check cooldown period
            if violation_key in self.last_alert_time:
                if current_time - self.last_alert_time[violation_key] < self.alert_cooldown:
                    continue
            
            # Trigger alert
            self._trigger_violation_alert(frame, violation, detections)
            self.last_alert_time[violation_key] = current_time
    
    def _trigger_violation_alert(self, frame, violation, detections):
        """Trigger alert for grooming violation"""
        self.total_violations += 1
        timestamp = datetime.now()
        
        # Save snapshot
        snapshot_filename = f'grooming_{self.channel_id}_{timestamp.strftime("%Y%m%d_%H%M%S")}.jpg'
        snapshot_path = os.path.join(self.snapshot_dir, snapshot_filename)
        
        # Draw violations on snapshot
        snapshot = self._draw_violation_snapshot(frame.copy(), violation, detections)
        cv2.imwrite(snapshot_path, snapshot)
        
        # Prepare alert data
        alert_data = {
            'channel_id': self.channel_id,
            'violation_type': violation['type'],
            'violation_item': violation['item'],
            'message': violation['message'],
            'timestamp': timestamp.isoformat(),
            'snapshot_filename': snapshot_filename,
            'total_violations': self.total_violations
        }
        
        # Save to database
        if self.db_manager:
            try:
                file_size = os.path.getsize(snapshot_path)
                self.db_manager.save_grooming_snapshot(
                    channel_id=self.channel_id,
                    snapshot_filename=snapshot_filename,
                    snapshot_path=snapshot_path,
                    alert_message=violation['message'],
                    alert_data=alert_data,
                    violation_type=violation['type'],
                    violation_item=violation['item'],
                    file_size=file_size
                )
                logger.info(f"Grooming violation saved to database: {violation['message']}")
            except Exception as e:
                logger.error(f"Failed to save grooming violation to database: {e}")
        
        # Send real-time alert via SocketIO
        if self.socketio:
            try:
                self.socketio.emit('grooming_violation_detected', alert_data)
                logger.info(f"Grooming violation alert sent: {violation['message']}")
            except Exception as e:
                logger.error(f"Failed to send grooming violation alert: {e}")
    
    def _draw_violation_snapshot(self, frame, violation, detections):
        """Draw violation details on snapshot image"""
        vis = frame.copy()
        
        # Draw all detections
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            
            # Color based on violation relevance
            if det['class'] == violation['item']:
                color = (0, 0, 255)  # Red for violating item
                thickness = 3
            else:
                color = (0, 255, 0)  # Green for other detections
                thickness = 2
            
            label = f"{det['class']} ({det['confidence']:.2f})"
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(vis, (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(vis, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw violation banner
        banner_height = 60
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, 0), (vis.shape[1], banner_height), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.7, vis, 0.3, 0, vis)
        
        cv2.putText(vis, f"VIOLATION: {violation['message']}",
                   (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        return vis
    
    def _draw_visualization(self, frame, detections, violations):
        """Draw visualization on frame for live display"""
        vis = frame.copy()
        
        # Draw detections (only show relevant ones)
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            
            # Only draw if it's relevant to grooming checks
            if det['class'] in self.required_classes + self.prohibited_classes + ['beard', 'beard_net']:
                # Determine color based on violation status
                is_violation = False
                for v in violations:
                    if v['item'] == det['class']:
                        is_violation = True
                        break
                
                if is_violation or det['class'] in self.prohibited_classes:
                    color = (0, 0, 255)  # Red for violations
                    thickness = 3
                else:
                    color = (0, 255, 0)  # Green for compliant
                    thickness = 2
                
                label = f"{det['class']} ({det['confidence']:.2f})"
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
                
                # Draw label
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(vis, (x1, y1 - label_size[1] - 10),
                             (x1 + label_size[0], y1), color, -1)
                cv2.putText(vis, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw violation alert banner if violations exist
        if violations:
            banner_height = 40
            overlay = vis.copy()
            cv2.rectangle(overlay, (0, 0), (vis.shape[1], banner_height), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)
            
            violation_text = f"VIOLATIONS: {', '.join([v['message'] for v in violations])}"
            cv2.putText(vis, violation_text, (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw statistics at bottom
        stats_text = f"Violations: {self.total_violations} | Frame: {self.frame_count}"
        cv2.putText(vis, stats_text, (10, vis.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return vis
    
    def get_statistics(self):
        """Get current detection statistics"""
        return {
            'total_violations': self.total_violations,
            'active_violations': len(self.active_violations),
            'frame_count': self.frame_count,
            'violations_detail': self.active_violations
        }
    
    def update_config(self, new_config):
        """Update detection configuration dynamically"""
        if 'confidence_threshold' in new_config:
            self.confidence_threshold = float(new_config['confidence_threshold'])
            logger.info(f"Updated confidence threshold to {self.confidence_threshold}")
        
        if 'alert_cooldown' in new_config:
            self.alert_cooldown = float(new_config['alert_cooldown'])
            logger.info(f"Updated alert cooldown to {self.alert_cooldown}")
        
        if 'required_classes' in new_config:
            self.required_classes = new_config['required_classes']
            logger.info(f"Updated required classes to {self.required_classes}")
        
        if 'prohibited_classes' in new_config:
            self.prohibited_classes = new_config['prohibited_classes']
            logger.info(f"Updated prohibited classes to {self.prohibited_classes}")
        
        if 'require_beard_net' in new_config:
            self.require_beard_net = bool(new_config['require_beard_net'])
            logger.info(f"Updated require_beard_net to {self.require_beard_net}")
    
    def cleanup(self):
        """Cleanup resources"""
        logger.info(f"Grooming detection cleanup for channel {self.channel_id}")
