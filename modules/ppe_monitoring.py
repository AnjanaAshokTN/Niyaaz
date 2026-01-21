"""
PPE (Personal Protective Equipment) Monitoring Module for Sakshi.AI
Detects PPE compliance including:
- Apron (required/not required)
- Gloves (required/not required)
- Hairnet (required/not required)
"""

import time
import cv2
import numpy as np
import logging
import torch
import os
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO
from modules.gif_recorder import AlertGifRecorder
from .model_manager import get_shared_model, release_shared_model

logger = logging.getLogger(__name__)


class PPEMonitoring:
    """PPE compliance monitoring with violation alerts"""
    
    def __init__(self, channel_id, socketio, db_manager=None, app=None):
        """
        Initialize PPE monitoring module
        
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
        
        # Model configuration - Use custom trained model
        self.model_weight = "models/best.pt"
        self.conf_threshold = 0.5
        self.nms_iou = 0.45
        
        # PPE classes from your best.pt model
        self.ppe_classes = {
            "Apron", "No_apron",
            "Gloves", "No_gloves",
            "Hairnet", "No_hairnet"
        }
        
        # Hairnet configuration
        self.hairnet_positive_class = "Hairnet"
        self.hairnet_violation_class = "No_hairnet"  
        
        # Required PPE items (can be overridden per camera via config)
        self.required_items = {
            "apron": True,      # Default: require apron
            "gloves": True,     # Default: require gloves
            "hairnet": True     # Default: require hairnet
        }
        
        # Alert configuration
        self.alert_cooldown = 30.0  # seconds between alerts for same violation
        self.violation_duration_threshold = 0.5  # seconds to confirm violation (reduced for faster alerts)

        # Hairnet false-positive suppression:
        # If the top of the person bbox is too close to the top of the frame, the head is likely cut off.
        # In that case, skip "Hairnet not detected" because we cannot see the head region reliably.
        self.hairnet_head_cut_margin_ratio = 0.03  # 3% of frame height
        
        # State tracking
        self.last_alert_time = {}  # employee_id -> timestamp
        self.violation_start_time = {}  # employee_id -> timestamp
        self.current_violations = {}  # employee_id -> list of violations
        
        # Statistics
        self.total_checks = 0
        self.total_violations = 0
        self.total_compliant = 0
        
        # Frame counter
        self.frame_count = 0
        
        # Track GIF recording state
        self._was_recording_alert = False
        self._last_ppe_alert_message = None
        self._last_ppe_alert_violations = []
        
        # Initialize YOLO detector with shared model manager
        logger.info(f"Loading shared YOLO model for PPE monitoring: {self.model_weight}")
        self.model = get_shared_model(self.model_weight)
        
        if self.model is None:
            raise RuntimeError(f"Failed to load model: {self.model_weight}")
        
        logger.info("Shared PPE monitoring model loaded successfully")
        
        # Get model class names
        if hasattr(self.model, 'names') and self.model.names:
            model_classes = list(self.model.names.values())
            logger.info(f"📋 Model has {len(model_classes)} classes: {model_classes}")
        
        # Initialize GIF recorder for alerts
        self.gif_recorder = AlertGifRecorder(
            buffer_size=90,
            gif_duration=3.0,
            fps=30
        )
        
        logger.info(f"PPE Monitoring initialized for channel {channel_id}")
        self.load_configuration()
    
    def load_configuration(self):
        """Load saved configuration from database"""
        if self.db_manager:
            try:
                from flask import has_app_context
                if not has_app_context():
                    logger.debug(f"Skipping config load from DB (no app context available)")
                    return
                
                config = self.db_manager.get_channel_config(
                    self.channel_id, 'PPEMonitoring', 'settings'
                )
                if config:
                    if 'required_items' in config:
                        self.required_items.update(config['required_items'])
                    if 'alert_cooldown' in config:
                        self.alert_cooldown = float(config['alert_cooldown'])
                    if 'violation_duration_threshold' in config:
                        self.violation_duration_threshold = float(config['violation_duration_threshold'])
                    if 'conf_threshold' in config:
                        self.conf_threshold = float(config['conf_threshold'])
                    logger.info(f"Loaded PPE configuration from database for {self.channel_id}")
            except Exception as e:
                logger.error(f"Failed to load PPE configuration: {e}")
    
    def set_settings(self, settings):
        """Update PPE monitoring settings"""
        if 'required_items' in settings:
            self.required_items.update(settings['required_items'])
        if 'alert_cooldown' in settings:
            self.alert_cooldown = float(settings['alert_cooldown'])
        if 'violation_duration_threshold' in settings:
            self.violation_duration_threshold = float(settings['violation_duration_threshold'])
        if 'conf_threshold' in settings:
            self.conf_threshold = float(settings['conf_threshold'])
        
        logger.info(f"PPE settings updated for channel {self.channel_id}")
        
        # Save to database
        if self.db_manager:
            try:
                from flask import has_app_context
                if has_app_context():
                    self.db_manager.save_channel_config(
                        self.channel_id,
                        'PPEMonitoring',
                        'settings',
                        {
                            'required_items': self.required_items,
                            'alert_cooldown': self.alert_cooldown,
                            'violation_duration_threshold': self.violation_duration_threshold,
                            'conf_threshold': self.conf_threshold
                        }
                    )
            except Exception as e:
                logger.error(f"Failed to save PPE settings: {e}")
    
    def _is_head_visible(self, detections, frame_shape):
        """
        Heuristic: head is considered visible if we have at least one Person detection whose bbox top (y1)
        is not near the top edge of the frame (i.e., not truncated).
        """
        if not detections or frame_shape is None:
            return True  # can't decide; keep existing behavior
        h = int(frame_shape[0])
        margin_px = int(max(5, self.hairnet_head_cut_margin_ratio * h))

        persons = [d for d in detections if str(d.get('class', '')).lower() == 'person' and d.get('bbox')]
        if not persons:
            return True  # can't decide; keep existing behavior

        # If ANY person has y1 > margin, we assume at least one head is visible in frame.
        for p in persons:
            try:
                y1 = int(p['bbox'][1])
                if y1 > margin_px:
                    return True
            except Exception:
                continue
        return False

    def check_compliance(self, detected_classes, detections=None, frame_shape=None):
        """
        Check PPE compliance based on detected classes
        
        Args:
            detected_classes: Set of detected class names
            
        Returns:
            tuple: (is_compliant, violations_list)
        """
        violations = []
        
        # Check apron requirement
        if self.required_items.get('apron', False):
            if 'No_apron' in detected_classes and 'Apron' not in detected_classes:
                violations.append('Apron not detected')
            elif 'Apron' in detected_classes:
                pass  # Compliant
        
        # Check gloves requirement
        if self.required_items.get('gloves', False):
            if 'No_gloves' in detected_classes and 'Gloves' not in detected_classes:
                violations.append('Gloves not detected')
            elif 'Gloves' in detected_classes:
                pass  # Compliant
        
        # Check hairnet requirement
        # Note: Model only has "Hairnet" class (no "No_hairnet")
        # If hairnet is required and not detected, it's a violation
        # But only flag if we have other detections (person/PPE) to avoid false positives
        if self.required_items.get('hairnet', False):
            # Only check hairnet if we have other detections (person or other PPE items)
            # This prevents false violations when no one is in frame
            has_other_detections = (
                'Person' in detected_classes or
                'Apron' in detected_classes or
                'No_apron' in detected_classes or
                'Gloves' in detected_classes or
                'No_gloves' in detected_classes
            )
            
            if has_other_detections and self.hairnet_positive_class not in detected_classes:
                # Suppress hairnet violations when heads are cut off at the top of the frame
                head_visible = self._is_head_visible(detections or [], frame_shape)
                if not head_visible:
                    logger.debug(
                        f"[{self.channel_id}] Skipping hairnet violation (head likely cut off in frame)"
                    )
                else:
                    violations.append('Hairnet not detected')
        
        is_compliant = len(violations) == 0
        return is_compliant, violations
    
    def process_frame(self, frame):
        """
        Process a single frame for PPE compliance
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Annotated frame with PPE compliance information
        """
        self.frame_count += 1
        original_frame = frame.copy()
        
        # Debug: Log first frame processing
        if self.frame_count == 1:
            logger.info(f"🔍 PPE Monitoring: Processing first frame for channel {self.channel_id}, frame shape: {frame.shape}")
        
        # Clear CUDA cache periodically
        if torch.cuda.is_available() and self.frame_count % 100 == 0:
            torch.cuda.empty_cache()
        
        # Add frame to GIF recorder buffer
        self.gif_recorder.add_frame(original_frame)
        
        # Run YOLO detection
        try:
            if self.model is None:
                logger.error(f"PPE model is None for channel {self.channel_id}")
                return frame
            
            results = self.model.predict(
                frame,
                conf=self.conf_threshold,
                iou=self.nms_iou,
                verbose=False,
                classes=None  # Detect all classes, we'll filter for PPE
            )[0]
            
            # Debug logging for first few frames
            if self.frame_count <= 5:
                logger.info(f"🔍 PPE Frame {self.frame_count}: Running detection on frame shape={frame.shape}")
        except Exception as e:
            logger.error(f"Error during PPE detection for channel {self.channel_id}: {e}", exc_info=True)
            return frame
        
        # Extract detections
        detections = []
        detected_classes = set()
        
        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                if cls_id < len(self.model.names):
                    class_name = self.model.names[cls_id]
                    
                    # Process PPE-related classes and Person class (for context)
                    if class_name in self.ppe_classes or class_name == 'Person':
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        detections.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'class': class_name,
                            'confidence': conf
                        })
                        detected_classes.add(class_name)
        
        # Log detection summary periodically and for first few frames
        if self.frame_count <= 5 or self.frame_count % 300 == 0:
            logger.info(f"📊 PPE Frame {self.frame_count}: Total detections: {len(detections)}")
            logger.info(f"   Detected PPE classes: {detected_classes}")
            # Log all detected classes (not just PPE)
            if results.boxes is not None and len(results.boxes) > 0:
                all_classes = set()
                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    if cls_id < len(self.model.names):
                        all_classes.add(self.model.names[cls_id])
                logger.info(f"   All detected classes in frame: {all_classes}")
        
        # Check compliance for each detected person/area
        # For simplicity, we'll check overall compliance in the frame
        # In a more sophisticated implementation, you'd track individual people
        is_compliant, violations = self.check_compliance(detected_classes, detections=detections, frame_shape=frame.shape)
        
        # Log compliance check result
        if self.frame_count <= 5 or violations:
            logger.info(f"🔍 PPE Compliance Check Frame {self.frame_count}: is_compliant={is_compliant}, violations={violations}")
        
        # Track violations over time
        employee_id = "employee_0"  # Simplified - in production, track individual employees
        
        current_time = time.time()
        
        if violations:
            # Violation detected
            if employee_id not in self.violation_start_time:
                self.violation_start_time[employee_id] = current_time
                logger.warning(f"⚠️ PPE violation STARTED for {employee_id}: {violations} (threshold: {self.violation_duration_threshold}s)")
            
            violation_duration = current_time - self.violation_start_time[employee_id]
            
            # Log violation duration progress every few frames
            if violations and (self.frame_count % 5 == 0 or violation_duration >= self.violation_duration_threshold * 0.9):
                logger.info(f"⏱️ PPE violation duration: {violation_duration:.2f}s / {self.violation_duration_threshold}s (frame {self.frame_count})")
            
            if violation_duration >= self.violation_duration_threshold:
                # Violation sustained, check if we should alert
                if (employee_id not in self.last_alert_time or 
                    current_time - self.last_alert_time[employee_id] >= self.alert_cooldown):
                    
                    # Trigger alert
                    violation_msg = ", ".join(violations)
                    alert_message = f"PPE violation: {violation_msg}"
                    
                    logger.warning(f"🚨 PPE violation ALERT TRIGGERED for {employee_id}: {violation_msg} (duration: {violation_duration:.2f}s)")
                    
                    # Save violation snapshot (similar to dress code monitoring)
                    snapshot_path = self._save_violation_snapshot(original_frame, violations, employee_id)
                    
                    # Store alert info for later use when saving GIF
                    self._last_ppe_alert_message = alert_message
                    self._last_ppe_alert_violations = violations
                    
                    # Start GIF recording
                    alert_info = {
                        'type': 'ppe_alert',
                        'message': alert_message,
                        'violations': violations,
                        'timestamp': datetime.now().isoformat(),
                        'channel_id': self.channel_id
                    }
                    self.gif_recorder.start_alert_recording(alert_info)
                    self.gif_recorder.add_alert_frame(original_frame)
                    
                    # Save violation to database (with snapshot path)
                    if self.db_manager:
                        try:
                            # Save to PPE alerts table
                            if self.app:
                                with self.app.app_context():
                                    self.db_manager.add_ppe_alert(
                                        channel_id=self.channel_id,
                                        violations=violations,
                                        violation_types=violations,  # List of violation types
                                        employee_id=None,  # Don't store employee_id
                                        snapshot_path=snapshot_path,
                                        alert_data={'violations': violations, 'message': alert_message}
                                    )
                                    # Also log to general alerts table
                                    self.db_manager.log_alert(
                                        self.channel_id,
                                        'ppe_alert',
                                        alert_message,
                                        alert_data={'violations': violations}
                                    )
                            else:
                                self.db_manager.add_ppe_alert(
                                    channel_id=self.channel_id,
                                    violations=violations,
                                    violation_types=violations,
                                    employee_id=None,  # Don't store employee_id
                                    snapshot_path=snapshot_path,
                                    alert_data={'violations': violations, 'message': alert_message}
                                )
                                self.db_manager.log_alert(
                                    self.channel_id,
                                    'ppe_alert',
                                    alert_message,
                                    alert_data={'violations': violations}
                                )
                            logger.info(f"✅ PPE violation saved to database: {violation_msg}")
                        except Exception as e:
                            logger.error(f"Database logging error for PPE alert: {e}")
                    
                    # Emit socket event
                    self.socketio.emit('ppe_alert', alert_info)
                    
                    self.last_alert_time[employee_id] = current_time
                    self.total_violations += 1
            else:
                # Violation just started, not yet sustained
                pass
        else:
            # No violations - but don't clear immediately, wait a bit to avoid flickering
            # Only clear if we've been compliant for a while
            if employee_id in self.violation_start_time:
                # Keep tracking for a short grace period to avoid false clears
                violation_duration = current_time - self.violation_start_time[employee_id]
                if violation_duration < 0.1:  # Very short violation, might be flickering
                    # Don't clear yet, might come back
                    if self.frame_count % 10 == 0:
                        logger.debug(f"⏸️ PPE violation cleared too quickly ({violation_duration:.2f}s), might be flickering - keeping track")
                else:
                    # Violation was real but now cleared
                    del self.violation_start_time[employee_id]
                    logger.info(f"✅ PPE violation cleared for {employee_id} (was tracked for {violation_duration:.2f}s)")
            self.total_compliant += 1
        
        self.total_checks += 1
        
        # Handle GIF recording
        was_recording = self.gif_recorder.is_recording_alert
        
        if was_recording:
            self.gif_recorder.add_alert_frame(original_frame)
            # stop_alert_recording() is called automatically by add_alert_frame when duration is reached
        
        # Check if recording just finished (was recording, now stopped)
        if self._was_recording_alert and not self.gif_recorder.is_recording_alert:
            # Recording just finished - get GIF info and save to DB
            gif_info = self.gif_recorder.get_last_gif_info()
            if gif_info and self.db_manager and self._last_ppe_alert_message:
                try:
                    # Extract filename from path
                    gif_path = gif_info.get('gif_path', '')
                    gif_filename = os.path.basename(gif_path) if gif_path else f"ppe_alert_{self.channel_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.gif"
                    
                    if self.app:
                        with self.app.app_context():
                            self.db_manager.save_alert_gif(
                                self.channel_id,
                                'ppe_alert',
                                {'filename': gif_filename, 'path': gif_path},
                                alert_message=self._last_ppe_alert_message,
                                alert_data={'violations': self._last_ppe_alert_violations}
                            )
                    else:
                        self.db_manager.save_alert_gif(
                            self.channel_id,
                            'ppe_alert',
                            {'filename': gif_filename, 'path': gif_path},
                            alert_message=self._last_ppe_alert_message,
                            alert_data={'violations': self._last_ppe_alert_violations}
                        )
                    logger.info(f"PPE alert GIF saved to database: {gif_filename}")
                    # Clear stored alert info
                    self._last_ppe_alert_message = None
                    self._last_ppe_alert_violations = []
                except Exception as e:
                    logger.error(f"Failed to save PPE alert GIF to DB: {e}")
        
        # Track recording state for next frame
        self._was_recording_alert = was_recording
        
        # Annotate frame
        annotated_frame = self._annotate_frame(frame, detections, is_compliant, violations)
        
        return annotated_frame
    
    def _annotate_frame(self, frame, detections, is_compliant, violations):
        """Annotate frame with PPE detection results"""
        annotated = frame.copy()
        
        # Draw bounding boxes
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class']
            conf = det['confidence']
            
            # Color coding: green for positive PPE, red for negative
            if class_name in ['Apron', 'Gloves', 'Hairnet']:
                color = (0, 255, 0)  # Green
            else:  # No_apron, No_gloves
                color = (0, 0, 255)  # Red
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(annotated, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Display compliance status
        status_text = "PPE: COMPLIANT" if is_compliant else f"PPE: VIOLATION - {', '.join(violations)}"
        status_color = (0, 255, 0) if is_compliant else (0, 0, 255)
        cv2.putText(annotated, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Display statistics
        stats_text = f"Checks: {self.total_checks} | Violations: {self.total_violations} | Compliant: {self.total_compliant}"
        cv2.putText(annotated, stats_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated
    
    def _save_violation_snapshot(self, frame, violations, employee_id):
        """Save snapshot of PPE violation (similar to dress code monitoring)"""
        timestamp = datetime.now()
        
        # Generate unique filename (without employee_id)
        filename = f"ppe_{self.channel_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = Path('static/ppe_snapshots') / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Add violation text to frame
        annotated_frame = frame.copy()
        y_offset = 30
        cv2.putText(annotated_frame, "PPE VIOLATION", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        y_offset += 30
        for violation in violations[:3]:  # Show max 3 violations
            cv2.putText(annotated_frame, f"- {violation}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            y_offset += 25
        
        # Save snapshot
        cv2.imwrite(str(filepath), annotated_frame)
        
        logger.info(f"PPE violation snapshot saved: {filename}")
        
        return str(filepath)
    
    def get_status(self):
        """Get current status of PPE monitoring"""
        return {
            'module': 'PPEMonitoring',
            'channel_id': self.channel_id,
            'status': 'active',
            'total_checks': self.total_checks,
            'total_violations': self.total_violations,
            'total_compliant': self.total_compliant,
            'required_items': self.required_items,
            'frame_count': self.frame_count
        }

