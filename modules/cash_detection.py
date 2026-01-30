"""
Cash Detection Module for Sakshi.AI
- Detects cash and open drawer using custom YOLO model (best.pt)
- Triggers alerts ONLY when both cash AND open_drawer are detected simultaneously
- Takes snapshot when alert conditions are met
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
from .gif_recorder import AlertGifRecorder

logger = logging.getLogger(__name__)


class CashDetection:
    """Cash detection with snapshot capture and alerting"""
    
    def __init__(self, channel_id, socketio, db_manager=None, app=None):
        """
        Initialize cash detection module
        
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
        self.conf_threshold = 0.7  # Confidence threshold for detections
        self.nms_iou = 0.45
        
        # Target classes for detection (from best.pt model)
        # Class 1: Cashdraw-open
        # Model structure: {0: 'Apron', 1: 'Cashdraw-open', ..., 12: 'Person', ..., 15: 'Table_clean', 16: 'Table_unclean'}
        self.drawer_class = "Cashdraw-open"  # Updated to match model class name
        # Note: Model doesn't have separate "cash" class, only "Cashdraw-open"
        
        # Alert configuration
        self.alert_cooldown = 240  # seconds between repeated alerts for same detection
        self.detection_duration_threshold = 1.0  # seconds - cash must be detected for this long before alert
        self.snapshot_dir = Path("static/cash_snapshots")
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize YOLO detector with shared model manager
        logger.info(f"Loading shared YOLO model for cash detection: {self.model_weight}")
        try:
            self.yolo = get_shared_model(self.model_weight, device='auto')
            logger.info("Shared cash detection model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load shared cash detection model: {e}")
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
        self.first_detection_time = None  # When cash was first detected
        self.consecutive_detection = False  # Whether cash is currently being detected
        
        # Frame processing
        self.frame_count = 0
        self.last_update_time = time.time()
        
        # Initialize GIF recorder for alert snapshots
        self.gif_recorder = AlertGifRecorder(buffer_size=90, gif_duration=3.0, fps=5)
        
        # Track recording state for GIF management
        self._was_recording_alert = False
        self._pending_alert_info = None  # Store alert info for database saving when GIF completes
        self._pending_snapshot_id = None  # Store snapshot ID to update with GIF path when complete
        self._pending_alert_timestamp = None  # Track when alert was triggered to prevent premature clearing
        
        logger.info(f"CashDetection initialized for channel {channel_id}")
    
    def __del__(self):
        """Cleanup: Release shared model reference when cash detection is destroyed"""
        try:
            if hasattr(self, 'model_weight'):
                release_shared_model(self.model_weight, device='auto')
                logger.debug(f"Released shared model reference: {self.model_weight}")
        except Exception as e:
            logger.warning(f"Error releasing shared model: {e}")
    
    def process_frame(self, frame):
        """
        Process a single frame for cash detection
        
        Args:
            frame: Input video frame
            
        Returns:
            Annotated frame with detection boxes and alerts
        """
        if frame is None:
            return None
        
        self.frame_count += 1
        t_now = time.time()
        
        # Add frame to GIF recorder buffer (always running)
        self.gif_recorder.add_frame(frame)
        
        # Clear GPU cache periodically to prevent memory fragmentation
        if torch.cuda.is_available() and self.frame_count % 100 == 0:
            torch.cuda.empty_cache()
        
        # YOLO inference
        results = self.yolo(frame, conf=self.conf_threshold, iou=self.nms_iou, verbose=False)
        
        # Process detections - looking for "Cashdraw-open" class
        detections = []
        drawer_detected = False
        
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = self.yolo.names.get(cls_id, str(cls_id))
                # Only process Cashdraw-open class
                if cls_name != self.drawer_class:
                    continue
                x1, y1, x2, y2 = map(int, xyxy.tolist())
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf,
                    "class_name": cls_name,
                    "class_id": cls_id
                })
                # Check for Cashdraw-open detection
                if cls_name == self.drawer_class:
                    drawer_detected = True
        
        self.current_detections = detections
        self.detection_count = len(detections)
        
        # Alert condition: Cashdraw-open detected (model only has this class, not separate cash)
        alert_condition_met = drawer_detected
        cash_detected = drawer_detected  # For backward compatibility with existing code
        
        # Update enhanced metrics
        if alert_condition_met:
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
        if alert_condition_met:
            # If this is the first detection, start the timer
            if self.first_detection_time is None:
                self.first_detection_time = t_now
                self.consecutive_detection = True
                self.detection_sessions += 1
                logger.info(f"💰 Cashdraw-open detected at {t_now}")
            
            # Check if both have been detected continuously for required duration
            detection_duration = t_now - self.first_detection_time
            
            # Trigger alert if duration threshold met and cooldown passed
            if (detection_duration >= self.detection_duration_threshold and 
                (t_now - self.last_alert_time >= self.alert_cooldown)):
                self._trigger_cash_alert(frame, detections, t_now, cash_detected, drawer_detected)
                self.last_alert_time = t_now
                # Reset detection timer after alert
                self.first_detection_time = None
                self.consecutive_detection = False
        else:
            # Alert condition not met - reset the timer
            if self.first_detection_time is not None:
                logger.debug(f"Cash+Drawer detection interrupted after {t_now - self.first_detection_time:.2f}s")
            self.first_detection_time = None
            self.consecutive_detection = False
        
        # Handle GIF recording state (same pattern as queue_monitor)
        was_recording = self.gif_recorder.is_recording_alert
        
        if was_recording:
            # Add frame during alert recording
            self.gif_recorder.add_alert_frame(frame)
            # stop_alert_recording() is called automatically by add_alert_frame when duration is reached
            # Log periodically to track recording progress (every 10 frames since processing is slow)
            if self.frame_count % 10 == 0:  # Every 10 frames to catch progress even with slow processing
                elapsed = (datetime.now() - self.gif_recorder.alert_start_time).total_seconds() if self.gif_recorder.alert_start_time else 0
                frame_count = len(self.gif_recorder.alert_frames) if hasattr(self.gif_recorder, 'alert_frames') else 0
                logger.info(f"[{self.channel_id}] 📹 GIF recording in progress: {elapsed:.1f}s / {self.gif_recorder.gif_duration:.1f}s, alert_frames={frame_count}, is_recording={self.gif_recorder.is_recording_alert}")
        
        # Check current state after potentially adding frame
        is_recording_now = self.gif_recorder.is_recording_alert
        
        # Check if recording just finished (was recording, now stopped)
        # This detects the transition from recording to stopped
        # Pattern matches queue_monitor: check if we WERE recording (previous frame) and NOW we're not (after adding frame)
        if self._was_recording_alert and not is_recording_now:
            # Recording just finished - get GIF info and save to DB
            logger.info(f"[{self.channel_id}] 🎬 GIF recording completed! Previous state: {self._was_recording_alert}, Current state: {self.gif_recorder.is_recording_alert}")
            logger.info(f"[{self.channel_id}] 📊 GIF recorder state check: was_recording={was_recording}, is_recording_now={self.gif_recorder.is_recording_alert}, _was_recording_alert={self._was_recording_alert}")
            
            # Get GIF info - this should have been set by stop_alert_recording()
            gif_info = self.gif_recorder.get_last_gif_info()
            
            # If get_last_gif_info() returns None, try to get it directly from the recorder
            if not gif_info and hasattr(self.gif_recorder, 'last_gif_info'):
                gif_info = self.gif_recorder.last_gif_info
                logger.info(f"[{self.channel_id}] ℹ️ Retrieved GIF info from last_gif_info attribute: {gif_info is not None}")
            
            # Log diagnostic info with more details
            if not gif_info:
                logger.warning(f"[{self.channel_id}] ⚠️ GIF recording completed but get_last_gif_info() returned None")
            else:
                logger.info(f"[{self.channel_id}] ✅ Retrieved GIF info: path={gif_info.get('gif_path', 'N/A')}, frames={gif_info.get('frame_count', 0)}")
            
            if not self.db_manager:
                logger.warning(f"[{self.channel_id}] ⚠️ GIF recording completed but db_manager is None")
            
            if not self._pending_alert_info:
                logger.error(f"[{self.channel_id}] ❌ GIF recording completed but _pending_alert_info is None! This means alert info was lost. GIF path: {gif_info.get('gif_path', 'N/A') if gif_info else 'N/A'}")
                # Try to save with minimal info if we have GIF but no alert info
                if gif_info and self.db_manager:
                    logger.warning(f"[{self.channel_id}] ⚠️ Attempting to save GIF with minimal info since _pending_alert_info is missing")
                    try:
                        gif_path = gif_info.get('gif_path', '')
                        gif_filename = os.path.basename(gif_path) if gif_path else f"cash_alert_{self.channel_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.gif"
                        gif_payload = {
                            'gif_filename': gif_filename,
                            'gif_path': gif_path,
                            'frame_count': gif_info.get('frame_count', 0),
                            'duration': gif_info.get('duration', 0.0)
                        }
                        fallback_message = f"Cashdraw-open detected on {self.channel_id}"
                        if self.app:
                            with self.app.app_context():
                                self.db_manager.save_alert_gif(
                                    self.channel_id,
                                    'cash_detection_alert',
                                    gif_payload,
                                    alert_message=fallback_message,
                                    alert_data={'channel_id': self.channel_id, 'recovered': True}
                                )
                        else:
                            self.db_manager.save_alert_gif(
                                self.channel_id,
                                'cash_detection_alert',
                                gif_payload,
                                alert_message=fallback_message,
                                alert_data={'channel_id': self.channel_id, 'recovered': True}
                            )
                        logger.info(f"[{self.channel_id}] ✅ Recovered and saved GIF with minimal info: {gif_filename}")
                    except Exception as e:
                        logger.error(f"[{self.channel_id}] ❌ Failed to save recovered GIF: {e}", exc_info=True)
            else:
                logger.info(f"[{self.channel_id}] ✅ _pending_alert_info available: message='{self._pending_alert_info.get('message', 'N/A')}'")
            
            if not self._pending_snapshot_id:
                logger.info(f"[{self.channel_id}] ℹ️ GIF recording completed but _pending_snapshot_id is None - will save to alert_gifs only")
            
            # Save GIF if we have the required info
            if gif_info and self.db_manager and self._pending_alert_info:
                try:
                    # Extract filename from path
                    gif_path = gif_info.get('gif_path', '')
                    gif_filename = os.path.basename(gif_path) if gif_path else f"cash_alert_{self.channel_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.gif"
                    
                    # Verify GIF file exists
                    if gif_path and not os.path.exists(gif_path):
                        logger.error(f"[{self.channel_id}] ❌ GIF file does not exist at path: {gif_path}")
                        # Don't save to database if file doesn't exist
                        self._pending_alert_info = None
                        self._pending_snapshot_id = None
                        self._pending_alert_timestamp = None
                    else:
                        # Convert absolute path to relative path for database
                        if gif_path:
                            normalized_path = gif_path.replace('\\', '/')
                            if 'static/' in normalized_path:
                                snapshot_path = normalized_path.split('static/')[1]
                                snapshot_path = f"static/{snapshot_path}"
                            else:
                                snapshot_path = gif_path
                        else:
                            snapshot_path = None
                        
                        # Update CashSnapshot record with GIF path (if snapshot_id was set)
                        # Also save to alert_gifs table for consistency
                        gif_payload = {
                            'gif_filename': gif_filename,
                            'gif_path': gif_path,
                            'frame_count': gif_info.get('frame_count', 0),
                            'duration': gif_info.get('duration', 0.0)
                        }
                        
                        alert_info = self._pending_alert_info
                        alert_message = alert_info.get('message', 'Cash detection alert')
                        
                        # All database operations must be within app context
                        if self.app:
                            with self.app.app_context():
                                # Update CashSnapshot record with GIF path (if snapshot_id was set)
                                if self._pending_snapshot_id:
                                    snapshot = self.db_manager.CashSnapshot.query.get(self._pending_snapshot_id)
                                    if snapshot:
                                        snapshot.snapshot_path = snapshot_path
                                        snapshot.snapshot_filename = gif_filename
                                        # Update file_size if GIF exists
                                        if gif_path and os.path.exists(gif_path):
                                            snapshot.file_size = os.path.getsize(gif_path)
                                        self.db_manager.db.session.commit()
                                        logger.info(f"[{self.channel_id}] ✅ Updated cash snapshot {self._pending_snapshot_id} with GIF path: {snapshot_path}")
                                    else:
                                        logger.warning(f"[{self.channel_id}] ⚠️ Cash snapshot {self._pending_snapshot_id} not found for update")
                                else:
                                    logger.info(f"[{self.channel_id}] ℹ️ No snapshot_id to update, saving to alert_gifs only")
                                
                                # Save to alert_gifs table
                                self.db_manager.save_alert_gif(
                                    self.channel_id,
                                    'cash_detection_alert',
                                    gif_payload,
                                    alert_message=alert_message,
                                    alert_data=alert_info
                                )
                                
                                logger.info(f"[{self.channel_id}] ✅ Cash alert GIF saved to database: {gif_filename} (path: {gif_path})")
                                # Clear stored alert info only after successful save
                                self._pending_alert_info = None
                                self._pending_snapshot_id = None
                                self._pending_alert_timestamp = None
                        else:
                            logger.error(f"[{self.channel_id}] ❌ Cannot save GIF: app context not available")
                except Exception as e:
                    logger.error(f"[{self.channel_id}] ❌ Failed to save cash alert GIF to DB: {e}", exc_info=True)
            else:
                logger.error(f"[{self.channel_id}] ❌ Cannot save GIF: gif_info={gif_info is not None}, db_manager={self.db_manager is not None}, alert_info={self._pending_alert_info is not None}")
        
        # Track recording state for next frame (use state BEFORE adding frame, like queue_monitor)
        # But if recording just stopped, use the current state (False) so we don't trigger completion again
        if self._was_recording_alert and not is_recording_now:
            # Recording just finished - set to False to prevent re-triggering
            self._was_recording_alert = False
        else:
            # Normal case: use state before adding frame
            self._was_recording_alert = was_recording
        
        # Draw visualization
        vis_frame = self._draw_visualization(frame, detections, cash_detected, drawer_detected)
        
        # Send real-time updates
        if t_now - self.last_update_time >= 1.0:
            self._send_realtime_update()
            self.last_update_time = t_now
        
        # Return structured result format consistent with other modules
        return {
            'frame': vis_frame,
            'status': {
                'cash_detected': cash_detected,
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
    
    def _trigger_cash_alert(self, frame, detections, timestamp, cash_detected, drawer_detected):
        """Trigger cash detection alert with GIF recording when Cashdraw-open is detected"""
        self.total_alerts += 1
        
        dt = datetime.fromtimestamp(timestamp)
        
        # Prepare alert data
        alert_message = f"ALERT: Cashdraw-open detected"
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
            'cash_detected': cash_detected,
            'drawer_detected': drawer_detected,
            'channel_id': self.channel_id
        }
        
        # Check if a GIF is already recording - if so, skip this alert to avoid overwriting pending info
        if self.gif_recorder.is_recording_alert:
            logger.warning(f"[{self.channel_id}] 💰🗄️ ALERT: Cashdraw-open detected but GIF already recording - skipping to avoid overwriting pending alert info")
            return
        
        # Also check if we have pending info from a previous alert that hasn't completed yet
        # Only clear it if we're NOT currently recording AND enough time has passed (to avoid losing data for an in-progress GIF)
        if self._pending_alert_info is not None:
            if self.gif_recorder.is_recording_alert:
                # This should have been caught by the first check, but just in case
                logger.error(f"[{self.channel_id}] 💰🗄️ ALERT: Cashdraw-open detected but both _pending_alert_info is set AND recording is active - this should not happen! Skipping alert.")
                return
            elif self._pending_alert_timestamp is not None:
                time_since_alert = time.time() - self._pending_alert_timestamp
                if time_since_alert < 5.0:  # Less than 5 seconds (GIF takes 3 seconds, so this is safe)
                    logger.warning(f"[{self.channel_id}] 💰🗄️ ALERT: Cashdraw-open detected but _pending_alert_info is set and only {time_since_alert:.1f}s have passed - previous GIF may still be processing. Skipping alert.")
                    return
                else:
                    logger.warning(f"[{self.channel_id}] 💰🗄️ ALERT: Cashdraw-open detected but _pending_alert_info is set from {time_since_alert:.1f}s ago (recording not active) - clearing stale info.")
                    # Clear old pending info to allow new alert (only if enough time has passed)
                    self._pending_alert_info = None
                    self._pending_snapshot_id = None
                    self._pending_alert_timestamp = None
            else:
                # No timestamp, so clear it
                logger.warning(f"[{self.channel_id}] 💰🗄️ ALERT: Cashdraw-open detected but _pending_alert_info is set without timestamp - clearing stale info.")
                self._pending_alert_info = None
                self._pending_snapshot_id = None
        
        # Start GIF recording for this alert
        alert_info = {
            'type': 'cash_detection_alert',
            'message': alert_message,
            'detection_count': len(detections),
            'timestamp': dt.isoformat(),
            **alert_data
        }
        
        logger.info(f"[{self.channel_id}] 💰🗄️ ALERT: Cashdraw-open detected! Starting GIF recording...")
        self.gif_recorder.start_alert_recording(alert_info)
        # Store alert_info for database saving when GIF completes
        self._pending_alert_info = alert_info
        self._pending_snapshot_id = None  # Will be set when snapshot is saved
        
        self._pending_alert_timestamp = time.time()  # Track when this alert was triggered
        logger.info(f"[{self.channel_id}] ✅ GIF recording started, _pending_alert_info set: message='{alert_message}', timestamp='{dt.isoformat()}'")
        logger.info(f"[{self.channel_id}] 📊 GIF recorder state: is_recording_alert={self.gif_recorder.is_recording_alert}")
        
        # Save to database immediately (with placeholder snapshot_path - will be updated when GIF completes)
        if self.db_manager and self.app:
            try:
                with self.app.app_context():
                    # Generate placeholder filename (will be replaced with GIF filename)
                    placeholder_filename = f"cash_{self.channel_id}_{dt.strftime('%Y%m%d_%H%M%S')}.gif"
                    # Use a placeholder path since snapshot_path is NOT NULL in database
                    # This will be updated with the actual GIF path when recording completes
                    placeholder_path = f"static/cash_snapshots/{placeholder_filename}"
                    
                    snapshot_id = self.db_manager.save_cash_snapshot(
                        channel_id=self.channel_id,
                        snapshot_filename=placeholder_filename,
                        snapshot_path=placeholder_path,  # Placeholder - will be updated when GIF completes
                        alert_message=alert_message,
                        alert_data=alert_data,
                        file_size=0,  # Will be updated when GIF completes
                        detection_count=len(detections)
                    )
                    
                    # Store snapshot ID to update with GIF path later
                    if snapshot_id:
                        self._pending_snapshot_id = snapshot_id
                        logger.info(f"[{self.channel_id}] Cash snapshot record created in database: ID {snapshot_id} (GIF recording in progress)")
                    else:
                        logger.warning(f"[{self.channel_id}] ⚠️ Cash snapshot save returned None - GIF will be saved to alert_gifs only")
                        self._pending_snapshot_id = None
                    
                    # Emit real-time notification
                    self.socketio.emit('cash_detected', {
                        'snapshot_id': snapshot_id,
                        'channel_id': self.channel_id,
                        'snapshot_filename': placeholder_filename,
                        'snapshot_url': None,  # Will be available when GIF completes
                        'alert_message': alert_message,
                        'detection_count': len(detections),
                        'timestamp': dt.isoformat()
                    })
                    
            except Exception as e:
                logger.error(f"[{self.channel_id}] Error saving cash snapshot to database: {e}")
                import traceback
                logger.error(traceback.format_exc())
                # Don't set _pending_snapshot_id if save failed - GIF will be saved to alert_gifs only
                self._pending_snapshot_id = None
        
        # Emit Socket.IO alert
        self.socketio.emit('cash_alert', {
            'channel_id': self.channel_id,
            'alert_message': alert_message,
            'detection_count': len(detections),
            'snapshot_filename': None,  # Will be available when GIF completes
            'snapshot_url': None,
            'timestamp': dt.isoformat()
        })
    
    def _draw_visualization(self, frame, detections, cash_detected, drawer_detected):
        """Draw bounding boxes and detection info on frame"""
        vis = frame.copy()
        
        alert_condition = cash_detected and drawer_detected
        
        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            cls_name = det['class_name']
            # Visualize Cashdraw-open
            if cls_name == self.drawer_class:
                color = (0, 255, 255)  # Yellow for Cashdraw-open
            else:
                continue
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)
            # Draw label with confidence
            label = f"{cls_name}: {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            # Background for label
            cv2.rectangle(vis, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            # Label text
            cv2.putText(vis, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Draw alert banner if Cashdraw-open detected
        if alert_condition:
            # Calculate detection duration
            if self.first_detection_time is not None:
                current_time = time.time()
                detection_duration = current_time - self.first_detection_time
                remaining_time = max(0, self.detection_duration_threshold - detection_duration)
                
                # Color changes from yellow to red as we approach threshold
                if detection_duration >= self.detection_duration_threshold:
                    banner_color = (0, 0, 255)  # Red - alert triggered
                    text = f"⚠️ CASHDRAW-OPEN DETECTED - ALERT!"
                else:
                    banner_color = (0, 165, 255)  # Orange - detecting
                    text = f"💰 Cashdraw-open Detected - Alert in {remaining_time:.1f}s"
            else:
                banner_color = (0, 165, 255)  # Orange
                text = f"💰 CASHDRAW-OPEN DETECTED"
            
            # Banner at top
            cv2.rectangle(vis, (0, 0), (vis.shape[1], 50), banner_color, -1)
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            text_x = (vis.shape[1] - text_size[0]) // 2
            cv2.putText(vis, text, (text_x, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        elif drawer_detected:
            # Show info if Cashdraw-open detected but alert not yet triggered
            text = "🗄️ Cashdraw-open detected (waiting for duration threshold...)"
            
            banner_color = (128, 128, 128)  # Gray - partial detection
            cv2.rectangle(vis, (0, 0), (vis.shape[1], 40), banner_color, -1)
            cv2.putText(vis, text, (10, 28), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw statistics at bottom
        stats_y = vis.shape[0] - 60
        stats_bg = (50, 50, 50)
        cv2.rectangle(vis, (0, stats_y - 10), (500, vis.shape[0]), stats_bg, -1)
        
        cv2.putText(vis, f"Cashdraw-open: {'YES' if drawer_detected else 'NO'}", 
                   (10, stats_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis, f"Current Detections: {len(detections)}", (10, stats_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis, f"Total Alerts: {self.total_alerts}", (10, stats_y + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return vis
    
    def _send_realtime_update(self):
        """Send real-time statistics update via Socket.IO"""
        self.socketio.emit('cash_detection_update', {
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
        
        logger.info(f"CashDetection config updated for channel {self.channel_id}")
