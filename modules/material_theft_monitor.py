"""
Material Theft / Misuse Monitor
-------------------------------
Detects when an object is placed on the weighing machine (within a configured ROI)
and remains still for a configurable number of frames. Triggers an alert, saves a
snapshot, and emits Socket.IO events. Designed for camera 15 by default.
"""

import cv2
import numpy as np
import time
import logging
import os
from datetime import datetime
from pathlib import Path

from .gif_recorder import AlertGifRecorder
from .model_manager import get_shared_model, release_shared_model  # kept for consistency; not used here
from .yolo_detector import YOLODetector

logger = logging.getLogger(__name__)


class MaterialTheftMonitor:
    def __init__(self, channel_id, socketio, db_manager=None, app=None, config=None):
        self.channel_id = channel_id
        self.socketio = socketio
        self.db_manager = db_manager
        self.app = app

        # Settings (can be overridden via config)
        cfg = config or {}
        self.alert_cooldown = cfg.get("alert_cooldown", 5.0)  # Alert cooldown in seconds (default: 5s like Material_theft.py)
        self.person_proximity_threshold = cfg.get("person_proximity_threshold", 50)  # pixels from ROI border
        self.detect_persons = cfg.get("detect_persons", True)  # Enable person detection
        self.confidence_threshold = cfg.get("confidence_threshold", 0.25)  # YOLO confidence threshold

        # ROI points: allow normalized (0-1) or absolute pixels
        default_roi = np.array([
            [1938, 536],
            [1616, 396],
            [1384, 546],
            [1904, 802],
        ], dtype=np.float32)
        roi_points = cfg.get("roi_points", default_roi.tolist())
        self.roi_points = np.array(roi_points, dtype=np.float32)

        # Snapshot/GIF paths
        self.snapshot_dir = Path("static/material_theft_snapshots")
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        # YOLO model for detecting weighing_machine_item class (like Material_theft.py)
        self.model_path = cfg.get("model_path", "models/best.pt")
        self.target_class = cfg.get("target_class", "weighing_machine_item")
        self.item_model = get_shared_model(self.model_path)
        logger.info(f"[{self.channel_id}] Loaded YOLO model: {self.model_path} for class: {self.target_class}")

        # YOLO detector for person detection (if enabled)
        self.person_detector = None
        if self.detect_persons:
            self.person_detector = YOLODetector(
                model_path="models/yolo11n.pt",
                confidence_threshold=0.25,
                img_size=640,
                person_class_id=0  # Person class in yolo11n.pt
            )
            logger.info(f"[{self.channel_id}] Person detection enabled for MaterialTheftMonitor")

        # GIF recorder (optional, reusing alert GIF workflow)
        self.gif_recorder = AlertGifRecorder(buffer_size=90, gif_duration=3.0, fps=5)
        self._was_recording_alert = False
        self._last_alert_message = None
        self._last_alert_data = None

        # State
        self.last_alert_time = 0
        self.last_person_alert_time = 0
        self.prev_detected = False  # Track previous detection state (for edge detection)
        self.frame_count = 0
        self._persons_near_roi = []  # Store detected persons near ROI for visualization

        logger.info(f"[{self.channel_id}] MaterialTheftMonitor initialized")

    def __del__(self):
        try:
            release_shared_model(self.channel_id, device='auto')
        except Exception:
            pass
    
    def _point_in_polygon(self, point, polygon_points):
        """Check if a point is inside a polygon using ray casting algorithm"""
        x, y = point
        n = len(polygon_points)
        inside = False
        
        p1x, p1y = polygon_points[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon_points[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def _point_near_polygon(self, point, polygon_points, threshold):
        """Check if a point is within threshold distance from polygon border"""
        x, y = point
        
        # First check if inside polygon
        if self._point_in_polygon(point, polygon_points):
            return True
        
        # Check distance to each edge
        n = len(polygon_points)
        min_dist = float('inf')
        
        for i in range(n):
            p1 = polygon_points[i]
            p2 = polygon_points[(i + 1) % n]
            
            # Calculate distance from point to line segment
            A = x - p1[0]
            B = y - p1[1]
            C = p2[0] - p1[0]
            D = p2[1] - p1[1]
            
            dot = A * C + B * D
            len_sq = C * C + D * D
            
            if len_sq == 0:
                # Point to point distance
                dist = np.sqrt((x - p1[0])**2 + (y - p1[1])**2)
            else:
                param = dot / len_sq
                if param < 0:
                    xx, yy = p1
                elif param > 1:
                    xx, yy = p2
                else:
                    xx = p1[0] + param * C
                    yy = p1[1] + param * D
                
                dist = np.sqrt((x - xx)**2 + (y - yy)**2)
            
            min_dist = min(min_dist, dist)
        
        return min_dist <= threshold

    def process_frame(self, frame):
        if frame is None or frame.size == 0:
            logger.warning(f"[{self.channel_id}] MaterialTheftMonitor: Received empty frame")
            return {'frame': frame, 'object_present': False, 'still_counter': 0, 'persons_near_roi': 0}

        self.frame_count += 1
        
        # Log first frame to confirm module is being called
        if self.frame_count == 1:
            logger.info(f"[{self.channel_id}] MaterialTheftMonitor: Processing first frame, shape={frame.shape}")
            logger.info(f"[{self.channel_id}] ROI points: {self.roi_points}")
            logger.info(f"[{self.channel_id}] Settings: target_class={self.target_class}, cooldown={self.alert_cooldown}s")
        
        now_ts = time.time()
        current_time = datetime.now()
        h, w = frame.shape[:2]

        # Use YOLO to detect weighing_machine_item class (like Material_theft.py)
        item_detected = False
        detected_boxes = []
        detected_classes = []
        
        try:
            # Run YOLO inference
            results = self.item_model(frame, conf=self.confidence_threshold, verbose=False)
            
            # Get ROI points in pixels for checking if detection is within ROI
            roi_points_px = self._get_roi_points_pixels(w, h).astype(np.int32)
            
            # Process detections
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                class_names = results[0].names
                
                for box in boxes:
                    cls_id = int(box.cls[0])
                    cls_name = class_names[cls_id]
                    cls_name_normalized = cls_name.lower().replace(" ", "_")
                    detected_classes.append(cls_name_normalized)
                    
                    # Check if this is the target class
                    if cls_name_normalized == self.target_class:
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        
                        # Check if detection center is within ROI
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        center_point = (center_x, center_y)
                        
                        if self._point_in_polygon(center_point, roi_points_px):
                            item_detected = True
                            detected_boxes.append({
                                'bbox': [x1, y1, x2, y2],
                                'confidence': conf,
                                'class': cls_name
                            })
                            
                            if self.frame_count <= 10 or self.frame_count % 30 == 0:
                                logger.info(f"[{self.channel_id}] ✅ {self.target_class} detected in ROI: conf={conf:.2f}, center=({center_x:.0f}, {center_y:.0f})")
            
            # Log detected classes for debugging
            if detected_classes and (self.frame_count <= 10 or self.frame_count % 60 == 0):
                logger.debug(f"[{self.channel_id}] Detected classes: {detected_classes}")
                
        except Exception as e:
            logger.error(f"[{self.channel_id}] Error in YOLO detection: {e}", exc_info=True)
        
        # Alert logic (like Material_theft.py): alert when item is detected and wasn't detected before (edge detection)
        alert_triggered = False
        time_since_last_alert = now_ts - self.last_alert_time
        
        if item_detected and not self.prev_detected:
            # Item just appeared (edge detection)
            if time_since_last_alert >= self.alert_cooldown:
                alert_triggered = True
                self.last_alert_time = now_ts
                logger.warning(f"[{self.channel_id}] 🚨 ALERT: {self.target_class} detected on weighing machine!")
                self._trigger_alert(frame.copy(), current_time, len(detected_boxes), alert_type="object_placed")
            else:
                logger.debug(f"[{self.channel_id}] Alert condition met but in cooldown: "
                           f"cooldown_remaining={self.alert_cooldown - time_since_last_alert:.1f}s")
        
        # Update previous detection state
        self.prev_detected = item_detected
        
        # Debug logging
        if self.frame_count <= 10 or self.frame_count % 30 == 0 or item_detected:
            logger.info(f"[{self.channel_id}] MaterialTheft Frame {self.frame_count}: "
                       f"item_detected={item_detected}, prev_detected={self.prev_detected}, "
                       f"cooldown_remaining={max(0, self.alert_cooldown - time_since_last_alert):.1f}s")
        
        # Person detection (check every 5 frames for performance)
        persons_near_roi = []
        if self.person_detector and self.frame_count % 5 == 0:
            try:
                person_detections = self.person_detector.detect_persons(frame)
                roi_points_px = self._get_roi_points_pixels(w, h).astype(np.int32)
                
                # Log total person detections for debugging
                if self.frame_count <= 10 or self.frame_count % 60 == 0:
                    logger.info(f"[{self.channel_id}] Person detection: Found {len(person_detections)} person(s) in frame")
                
                for detection in person_detections:
                    # Check if person's bottom center (feet position) is near ROI
                    bottom_center = detection.get('bottom_center', detection.get('center'))
                    if bottom_center:
                        is_near = self._point_near_polygon(bottom_center, roi_points_px, self.person_proximity_threshold)
                        if is_near:
                            persons_near_roi.append(detection)
                            if self.frame_count <= 10 or self.frame_count % 60 == 0:
                                logger.info(f"[{self.channel_id}] 👤 Person near ROI detected at ({bottom_center[0]:.0f}, {bottom_center[1]:.0f})")
                
                # Store for visualization
                self._persons_near_roi = persons_near_roi
                
                # Log person detection status
                if persons_near_roi:
                    logger.info(f"[{self.channel_id}] 👤 Person(s) detected near ROI: {len(persons_near_roi)} person(s) - No alert (persons near weighing machine are expected)")
                
                # No alerts for persons near weighing machine - they are expected to be there
                # Don't reset still counter - we still want to alert when objects are placed, even if person is nearby
            except Exception as e:
                logger.error(f"[{self.channel_id}] Error in person detection: {e}", exc_info=True)
                self._persons_near_roi = []
        else:
            # Keep previous person detections for visualization between detection frames
            if not hasattr(self, '_persons_near_roi'):
                self._persons_near_roi = []

        # Add frame to GIF buffer (always buffer frames)
        self.gif_recorder.add_frame(frame)
        
        # Only start GIF recording when alert is actually triggered
        if alert_triggered and not self.gif_recorder.is_recording_alert:
            alert_info = {
                'type': 'material_theft_alert',
                'message': self._last_alert_message or f"Object placed on scale - {self.channel_id}",
                'timestamp': current_time.isoformat()
            }
            self.gif_recorder.start_alert_recording(alert_info)
            logger.info(f"[{self.channel_id}] Started GIF recording for material theft alert")
        elif self.gif_recorder.is_recording_alert:
            # Continue recording if already recording
            self.gif_recorder.add_alert_frame(frame)

        # When recording finishes, save GIF entry to DB
        if self._was_recording_alert and not self.gif_recorder.is_recording_alert:
            gif_info = self.gif_recorder.get_last_gif_info()
            logger.info(f"[{self.channel_id}] GIF recording finished. gif_info={gif_info}, has_db_manager={self.db_manager is not None}, has_alert_message={self._last_alert_message is not None}")
            if gif_info and self.db_manager and self._last_alert_message:
                try:
                    # Normalize path separators for cross-platform compatibility
                    gif_path = gif_info.get('gif_path', '').replace('\\', '/')
                    gif_filename = gif_info.get('gif_filename', '')
                    
                    payload = {
                        'gif_filename': gif_filename,
                        'gif_path': gif_path,
                        'frame_count': gif_info.get('frame_count', 0),
                        'duration': gif_info.get('duration', 0.0)  # Fixed: use 'duration' not 'gif_duration'
                    }
                    logger.info(f"[{self.channel_id}] Saving material theft alert GIF to database: {gif_filename} (path: {gif_path})")
                    if self.app:
                        with self.app.app_context():
                            # Save GIF to database
                            gif_id = self.db_manager.save_alert_gif(
                                self.channel_id,
                                'material_theft_alert',
                                payload,
                                alert_message=self._last_alert_message,
                                alert_data=self._last_alert_data
                            )
                            logger.info(f"[{self.channel_id}] ✅ Material theft alert GIF saved to database: ID={gif_id}, {payload.get('gif_filename')}")
                    else:
                        # Save GIF to database
                        gif_id = self.db_manager.save_alert_gif(
                            self.channel_id,
                            'material_theft_alert',
                            payload,
                            alert_message=self._last_alert_message,
                            alert_data=self._last_alert_data
                        )
                        logger.info(f"[{self.channel_id}] ✅ Material theft alert GIF saved to database: ID={gif_id}, {payload.get('gif_filename')}")
                except Exception as e:
                    logger.error(f"[{self.channel_id}] ❌ Error saving material theft GIF to database: {e}", exc_info=True)
                finally:
                    # Clear stored alert info after saving
                    self._last_alert_message = None
                    self._last_alert_data = None
            elif not self._last_alert_message:
                logger.warning(f"[{self.channel_id}] ⚠️ GIF recording finished but no alert message stored - alert may not have been triggered")
        self._was_recording_alert = self.gif_recorder.is_recording_alert

        # Draw ROI overlay with enhanced visibility
        annotated = frame.copy()
        roi_points_px = self._get_roi_points_pixels(w, h).astype(np.int32)
        
        # Log ROI points on first frame for debugging
        if self.frame_count == 1:
            logger.info(f"[{self.channel_id}] ✅ ROI points in pixels: {roi_points_px.tolist()}")
        
        # Draw semi-transparent fill for ROI area
        roi_overlay = annotated.copy()
        cv2.fillPoly(roi_overlay, [roi_points_px], (0, 255, 0))
        cv2.addWeighted(annotated, 0.7, roi_overlay, 0.3, 0, annotated)
        
        # Draw ROI border with thicker, more visible line
        cv2.polylines(annotated, [roi_points_px], True, (0, 255, 0), 3)
        
        # Add ROI label at top-left corner of ROI
        if len(roi_points_px) > 0:
            # Find topmost point for label placement
            top_point = tuple(min(roi_points_px, key=lambda p: p[1]))
            label_y = max(10, top_point[1] - 10)
            label_x = top_point[0]
            
            # Background for text
            (text_width, text_height), _ = cv2.getTextSize("MONITORING AREA", cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(annotated, 
                         (label_x - 5, label_y - text_height - 5), 
                         (label_x + text_width + 5, label_y + 5), 
                         (0, 255, 0), -1)
            cv2.putText(annotated, "MONITORING AREA", (label_x, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Draw detected items (like Material_theft.py)
        if item_detected and detected_boxes:
            for det in detected_boxes:
                x1, y1, x2, y2 = det['bbox']
                conf = det['confidence']
                
                # Draw bounding box (red like Material_theft.py)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # Draw label
                label_text = f"ITEM ON WEIGHING MACHINE ({conf:.2f})"
                cv2.putText(annotated, label_text, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw detected persons near ROI
        if hasattr(self, '_persons_near_roi') and self._persons_near_roi:
            for detection in self._persons_near_roi:
                x1, y1, x2, y2 = detection['bbox']
                confidence = detection['confidence']
                bottom_center = detection.get('bottom_center', detection.get('center'))
                
                # Draw bounding box for person (blue for person detection)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 165, 0), 3)  # Orange color
                
                # Draw label
                label_text = f"PERSON NEAR ROI ({confidence:.2f})"
                (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated, 
                             (x1, y1 - text_height - 10), 
                             (x1 + text_width + 5, y1), 
                             (255, 165, 0), -1)
                cv2.putText(annotated, label_text, (x1 + 2, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
                # Draw bottom center point (feet position)
                if bottom_center:
                    cv2.circle(annotated, bottom_center, 8, (255, 165, 0), -1)
                    cv2.circle(annotated, bottom_center, 12, (255, 165, 0), 2)
        
        # Enhanced status bar at top
        status_txt = f"Material Theft / Misuse Monitor | Item Detected: {'YES' if item_detected else 'NO'}"
        if item_detected:
            status_txt += " | ALERT READY"
        
        # Status bar background
        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 40), (40, 40, 40), -1)
        
        # Status text
        cv2.putText(annotated, status_txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add channel info in top right
        channel_info = f"Channel: {self.channel_id}"
        (text_width, _), _ = cv2.getTextSize(channel_info, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(annotated, channel_info, (annotated.shape[1] - text_width - 10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Return result in the format expected by SharedMultiModuleVideoProcessor
        return {
            'frame': annotated,
            'object_present': item_detected,
            'still_counter': 1 if item_detected else 0,  # Keep for compatibility
            'persons_near_roi': len(self._persons_near_roi) if hasattr(self, '_persons_near_roi') else 0
        }

    def set_roi(self, roi_points):
        """
        Set ROI points (can be normalized 0-1 or absolute pixels)
        """
        if roi_points:
            self.roi_points = np.array(roi_points, dtype=np.float32)
            # Reset mask so it rebuilds on next frame
            self.roi_mask = None
            self.frame_size = None
            logger.info(f"[{self.channel_id}] ROI updated: {len(roi_points)} points")
    
    def _get_roi_points_pixels(self, w, h):
        pts = self.roi_points.copy()
        # Detect if points are normalized (0-1 range)
        if np.max(pts) <= 5.0:
            pts[:, 0] *= w
            pts[:, 1] *= h
        else:
            # Points are absolute pixels - check if they need scaling
            # If ROI points are larger than frame, assume they're for a different resolution
            # Scale them down proportionally
            max_x = np.max(pts[:, 0])
            max_y = np.max(pts[:, 1])
            if max_x > w or max_y > h:
                # Scale down to fit current frame - use separate scales for X and Y
                # This handles cases where frame aspect ratio differs from ROI aspect ratio
                scale_x = w / max_x if max_x > w else 1.0
                scale_y = h / max_y if max_y > h else 1.0
                # Apply scaling
                pts[:, 0] *= scale_x
                pts[:, 1] *= scale_y
                logger.info(f"[{self.channel_id}] Scaled ROI from {max_x:.0f}x{max_y:.0f} to fit frame {w}x{h} (scale_x={scale_x:.3f}, scale_y={scale_y:.3f})")
                logger.info(f"[{self.channel_id}] Scaled ROI points: {pts.tolist()}")
        return pts

    def _trigger_alert(self, frame, current_time, detection_count, alert_type="object_placed", person_count=None):
        ts = current_time.strftime("%Y%m%d_%H%M%S")
        filename = f"material_theft_{self.channel_id}_{ts}.jpg"
        filepath = self.snapshot_dir / filename
        try:
            cv2.imwrite(str(filepath), frame)
            file_size = os.path.getsize(filepath)

            if alert_type == "person_near":
                alert_message = f"👤 Person detected near weighing machine (count: {person_count})"
                alert_data = {
                    "channel_id": self.channel_id,
                    "person_count": person_count,
                    "alert_type": "person_near",
                    "snapshot_path": str(filepath),
                    "timestamp": current_time.isoformat()
                }
                logger.info(f"[{self.channel_id}] ✅ Setting person alert message: {alert_message}")
            else:  # object_placed
                alert_message = f"📦 {self.target_class} detected on weighing machine"
                alert_data = {
                    "channel_id": self.channel_id,
                    "detection_count": detection_count,
                    "alert_type": "object_placed",
                    "target_class": self.target_class,
                    "snapshot_path": str(filepath),
                    "timestamp": current_time.isoformat()
                }
                logger.info(f"[{self.channel_id}] ✅ Setting object alert message: {alert_message}")

            # Store for GIF saving hookup
            self._last_alert_message = alert_message
            self._last_alert_data = alert_data

            # Save alert to database (use log_alert for snapshot-based alerts)
            if self.db_manager:
                try:
                    if self.app:
                        with self.app.app_context():
                            # Log alert to database
                            self.db_manager.log_alert(
                                self.channel_id,
                                'material_theft_alert',
                                alert_message,
                                alert_data=alert_data
                            )
                            # Also save as alert GIF entry (for snapshot)
                            self.db_manager.save_alert_gif(
                                self.channel_id,
                                'material_theft_alert',
                                {'gif_filename': filename, 'gif_path': str(filepath), 'frame_count': 0, 'duration': 0.0},
                                alert_message=alert_message,
                                alert_data=alert_data
                            )
                            logger.info(f"[{self.channel_id}] Material theft alert saved to database: {alert_message}")
                    else:
                        # Log alert to database
                        self.db_manager.log_alert(
                            self.channel_id,
                            'material_theft_alert',
                            alert_message,
                            alert_data=alert_data
                        )
                        # Also save as alert GIF entry (for snapshot)
                        self.db_manager.save_alert_gif(
                            self.channel_id,
                            'material_theft_alert',
                            {'gif_filename': filename, 'gif_path': str(filepath), 'frame_count': 0, 'duration': 0.0},
                            alert_message=alert_message,
                            alert_data=alert_data
                        )
                        logger.info(f"[{self.channel_id}] Material theft alert saved to database: {alert_message}")
                except Exception as db_error:
                    logger.error(f"[{self.channel_id}] Error saving material theft alert to database: {db_error}", exc_info=True)

            # Emit socket event
            if self.socketio:
                emit_data = {
                    'channel_id': self.channel_id,
                    'message': alert_message,
                    'timestamp': current_time.isoformat(),
                    'snapshot_url': f"/static/material_theft_snapshots/{filename}",
                    'alert_type': alert_type
                }
                if alert_type == "person_near":
                    emit_data['person_count'] = person_count
                else:
                    emit_data['detection_count'] = detection_count
                self.socketio.emit('material_theft_alert', emit_data)

            logger.warning(f"[{self.channel_id}] Material theft alert triggered, snapshot: {filename} ({file_size} bytes)")
            
            # Debug: Check if db_manager is available
            if not self.db_manager:
                logger.warning(f"[{self.channel_id}] ⚠️ db_manager is None - alerts will not be saved to database!")
            else:
                logger.info(f"[{self.channel_id}] ✅ db_manager is available - alert should be saved")
                
        except Exception as e:
            logger.error(f"[{self.channel_id}] Error saving material theft snapshot: {e}", exc_info=True)

    def get_status(self):
        return {
            "module": "MaterialTheftMonitor",
            "channel_id": self.channel_id,
            "last_alert_time": self.last_alert_time,
            "item_detected": self.prev_detected,
            "frame_count": self.frame_count,
            "target_class": self.target_class
        }


