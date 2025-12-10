"""
Restricted Area Monitor - Allow only specific uniform colors in defined ROI
Detects unauthorized personnel (non-black uniform) entering restricted zones
"""

import cv2
import numpy as np
from datetime import datetime, timedelta
import logging
import os
import json
from .yolo_detector import YOLODetector

logger = logging.getLogger(__name__)

class RestrictedAreaMonitor:
    def __init__(self, channel_id, model_path, db_manager=None, socketio=None, config=None):
        """
        Initialize Restricted Area Monitor
        
        Args:
            channel_id: Channel identifier
            model_path: Path to YOLO model
            db_manager: Database manager instance
            socketio: SocketIO instance for real-time alerts
            config: Configuration dictionary
        """
        self.channel_id = channel_id
        self.model_path = model_path
        self.db_manager = db_manager
        self.socketio = socketio
        
        # Initialize YOLO detector with very low confidence threshold to catch uniform_beige
        self.detector = YOLODetector(model_path=model_path, confidence_threshold=0.15, img_size=640)
        
        # Default configuration
        self.config = config or {}
        self.confidence_threshold = self.config.get('confidence_threshold', 0.15)  # Very low threshold for uniform_beige
        self.alert_cooldown = self.config.get('alert_cooldown', 10)  # seconds between alerts
        
        # Target detection: only uniform_beige (all other classes are ignored)
        self.target_class = 'uniform_beige'
        self.beige_threshold = 0.15  # Ultra-low threshold specifically for uniform_beige
        
        # Allowed class in restricted area - DISABLED (detect only uniform_beige)
        self.allowed_class = None
        
        # Unauthorized classes - ONLY uniform_beige triggers alert
        self.unauthorized_classes = ['uniform_beige']
        
        # ROI points (will be loaded from config or database)
        self.roi_points = self.config.get('roi_points', [])
        
        # Alert tracking
        self.last_alert_time = None
        self.violation_count = 0
        self.total_checks = 0
        
        # Statistics
        self.stats = {
            'total_violations': 0,
            'unauthorized_entries': 0,
            'last_violation_time': None,
            'violation_by_class': {}
        }
        
        # Snapshot directory
        self.snapshot_dir = 'static/restricted_area_snapshots'
        os.makedirs(self.snapshot_dir, exist_ok=True)
        
        logger.info(f"Restricted Area Monitor initialized for channel {channel_id}")
        logger.info(f"🎯 TARGET DETECTION: Only '{self.target_class}' class will trigger alerts")
        logger.info(f"📊 Confidence Threshold: uniform_beige={self.beige_threshold}, others={self.confidence_threshold}")
        logger.info(f"Unauthorized classes: {self.unauthorized_classes}")
    
    def set_roi_points(self, points):
        """Set ROI polygon points"""
        # Validate points format
        if isinstance(points, dict) and 'main' in points:
            points = points['main']
            logger.info(f"Extracted 'main' from dict for channel {self.channel_id}")
        
        # Validate each point
        validated_points = []
        for i, point in enumerate(points):
            if isinstance(point, dict) and 'x' in point and 'y' in point:
                try:
                    validated_points.append([float(point['x']), float(point['y'])])
                except (ValueError, TypeError) as e:
                    logger.error(f"Invalid dict point {i}: {point}, error: {e}")
            elif isinstance(point, (list, tuple)) and len(point) == 2:
                try:
                    validated_points.append([float(point[0]), float(point[1])])
                except (ValueError, TypeError) as e:
                    logger.error(f"Invalid point {i}: {point}, error: {e}")
            else:
                logger.error(f"Point {i} has invalid format: {point}")
        
        self.roi_points = validated_points
        logger.info(f"✅ ROI points validated and set for channel {self.channel_id}: {len(self.roi_points)} valid points")
    
    def get_roi_points(self):
        """Get current ROI points"""
        return self.roi_points
    
    def set_roi(self, roi_data):
        """Set ROI configuration (wrapper for compatibility with generic API)"""
        # Handle both formats: direct list or dict with 'main' key
        if isinstance(roi_data, dict) and 'main' in roi_data:
            points_list = roi_data['main']
            logger.info(f"ROI configuration set (dict format) for channel {self.channel_id}: {len(points_list)} points")
        elif isinstance(roi_data, list):
            points_list = roi_data
            logger.info(f"ROI configuration set (list format) for channel {self.channel_id}: {len(points_list)} points")
        else:
            logger.error(f"Invalid ROI data format: {type(roi_data)}, data: {roi_data}")
            self.roi_points = []
            return
        
        # Validate and clean the points list
        validated_points = []
        for i, point in enumerate(points_list):
            if isinstance(point, dict):
                # Handle dict points with x, y keys
                if 'x' in point and 'y' in point:
                    try:
                        validated_points.append([float(point['x']), float(point['y'])])
                    except (ValueError, TypeError) as e:
                        logger.error(f"Could not convert dict point {i}: {point}, error: {e}")
                else:
                    logger.error(f"Dict point {i} missing x/y keys: {point}")
            elif isinstance(point, (list, tuple)) and len(point) == 2:
                try:
                    validated_points.append([float(point[0]), float(point[1])])
                except (ValueError, TypeError) as e:
                    logger.error(f"Invalid coordinates in point {i}: {point}, error: {e}")
            else:
                logger.error(f"Point {i} has invalid format: {point}")
        
        self.roi_points = validated_points
        logger.info(f"✅ ROI validated and set: {len(self.roi_points)} valid points for channel {self.channel_id}")
    
    def get_roi(self):
        """Get ROI configuration (wrapper for compatibility with generic API)"""
        return {'main': self.roi_points, 'secondary': []}
    
    def point_in_polygon(self, point, polygon):
        """Check if point is inside polygon using ray casting algorithm"""
        if len(polygon) < 3:
            return False
        
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def process_frame(self, frame, detections=None):
        """
        Process frame and detect unauthorized personnel in restricted area
        
        Args:
            frame: Video frame (numpy array)
            detections: Optional list of detection dicts with keys: class_name, confidence, bbox
                       If None, will run detection using internal detector
            
        Returns:
            dict: Processing result with frame, violations, and metadata
        """
        self.total_checks += 1
        
        if frame is None:
            return {
                'frame': frame,
                'violations': [],
                'metadata': {
                    'roi_defined': len(self.roi_points) >= 3,
                    'total_checks': self.total_checks
                }
            }
        
        violations = []
        frame_with_overlay = frame.copy()
        
        # DEBUG: Draw ULTRA VISIBLE indicators to confirm overlay is working
        # Draw bright red bar at top
        cv2.rectangle(frame_with_overlay, (0, 0), (frame.shape[1], 50), (0, 0, 255), -1)
        cv2.putText(frame_with_overlay, "RESTRICTED AREA MONITOR ACTIVE", (10, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        
        # Draw bright green bar at bottom to double-confirm
        cv2.rectangle(frame_with_overlay, (0, frame.shape[0]-50), (frame.shape[1], frame.shape[0]), (0, 255, 0), -1)
        cv2.putText(frame_with_overlay, f"ROI POINTS: {len(self.roi_points)}", (10, frame.shape[0]-15),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)
        
        # If no detections provided, run detection WITHOUT class filtering
        if detections is None:
            # Detect ALL classes - no filtering at YOLO level
            detections = self.detector.detect_all_classes(frame, target_classes=None)
            
            # Debug: Log raw detections every 50 frames
            if self.total_checks % 50 == 1:
                logger.info(f"🔍 DEBUG camera_14: YOLO returned {len(detections)} detections (all classes)")
                if len(detections) > 0:
                    for i, det in enumerate(detections[:5]):  # Log first 5
                        logger.info(f"   Detection {i}: {det.get('class_name')} conf={det.get('confidence'):.2f} bbox={det.get('bbox')}")
        
        # Validate and fix ROI points format - ensure it's a list of coordinate pairs
        if isinstance(self.roi_points, dict):
            if 'main' in self.roi_points:
                self.roi_points = self.roi_points['main']
                logger.warning(f"Converted ROI from dict to list: {len(self.roi_points)} points")
            else:
                logger.error(f"ROI dict has no 'main' key: {self.roi_points}")
                self.roi_points = []
        
        # Validate that roi_points is a list
        if not isinstance(self.roi_points, list):
            logger.error(f"ROI points is not a list, type: {type(self.roi_points)}, value: {self.roi_points}")
            self.roi_points = []
        
        # Validate each point in roi_points is a list/tuple of 2 numbers
        if isinstance(self.roi_points, list) and len(self.roi_points) > 0:
            validated_points = []
            try:
                for i, point in enumerate(self.roi_points):
                    if isinstance(point, (list, tuple)) and len(point) == 2:
                        try:
                            validated_points.append([float(point[0]), float(point[1])])
                        except (ValueError, TypeError) as e:
                            logger.error(f"Invalid point at index {i}: {point}, error: {e}")
                    elif isinstance(point, dict):
                        logger.error(f"Point {i} is a dict (should be list/tuple): {point}")
                        # Try to extract x, y from dict if present
                        if 'x' in point and 'y' in point:
                            try:
                                validated_points.append([float(point['x']), float(point['y'])])
                                logger.info(f"Converted dict point to list: [{point['x']}, {point['y']}]")
                            except (ValueError, TypeError) as e:
                                logger.error(f"Could not convert dict point: {e}")
                    else:
                        logger.error(f"Point {i} has invalid format: {point}")
            except Exception as e:
                logger.error(f"Error validating ROI points: {e}, roi_points type: {type(self.roi_points)}, content: {self.roi_points}")
                validated_points = []
            
            if len(validated_points) != len(self.roi_points):
                logger.warning(f"ROI validation: {len(self.roi_points)} points -> {len(validated_points)} valid points")
            self.roi_points = validated_points
        
        # Early return if ROI not defined - but still draw frame with info
        if not isinstance(self.roi_points, list) or len(self.roi_points) < 3:
            cv2.putText(frame_with_overlay, "ROI not configured", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return {
                'frame': frame_with_overlay,
                'violations': [],
                'metadata': {
                    'roi_defined': False,
                    'total_checks': self.total_checks
                }
            }
        
        # Draw ROI polygon - ALWAYS VISIBLE with thick bright lines
        if len(self.roi_points) >= 3:
            # Convert normalized coordinates (0.0-1.0) to pixel coordinates
            frame_height, frame_width = frame_with_overlay.shape[:2]
            roi_pixels = []
            for point in self.roi_points:
                px = int(point[0] * frame_width)
                py = int(point[1] * frame_height)
                roi_pixels.append([px, py])
            
            # Log ROI drawing with coordinates ALWAYS for debugging
            logger.info(f"🔷 Drawing ROI on channel {self.channel_id}: {len(self.roi_points)} normalized points")
            logger.info(f"   Normalized: {self.roi_points}")
            logger.info(f"   Pixels: {roi_pixels} (frame: {frame_width}x{frame_height})")
            
            roi_array = np.array(roi_pixels, dtype=np.int32)
            
            # Draw EXTRA THICK BRIGHT borders for maximum visibility
            # Outer thick cyan border
            cv2.polylines(frame_with_overlay, [roi_array], True, (0, 255, 255), 8)  # Very thick - 8px
            
            # Middle red line for contrast
            cv2.polylines(frame_with_overlay, [roi_array], True, (0, 0, 255), 4)  # Red 4px
            
            # Inner white line
            cv2.polylines(frame_with_overlay, [roi_array], True, (255, 255, 255), 2)  # White 2px
            
            # Semi-transparent colored overlay (more visible)
            overlay = frame_with_overlay.copy()
            cv2.fillPoly(overlay, [roi_array], (0, 255, 255))
            cv2.addWeighted(overlay, 0.25, frame_with_overlay, 0.75, 0, frame_with_overlay)  # 25% opacity
            
            # Draw LARGE corner points for visibility
            for idx, point in enumerate(self.roi_points):
                # Large green circles at corners
                cv2.circle(frame_with_overlay, (int(point[0]), int(point[1])), 12, (0, 255, 0), -1)  # Larger green circles
                cv2.circle(frame_with_overlay, (int(point[0]), int(point[1])), 12, (255, 255, 255), 3)  # Thicker white border
                # Add point number
                cv2.putText(frame_with_overlay, str(idx+1), (int(point[0])-5, int(point[1])+5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            # Log when ROI is not drawn
            if self.total_checks % 100 == 1:
                logger.info(f"ROI not drawn on channel {self.channel_id}: only {len(self.roi_points)} points")
        
        # Log detection activity every 100 frames
        if self.total_checks % 100 == 1:
            logger.info(f"RestrictedAreaMonitor {self.channel_id}: Processing {len(detections)} detections")
        
        # Convert ROI to pixel coordinates for detection checking
        frame_height, frame_width = frame_with_overlay.shape[:2]
        roi_pixels_for_check = []
        for point in self.roi_points:
            px = int(point[0] * frame_width)
            py = int(point[1] * frame_height)
            roi_pixels_for_check.append([px, py])
        
        # Check each detection
        for det in detections:
            class_name = det.get('class_name', '')
            confidence = det.get('confidence', 0)
            bbox = det.get('bbox', [])
            
            # Use ultra-low threshold for uniform_beige, normal threshold for others
            min_confidence = self.beige_threshold if class_name == 'uniform_beige' else self.confidence_threshold
            if confidence < min_confidence:
                continue
            
            if len(bbox) != 4:
                continue
            
            x1, y1, x2, y2 = bbox
            
            # Calculate center point of bounding box
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            # Check if object is in restricted area (using pixel coordinates)
            in_roi = self.point_in_polygon((center_x, center_y), roi_pixels_for_check)
            
            if in_roi:
                # Log when any object enters ROI (debug every 50 frames)
                if self.total_checks % 50 == 1:
                    logger.info(f"Object in ROI: {class_name} (conf: {confidence:.2f}) at center ({center_x}, {center_y})")
                
                # ONLY TRIGGER ALERT for uniform_beige
                if class_name == 'uniform_beige':
                    logger.warning(f"🚨 ALERT: uniform_beige detected (conf: {confidence:.2f}) in restricted area on {self.channel_id}")
                    
                    violations.append({
                        'class_name': class_name,
                        'confidence': confidence,
                        'bbox': bbox,
                        'center': (center_x, center_y)
                    })
                    
                    # Draw RED box for uniform_beige in ROI (ALERT)
                    cv2.rectangle(frame_with_overlay, (int(x1), int(y1)), (int(x2), int(y2)), 
                                (0, 0, 255), 3)
                    
                    # Label
                    label = f"ALERT: {class_name} {confidence:.2f}"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame_with_overlay, 
                                (int(x1), int(y1) - label_size[1] - 10),
                                (int(x1) + label_size[0], int(y1)),
                                (0, 0, 255), -1)
                    cv2.putText(frame_with_overlay, label,
                              (int(x1), int(y1) - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Update statistics
                    self.stats['violation_by_class'][class_name] = \
                        self.stats['violation_by_class'].get(class_name, 0) + 1
                else:
                    # Draw gray box for other objects IN ROI (non-alert)
                    cv2.rectangle(frame_with_overlay, (int(x1), int(y1)), (int(x2), int(y2)), 
                                (128, 128, 128), 2)
                    label = f"{class_name} {confidence:.2f}"
                    cv2.putText(frame_with_overlay, label,
                              (int(x1), int(y1) - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 2)
            else:
                # Draw boxes for ALL detections OUTSIDE ROI too
                if class_name == 'uniform_beige':
                    # Draw ORANGE box for uniform_beige OUTSIDE ROI (warning)
                    cv2.rectangle(frame_with_overlay, (int(x1), int(y1)), (int(x2), int(y2)), 
                                (0, 165, 255), 2)
                    label = f"uniform_beige {confidence:.2f}"
                    cv2.putText(frame_with_overlay, label,
                              (int(x1), int(y1) - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                else:
                    # Draw LIGHT GRAY box for other objects outside ROI
                    cv2.rectangle(frame_with_overlay, (int(x1), int(y1)), (int(x2), int(y2)), 
                                (200, 200, 200), 1)
                    label = f"{class_name} {confidence:.2f}"
                    cv2.putText(frame_with_overlay, label,
                              (int(x1), int(y1) - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Trigger alert if violations detected
        if violations:
            self._trigger_violation_alert(frame_with_overlay, violations)
        
        # Add info text
        info_text = f"Checks: {self.total_checks} | Violations: {self.stats['total_violations']}"
        cv2.putText(frame_with_overlay, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show ROI status
        roi_status = f"ROI: {len(self.roi_points)} points" if len(self.roi_points) >= 3 else "ROI: Not Configured"
        cv2.putText(frame_with_overlay, roi_status, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Show detection target
        target_text = f"Target: {self.target_class}"
        cv2.putText(frame_with_overlay, target_text, (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        if violations:
            cv2.putText(frame_with_overlay, f"ALERT: {len(violations)} Unauthorized Entry", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Log frame return every 100 frames
        if self.total_checks % 100 == 0:
            logger.info(f"🎬 Returning annotated frame: shape={frame_with_overlay.shape}, ROI points={len(self.roi_points)}, detections={len(detections)}")
        
        return {
            'frame': frame_with_overlay,
            'violations': violations,
            'metadata': {
                'total_checks': self.total_checks,
                'total_violations': self.stats['total_violations'],
                'current_violations': len(violations),
                'roi_defined': len(self.roi_points) >= 3
            }
        }
    
    def process_frame_with_detections(self, frame, detections):
        """
        Process frame with pre-computed detections (for shared detection optimization)
        
        Args:
            frame: Video frame (numpy array)
            detections: Pre-computed list of detection dicts
            
        Returns:
            dict: Processing result with frame, violations, and metadata
        """
        return self.process_frame(frame, detections)
    
    def _trigger_violation_alert(self, frame, violations):
        """Trigger alert for unauthorized entry"""
        current_time = datetime.now()
        
        # Check cooldown
        if self.last_alert_time:
            time_diff = (current_time - self.last_alert_time).total_seconds()
            if time_diff < self.alert_cooldown:
                return
        
        self.last_alert_time = current_time
        self.stats['total_violations'] += 1
        self.stats['unauthorized_entries'] += len(violations)
        self.stats['last_violation_time'] = current_time.isoformat()
        
        # Save snapshot
        timestamp = current_time.strftime("%Y%m%d_%H%M%S")
        snapshot_filename = f"restricted_area_{self.channel_id}_{timestamp}.jpg"
        snapshot_path = os.path.join(self.snapshot_dir, snapshot_filename)
        
        cv2.imwrite(snapshot_path, frame)
        file_size = os.path.getsize(snapshot_path)
        
        # Create alert message
        violation_classes = [v['class_name'] for v in violations]
        alert_message = f"UNAUTHORIZED ENTRY: {', '.join(set(violation_classes))} detected in restricted area"
        
        alert_data = {
            'channel_id': self.channel_id,
            'timestamp': current_time.isoformat(),
            'violation_count': len(violations),
            'violation_classes': violation_classes,
            'allowed_class': self.allowed_class
        }
        
        logger.warning(f"🚨 {alert_message} on channel {self.channel_id}")
        
        # Save to database
        if self.db_manager:
            try:
                snapshot_id = self.db_manager.save_restricted_area_snapshot(
                    channel_id=self.channel_id,
                    snapshot_filename=snapshot_filename,
                    snapshot_path=snapshot_path,
                    alert_message=alert_message,
                    alert_data=alert_data,
                    file_size=file_size,
                    violation_count=len(violations),
                    detection_time=current_time
                )
                logger.info(f"Restricted area violation saved to database: ID {snapshot_id}")
            except Exception as e:
                logger.error(f"Error saving restricted area violation to database: {e}")
        
        # Send real-time alert via SocketIO
        if self.socketio:
            try:
                self.socketio.emit('restricted_area_alert', {
                    'channel_id': self.channel_id,
                    'message': alert_message,
                    'violation_count': len(violations),
                    'violation_classes': violation_classes,
                    'snapshot_url': f"/static/restricted_area_snapshots/{snapshot_filename}",
                    'timestamp': current_time.isoformat()
                })
            except Exception as e:
                logger.error(f"Error sending SocketIO alert: {e}")
    
    def get_statistics(self):
        """Get monitoring statistics"""
        return {
            'total_checks': self.total_checks,
            'total_violations': self.stats['total_violations'],
            'unauthorized_entries': self.stats['unauthorized_entries'],
            'last_violation_time': self.stats['last_violation_time'],
            'violation_by_class': self.stats['violation_by_class'],
            'allowed_class': self.allowed_class,
            'unauthorized_classes': self.unauthorized_classes,
            'roi_points_count': len(self.roi_points)
        }
    
    def get_status(self):
        """Get current status"""
        return {
            'active': True,
            'channel_id': self.channel_id,
            'roi_defined': len(self.roi_points) >= 3,
            'statistics': self.get_statistics()
        }
