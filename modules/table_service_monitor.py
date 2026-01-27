"""
Cleanliness & Table Reset Monitor Module
-----------------------------------------

Monitors table cleanliness and reset times after customers leave.

CV Problem Definition:
- Detect unclean tables (Table_unclean class)
- Track when customers leave tables
- Monitor reset time: time from when customers leave until table is cleaned
- Violations:
  1. Unclean table with no customers (needs cleaning)
  2. Slow reset: table remains unclean too long after customers leave

Rules:
- Table ROI = polygon around the table area
- Customer = person in table ROI (any person)
- Unclean table = Table_unclean class detected
- Clean table = Table_clean class detected
"""

import cv2
import numpy as np
import logging
import math
from datetime import datetime
import os
import json
from pathlib import Path

from ultralytics import YOLO
from .model_manager import get_shared_model, release_shared_model
from .gif_recorder import AlertGifRecorder
from .yolo_detector import YOLODetector

logger = logging.getLogger(__name__)


class TableServiceMonitor:
    def __init__(self, channel_id, socketio, db_manager=None, app=None):
        """
        Initialize Table Service Monitor

        Args:
            channel_id: Unique identifier for the video channel
            socketio: SocketIO instance for real-time updates
            db_manager: Database manager (optional)
            app: Flask application instance (optional)
        """
        self.channel_id = channel_id
        self.socketio = socketio
        self.db_manager = db_manager
        self.app = app

        # Model configuration - Use custom trained model (best.pt)
        self.model_weight = "models/best.pt"
        self.conf_threshold = 0.5  # General confidence threshold for all detections
        self.unclean_conf_threshold = 0.75  # Higher threshold specifically for unclean detections (reduces false positives)
        self.nms_iou = 0.45

        # Allowed uniform classes for cameras 5 & 6 (matching script logic)
        # Any uniform NOT in this list will trigger a violation
        self.allowed_uniforms = {
            "uniform_black",
            "uniform_cream",
            "uniform_grey"
        }
        
        # Track uniform violation cooldown to prevent spam
        self.last_uniform_alert_time = None
        self.uniform_alert_cooldown = 30.0  # 30 seconds between uniform violation alerts

        # Table cleanliness classes (best.pt uses lowercase)
        self.table_clean_class = "table_clean"      # Class 15
        self.table_unclean_class = "table_unclean"  # Class 16

        # Person class ID (best.pt person is class 0)
        self.person_class_id = 0

        # Load shared YOLO model for tables
        self.model = get_shared_model(self.model_weight)
        
        # Person detector (using yolo11n.pt like simple script)
        self.person_detector = YOLODetector(
            model_path="models/yolo11n.pt",
            confidence_threshold=0.25,  # Lower threshold to detect more persons
            img_size=640,
            person_class_id=0  # Person class in yolo11n.pt (class 0 = person)
        )
        
        # Person proximity settings (matching simple script)
        self.PERSON_NEAR_DISTANCE = 250  # pixels - increased to better detect persons at tables
        self.TABLE_MATCH_DISTANCE = 100  # pixels - for table matching (if not using ROIs)

        # Table ROI configuration
        # Format: {table_id: {"polygon": [(x1,y1), (x2,y2), ...], "bbox": (min_x, min_y, max_x, max_y)}}
        self.table_rois = {}

        # Settings
        self.settings = {
            "unclean_alert_cooldown": 180.0,  # 3 minutes between unclean table alerts
            "unclean_duration_threshold": 0.0,  # Changed to 0.0 to match provided code (immediate alert)
            # Note: Provided code alerts immediately, but we keep cooldown to prevent spam
        }
        
        # Status update tracking
        self.frame_count = 0
        self.total_alerts = 0
        self.last_status_update = 0
        self.status_update_interval = 2.0  # Update status every 2 seconds
        
        # Performance optimization: frame skipping
        # Process every Nth frame to reduce YOLO inference load
        self.process_every_n_frames = 5  # Process every 5th frame (adjust based on performance needs)
        self.last_detections = []  # Cache last detections for skipped frames
        self.last_table_detections = {}  # Cache last table detections
        self.last_persons = []  # Cache last person positions for skipped frames

        # Tracking data per table
        # Format: {table_id: {
        #     "cleanliness_state": "clean"/"unclean"/None,
        #     "unclean_start_time": timestamp,  # When table became unclean
        #     "last_unclean_alert_time": timestamp,
        #     "last_detection_bbox": [x1, y1, x2, y2],  # Last detected table bounding box
        #     "last_detection_time": timestamp
        #         }}
        self.table_tracking = {}

        # Initialize GIF recorder for violation snapshots
        self.gif_recorder = AlertGifRecorder(buffer_size=90, gif_duration=3.0, fps=5)
        
        # Track recording state for GIF management
        self._was_recording_alert = False
        self._last_alert_message = None
        self._last_alert_data = None
        
        # Track current violations for frame display (matching script behavior)
        self._current_wrong_uniforms = []
        self._current_unclean_tables = []

        # Load configuration from database
        self.load_configuration()

    def load_configuration(self):
        """Load table ROIs and settings from database"""
        try:
            if not self.db_manager:
                return
            
            # Ensure we have an application context for DB access
            # Otherwise Flask-SQLAlchemy will raise "Working outside of application context"
            if self.app:
                with self.app.app_context():
                    self._load_configuration_from_db()
            else:
                # No app provided; best effort without context (may still work in some setups)
                self._load_configuration_from_db()

            logger.info(f"[{self.channel_id}] Loaded table service configuration: {len(self.table_rois)} tables")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}", exc_info=True)

    def _load_configuration_from_db(self):
        """Internal helper to read ROI + settings from DB (expects app context if required)."""
        # Load table ROIs
        roi_config = self.db_manager.get_channel_config(
            self.channel_id, "TableServiceMonitor", "table_rois"
        )
        if roi_config:
            self.table_rois = roi_config

        # Load settings
        settings = self.db_manager.get_channel_config(
            self.channel_id, "TableServiceMonitor", "settings"
        )
        if settings:
            self.settings.update(settings)

    def set_table_roi(self, table_id, polygon_points):
        """
        Set ROI for a table

        Args:
            table_id: Unique identifier for the table
            polygon_points: List of (x, y) points defining the table polygon (normalized 0-1)
        """
        if not polygon_points or len(polygon_points) < 3:
            logger.warning(f"Invalid polygon for table {table_id}")
            return

        # Store polygon and calculate bounding box
        min_x = min(p[0] for p in polygon_points)
        min_y = min(p[1] for p in polygon_points)
        max_x = max(p[0] for p in polygon_points)
        max_y = max(p[1] for p in polygon_points)

        self.table_rois[table_id] = {
            "polygon": polygon_points,
            "bbox": (min_x, min_y, max_x, max_y)
        }

        # Initialize tracking for this table
        if table_id not in self.table_tracking:
            self.table_tracking[table_id] = {
                "customer_tracks": [],
                "server_tracks": [],
                "last_alert_time": None,
                "cleanliness_state": None,  # "clean", "unclean", or None
                "unclean_start_time": None,
                "last_unclean_alert_time": None
            }

        # Save to database
        if self.db_manager:
            try:
                self.db_manager.set_channel_config(
                    self.channel_id, "TableServiceMonitor", "table_rois", self.table_rois
                )
            except Exception as e:
                logger.error(f"Failed to save table ROI to database: {e}")

        logger.info(f"[{self.channel_id}] Set ROI for table {table_id}")

    def _point_in_polygon(self, point, polygon, bbox):
        """
        Check if a point is inside a polygon (optimized with bounding box check)

        Args:
            point: (x, y) tuple
            polygon: List of (x, y) tuples
            bbox: (min_x, min_y, max_x, max_y) tuple

        Returns:
            bool: True if point is inside polygon
        """
        x, y = point
        min_x, min_y, max_x, max_y = bbox

        # Quick bounding box check
        if x < min_x or x > max_x or y < min_y or y > max_y:
            return False

        # Ray casting algorithm
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

    def _classify_detections(self, detections, frame_shape):
        """
        Classify detections to find clean and unclean tables with bounding boxes
        Only processes Table_clean and Table_unclean classes, ignores people
        If no ROIs are configured, detects all tables in the frame

        Args:
            detections: List of detection dicts from YOLO
            frame_shape: (height, width) of the frame

        Returns:
            dict: {table_id: {"cleanliness": "clean"/"unclean"/None, "bbox": [x1, y1, x2, y2], "confidence": float}}
        """
        h, w = frame_shape[:2]
        table_detections = {}
        
        # If ROIs are configured, initialize with ROI table IDs
        if self.table_rois:
            table_detections = {
                table_id: {
                    "cleanliness": None,
                    "bbox": None,
                    "confidence": 0.0
                } for table_id in self.table_rois.keys()
            }
        else:
            # No ROIs - track tables by their detection index
            # We'll assign IDs based on detection position
            pass

        for det in detections:
            class_name = det.get("class_name", "").lower()
            bbox = det.get("bbox", [])
            confidence = det.get("confidence", 0.0)
            
            if len(bbox) != 4:
                continue

            # Only process table cleanliness detections
            if class_name not in [self.table_clean_class, self.table_unclean_class]:
                continue

            # Apply stricter confidence threshold for unclean detections to reduce false positives
            # Standard table settings (condiment holders, napkin dispensers) should not trigger unclean alerts
            if class_name == self.table_unclean_class:
                if confidence < self.unclean_conf_threshold:
                    logger.debug(
                        f"[{self.channel_id}] Filtering low-confidence unclean detection: "
                        f"conf={confidence:.3f} < threshold={self.unclean_conf_threshold}"
                    )
                    continue  # Skip this unclean detection - confidence too low

            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            point_to_check = (center_x / w, center_y / h)  # Normalize to 0-1

            if self.table_rois:
                # ROIs configured - check which ROI this detection belongs to
                for table_id, roi_info in self.table_rois.items():
                    polygon = roi_info["polygon"]
                    bbox_norm = roi_info["bbox"]

                    if self._point_in_polygon(point_to_check, polygon, bbox_norm):
                        # Track both clean and unclean detections separately
                        current_cleanliness = table_detections[table_id]["cleanliness"]
                        current_confidence = table_detections[table_id]["confidence"]
                        
                        # If we already have a clean detection, prioritize it over unclean
                        # Only update to unclean if:
                        # 1. No clean detection exists, OR
                        # 2. Clean detection has very low confidence (< 0.3) and unclean has high confidence (> 0.7)
                        if class_name == self.table_clean_class:
                            # Always update if clean detection has higher confidence
                            if confidence > current_confidence:
                                table_detections[table_id]["cleanliness"] = "clean"
                                table_detections[table_id]["bbox"] = bbox
                                table_detections[table_id]["confidence"] = confidence
                        elif class_name == self.table_unclean_class:
                            # Stricter requirements for unclean detection to reduce false positives:
                            # 1. Must meet minimum confidence threshold (already filtered above)
                            # 2. If clean detection exists, require significant confidence gap
                            # 3. If no clean detection, still require high confidence
                            
                            if current_cleanliness != "clean":
                                # No clean detection - require high confidence for unclean (already filtered to >= unclean_conf_threshold)
                                # Also require confidence to be higher than any existing unclean detection
                                if confidence > current_confidence:
                                    table_detections[table_id]["cleanliness"] = "unclean"
                                    table_detections[table_id]["bbox"] = bbox
                                    table_detections[table_id]["confidence"] = confidence
                                    logger.debug(
                                        f"[{self.channel_id}] Table {table_id}: Unclean detection (conf={confidence:.3f}, "
                                        f"threshold={self.unclean_conf_threshold})"
                                    )
                            else:
                                # Clean detection exists - require very high confidence gap to override
                                # Unclean must be at least 0.4 higher than clean, AND clean must be < 0.4
                                confidence_gap = confidence - current_confidence
                                min_gap_required = 0.4  # Require 40% confidence gap
                                
                                if current_confidence < 0.4 and confidence_gap >= min_gap_required and confidence > 0.75:
                                    logger.warning(
                                        f"[{self.channel_id}] Table {table_id}: Overriding clean detection "
                                        f"(conf={current_confidence:.3f}) with unclean (conf={confidence:.3f}, "
                                        f"gap={confidence_gap:.3f})"
                                    )
                                    table_detections[table_id]["cleanliness"] = "unclean"
                                    table_detections[table_id]["bbox"] = bbox
                                    table_detections[table_id]["confidence"] = confidence
                                else:
                                    # Clean detection is reliable - keep it, ignore unclean
                                    logger.debug(
                                        f"[{self.channel_id}] Table {table_id}: Ignoring unclean detection "
                                        f"(conf={confidence:.3f}) because clean detection exists (conf={current_confidence:.3f}, "
                                        f"gap={confidence_gap:.3f} < required={min_gap_required})"
                                    )
                        break  # Detection can only be in one table
            else:
                # No ROIs - track each detection as a separate table
                # Use center position to create a unique table ID
                table_id = f"table_{int(center_x)}_{int(center_y)}"
                
                # Match to existing table if bbox overlaps significantly
                matched_table_id = None
                for existing_id, existing_data in table_detections.items():
                    if existing_data["bbox"] is not None:
                        ex1, ey1, ex2, ey2 = existing_data["bbox"]
                        # Check if bounding boxes overlap significantly
                        overlap_x = max(0, min(x2, ex2) - max(x1, ex1))
                        overlap_y = max(0, min(y2, ey2) - max(y1, ey1))
                        overlap_area = overlap_x * overlap_y
                        det_area = (x2 - x1) * (y2 - y1)
                        if overlap_area > 0.5 * det_area:  # 50% overlap threshold
                            matched_table_id = existing_id
                            break
                
                if matched_table_id:
                    table_id = matched_table_id
                else:
                    # New table detection
                    table_detections[table_id] = {
                        "cleanliness": None,
                        "bbox": None,
                        "confidence": 0.0
                    }
                
                # Track both clean and unclean detections separately
                current_cleanliness = table_detections[table_id]["cleanliness"]
                current_confidence = table_detections[table_id]["confidence"]
                
                # If we already have a clean detection, prioritize it over unclean
                # Only update to unclean if:
                # 1. No clean detection exists, OR
                # 2. Clean detection has very low confidence (< 0.3) and unclean has high confidence (> 0.7)
                if class_name == self.table_clean_class:
                    # Always update if clean detection has higher confidence
                    if confidence > current_confidence:
                        table_detections[table_id]["cleanliness"] = "clean"
                        table_detections[table_id]["bbox"] = bbox
                        table_detections[table_id]["confidence"] = confidence
                elif class_name == self.table_unclean_class:
                    # Stricter requirements for unclean detection to reduce false positives:
                    # 1. Must meet minimum confidence threshold (already filtered above)
                    # 2. If clean detection exists, require significant confidence gap
                    # 3. If no clean detection, still require high confidence
                    
                    if current_cleanliness != "clean":
                        # No clean detection - require high confidence for unclean (already filtered to >= unclean_conf_threshold)
                        # Also require confidence to be higher than any existing unclean detection
                        if confidence > current_confidence:
                            table_detections[table_id]["cleanliness"] = "unclean"
                            table_detections[table_id]["bbox"] = bbox
                            table_detections[table_id]["confidence"] = confidence
                            logger.debug(
                                f"[{self.channel_id}] Table {table_id}: Unclean detection (conf={confidence:.3f}, "
                                f"threshold={self.unclean_conf_threshold})"
                            )
                    else:
                        # Clean detection exists - require very high confidence gap to override
                        # Unclean must be at least 0.4 higher than clean, AND clean must be < 0.4
                        confidence_gap = confidence - current_confidence
                        min_gap_required = 0.4  # Require 40% confidence gap
                        
                        if current_confidence < 0.4 and confidence_gap >= min_gap_required and confidence > 0.75:
                            logger.warning(
                                f"[{self.channel_id}] Table {table_id}: Overriding clean detection "
                                f"(conf={current_confidence:.3f}) with unclean (conf={confidence:.3f}, "
                                f"gap={confidence_gap:.3f})"
                            )
                            table_detections[table_id]["cleanliness"] = "unclean"
                            table_detections[table_id]["bbox"] = bbox
                            table_detections[table_id]["confidence"] = confidence
                        else:
                            # Clean detection is reliable - keep it, ignore unclean
                            logger.debug(
                                f"[{self.channel_id}] Table {table_id}: Ignoring unclean detection "
                                f"(conf={confidence:.3f}) because clean detection exists (conf={current_confidence:.3f}, "
                                f"gap={confidence_gap:.3f} < required={min_gap_required})"
                            )

        return table_detections

    def _distance(self, p1, p2):
        """Calculate Euclidean distance between two points"""
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])
    
    def _check_person_near_table(self, table_center, persons, table_roi=None):
        """
        Check if any person is near the table or within the table ROI
        
        Args:
            table_center: (x, y) tuple of table center
            persons: List of (x, y) tuples of person centers
            table_roi: Optional table ROI polygon for more accurate checking
            
        Returns:
            bool: True if any person is within PERSON_NEAR_DISTANCE of table center or within ROI
        """
        if not persons or len(persons) == 0:
            return False
        
        for person_center in persons:
            # Check distance from table center
            distance = self._distance(table_center, person_center)
            if distance < self.PERSON_NEAR_DISTANCE:
                return True
            
            # If ROI is provided, also check if person is within the table ROI polygon
            if table_roi is not None:
                polygon = table_roi.get("polygon")
                bbox_norm = table_roi.get("bbox")
                if polygon and bbox_norm:
                    # Normalize person center to 0-1 range (assuming frame dimensions)
                    # Note: This requires frame dimensions, but we'll use a simpler check
                    # Check if person is within expanded bounding box
                    min_x, min_y, max_x, max_y = bbox_norm
                    # Expand bbox by 10% for better detection
                    expand = 0.1
                    expanded_min_x = max(0, min_x - expand)
                    expanded_min_y = max(0, min_y - expand)
                    expanded_max_x = min(1, max_x + expand)
                    expanded_max_y = min(1, max_y + expand)
                    
                    # Check if person center (normalized) is within expanded bbox
                    # Note: person_center is in pixels, we need frame dimensions to normalize
                    # For now, rely on distance check which should work for most cases
                    pass
        
        return False

    def _update_table_tracking(self, table_id, cleanliness, bbox, confidence, current_time, frame=None, persons=None):
        """
        Update tracking for a table and check for violations (cleanliness only)
        Matches simple script logic: only alert if table is unclean AND no person is near

        Args:
            table_id: Table identifier
            cleanliness: "clean", "unclean", or None
            bbox: Bounding box of detected table [x1, y1, x2, y2] or None
            confidence: Detection confidence (0.0-1.0)
            current_time: Current timestamp
            frame: Optional frame for snapshot saving
            persons: List of (x, y) tuples of person centers (for proximity check)
        """
        if table_id not in self.table_tracking:
            self.table_tracking[table_id] = {
                "cleanliness_state": None,
                "unclean_start_time": None,
                "last_unclean_alert_time": None,
                "last_detection_bbox": None,
                "last_detection_time": None,
                "center": None  # Table center for proximity checking
            }

        tracking = self.table_tracking[table_id]
        now_ts = current_time.timestamp()

        # Update detection info
        if bbox is not None:
            tracking["last_detection_bbox"] = bbox
            tracking["last_detection_time"] = now_ts
            # Calculate table center
            x1, y1, x2, y2 = bbox
            tracking["center"] = ((x1 + x2) / 2, (y1 + y2) / 2)

        # Get table center for proximity check
        table_center = tracking.get("center")
        if table_center is None and bbox is not None:
            x1, y1, x2, y2 = bbox
            table_center = ((x1 + x2) / 2, (y1 + y2) / 2)

        # Check person proximity FIRST (matching simple script logic)
        person_near = False
        if persons is not None and len(persons) > 0 and table_center is not None:
            # Get table ROI if available for more accurate person detection
            table_roi = self.table_rois.get(table_id) if table_id in self.table_rois else None
            person_near = self._check_person_near_table(table_center, persons, table_roi)
            if person_near:
                logger.debug(f"[{self.channel_id}] Table {table_id}: Person detected near table ({len(persons)} persons detected, distance threshold={self.PERSON_NEAR_DISTANCE}px)")
        elif persons is None or len(persons) == 0:
            # No person detections available - assume no person is near
            person_near = False

        # Update cleanliness state
        if cleanliness is not None:
            if cleanliness == "unclean":
                # If person is present, don't mark as unclean or reset any unclean state
                if person_near:
                    # ✅ PERSON PRESENT → NO ALERT, NO UNclean STATE (suppress completely)
                    if tracking["cleanliness_state"] == "unclean":
                        # Person arrived - reset unclean state
                        tracking["cleanliness_state"] = None  # Reset to unknown/neutral
                        tracking["unclean_start_time"] = None
                        logger.info(f"[{self.channel_id}] Table {table_id}: Person present - suppressing unclean detection")
                    # Don't set unclean state if person is present
                    return  # Exit early - no processing needed when person is present
                
                # No person present - proceed with unclean detection
                if tracking["cleanliness_state"] != "unclean":
                    # State changed to unclean
                    tracking["cleanliness_state"] = "unclean"
                    tracking["unclean_start_time"] = now_ts
                    logger.debug(f"[{self.channel_id}] Table {table_id}: State changed to unclean (no person present)")
                else:
                    # Already unclean - check if unclean for long enough to trigger alert
                    if tracking["unclean_start_time"] is not None:
                        unclean_duration = now_ts - tracking["unclean_start_time"]
                        threshold = self.settings.get("unclean_duration_threshold", 0.0)
                        if unclean_duration >= threshold:
                            self._check_unclean_violation(table_id, tracking, unclean_duration, current_time, frame)
                        
            elif cleanliness == "clean":
                # State changed to clean - reset tracking
                if tracking["cleanliness_state"] == "unclean":
                    tracking["cleanliness_state"] = "clean"
                    tracking["unclean_start_time"] = None
                    logger.debug(f"[{self.channel_id}] Table {table_id}: State changed to clean")
            # If cleanliness is None, keep current state (no detection in this frame)
        
        # Final safety check: If person is present but table is still marked unclean, reset it
        # This handles cases where person detection happens after unclean state is set
        if person_near and tracking.get("cleanliness_state") == "unclean":
            tracking["cleanliness_state"] = None
            tracking["unclean_start_time"] = None
            tracking["last_unclean_alert_time"] = None  # Reset alert cooldown
            logger.info(f"[{self.channel_id}] Table {table_id}: Person detected - resetting unclean state (safety check)")

    def _check_unclean_violation(self, table_id, tracking, unclean_duration, current_time, frame=None):
        """
        Check for unclean table violations and generate alerts
        IMPORTANT: This should only be called when NO person is present at the table

        Args:
            table_id: Table identifier
            tracking: Tracking data for this table
            unclean_duration: How long the table has been unclean (seconds)
            current_time: Current timestamp
            frame: Optional frame for snapshot saving
        """
        # Safety check: Verify no person is present before alerting
        # This is a final safeguard in case person detection wasn't passed correctly
        table_center = tracking.get("center")
        if table_center is None:
            logger.warning(f"[{self.channel_id}] Table {table_id}: Cannot verify person presence - no table center, skipping alert")
            return
        
        # Note: We can't check persons here since they're not passed to this method
        # The main check happens in _update_table_tracking before this is called
        # This is just a safety check for edge cases
        
        now_ts = current_time.timestamp()
        cooldown = self.settings["unclean_alert_cooldown"]
        last_alert = tracking.get("last_unclean_alert_time")

        # Check cooldown
        if last_alert is not None and (now_ts - last_alert) < cooldown:
            return  # Still in cooldown

        # Table is unclean - trigger violation
        logger.warning(
            f"[{self.channel_id}] Table {table_id} unclean violation: "
            f"Table has been unclean for {unclean_duration:.1f}s"
        )

        # Save snapshot
        snapshot_path = self._save_unclean_snapshot(table_id, unclean_duration, current_time, frame)

        # Emit socket event
        alert_message = f"Unclean table {table_id} detected (unclean for {unclean_duration:.1f}s)"
        if self.socketio:
            self.socketio.emit("table_unclean_alert", {
                "channel_id": self.channel_id,
                "table_id": table_id,
                "unclean_duration": round(unclean_duration, 1),
                "timestamp": current_time.isoformat(),
                "snapshot_path": snapshot_path,
                "message": alert_message
            })

        # Save to database (NEW: table_cleanliness_violations table)
        if self.db_manager:
            try:
                payload = {
                    "violation_type": "unclean_table",
                    "unclean_duration": unclean_duration,
                    "message": alert_message,
                }

                if self.app:
                    with self.app.app_context():
                        self.db_manager.add_table_cleanliness_violation(
                            channel_id=self.channel_id,
                            table_id=table_id,
                            violation_type="unclean_table",
                            snapshot_path=snapshot_path,
                            timestamp=current_time,
                            alert_data=payload,
                        )
                        self.db_manager.log_alert(
                            self.channel_id,
                            "table_cleanliness_alert",
                            alert_message,
                            alert_data={
                                "violation_type": "unclean_table",
                                "table_id": table_id,
                                "unclean_duration": unclean_duration,
                            },
                        )
                else:
                    self.db_manager.add_table_cleanliness_violation(
                        channel_id=self.channel_id,
                        table_id=table_id,
                        violation_type="unclean_table",
                        snapshot_path=snapshot_path,
                        timestamp=current_time,
                        alert_data=payload,
                    )
                    self.db_manager.log_alert(
                        self.channel_id,
                        "table_cleanliness_alert",
                        alert_message,
                        alert_data={
                            "violation_type": "unclean_table",
                            "table_id": table_id,
                            "unclean_duration": unclean_duration,
                        },
                    )

                logger.info(f"[{self.channel_id}] ✅ Table cleanliness saved: unclean_table for {table_id}")
            except Exception as e:
                logger.error(f"Failed to save table cleanliness (unclean_table): {e}", exc_info=True)

        tracking["last_unclean_alert_time"] = now_ts
        self.total_alerts += 1

    def _save_unclean_snapshot(self, table_id, unclean_duration, current_time, frame=None):
        """
        Save a snapshot of the unclean table violation

        Args:
            table_id: Table identifier
            unclean_duration: Duration table has been unclean
            current_time: Current timestamp
            frame: Optional frame to save

        Returns:
            str: Path to saved snapshot (relative to static/)
        """
        try:
            snapshot_dir = Path("static/table_service_violations")
            snapshot_dir.mkdir(parents=True, exist_ok=True)

            timestamp_str = current_time.strftime("%Y%m%d_%H%M%S")
            filename = f"unclean_table_{table_id}_{self.channel_id}_{timestamp_str}.jpg"
            snapshot_path = snapshot_dir / filename

            # Save frame if provided
            if frame is not None:
                annotated = frame.copy()
                
                # Draw bounding box if available
                tracking = self.table_tracking.get(table_id, {})
                bbox = tracking.get("last_detection_bbox")
                if bbox is not None and len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
                    cv2.putText(annotated, "UNCLEAN TABLE", (int(x1), int(y1) - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Draw text annotation
                cv2.putText(annotated, f"Table {table_id}: UNCLEAN ({unclean_duration:.1f}s)",
                           (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                cv2.putText(annotated, "VIOLATION: Unclean table detected",
                           (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                cv2.imwrite(str(snapshot_path), annotated)
                logger.info(f"Unclean table violation snapshot saved: {snapshot_path}")

            return str(snapshot_path.relative_to("static"))
        except Exception as e:
            logger.error(f"Failed to save unclean snapshot: {e}")
            return None

    def _check_slow_reset_violation(self, table_id, tracking, reset_duration, current_time, frame=None):
        """
        Check for slow reset violations (table not cleaned within threshold after customers leave)

        Args:
            table_id: Table identifier
            tracking: Tracking data for this table
            reset_duration: How long table has been unclean since customers left (seconds)
            current_time: Current timestamp
            frame: Optional frame for snapshot saving
        """
        now_ts = current_time.timestamp()
        cooldown = self.settings["slow_reset_alert_cooldown"]
        last_alert = tracking.get("last_slow_reset_alert_time")

        # Check cooldown
        if last_alert is not None and (now_ts - last_alert) < cooldown:
            return  # Still in cooldown

        # Check if customers are back (reset no longer needed)
        active_customers = [
            track for track in tracking.get("customer_tracks", [])
            if (now_ts - track.get("last_seen", 0)) < self.settings["track_timeout"]
        ]
        
        if len(active_customers) > 0:
            # Customers are back - reset timer should be cleared
            tracking["reset_start_time"] = None
            return

        # Slow reset violation: table still unclean after threshold time
        logger.warning(
            f"[{self.channel_id}] Table {table_id} slow reset violation: "
            f"Table not cleaned within {reset_duration:.1f}s after customers left"
        )

        # Save snapshot
        snapshot_path = self._save_slow_reset_snapshot(table_id, reset_duration, current_time, frame)

        # Emit socket event
        alert_message = f"Slow reset: Table {table_id} not cleaned within {reset_duration:.1f}s after customers left"
        if self.socketio:
            self.socketio.emit("table_slow_reset_alert", {
                "channel_id": self.channel_id,
                "table_id": table_id,
                "reset_duration": round(reset_duration, 1),
                "timestamp": current_time.isoformat(),
                "snapshot_path": snapshot_path,
                "message": alert_message
            })

        # Save to database (NEW: table_cleanliness_violations table)
        if self.db_manager:
            try:
                payload = {
                    "violation_type": "slow_reset",
                    "reset_duration": reset_duration,
                    "message": alert_message,
                }
                if self.app:
                    with self.app.app_context():
                        self.db_manager.add_table_cleanliness_violation(
                            channel_id=self.channel_id,
                            table_id=table_id,
                            violation_type="slow_reset",
                            snapshot_path=snapshot_path,
                            timestamp=current_time,
                            alert_data=payload,
                        )
                        self.db_manager.log_alert(
                            self.channel_id,
                            "table_cleanliness_alert",
                            alert_message,
                            alert_data={
                                "violation_type": "slow_reset",
                                "table_id": table_id,
                                "reset_duration": reset_duration,
                            },
                        )
                else:
                    self.db_manager.add_table_cleanliness_violation(
                        channel_id=self.channel_id,
                        table_id=table_id,
                        violation_type="slow_reset",
                        snapshot_path=snapshot_path,
                        timestamp=current_time,
                        alert_data=payload,
                    )
                    self.db_manager.log_alert(
                        self.channel_id,
                        "table_cleanliness_alert",
                        alert_message,
                        alert_data={
                            "violation_type": "slow_reset",
                            "table_id": table_id,
                            "reset_duration": reset_duration,
                        },
                    )
                logger.info(f"[{self.channel_id}] ✅ Table cleanliness saved: slow_reset for {table_id}")
            except Exception as e:
                logger.error(f"Failed to save table cleanliness (slow_reset): {e}", exc_info=True)

        tracking["last_slow_reset_alert_time"] = now_ts

    def _save_slow_reset_snapshot(self, table_id, reset_duration, current_time, frame=None):
        """
        Save a snapshot of the slow reset violation

        Args:
            table_id: Table identifier
            reset_duration: Duration table has been unclean since customers left
            current_time: Current timestamp
            frame: Optional frame to save

        Returns:
            str: Path to saved snapshot (relative to static/)
        """
        try:
            snapshot_dir = Path("static/table_service_violations")
            snapshot_dir.mkdir(parents=True, exist_ok=True)

            timestamp_str = current_time.strftime("%Y%m%d_%H%M%S")
            filename = f"slow_reset_{table_id}_{self.channel_id}_{timestamp_str}.jpg"
            snapshot_path = snapshot_dir / filename

            # Save frame if provided
            if frame is not None:
                annotated = frame.copy()
                # Draw text annotation
                cv2.putText(annotated, f"Table {table_id}: SLOW RESET ({reset_duration:.1f}s)",
                           (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)
                cv2.putText(annotated, "VIOLATION: Table not cleaned after customers left",
                           (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

                cv2.imwrite(str(snapshot_path), annotated)
                logger.info(f"Slow reset violation snapshot saved: {snapshot_path}")

            return str(snapshot_path.relative_to("static"))
        except Exception as e:
            logger.error(f"Failed to save slow reset snapshot: {e}")
            return None

    def _trigger_violation_alert(self, table_id, customer_track, waiting_time, current_time, frame=None):
        """
        DEPRECATED: This method is no longer used (wait time tracking removed)
        Kept for compatibility but should not be called

        Args:
            table_id: Table identifier
            customer_track: Customer track data
            waiting_time: Waiting time in seconds
            current_time: Current timestamp
            frame: Optional frame for snapshot saving
        """
        logger.warning(
            f"[{self.channel_id}] Table {table_id} violation: Customer waiting {waiting_time:.1f}s "
            f"(threshold: {self.settings['wait_time_threshold']}s)"
        )

        # Save snapshot
        snapshot_path = self._save_violation_snapshot(table_id, customer_track, waiting_time, current_time, frame)

        # Emit socket event
        if self.socketio:
            self.socketio.emit("table_service_alert", {
                "channel_id": self.channel_id,
                "table_id": table_id,
                "waiting_time": round(waiting_time, 1),
                "threshold": self.settings["wait_time_threshold"],
                "timestamp": current_time.isoformat(),
                "snapshot_path": snapshot_path
            })

        # Save to database
        if self.db_manager:
            try:
                self.db_manager.add_table_service_violation(
                    channel_id=self.channel_id,
                    table_id=table_id,
                    waiting_time=waiting_time,
                    snapshot_path=snapshot_path,
                    timestamp=current_time
                )
            except Exception as e:
                logger.error(f"Failed to save table service violation to database: {e}")

    def _save_violation_snapshot(self, table_id, customer_track, waiting_time, current_time, frame=None):
        """
        Save a snapshot of the violation

        Args:
            table_id: Table identifier
            customer_track: Customer track data
            waiting_time: Waiting time in seconds
            current_time: Current timestamp
            frame: Optional frame to save (if None, just return path)

        Returns:
            str: Path to saved snapshot (relative to static/)
        """
        try:
            snapshot_dir = Path("static/table_service_violations")
            snapshot_dir.mkdir(parents=True, exist_ok=True)

            timestamp_str = current_time.strftime("%Y%m%d_%H%M%S")
            filename = f"table_{table_id}_{self.channel_id}_{timestamp_str}.jpg"
            snapshot_path = snapshot_dir / filename

            # Save frame if provided
            if frame is not None:
                # Draw annotation on frame
                annotated = frame.copy()
                center = customer_track.get("center", [0, 0])
                cv2.circle(annotated, (int(center[0]), int(center[1])), 20, (0, 0, 255), -1)
                cv2.putText(annotated, f"Table {table_id}: {waiting_time:.1f}s", 
                           (int(center[0]) - 100, int(center[1]) - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.imwrite(str(snapshot_path), annotated)
                logger.info(f"Table service violation snapshot saved: {snapshot_path}")

            return str(snapshot_path.relative_to("static"))
        except Exception as e:
            logger.error(f"Failed to save snapshot: {e}")
            return None

    def process_frame(self, frame):
        """
        Process a video frame

        Args:
            frame: BGR frame (numpy array)

        Returns:
            numpy.ndarray: Annotated frame
        """
        self.frame_count += 1
        current_time = datetime.now()

        if frame is None or frame.size == 0:
            return frame

        h, w = frame.shape[:2]

        # Performance optimization: Skip YOLO inference on some frames
        # Only run expensive YOLO inference every Nth frame
        should_run_detection = (self.frame_count % self.process_every_n_frames == 0) or (self.frame_count == 1)
        
        if should_run_detection:
            # Run YOLO detection (expensive operation)
            try:
                results = self.model(frame, conf=self.conf_threshold, iou=self.nms_iou, verbose=False)
                detections = []

                if len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes
                    class_names = results[0].names

                    for box in boxes:
                        class_id = int(box.cls[0])
                        class_name = class_names[class_id]
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                        detections.append({
                            "class_id": class_id,
                            "class_name": class_name,
                            "confidence": conf,
                            "bbox": [int(x1), int(y1), int(x2), int(y2)]
                        })

                # Cache detections for skipped frames
                self.last_detections = detections
                
                # Check for uniform violations (matching provided code behavior)
                self._check_uniform_violations(detections, current_time, frame)
                
                # Update current violations for frame display (matching script)
                self._update_current_violations(detections)
                
                # Classify detections per table (only clean/unclean tables)
                table_detections = self._classify_detections(detections, frame.shape)
                
                # Cache table detections
                self.last_table_detections = table_detections
                
                # Run person detection for proximity checks
                try:
                    person_detections = self.person_detector.detect_persons(frame)
                    persons_centers = [d['center'] for d in person_detections]
                    self.last_persons = persons_centers
                except Exception as e:
                    logger.error(f"Error in person detection: {e}")
                    persons_centers = self.last_persons if hasattr(self, 'last_persons') else []

                # Update tracking for each detected table
                for table_id in table_detections.keys():
                    cleanliness = table_detections[table_id]["cleanliness"]
                    bbox = table_detections[table_id]["bbox"]
                    confidence = table_detections[table_id]["confidence"]
                    self._update_table_tracking(table_id, cleanliness, bbox, confidence, current_time, frame, persons=persons_centers)
                
            except Exception as e:
                logger.error(f"Error processing frame: {e}")
                # On error, use cached detections
                detections = self.last_detections
                table_detections = self.last_table_detections
                persons = self.last_persons if hasattr(self, 'last_persons') else []
                
                # Still update tracking with cached data (but won't alert on cached data)
                for table_id in table_detections.keys():
                    cleanliness = table_detections[table_id]["cleanliness"]
                    bbox = table_detections[table_id]["bbox"]
                    confidence = table_detections[table_id]["confidence"]
                    self._update_table_tracking(table_id, cleanliness, bbox, confidence, current_time, frame, persons=persons)
        else:
            # Use cached detections from last processed frame
            detections = self.last_detections
            table_detections = self.last_table_detections
            persons = self.last_persons if hasattr(self, 'last_persons') else []
            
            # Still update violations display (lightweight operation)
            self._update_current_violations(detections)
            
            # Update tracking with cached data (lightweight - won't trigger new alerts)
            for table_id in table_detections.keys():
                cleanliness = table_detections[table_id]["cleanliness"]
                bbox = table_detections[table_id]["bbox"]
                confidence = table_detections[table_id]["confidence"]
                self._update_table_tracking(table_id, cleanliness, bbox, confidence, current_time, frame, persons=persons)
        
        # Update unclean tables list for display (always update for visualization)
        self._current_unclean_tables = [
            table_id for table_id, data in table_detections.items()
            if data.get("cleanliness") == "unclean"
        ]

        # Draw annotations (always draw for smooth visualization)
        annotated_frame = self._draw_annotations(frame, table_detections, current_time)

        return annotated_frame

    def _draw_annotations(self, frame, table_detections, current_time):
        """
        Draw annotations on the frame - table bounding boxes and cleanliness status
        Also displays alerts matching script behavior:
        - "⚠ WRONG UNIFORM DETECTED!" when wrong uniform found
        - "⚠ TABLE UNCLEAN!" when table_unclean detected

        Args:
            frame: BGR frame
            table_detections: Detection data per table
            current_time: Current timestamp

        Returns:
            numpy.ndarray: Annotated frame
        """
        h, w = frame.shape[:2]
        annotated = frame.copy()
        
        # Draw alerts matching script behavior
        y_offset = 0
        
        # 1. Wrong uniform alert (matching script)
        if self._current_wrong_uniforms:
            alert_text = "⚠ WRONG UNIFORM DETECTED!"
            cv2.putText(annotated, alert_text, (20, 40 + y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
            y_offset += 50
        
        # 2. Table unclean alert (matching script)
        if self._current_unclean_tables:
            alert_text = "⚠ TABLE UNCLEAN!"
            cv2.putText(annotated, alert_text, (20, 40 + y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
            y_offset += 50

        # Draw table ROIs if configured
        if self.table_rois:
            for table_id, roi_info in self.table_rois.items():
                polygon = roi_info["polygon"]
                # Convert normalized coordinates to pixel coordinates
                polygon_pixels = [(int(p[0] * w), int(p[1] * h)) for p in polygon]

                # Get cleanliness status
                cleanliness = table_detections.get(table_id, {}).get("cleanliness")
                bbox = table_detections.get(table_id, {}).get("bbox")
                confidence = table_detections.get(table_id, {}).get("confidence", 0.0)
                
                # Determine color and status text
                if cleanliness == "unclean":
                    roi_color = (0, 0, 255)  # Red for unclean
                    status_text = "UNCLEAN"
                elif cleanliness == "clean":
                    roi_color = (0, 255, 0)  # Green for clean
                    status_text = "CLEAN"
                else:
                    roi_color = (0, 255, 255)  # Yellow for unknown
                    status_text = "UNKNOWN"
                
                # Draw table ROI polygon
                cv2.polylines(annotated, [np.array(polygon_pixels, np.int32)], True, roi_color, 2)
                cv2.putText(annotated, f"Table {table_id} - {status_text}", 
                           (polygon_pixels[0][0], polygon_pixels[0][1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, roi_color, 2)

                # Draw detected table bounding box if available
                if bbox is not None and len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    # Draw bounding box
                    cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), roi_color, 3)
                    # Draw label with confidence
                    label = f"{status_text} ({confidence:.2f})"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(annotated, (int(x1), int(y1) - label_size[1] - 10), 
                                 (int(x1) + label_size[0], int(y1)), roi_color, -1)
                    cv2.putText(annotated, label, (int(x1), int(y1) - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            # No ROIs configured - draw all detected tables with bounding boxes
            for table_id, detection_data in table_detections.items():
                cleanliness = detection_data.get("cleanliness")
                bbox = detection_data.get("bbox")
                confidence = detection_data.get("confidence", 0.0)
                
                if bbox is None or len(bbox) != 4:
                    continue
                
                # Determine color and status text
                if cleanliness == "unclean":
                    bbox_color = (0, 0, 255)  # Red for unclean
                    status_text = "UNCLEAN"
                elif cleanliness == "clean":
                    bbox_color = (0, 255, 0)  # Green for clean
                    status_text = "CLEAN"
                else:
                    bbox_color = (0, 255, 255)  # Yellow for unknown
                    status_text = "UNKNOWN"
                
                x1, y1, x2, y2 = bbox
                # Draw bounding box
                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), bbox_color, 3)
                
                # Draw label with table ID, status, and confidence
                label = f"Table {table_id} - {status_text} ({confidence:.2f})"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(annotated, (int(x1), int(y1) - label_size[1] - 10), 
                             (int(x1) + label_size[0], int(y1)), bbox_color, -1)
                cv2.putText(annotated, label, (int(x1), int(y1) - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Count clean and unclean tables
        clean_count = sum(1 for data in table_detections.values() if data.get("cleanliness") == "clean")
        unclean_count = sum(1 for data in table_detections.values() if data.get("cleanliness") == "unclean")
        
        # Draw footer with table counts at the bottom of the frame
        footer_y = h - 20
        footer_text = f"Clean Tables: {clean_count} | Unclean Tables: {unclean_count}"
        text_size, _ = cv2.getTextSize(footer_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        
        # Draw semi-transparent background for footer
        footer_bg_height = text_size[1] + 20
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, footer_y - footer_bg_height), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)
        
        # Draw footer text
        cv2.putText(annotated, footer_text, (10, footer_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return annotated

    def _update_current_violations(self, detections):
        """
        Update current violations for frame display (matching script behavior)
        This tracks violations in real-time for display, separate from alert triggering
        
        Args:
            detections: List of detection dicts from YOLO
        """
        # Get all detected uniform classes (normalize to lowercase with underscores)
        detected_uniforms = []
        for det in detections:
            class_name = det.get("class_name", "")
            class_clean = class_name.lower().replace(" ", "_")
            
            # Check if it's a uniform class
            if class_clean.startswith("uniform_"):
                detected_uniforms.append(class_clean)
        
        # Check for wrong uniforms (matching script logic)
        wrong_uniforms = []
        for uni in detected_uniforms:
            if uni not in self.allowed_uniforms:
                wrong_uniforms.append(uni)
        
        self._current_wrong_uniforms = wrong_uniforms
    
    def _check_uniform_violations(self, detections, current_time, frame=None):
        """
        Check for wrong uniform violations (matching script logic)
        
        Logic from script:
        - Detect all uniforms (classes starting with "uniform_")
        - If any uniform is NOT in ALLOWED_UNIFORMS, trigger alert
        
        Args:
            detections: List of detection dicts from YOLO
            current_time: Current timestamp
            frame: Optional frame for snapshot saving
        """
        now_ts = current_time.timestamp()
        
        # Check cooldown
        if self.last_uniform_alert_time is not None:
            if (now_ts - self.last_uniform_alert_time) < self.uniform_alert_cooldown:
                return  # Still in cooldown
        
        # Get wrong uniforms from current violations (already computed)
        wrong_uniforms = self._current_wrong_uniforms
        
        # Trigger alert if wrong uniform detected
        if wrong_uniforms:
            self._trigger_uniform_violation_alert(wrong_uniforms, current_time, frame)
            self.last_uniform_alert_time = now_ts
    
    def _trigger_uniform_violation_alert(self, wrong_uniforms, current_time, frame=None):
        """
        Trigger alert for wrong uniform detection
        
        Args:
            wrong_uniforms: List of wrong uniform class names detected
            current_time: Current timestamp
            frame: Optional frame for snapshot saving
        """
        uniform_names = ", ".join([u.upper() for u in wrong_uniforms])
        alert_message = f"⚠ WRONG UNIFORM DETECTED: {uniform_names}"
        
        logger.warning(
            f"[{self.channel_id}] Wrong uniform violation: {uniform_names}"
        )
        
        # Save snapshot
        snapshot_path = self._save_uniform_violation_snapshot(wrong_uniforms, current_time, frame)
        
        # Emit socket event
        if self.socketio:
            self.socketio.emit("table_service_uniform_alert", {
                "channel_id": self.channel_id,
                "wrong_uniforms": wrong_uniforms,
                "timestamp": current_time.isoformat(),
                "snapshot_path": snapshot_path,
                "message": alert_message
            })
        
        # Save to database
        if self.db_manager:
            try:
                if self.app:
                    with self.app.app_context():
                        self.db_manager.add_table_service_violation(
                            channel_id=self.channel_id,
                            table_id="N/A",  # Uniform violation is not table-specific
                            waiting_time=0.0,  # Uniform violation doesn't have waiting time
                            snapshot_path=snapshot_path,
                            timestamp=current_time,
                            alert_data={
                                "violation_type": "wrong_uniform",
                                "wrong_uniforms": wrong_uniforms,
                                "message": alert_message
                            }
                        )
                        # Also log to general alerts table
                        self.db_manager.log_alert(
                            self.channel_id,
                            'table_service_alert',
                            alert_message,
                            alert_data={
                                "violation_type": "wrong_uniform",
                                "wrong_uniforms": wrong_uniforms
                            }
                        )
                        logger.info(f"[{self.channel_id}] ✅ Uniform violation alert saved to database: {uniform_names}")
                else:
                    self.db_manager.add_table_service_violation(
                        channel_id=self.channel_id,
                        table_id="N/A",  # Uniform violation is not table-specific
                        waiting_time=0.0,  # Uniform violation doesn't have waiting time
                        snapshot_path=snapshot_path,
                        timestamp=current_time,
                        alert_data={
                            "violation_type": "wrong_uniform",
                            "wrong_uniforms": wrong_uniforms,
                            "message": alert_message
                        }
                    )
                    # Also log to general alerts table
                    self.db_manager.log_alert(
                        self.channel_id,
                        'table_service_alert',
                        alert_message,
                        alert_data={
                            "violation_type": "wrong_uniform",
                            "wrong_uniforms": wrong_uniforms
                        }
                    )
                    logger.info(f"[{self.channel_id}] ✅ Uniform violation alert saved to database: {uniform_names}")
            except Exception as e:
                logger.error(f"Failed to save uniform violation to database: {e}", exc_info=True)
        
        self.total_alerts += 1
    
    def _save_uniform_violation_snapshot(self, wrong_uniforms, current_time, frame=None):
        """
        Save a snapshot of the wrong uniform violation
        
        Args:
            wrong_uniforms: List of wrong uniform class names
            current_time: Current timestamp
            frame: Optional frame to save
            
        Returns:
            str: Path to saved snapshot (relative to static/)
        """
        try:
            snapshot_dir = Path("static/table_service_violations")
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp_str = current_time.strftime("%Y%m%d_%H%M%S")
            uniform_str = "_".join(wrong_uniforms)
            filename = f"wrong_uniform_{self.channel_id}_{uniform_str}_{timestamp_str}.jpg"
            snapshot_path = snapshot_dir / filename
            
            # Save frame if provided
            if frame is not None:
                annotated = frame.copy()
                
                # Draw alert text (matching script style)
                alert_text = f"⚠ WRONG UNIFORM DETECTED!"
                cv2.putText(annotated, alert_text, (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
                
                # Add uniform names
                uniform_names = ", ".join([u.upper() for u in wrong_uniforms])
                cv2.putText(annotated, uniform_names, (20, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                
                cv2.imwrite(str(snapshot_path), annotated)
                logger.info(f"Wrong uniform violation snapshot saved: {snapshot_path}")
            
            return str(snapshot_path.relative_to("static"))
        except Exception as e:
            logger.error(f"Failed to save uniform violation snapshot: {e}")
            return None

    def get_status(self):
        """Get current status of table service monitoring"""
        # Count clean/unclean tables
        clean_count = 0
        unclean_count = 0
        unknown_count = 0
        
        for table_id, tracking in self.table_tracking.items():
            state = tracking.get("cleanliness_state")
            if state == "clean":
                clean_count += 1
            elif state == "unclean":
                unclean_count += 1
            else:
                unknown_count += 1

        status = {
            "tables_monitored": len(self.table_rois) if self.table_rois else len(self.table_tracking),
            "clean_tables": clean_count,
            "unclean_tables": unclean_count,
            "unknown_tables": unknown_count
        }

        return status
    
    def _send_status_update(self, current_time):
        """Send status update via Socket.IO"""
        if not self.socketio:
            return
        
        now_ts = current_time.timestamp()
        if (now_ts - self.last_status_update) < self.status_update_interval:
            return
        
        status = self.get_status()
        self.socketio.emit("table_service_update", {
            "channel_id": self.channel_id,
            "clean_tables": status["clean_tables"],
            "unclean_tables": status["unclean_tables"],
            "unknown_tables": status["unknown_tables"],
            "tables_monitored": status["tables_monitored"],
            "total_alerts": self.total_alerts,
            "timestamp": current_time.isoformat()
        })
        
        self.last_status_update = now_ts

