"""
Queue Monitor Module
--------------------

- Counts how many people are in the queue (queue_roi / "main")
- Counts how many people are at the counter (counter_roi / "secondary")
- Uses bottom-center of the bbox for ROI checks (feet position)
- Violations:
    V1: queue_count > queue_alert_threshold
    V2: any person in queue waits >= wait_time_threshold seconds
    V3: queue_count > 0 and counter_count < counter_threshold

Emits:
    - socketio.emit('queue_update', {...})
    - socketio.emit('queue_alert', {...})
    - Optional Telegram message if bot_token & chat_id are set in env
"""

import cv2
import numpy as np
import logging
import math
from datetime import datetime
import os
import requests

from .yolo_detector import YOLODetector
from .gif_recorder import AlertGifRecorder

logger = logging.getLogger(__name__)

# Optional Telegram integration (configured via environment variables)
# Support both naming conventions: bot_token/chat_id and TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("bot_token")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID") or os.getenv("chat_id")


def send_telegram_message(text: str) -> bool:
    """
    Send a message to a Telegram chat.

    Uses environment variables (checks both naming conventions):
      - TELEGRAM_BOT_TOKEN or bot_token
      - TELEGRAM_CHAT_ID or chat_id

    Returns:
        True if message was sent successfully, False otherwise
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("⚠️ Telegram credentials not set (bot_token or chat_id missing); skipping Telegram notification")
        return False

    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
        resp = requests.post(url, data=data, timeout=5)
        if resp.status_code == 200:
            logger.info("✅ Telegram message sent successfully")
            return True
        else:
            logger.error(f"❌ Telegram API error: {resp.status_code} - {resp.text}")
            return False
    except Exception as e:
        logger.error(f"❌ Error sending Telegram message: {e}")
        return False


class QueueMonitor:
    def __init__(self, channel_id, socketio, db_manager=None, app=None):
        """
        Initialize Queue Monitor

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

        # --- YOLO detector ---
        # Uses YOLOv11n.pt for person detection (Person = class 0)
        # Note: Custom best.pt model has Person = class 12:
        # {0: 'Apron', 1: 'Cashdraw-open', 2: 'Gloves', 3: 'Hairnet', 4: 'No_apron',
        #  5: 'No_gloves', 6: 'Shoe', 7: 'Uniform_black', 8: 'Uniform_blue', 9: 'Uniform_cream',
        #  10: 'Smoke', 11: 'Fire', 12: 'Person', 13: 'Uniform_grey', 14: 'No_hairnet',
        #  15: 'Table_clean', 16: 'Table_unclean'}
        self.detector = YOLODetector(
            model_path="models/yolo11n.pt",
            # model_path="models/best.pt",
            confidence_threshold=0.5,
            img_size=640,
            person_class_id=0,
        )
        
        # --- Uniform detector (fallback for counter area) ---
        # Use best.pt to detect uniforms when person detection fails
        # This helps in cases where person detection misses staff but uniform detection works
        try:
            from .model_manager import get_shared_model
            self.uniform_detector = get_shared_model("models/best.pt", device='auto')
            self.use_uniform_fallback = True
            self.uniform_classes = {"Uniform_black", "Uniform_grey", "Uniform_cream", "Uniform_blue"}
            logger.info(f"[{self.channel_id}] ✅ Uniform detector initialized for counter area fallback")
        except Exception as e:
            logger.warning(f"[{self.channel_id}] ⚠️ Failed to initialize uniform detector fallback: {e}")
            self.uniform_detector = None
            self.use_uniform_fallback = False

        # ROI configuration (normalized 0–1 coordinates)
        self.roi_config = {
            "main": [],      # Queue area
            "secondary": []  # Counter area
        }

        # Settings (overridable via set_settings / DB)
        self.settings = {
            "dwell_time_threshold": 0.0,   # For counter area (instant counting for staff)
            "queue_dwell_time_threshold": 5.0,  # For queue area (filter out passing people - must stay 5+ seconds to be counted)
            "queue_alert_threshold": 3,    # V1: queue > 3
            "counter_threshold": 1,        # V3: need at least 1 at counter
            "alert_cooldown": 60.0,        # seconds between alerts
            "wait_time_threshold": 120.0,  # V2: any wait >= 120s (2 minutes)
            "counter_capacity_max": None,  # V4: max people allowed at counter (None = no limit, set per channel)
        }
        
        # Set counter capacity limits per channel
        if channel_id == "camera_1":
            self.settings["counter_capacity_max"] = 4
        elif channel_id == "camera_2":
            self.settings["counter_capacity_max"] = 3

        # Tracking (simple nearest-neighbor like your prototype)
        self.person_tracking = []  # list of dict tracks
        self.next_track_id = 0
        self.MAX_TRACK_DIST = 80.0
        self.TRACK_TIMEOUT = 2.0

        # Counts
        self.queue_count = 0
        self.counter_count = 0

        # Alert state
        self.last_alert_time = None
        self.alert_condition_start_time = None
        self.alert_condition_sustained_duration = 0.5  # seconds (reduced for faster detection)

        # ROI cache for current frame size
        self.roi_cache = {"main": None, "secondary": None}
        self.roi_cache_frame_size = None

        # Performance
        self.frame_count = 0
        
        # Uniform detection cache (for fallback)
        self.uniform_detection_cache = None
        self.uniform_cache_frame_count = 0
        self.uniform_cache_interval = 10  # Cache uniform detections for 10 frames
        self.detection_cache = None
        self.cache_frame_count = 0
        self.cache_interval = 2  # run YOLO every 2 frames
        self.socketio_update_interval = 15
        self.db_log_interval = 600

        # Initialize GIF recorder for alert snapshots
        self.gif_recorder = AlertGifRecorder(buffer_size=90, gif_duration=3.0, fps=5)
        
        # Track recording state for GIF management
        self._was_recording_alert = False
        self._pending_alert_info = None  # Store alert_info for database saving when GIF completes
        self._pending_violation_ids = []  # Store violation IDs to update with GIF path when complete

        logger.info(f"QueueMonitor initialized for channel {channel_id}")
        logger.info(
            f"  Model: {self.detector.model_path}, "
            f"person_class_id={self.detector.person_class_id}, "
            f"conf={self.detector.confidence_threshold}"
        )

    # ------------------------------------------------------------------ #
    # Configuration / ROI helpers
    # ------------------------------------------------------------------ #

    def set_roi(self, roi_points: dict):
        """Update ROI configuration (expects {'main': [...], 'secondary': [...]})."""
        self.roi_config = roi_points or {"main": [], "secondary": []}
        self.roi_cache = {"main": None, "secondary": None}
        logger.info(f"[{self.channel_id}] ROI updated")

        if self.db_manager:
            try:
                from flask import has_app_context

                if has_app_context():
                    self.db_manager.save_channel_config(
                        self.channel_id, "QueueMonitor", "roi", self.roi_config
                    )
            except Exception as e:
                logger.error(f"Failed to save ROI config: {e}")

    def get_roi(self):
        return self.roi_config.copy()

    def set_settings(self, settings: dict):
        """Update settings (dwell_time_threshold, queue_alert_threshold, etc.)."""
        self.settings.update(settings or {})
        logger.info(f"[{self.channel_id}] Settings updated: {self.settings}")

        if self.db_manager:
            try:
                from flask import has_app_context

                if has_app_context():
                    self.db_manager.save_channel_config(
                        self.channel_id, "QueueMonitor", "settings", self.settings
                    )
            except Exception as e:
                logger.error(f"Failed to save settings: {e}")

    def load_configuration(self):
        """Load saved ROI + settings from DB (if available)."""
        if not self.db_manager:
            return
        try:
            from flask import has_app_context

            if not has_app_context():
                return

            roi_cfg = self.db_manager.get_channel_config(
                self.channel_id, "QueueMonitor", "roi"
            )
            if roi_cfg:
                self.roi_config = roi_cfg

            settings = self.db_manager.get_channel_config(
                self.channel_id, "QueueMonitor", "settings"
            )
            if settings:
                # Update settings
                self.settings.update(settings)
                # Ensure counter area uses instant counting (dwell_time_threshold = 0.0)
                # But queue area can have a minimum dwell time to filter passing people
                if "dwell_time_threshold" not in self.settings:
                    self.settings["dwell_time_threshold"] = 0.0  # Counter area: instant counting
                if "queue_dwell_time_threshold" not in self.settings:
                    self.settings["queue_dwell_time_threshold"] = 5.0  # Queue area: 5 seconds minimum to filter passing people

            logger.info(f"[{self.channel_id}] Loaded configuration from DB")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")

    def _update_roi_cache(self, frame_w: int, frame_h: int):
        """Convert normalized ROI points to pixel polygons once per frame size."""
        if self.roi_cache_frame_size == (frame_w, frame_h):
            return

        def _build_polygon(area_key: str):
            pts_norm = self.roi_config.get(area_key) or []
            if not pts_norm:
                return None
            pts = []
            for p in pts_norm:
                x = int(p["x"] * frame_w)
                y = int(p["y"] * frame_h)
                pts.append([x, y])
            poly = np.array(pts, dtype=np.int32)
            xs = poly[:, 0]
            ys = poly[:, 1]
            bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
            return {"polygon": poly, "bbox": bbox}

        self.roi_cache["main"] = _build_polygon("main")  # Queue area
        self.roi_cache["secondary"] = _build_polygon("secondary")  # Counter area
        self.roi_cache_frame_size = (frame_w, frame_h)
        
        # Debug: Log ROI info on first few frames
        if self.frame_count <= 5:
            logger.info(f"[{self.channel_id}] ROI cache updated (Frame {self.frame_count}):")
            logger.info(f"  Queue ROI (main): {'configured' if self.roi_cache['main'] else 'not configured'}")
            if self.roi_cache["main"]:
                logger.info(f"    Polygon points: {len(self.roi_cache['main']['polygon'])}")
                logger.info(f"    Bbox: {self.roi_cache['main']['bbox']}")
            logger.info(f"  Counter ROI (secondary): {'configured' if self.roi_cache['secondary'] else 'not configured'}")
            if self.roi_cache["secondary"]:
                logger.info(f"    Polygon points: {len(self.roi_cache['secondary']['polygon'])}")
                logger.info(f"    Bbox: {self.roi_cache['secondary']['bbox']}")

    @staticmethod
    def _point_in_polygon_optimized(point, polygon, bbox):
        """Fast ROI check using bbox + cv2.pointPolygonTest."""
        x, y = point
        (min_x, min_y, max_x, max_y) = bbox
        if x < min_x or x > max_x or y < min_y or y > max_y:
            return False
        return cv2.pointPolygonTest(polygon, (float(x), float(y)), False) >= 0
    
    @staticmethod
    def _bbox_overlaps_roi(bbox, roi_polygon, roi_bbox, min_overlap_ratio=0.3):
        """
        Check if a person's bounding box overlaps with ROI polygon.
        More forgiving than point check - useful for counter area where staff may be at edges.
        
        Args:
            bbox: Person bounding box [x1, y1, x2, y2]
            roi_polygon: ROI polygon points
            roi_bbox: ROI bounding box (min_x, min_y, max_x, max_y)
            min_overlap_ratio: Minimum ratio of bbox that must be in ROI (0.0-1.0)
        
        Returns:
            True if bbox overlaps ROI sufficiently
        """
        x1, y1, x2, y2 = bbox
        min_x, min_y, max_x, max_y = roi_bbox
        
        # Quick bbox check first - but be more forgiving (allow partial overlap)
        # If bbox is completely outside ROI bbox, definitely not in ROI
        if x2 < min_x or x1 > max_x or y2 < min_y or y1 > max_y:
            return False
        
        # For counter area, be very forgiving - check multiple points including top of bbox
        # Staff may be standing behind counter, so we check:
        # - Center point (most reliable)
        # - Top center (head position - staff behind counter)
        # - Bottom center (feet position)
        # - Top-left and top-right corners (for staff at edges)
        bbox_center = ((x1 + x2) / 2, (y1 + y2) / 2)
        bbox_top_center = ((x1 + x2) / 2, y1)  # Head position - important for staff behind counter
        bbox_bottom_center = ((x1 + x2) / 2, y2)
        bbox_top_left = (x1, y1)
        bbox_top_right = (x2, y1)
        bbox_bottom_left = (x1, y2)
        bbox_bottom_right = (x2, y2)
        
        # Check if any of these key points are in the ROI
        # Prioritize center and top-center (most reliable for staff detection)
        points_to_check = [
            bbox_center,      # Most reliable
            bbox_top_center,  # Important for staff behind counter
            bbox_bottom_center,
            bbox_top_left,
            bbox_top_right,
            bbox_bottom_left,
            bbox_bottom_right
        ]
        
        points_in_roi = 0
        for px, py in points_to_check:
            if cv2.pointPolygonTest(roi_polygon, (float(px), float(py)), False) >= 0:
                points_in_roi += 1
        
        # If center point OR top-center is in ROI, consider it a match
        # This is very forgiving for staff who may be standing behind the counter
        center_in = cv2.pointPolygonTest(roi_polygon, (float(bbox_center[0]), float(bbox_center[1])), False) >= 0
        top_center_in = cv2.pointPolygonTest(roi_polygon, (float(bbox_top_center[0]), float(bbox_top_center[1])), False) >= 0
        
        if center_in or top_center_in:
            return True
        
        # Otherwise, require at least 2 points (30% of checked points) to be in ROI
        # This handles edge cases where staff are partially visible
        return points_in_roi >= 2

    # ------------------------------------------------------------------ #
    # Detection classification & tracking
    # ------------------------------------------------------------------ #

    def _classify_detections(self, detections, frame_w: int, frame_h: int):
        """
        Split detections into queue vs counter vs none.

        Priority:
            - Use bottom_center point for ROI checks.
            - If in both counter & queue, classify as COUNTER.
        """
        self._update_roi_cache(frame_w, frame_h)
        queue_roi = self.roi_cache["main"]
        counter_roi = self.roi_cache["secondary"]

        queue_dets = []
        counter_dets = []

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]

            # Ensure we have center for tracking (if needed)
            if "center" not in det:
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                det["center"] = (cx, cy)
                det["bottom_center"] = (cx, int(y2))

            # For queue area: use bottom-right corner (feet position) - more accurate for customers
            # For counter area: use center point - better for staff who may be further back
            queue_point = (x2, y2)  # Bottom-right corner for queue
            counter_point = det["center"]  # Center point for counter (more forgiving for staff behind counter)

            # Check if in queue area first (reference uses if/elif, so person can only be in one area)
            if queue_roi and self._point_in_polygon_optimized(
                queue_point, queue_roi["polygon"], queue_roi["bbox"]
            ):
                det["area_type"] = "queue"
                queue_dets.append(det)
            # Check if in counter area (only if not in queue)
            # Use bbox overlap check for counter - more forgiving for staff at edges
            elif counter_roi:
                # Use bbox overlap for counter (more forgiving for staff behind counter)
                # Use very low overlap ratio (0.1 = 10%) to catch staff who are mostly in ROI
                in_counter = self._bbox_overlaps_roi(
                    det["bbox"], counter_roi["polygon"], counter_roi["bbox"], min_overlap_ratio=0.1
                )
                
                # Also check if person's center point is near the counter ROI (within 100 pixels)
                # This helps catch staff who are slightly outside the ROI but clearly at the counter
                # Increased from 50 to 100 pixels to be more forgiving for seated staff
                if not in_counter:
                    center = det.get("center")
                    if center:
                        # Check distance from center to ROI polygon
                        dist = cv2.pointPolygonTest(counter_roi["polygon"], (float(center[0]), float(center[1])), True)
                        # If within 100 pixels of ROI edge, consider them at counter
                        if dist > -100:  # Negative means outside, but if within 100px, count it
                            in_counter = True
                            if self.frame_count <= 10 or self.frame_count % 60 == 0:
                                logger.info(f"  Person near counter ROI (within 100px): center={center}, dist={dist:.1f}px")
                
                # Also check if any part of the bbox (top, center, bottom) is in the ROI
                # This is especially important for seated staff whose bbox might be positioned differently
                if not in_counter:
                    x1, y1, x2, y2 = det["bbox"]
                    bbox_top = ((x1 + x2) / 2, y1)
                    bbox_center = det.get("center", ((x1 + x2) / 2, (y1 + y2) / 2))
                    bbox_bottom = ((x1 + x2) / 2, y2)
                    
                    # Check if any key point is in ROI
                    for point in [bbox_top, bbox_center, bbox_bottom]:
                        if cv2.pointPolygonTest(counter_roi["polygon"], (float(point[0]), float(point[1])), False) >= 0:
                            in_counter = True
                            if self.frame_count <= 10 or self.frame_count % 60 == 0:
                                logger.info(f"  Person detected in counter via point check: point={point}")
                            break
                
                if in_counter:
                    det["area_type"] = "counter"
                    counter_dets.append(det)
                else:
                    det["area_type"] = "none"
            else:
                det["area_type"] = "none"
            
            # Debug logging for first few detections
            if self.frame_count <= 10 and len(detections) > 0:
                logger.info(f"  Detection: bbox={det['bbox']}, center={det.get('center')}, bottom_right=({x2}, {y2}), area={det['area_type']}")

        # Debug logging
        if self.frame_count <= 10 or self.frame_count % 30 == 0:
            logger.info(f"[{self.channel_id}] Frame {self.frame_count} detections:")
            logger.info(
                f"  total={len(detections)}, queue={len(queue_dets)}, counter={len(counter_dets)}, outside={len(detections) - len(queue_dets) - len(counter_dets)}"
            )
            logger.info(f"  Queue ROI configured: {queue_roi is not None}")
            logger.info(f"  Counter ROI configured: {counter_roi is not None}")
            if counter_roi:
                logger.info(f"  Counter ROI bbox: {counter_roi['bbox']}")
                logger.info(f"  Counter ROI polygon points: {len(counter_roi['polygon'])}")
            if len(detections) > 0:
                # Log all detections and their classification
                logger.info(f"  All detections classification:")
                for i, det in enumerate(detections[:5]):  # Show first 5
                    bbox = det.get('bbox', [])
                    center = det.get('center', 'N/A')
                    area_type = det.get('area_type', 'none')
                    logger.info(f"    Det {i+1}: bbox={bbox}, center={center}, area={area_type}")
                    if area_type == 'none' and counter_roi:
                        # Check why it's not in counter
                        bbox_overlap = self._bbox_overlaps_roi(
                            bbox, counter_roi["polygon"], counter_roi["bbox"]
                        )
                        center_check = self._point_in_polygon_optimized(
                            center, counter_roi["polygon"], counter_roi["bbox"]
                        ) if isinstance(center, tuple) else False
                        logger.info(f"      → Counter ROI check: bbox_overlap={bbox_overlap}, center_in={center_check}")
            if len(queue_dets) > 0:
                logger.info(f"  Queue detections (first 3):")
                for i, det in enumerate(queue_dets[:3]):
                    logger.info(f"    Queue det {i+1}: bbox={det['bbox']}, bottom_right={det.get('bottom_center', 'N/A')}")
            elif len(detections) > 0 and queue_roi:
                logger.warning(f"  ⚠️ WARNING: {len(detections)} detections but 0 in queue area! Queue ROI may need adjustment.")
                # Check first few detections to see why they're not in queue
                for i, det in enumerate(detections[:3]):
                    bottom_right = (det['bbox'][2], det['bbox'][3]) if len(det.get('bbox', [])) >= 4 else None
                    in_queue = self._point_in_polygon_optimized(
                        bottom_right, queue_roi["polygon"], queue_roi["bbox"]
                    ) if bottom_right else False
                    logger.info(f"    Det {i+1}: bbox={det.get('bbox')}, bottom_right={bottom_right}, in_queue={in_queue}")
            if len(counter_dets) > 0:
                logger.info(f"  Counter detections (first 3):")
                for i, det in enumerate(counter_dets[:3]):
                    logger.info(f"    Counter det {i+1}: bbox={det['bbox']}, center={det.get('center', 'N/A')}")
            elif len(detections) > 0 and counter_roi:
                logger.warning(f"  ⚠️ WARNING: {len(detections)} detections but 0 in counter area! Counter ROI may need adjustment.")

        return queue_dets, counter_dets

    @staticmethod
    def _euclidean(p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def _update_person_tracking(self, detections, area_type: str) -> int:
        """
        Simple nearest-neighbor tracking within a specific area (queue or counter).

        Returns:
            Count of people who have been in this area for >= dwell_time_threshold.
        """
        current_time = datetime.now()
        now_ts = current_time.timestamp()
        
        # Use different dwell times for queue vs counter
        # Counter: instant counting (staff should be counted immediately) - ALWAYS 0.0
        # Queue: require minimum dwell time to filter out passing people
        if area_type == "counter":
            dwell_thresh = 0.0  # Always instant counting for counter area
        else:  # queue area
            dwell_thresh = float(self.settings.get("queue_dwell_time_threshold", 2.0) or 2.0)
        
        # If dwell time = 0 → instant count (for counter area)
        # For counter, just count all detections immediately without tracking
        if dwell_thresh == 0.0:
            # Still update tracking for visualization, but count immediately
            centers = [d.get("center") for d in detections if d.get("center")]
            tracks = [t for t in self.person_tracking if t["area_type"] == area_type]
            
            # Quick tracking update for visualization
            for c in centers:
                best_track = None
                best_dist = float("inf")
                for t in tracks:
                    if t.get("matched"):
                        continue
                    d = self._euclidean(c, t["center"])
                    if d < best_dist and d < self.MAX_TRACK_DIST:
                        best_dist = d
                        best_track = t
                
                if best_track is None:
                    track = {
                        "id": self.next_track_id,
                        "center": c,
                        "last_seen": now_ts,
                        "in_roi": True,
                        "entered_roi_time": now_ts,
                        "counted": True,  # Always counted for counter
                        "matched": True,
                        "area_type": area_type,
                    }
                    self.next_track_id += 1
                    self.person_tracking.append(track)
                else:
                    best_track["center"] = c
                    best_track["last_seen"] = now_ts
                    best_track["matched"] = True
                    best_track["in_roi"] = True
                    best_track["counted"] = True  # Always counted for counter
                    if best_track.get("entered_roi_time") is None:
                        best_track["entered_roi_time"] = now_ts
            
            # Remove stale tracks
            self.person_tracking = [
                t for t in self.person_tracking
                if (now_ts - t["last_seen"]) < self.TRACK_TIMEOUT
            ]
            
            return len(detections)

        centers = [d.get("center") for d in detections if d.get("center")]

        # Filter tracks for this area
        tracks = [t for t in self.person_tracking if t["area_type"] == area_type]

        # Mark all tracks as unmatched
        for t in tracks:
            t["matched"] = False

        # Match detections to tracks
        for c in centers:
            best_track = None
            best_dist = float("inf")
            for t in tracks:
                if t.get("matched"):
                    continue
                d = self._euclidean(c, t["center"])
                if d < best_dist and d < self.MAX_TRACK_DIST:
                    best_dist = d
                    best_track = t

            if best_track is None:
                # New track
                track = {
                    "id": self.next_track_id,
                    "center": c,
                    "last_seen": now_ts,
                    "in_roi": True,
                    "entered_roi_time": now_ts,
                    "counted": False,
                    "matched": True,
                    "area_type": area_type,
                    "previous_center": c,  # Track previous position for movement detection
                    "position_history": [c],  # Keep history of positions to detect movement
                }
                self.next_track_id += 1
                self.person_tracking.append(track)
                tracks.append(track)
            else:
                # Update track with new position
                old_center = best_track.get("center", c)
                best_track["previous_center"] = old_center
                best_track["center"] = c
                best_track["last_seen"] = now_ts
                best_track["matched"] = True
                
                # Update position history (keep last 5 positions for movement analysis)
                if "position_history" not in best_track:
                    best_track["position_history"] = [old_center]
                best_track["position_history"].append(c)
                if len(best_track["position_history"]) > 5:
                    best_track["position_history"].pop(0)
                
                # Only update entered_roi_time if track was previously out of ROI
                if not best_track.get("in_roi", False):
                    # Track was out of ROI, now back in - reset entered_roi_time and position history
                    best_track["entered_roi_time"] = now_ts
                    best_track["in_roi"] = True
                    best_track["position_history"] = [c]  # Reset history when re-entering
                else:
                    # Track was already in ROI - preserve entered_roi_time (don't reset it!)
                    best_track["in_roi"] = True
                    # Only set entered_roi_time if it doesn't exist (shouldn't happen, but safety check)
                    if best_track.get("entered_roi_time") is None:
                        best_track["entered_roi_time"] = now_ts

        # Any unmatched tracks → mark as out-of-ROI
        for t in tracks:
            if not t.get("matched"):
                t["in_roi"] = False
                # Reset entered_roi_time when person leaves ROI
                t["entered_roi_time"] = None

        # Remove stale tracks
        self.person_tracking = [
            t
            for t in self.person_tracking
            if (now_ts - t["last_seen"]) < self.TRACK_TIMEOUT
        ]

        # Re-filter tracks after cleanup (tracks list might be stale)
        tracks = [t for t in self.person_tracking if t["area_type"] == area_type]

        # Count people whose time in ROI exceeds dwell threshold AND are relatively stationary
        count = 0
        for t in tracks:
            if t["in_roi"] and t["entered_roi_time"] is not None:
                time_in = now_ts - t["entered_roi_time"]
                
                # For queue area: also check if person is moving too fast (passing through)
                is_stationary = True
                if area_type == "queue" and len(t.get("position_history", [])) >= 3:
                    # Calculate total distance moved over last few positions
                    positions = t["position_history"]
                    total_distance = 0
                    for i in range(1, len(positions)):
                        dist = self._euclidean(positions[i-1], positions[i])
                        total_distance += dist
                    
                    # If person moved more than 100 pixels in last few frames, they're likely passing through
                    # (adjust threshold based on frame rate - assuming ~10-15 fps, 100 pixels over 3-5 frames is fast)
                    if total_distance > 100:
                        is_stationary = False
                        # Reset entered_roi_time if person is moving too fast (they're passing through)
                        t["entered_roi_time"] = now_ts
                
                if time_in >= dwell_thresh and is_stationary:
                    t["counted"] = True
                else:
                    # Person hasn't been in ROI long enough or is moving too fast - don't count yet
                    t["counted"] = False
            # Only count if person has been in ROI long enough and is relatively stationary
            if t.get("counted"):
                count += 1
        
        # Debug logging for tracking
        if (self.frame_count <= 10 or self.frame_count % 30 == 0):
            area_name = "Counter" if area_type == "counter" else "Queue"
            logger.info(f"[{self.channel_id}] {area_name} tracking debug (Frame {self.frame_count}):")
            logger.info(f"  Detections passed to tracking: {len(detections)}")
            logger.info(f"  Active tracks in {area_type} area: {len(tracks)}")
            logger.info(f"  Dwell time threshold: {dwell_thresh}s")
            logger.info(f"  Final count: {count}")
            for i, t in enumerate(tracks[:5]):  # Show first 5 tracks
                time_in = (now_ts - t["entered_roi_time"]) if t.get("entered_roi_time") else 0
                logger.info(f"  Track {i+1}: in_roi={t.get('in_roi')}, entered_time={t.get('entered_roi_time')}, time_in={time_in:.2f}s, counted={t.get('counted')}")

        return count

    # ------------------------------------------------------------------ #
    # Violation saving
    # ------------------------------------------------------------------ #

    def _save_violation_snapshot(self, frame, violation_type, violation_message, queue_count=None, counter_count=None):
        """
        Save a snapshot of a queue violation
        
        Args:
            frame: Frame to save (will be cleaned and annotated with specific violation)
            violation_type: Type of violation ('queue_overflow', 'wait_time_exceeded', 'no_counter_staff')
            violation_message: Human-readable violation message
            queue_count: Queue count at time of violation (optional, uses current if not provided)
            counter_count: Counter count at time of violation (optional, uses current if not provided)
            
        Returns:
            str: Path to saved snapshot (relative to static/)
        """
        try:
            from pathlib import Path
            timestamp = datetime.now()
            timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
            
            snapshot_dir = Path("static/queue_violations")
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            
            filename = f"queue_{self.channel_id}_{violation_type}_{timestamp_str}.jpg"
            snapshot_path = snapshot_dir / filename
            
            # Use provided counts or current counts
            q_count = queue_count if queue_count is not None else self.queue_count
            c_count = counter_count if counter_count is not None else self.counter_count
            
            # Create a clean copy of the frame and add ONLY this violation's text
            annotated = frame.copy()
            
            # Get frame dimensions
            h, w = annotated.shape[:2]
            
            # Don't black out the image - just draw text with background rectangles for readability
            # We'll draw background rectangles behind each text line instead of overlaying the whole area
            
            # Add violation text overlay at the TOP - this is the ONLY alert text on this snapshot
            # Use the SPECIFIC violation message passed to this function
            y_offset = 30
            
            # Draw "QUEUE VIOLATION" with light semi-transparent background
            text = "QUEUE VIOLATION"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
            text_w, text_h = text_size
            # Draw very light semi-transparent background (85% original, 15% dark gray)
            overlay = annotated.copy()
            cv2.rectangle(overlay, (5, y_offset - text_h - 5), (10 + text_w, y_offset + baseline + 5), (40, 40, 40), -1)
            cv2.addWeighted(annotated, 0.85, overlay, 0.15, 0, annotated)
            # Draw text
            cv2.putText(annotated, text, (10, y_offset),
                       font, font_scale, (0, 0, 255), thickness)
            y_offset += 35
            
            # CRITICAL: Use the violation_message parameter, not any text that might be on the frame
            # Wrap long messages to prevent text from going off-screen
            alert_text = f"Alert: {violation_message}"
            # Check if text is too long and wrap it if needed
            text_size, _ = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            max_text_width = w - 20  # Leave 10px margin on each side
            if text_size[0] > max_text_width:
                # Split text into multiple lines
                words = violation_message.split()
                lines = []
                current_line = "Alert:"
                for word in words:
                    test_line = f"{current_line} {word}"
                    test_size, _ = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    if test_size[0] > max_text_width:
                        lines.append(current_line)
                        current_line = word
                    else:
                        current_line = test_line
                lines.append(current_line)
                # Draw each line with light background
                for line in lines:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    thickness = 2
                    text_size, baseline = cv2.getTextSize(line, font, font_scale, thickness)
                    text_w, text_h = text_size
                    # Draw very light semi-transparent background (85% original, 15% dark gray)
                    overlay = annotated.copy()
                    cv2.rectangle(overlay, (5, y_offset - text_h - 3), (10 + text_w, y_offset + baseline + 3), (40, 40, 40), -1)
                    cv2.addWeighted(annotated, 0.85, overlay, 0.15, 0, annotated)
                    # Draw text
                    cv2.putText(annotated, line, (10, y_offset),
                               font, font_scale, (255, 255, 255), thickness)
                    y_offset += 28
            else:
                # Draw single line with light background
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                text_size, baseline = cv2.getTextSize(alert_text, font, font_scale, thickness)
                text_w, text_h = text_size
                # Draw very light semi-transparent background (85% original, 15% dark gray)
                overlay = annotated.copy()
                cv2.rectangle(overlay, (5, y_offset - text_h - 3), (10 + text_w, y_offset + baseline + 3), (40, 40, 40), -1)
                cv2.addWeighted(annotated, 0.85, overlay, 0.15, 0, annotated)
                # Draw text
                cv2.putText(annotated, alert_text, (10, y_offset),
                           font, font_scale, (255, 255, 255), thickness)
                y_offset += 28
            
            # Add counts (ensure it's within frame bounds)
            counts_text = f"Queue: {q_count} | Counter: {c_count}"
            if y_offset + 25 < h - 120:  # Make sure it's not in the bottom area
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                text_size, baseline = cv2.getTextSize(counts_text, font, font_scale, thickness)
                text_w, text_h = text_size
                # Draw very light semi-transparent background (85% original, 15% dark gray)
                overlay = annotated.copy()
                cv2.rectangle(overlay, (5, y_offset - text_h - 2), (10 + text_w, y_offset + baseline + 2), (40, 40, 40), -1)
                cv2.addWeighted(annotated, 0.85, overlay, 0.15, 0, annotated)
                # Draw text
                cv2.putText(annotated, counts_text, (10, y_offset), 
                           font, font_scale, (0, 255, 255), thickness)
            
            # Add channel, time, and alert info at the BOTTOM (within frame bounds)
            timestamp = datetime.now()
            time_str = timestamp.strftime("%d/%m/%Y, %I:%M:%S %p")
            
            bottom_y = h - 80  # Start 80 pixels from bottom
            max_text_width = w - 20  # Leave 10px margin on each side
            
            # Draw each bottom text line with background
            bottom_texts = [
                f"Channel: {self.channel_id}",
                f"Time: {time_str}",
            ]
            
            # Truncate alert message if too long for bottom display
            bottom_alert_text = f"Alert: {violation_message}"
            bottom_text_size, _ = cv2.getTextSize(bottom_alert_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            if bottom_text_size[0] > max_text_width:
                # Truncate with ellipsis
                truncated = violation_message[:50] + "..." if len(violation_message) > 50 else violation_message
                bottom_alert_text = f"Alert: {truncated}"
            bottom_texts.append(bottom_alert_text)
            
            # Draw each text with light background
            for text in bottom_texts:
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
                text_w, text_h = text_size
                # Draw very light semi-transparent background (85% original, 15% dark gray)
                overlay = annotated.copy()
                cv2.rectangle(overlay, (5, bottom_y - text_h - 2), (10 + text_w, bottom_y + baseline + 2), (40, 40, 40), -1)
                cv2.addWeighted(annotated, 0.85, overlay, 0.15, 0, annotated)
                # Draw text
                cv2.putText(annotated, text, (10, bottom_y),
                           font, font_scale, (255, 255, 255), thickness)
                bottom_y += 20
            
            # Log what we're saving to verify
            logger.info(f"[{self.channel_id}] Saving snapshot with violation_type={violation_type}, message='{violation_message}', queue={q_count}, counter={c_count}")
            
            # Save the image
            success = cv2.imwrite(str(snapshot_path), annotated)
            if not success:
                logger.error(f"Failed to write snapshot image to {snapshot_path}")
                return None
            
            # Verify file was created and has content
            import os
            if not os.path.exists(snapshot_path):
                logger.error(f"Snapshot file was not created: {snapshot_path}")
                return None
            
            file_size = os.path.getsize(snapshot_path)
            if file_size == 0:
                logger.error(f"Snapshot file is empty: {snapshot_path}")
                return None
            
            logger.info(f"Queue violation snapshot saved: {snapshot_path} ({file_size} bytes) with message: {violation_message}")
            
            # Return full path for database storage (use forward slashes for web compatibility)
            return str(snapshot_path).replace('\\', '/')
        except Exception as e:
            logger.error(f"Failed to save violation snapshot: {e}")
            return None

    def _save_violations(self, frame, alert_info):
        """
        Save violations to database with snapshots
        
        Args:
            frame: Annotated frame
            alert_info: Alert information dict from _check_alert_conditions
        """
        if not self.db_manager:
            logger.warning(f"[{self.channel_id}] Cannot save violations: db_manager not available")
            return
        
        if not alert_info:
            logger.warning(f"[{self.channel_id}] Cannot save violations: alert_info is None or empty")
            return
        
        current_time = datetime.now()
        message = alert_info.get("message", "")
        logger.info(f"[{self.channel_id}] Saving violations for message: {message}")
        
        # Parse violations from message (message can contain multiple violations separated by "; ")
        violations = []
        message_parts = [part.strip() for part in message.split(";")]
        
        for part in message_parts:
            if "Counter capacity exceeded" in part or ("capacity" in part.lower() and "exceeded" in part.lower()):
                violations.append(("counter_capacity_exceeded", part))
            elif "Queue too long" in part or ("Queue capacity exceeded" in part) or (">" in part and "queue" in part.lower() and "capacity" not in part.lower()):
                violations.append(("queue_overflow", part))
            elif "wait time" in part.lower() or "≥" in part or ">=" in part:
                violations.append(("wait_time_exceeded", part))
            elif "No staff at counter" in part or ("counter" in part.lower() and "staff" in part.lower() and "capacity" not in part.lower()):
                violations.append(("no_counter_staff", part))
            else:
                # Unknown violation type - log it but still save it
                logger.warning(f"[{self.channel_id}] Unknown violation type in message part: {part}")
                violations.append(("queue_alert", part))
        
        if not violations:
            # Fallback: save as general violation if parsing fails
            logger.warning(f"[{self.channel_id}] Could not parse violation types from message: {message}")
            violations.append(("queue_alert", message))
        
        logger.info(f"[{self.channel_id}] Parsed {len(violations)} violation(s) from alert message: {message}")
        
        # Save each violation type separately
        # IMPORTANT: Use the counts from alert_info (at time of alert) not current counts
        alert_queue_count = alert_info.get("queue_count", self.queue_count)
        alert_counter_count = alert_info.get("counter_count", self.counter_count)
        
        for violation_type, violation_message in violations:
            try:
                # Note: GIF recording is handled separately in process_frame
                # We no longer save static snapshots - GIF will be saved when recording completes
                # snapshot_path is set to None since we're using GIFs now
                snapshot_path = None
                
                # Save to database (with proper app context handling)
                # The GIF will be saved separately via save_alert_gif when recording completes
                violation_id = None
                if self.app:
                    from flask import has_app_context
                    if has_app_context():
                        violation_id = self.db_manager.add_queue_violation(
                            channel_id=self.channel_id,
                            violation_type=violation_type,
                            violation_message=violation_message,
                            queue_count=self.queue_count,
                            counter_count=self.counter_count,
                            wait_time_seconds=alert_info.get("max_wait_seconds", 0.0),
                            snapshot_path=snapshot_path,  # None - using GIFs now
                            alert_data=alert_info
                        )
                    else:
                        with self.app.app_context():
                            violation_id = self.db_manager.add_queue_violation(
                                channel_id=self.channel_id,
                                violation_type=violation_type,
                                violation_message=violation_message,
                                queue_count=self.queue_count,
                                counter_count=self.counter_count,
                                wait_time_seconds=alert_info.get("max_wait_seconds", 0.0),
                                snapshot_path=snapshot_path,  # None - using GIFs now
                                alert_data=alert_info
                            )
                else:
                    violation_id = self.db_manager.add_queue_violation(
                        channel_id=self.channel_id,
                        violation_type=violation_type,
                        violation_message=violation_message,
                        queue_count=self.queue_count,
                        counter_count=self.counter_count,
                        wait_time_seconds=alert_info.get("max_wait_seconds", 0.0),
                        snapshot_path=snapshot_path,  # None - using GIFs now
                        alert_data=alert_info
                    )
                
                if violation_id:
                    logger.info(f"[{self.channel_id}] ✅ Queue violation saved to database: ID {violation_id}, type: {violation_type}, message: {violation_message}")
                    logger.info(f"[{self.channel_id}] 📹 GIF recording in progress - will be saved when complete")
                    # Store violation ID to update with GIF path later
                    self._pending_violation_ids.append(violation_id)
                else:
                    logger.error(f"[{self.channel_id}] ❌ Failed to save queue violation (returned None)")
            except Exception as e:
                logger.error(f"[{self.channel_id}] ❌ Failed to save queue violation to database: {e}", exc_info=True)

    # ------------------------------------------------------------------ #
    # Alert logic
    # ------------------------------------------------------------------ #

    def _check_alert_conditions(self):
        """
        Check queue violation conditions.

        Violations:
        - V1: queue_count > queue_alert_threshold
        - V2: any person in queue waited >= wait_time_threshold seconds (2 minutes)
        - V3: queue_count > 0 and counter_count < counter_threshold
        - V4: counter_count > counter_capacity_max (if configured)
        """
        current_time = datetime.now()
        now_ts = current_time.timestamp()

        dwell_thresh = float(self.settings.get("dwell_time_threshold", 0.0) or 0.0)
        max_queue_without_violation = int(
            self.settings.get("queue_alert_threshold", 3) or 3
        )
        counter_required = int(self.settings.get("counter_threshold", 1) or 1)
        cooldown = float(self.settings.get("alert_cooldown", 60.0) or 60.0)
        wait_threshold = float(self.settings.get("wait_time_threshold", 120.0) or 120.0)  # 2 minutes
        counter_capacity_max = self.settings.get("counter_capacity_max")  # None or int

        # Compute queue wait times
        queue_wait_times = []
        for track in self.person_tracking:
            if track.get("area_type") != "queue":
                continue
            if track.get("in_roi") and track.get("entered_roi_time") is not None:
                t_in = now_ts - track["entered_roi_time"]
                if t_in >= dwell_thresh:
                    queue_wait_times.append(t_in)

        max_wait = max(queue_wait_times) if queue_wait_times else 0.0
        violations = []

        # V1: Queue too long
        if self.queue_count > max_queue_without_violation:
            violations.append(
                f"Queue capacity exceeded: {self.queue_count} > {max_queue_without_violation} (max allowed)"
            )
            logger.debug(f"[{self.channel_id}] V1 violation detected: queue_count={self.queue_count} > threshold={max_queue_without_violation}")

        # V2: Wait time exceeded
        if self.queue_count > 0 and max_wait >= wait_threshold:
            violations.append(
                f"Queue wait time {int(max_wait)}s ≥ {int(wait_threshold)}s"
            )
            logger.debug(f"[{self.channel_id}] V2 violation detected: max_wait={max_wait:.1f}s >= threshold={wait_threshold}s")

        # V3: No staff at counter
        # Only trigger if queue has people AND counter is actually empty (not just below threshold)
        # This prevents false positives when staff are present but detection is slightly off
        if self.queue_count > 0 and self.counter_count < counter_required:
            # Additional checks to prevent false positives:
            # 1. Check if we have any detections classified as "counter" (even if not counted yet)
            counter_detections = len([t for t in self.person_tracking if t.get("area_type") == "counter" and t.get("in_roi")])
            # 2. Check if we have any detections in counter ROI from current frame
            # (This catches cases where detection happens but tracking hasn't updated yet)
            current_counter_dets = getattr(self, '_last_counter_dets', [])
            current_counter_count = len(current_counter_dets)
            
            # If we have counter detections but count is 0, it's likely a tracking/dwell time issue
            if counter_detections > 0 or current_counter_count > 0:
                # People detected but not counted - likely tracking/dwell time issue, not absence
                logger.info(f"[{self.channel_id}] V3 check: {counter_detections} tracked + {current_counter_count} current detections in counter but count=0 (likely tracking issue, skipping alert)")
            else:
                # No detections at all in counter area - this is a real absence
                # Don't include counts in the message - they may be stale when displayed
                # The current counts are shown separately in the display
                violations.append(
                    "No staff at counter"
                )
                logger.info(f"[{self.channel_id}] V3 violation detected: queue={self.queue_count} > 0, counter={self.counter_count} < required={counter_required}, no counter detections found")

        # V4: Counter capacity exceeded
        if counter_capacity_max is not None and self.counter_count > counter_capacity_max:
            violations.append(
                f"Counter capacity exceeded: {self.counter_count} > {counter_capacity_max} (max allowed)"
            )
            logger.debug(f"[{self.channel_id}] V4 violation detected: counter_count={self.counter_count} > capacity_max={counter_capacity_max}")

        if not violations:
            self.alert_condition_start_time = None
            return None
        
        logger.debug(f"[{self.channel_id}] Found {len(violations)} violation(s): {violations}")

        # Require condition to be sustained for a short time
        if self.alert_condition_start_time is None:
            self.alert_condition_start_time = current_time
            logger.debug(f"[{self.channel_id}] Violation condition started, waiting for sustained duration ({self.alert_condition_sustained_duration}s)")
            return None

        sustained = (current_time - self.alert_condition_start_time).total_seconds()
        if sustained < self.alert_condition_sustained_duration:
            logger.debug(f"[{self.channel_id}] Violation condition sustained for {sustained:.2f}s / {self.alert_condition_sustained_duration}s")
            return None

        # Cooldown between alerts
        if self.last_alert_time is not None:
            time_since_last = (current_time - self.last_alert_time).total_seconds()
            if time_since_last < cooldown:
                logger.debug(f"[{self.channel_id}] Alert in cooldown: {time_since_last:.1f}s / {cooldown}s")
                return None

        message = "; ".join(violations)
        alert_info = {
            "type": "queue_alert",
            "channel_id": self.channel_id,
            "message": message,
            "queue_count": self.queue_count,
            "counter_count": self.counter_count,
            "max_wait_seconds": max_wait,
            "wait_threshold": wait_threshold,
            "queue_alert_threshold": max_queue_without_violation,
            "counter_threshold": counter_required,
            "counter_capacity_max": counter_capacity_max,
            "timestamp": current_time.isoformat(),
        }

        logger.info(f"[{self.channel_id}] ✅ Alert condition met and sustained! Violations: {message}")
        self.last_alert_time = current_time
        self.alert_condition_start_time = None
        return alert_info

    # ------------------------------------------------------------------ #
    # Main per-frame entrypoint
    # ------------------------------------------------------------------ #

    def process_frame(self, frame):
        """
        Main entrypoint called by the pipeline.

        Args:
            frame: BGR frame (numpy array)

        Returns:
            Processed frame (with overlays).
        """
        self.frame_count += 1

        original_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Add frame to GIF recorder buffer (always running)
        self.gif_recorder.add_frame(frame)

        # Run YOLO every N frames, reuse detections in between
        if (
            self.detection_cache is None
            or self.frame_count - self.cache_frame_count >= self.cache_interval
        ):
            detections = self.detector.detect_persons(frame)
            self.detection_cache = detections
            self.cache_frame_count = self.frame_count
        else:
            detections = self.detection_cache or []

        # Update ROI cache first (needed for fallback check)
        self._update_roi_cache(w, h)
        
        # Classify into queue/counter
        queue_dets, counter_dets = self._classify_detections(detections, w, h)
        
        # Fallback: If no person detections in counter area OR counter count is 0, try uniform detection
        # This helps when person detection fails but uniform detection works (e.g., camera_2)
        # Also check if we have detections but they're not being counted (tracking/dwell time issue)
        should_use_fallback = (
            self.use_uniform_fallback 
            and self.uniform_detector 
            and len(counter_dets) == 0
        )
        
        if should_use_fallback:
            counter_roi = self.roi_cache.get("secondary")
            if counter_roi and counter_roi.get("polygon") is not None:
                try:
                    # Run uniform detection (with caching for performance)
                    if (
                        self.uniform_detection_cache is None
                        or self.frame_count - self.uniform_cache_frame_count >= self.uniform_cache_interval
                    ):
                        uniform_results = self.uniform_detector(
                            frame,
                            conf=self.detector.confidence_threshold,
                            iou=0.45,
                            verbose=False
                        )
                        self.uniform_detection_cache = uniform_results
                        self.uniform_cache_frame_count = self.frame_count
                    else:
                        uniform_results = self.uniform_detection_cache
                    
                    # Extract uniform detections
                    uniform_dets = []
                    if uniform_results and len(uniform_results) > 0:
                        boxes = uniform_results[0].boxes
                        class_names = uniform_results[0].names
                        for box in boxes:
                            class_id = int(box.cls[0])
                            class_name = class_names[class_id]
                            conf = float(box.conf[0])
                            
                            # Only consider uniform classes
                            if class_name in self.uniform_classes:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                bbox = [int(x1), int(y1), int(x2), int(y2)]
                                center = [(x1 + x2) / 2, (y1 + y2) / 2]
                                bottom_center = [center[0], y2]  # Bottom center point
                                
                                # Check if uniform is in counter ROI
                                if self._point_in_polygon_optimized(
                                    bottom_center, 
                                    counter_roi["polygon"], 
                                    counter_roi["bbox"]
                                ):
                                    uniform_dets.append({
                                        "bbox": bbox,
                                        "center": center,
                                        "confidence": conf,
                                        "class_name": class_name,
                                        "from_uniform": True  # Mark as uniform detection
                                    })
                    
                    # If we found uniforms in counter area, add them as counter detections
                    if uniform_dets:
                        logger.info(f"[{self.channel_id}] 🔄 Fallback: Found {len(uniform_dets)} uniform(s) in counter area (person detection missed them)")
                        counter_dets.extend(uniform_dets)
                except Exception as e:
                    logger.warning(f"[{self.channel_id}] ⚠️ Uniform fallback detection failed: {e}")

        # Bounding boxes removed - only ROI and text overlays are shown

        # Draw ROIs (ROI cache already updated above)
        if self.roi_cache["main"] is not None:
            cv2.polylines(
                original_frame,
                [self.roi_cache["main"]["polygon"]],
                True,
                (0, 255, 255),
                2,
            )
        if self.roi_cache["secondary"] is not None:
            cv2.polylines(
                original_frame,
                [self.roi_cache["secondary"]["polygon"]],
                True,
                (0, 255, 0),
                2,
            )

        # Update tracking & counts
        self.queue_count = self._update_person_tracking(queue_dets, "queue")
        self.counter_count = self._update_person_tracking(counter_dets, "counter")
        
        # Second-level fallback: If counter count is still 0 after tracking, try uniform detection again
        # This catches cases where person detection found people but they weren't counted (tracking/dwell time issue)
        # This is especially useful for camera_2 where person detection sometimes fails
        if (self.use_uniform_fallback 
            and self.uniform_detector 
            and self.counter_count == 0):
            counter_roi = self.roi_cache.get("secondary")
            if counter_roi and counter_roi.get("polygon") is not None:
                try:
                    # Run uniform detection (with caching for performance)
                    if (
                        self.uniform_detection_cache is None
                        or self.frame_count - self.uniform_cache_frame_count >= self.uniform_cache_interval
                    ):
                        uniform_results = self.uniform_detector(
                            frame,
                            conf=self.detector.confidence_threshold,
                            iou=0.45,
                            verbose=False
                        )
                        self.uniform_detection_cache = uniform_results
                        self.uniform_cache_frame_count = self.frame_count
                    else:
                        uniform_results = self.uniform_detection_cache
                    
                    # Extract uniform detections in counter ROI
                    uniform_dets_in_counter = []
                    if uniform_results and len(uniform_results) > 0:
                        boxes = uniform_results[0].boxes
                        class_names = uniform_results[0].names
                        for box in boxes:
                            class_id = int(box.cls[0])
                            class_name = class_names[class_id]
                            conf = float(box.conf[0])
                            
                            # Only consider uniform classes
                            if class_name in self.uniform_classes:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                bbox = [int(x1), int(y1), int(x2), int(y2)]
                                center = [(x1 + x2) / 2, (y1 + y2) / 2]
                                bottom_center = [center[0], y2]  # Bottom center point
                                
                                # Check if uniform is in counter ROI
                                if self._point_in_polygon_optimized(
                                    bottom_center, 
                                    counter_roi["polygon"], 
                                    counter_roi["bbox"]
                                ):
                                    uniform_dets_in_counter.append({
                                        "bbox": bbox,
                                        "center": center,
                                        "confidence": conf,
                                        "class_name": class_name,
                                        "from_uniform": True
                                    })
                    
                    # If we found uniforms in counter area, directly set counter count
                    if uniform_dets_in_counter:
                        logger.warning(
                            f"[{self.channel_id}] 🔄 Second-level fallback: Found {len(uniform_dets_in_counter)} uniform(s) in counter area "
                            f"after tracking returned 0. Setting counter_count directly."
                        )
                        # Directly set counter count based on uniform detections
                        # This bypasses tracking/dwell time issues
                        self.counter_count = len(uniform_dets_in_counter)
                except Exception as e:
                    logger.warning(f"[{self.channel_id}] ⚠️ Second-level uniform fallback failed: {e}")
        
        # Store current detections for alert checking
        self._last_counter_dets = counter_dets
        self._last_queue_dets = queue_dets
        
        # Debug logging for counter tracking
        if self.frame_count <= 10 or self.frame_count % 30 == 0:
            logger.info(f"[{self.channel_id}] Frame {self.frame_count} tracking results:")
            logger.info(f"  Counter detections passed to tracking: {len(counter_dets)}")
            logger.info(f"  Counter count after tracking: {self.counter_count}")
            logger.info(f"  Dwell time threshold: {self.settings.get('dwell_time_threshold', 0.0)}")
            if len(counter_dets) > 0 and self.counter_count == 0:
                logger.warning(f"  ⚠️ WARNING: {len(counter_dets)} counter detections but count is 0! (dwell time filtering?)")
            if self.counter_count == 0 and self.use_uniform_fallback:
                logger.info(f"  Uniform fallback enabled: {self.uniform_detector is not None}")

        # Overlay counts
        cv2.putText(
            original_frame,
            f"Queue: {self.queue_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            original_frame,
            f"Counter: {self.counter_count}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        # Periodic debug logging for violation detection
        if self.frame_count % 60 == 0:  # Every 60 frames
            logger.info(f"[{self.channel_id}] 📊 Queue Monitor Status (Frame {self.frame_count}):")
            logger.info(f"  Queue count: {self.queue_count}, Counter count: {self.counter_count}")
            logger.info(f"  Settings: queue_threshold={self.settings.get('queue_alert_threshold', 3)}, wait_threshold={self.settings.get('wait_time_threshold', 3.0)}s")
            logger.info(f"  Last alert: {self.last_alert_time.isoformat() if self.last_alert_time else 'Never'}")
            if self.alert_condition_start_time:
                sustained = (datetime.now() - self.alert_condition_start_time).total_seconds()
                logger.info(f"  Condition sustained: {sustained:.2f}s / {self.alert_condition_sustained_duration}s")
        
        # Handle GIF recording state
        was_recording = self.gif_recorder.is_recording_alert
        
        if was_recording:
            # Add frame during alert recording
            self.gif_recorder.add_alert_frame(original_frame)
            # stop_alert_recording() is called automatically by add_alert_frame when duration is reached
        
        # Check if recording just finished (was recording, now stopped)
        if self._was_recording_alert and not self.gif_recorder.is_recording_alert:
            # Recording just finished - get GIF info and save to DB
            gif_info = self.gif_recorder.get_last_gif_info()
            if gif_info and self.db_manager and self._pending_alert_info:
                try:
                    # Extract filename from path
                    gif_path = gif_info.get('gif_path', '')
                    gif_filename = os.path.basename(gif_path) if gif_path else f"queue_alert_{self.channel_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.gif"
                    
                    # Convert absolute path to relative path for database (e.g., static/alerts/alert_xxx.gif)
                    if gif_path:
                        # Normalize path separators
                        normalized_path = gif_path.replace('\\', '/')
                        # Extract relative path (static/alerts/...)
                        if 'static/' in normalized_path:
                            snapshot_path = normalized_path.split('static/')[1]
                            snapshot_path = f"static/{snapshot_path}"
                        else:
                            snapshot_path = gif_path
                    else:
                        snapshot_path = None
                    
                    gif_payload = {
                        'gif_filename': gif_filename,
                        'gif_path': gif_path,
                        'frame_count': gif_info.get('frame_count', 0),
                        'duration': gif_info.get('duration', 0.0)
                    }
                    
                    alert_info = self._pending_alert_info
                    alert_message = alert_info.get('message', 'Queue violation detected')
                    
                    # Save GIF to alert_gifs table
                    if self.app:
                        with self.app.app_context():
                            self.db_manager.save_alert_gif(
                                self.channel_id,
                                'queue_alert',
                                gif_payload,
                                alert_message=alert_message,
                                alert_data=alert_info
                            )
                            
                            # Update queue_violations with GIF path for dashboard compatibility
                            if snapshot_path and self._pending_violation_ids:
                                for violation_id in self._pending_violation_ids:
                                    try:
                                        # Update the violation's snapshot_path using db_manager
                                        violation = self.db_manager.QueueViolation.query.get(violation_id)
                                        if violation:
                                            violation.snapshot_path = snapshot_path
                                            # Also update snapshot_filename for consistency
                                            if gif_filename:
                                                violation.snapshot_filename = gif_filename
                                            # Update file_size if GIF exists
                                            if gif_path and os.path.exists(gif_path):
                                                violation.file_size = os.path.getsize(gif_path)
                                            self.db_manager.db.session.commit()
                                            logger.info(f"[{self.channel_id}] ✅ Updated violation {violation_id} with GIF path: {snapshot_path}")
                                        else:
                                            logger.warning(f"[{self.channel_id}] ⚠️ Violation {violation_id} not found for update")
                                    except Exception as update_error:
                                        logger.warning(f"[{self.channel_id}] ⚠️ Could not update violation {violation_id} with GIF path: {update_error}", exc_info=True)
                    else:
                        self.db_manager.save_alert_gif(
                            self.channel_id,
                            'queue_alert',
                            gif_payload,
                            alert_message=alert_message,
                            alert_data=alert_info
                        )
                    
                    logger.info(f"[{self.channel_id}] ✅ Queue alert GIF saved to database: {gif_filename}")
                    # Clear stored alert info and violation IDs
                    self._pending_alert_info = None
                    self._pending_violation_ids = []
                except Exception as e:
                    logger.error(f"[{self.channel_id}] ❌ Failed to save queue alert GIF to DB: {e}", exc_info=True)
        
        # Track recording state for next frame
        self._was_recording_alert = was_recording
        
        # Check for alerts and violations
        alert_info = self._check_alert_conditions()
        if alert_info:
            logger.warning(f"[{self.channel_id}] 🚨 Queue alert triggered: {alert_info['message']}")
            logger.info(f"[{self.channel_id}] Alert details: queue={self.queue_count}, counter={self.counter_count}, wait={alert_info.get('max_wait_seconds', 0):.1f}s")
            
            # Start GIF recording for this alert
            self.gif_recorder.start_alert_recording(alert_info)
            # Store alert_info for database saving when GIF completes
            self._pending_alert_info = alert_info
            # Clear any previous violation IDs
            self._pending_violation_ids = []
            
            # Still save violations to database (but without static snapshot)
            # The GIF will be saved separately when recording completes
            # Store violation IDs so we can update them with GIF path later
            self._save_violations(original_frame, alert_info)
            
            # Socket.IO alert
            self.socketio.emit("queue_alert", alert_info)

            # Optionally send Telegram
            try:
                msg_lines = [
                    "⚠️ Queue Violation",
                    f"Channel: {self.channel_id}",
                    f"Queue: {self.queue_count}",
                    f"Counter: {self.counter_count}",
                    f"Issues: {alert_info['message']}",
                ]
                send_telegram_message("\n".join(msg_lines))
            except Exception as e:
                logger.error(f"Failed to send Telegram alert: {e}")

            # Optional DB logging
            if self.db_manager:
                try:
                    if self.app:
                        from flask import has_app_context

                        if has_app_context():
                            with self.app.app_context():
                                self.db_manager.log_queue_analytics(
                                    self.channel_id,
                                    self.queue_count,
                                    self.counter_count,
                                    alert_triggered=True,
                                    alert_message=alert_info["message"],
                                )
                    else:
                        self.db_manager.log_queue_analytics(
                            self.channel_id,
                            self.queue_count,
                            self.counter_count,
                            alert_triggered=True,
                            alert_message=alert_info["message"],
                        )
                except Exception as e:
                    logger.error(f"DB logging error: {e}")

        # Periodic DB + Socket.IO updates
        if self.frame_count % self.db_log_interval == 0 and self.db_manager:
            try:
                if self.app:
                    from flask import has_app_context

                    if has_app_context():
                        with self.app.app_context():
                            self.db_manager.log_queue_analytics(
                                self.channel_id,
                                self.queue_count,
                                self.counter_count,
                            )
                else:
                    self.db_manager.log_queue_analytics(
                        self.channel_id,
                        self.queue_count,
                        self.counter_count,
                    )
            except Exception as e:
                logger.error(f"DB logging error: {e}")

        if self.frame_count % self.socketio_update_interval == 0:
            self.socketio.emit(
                "queue_update",
                {
                    "channel_id": self.channel_id,
                    "queue_count": self.queue_count,
                    "counter_count": self.counter_count,
                },
            )

        return original_frame

    # ------------------------------------------------------------------ #
    # Status helpers
    # ------------------------------------------------------------------ #

    def get_current_status(self):
        return {
            "queue_count": self.queue_count,
            "counter_count": self.counter_count,
            "alert_threshold": self.settings["queue_alert_threshold"],
            "last_alert": self.last_alert_time.isoformat()
            if self.last_alert_time
            else None,
            "roi_configured": bool(self.roi_config["main"] or self.roi_config["secondary"]),
        }

    def get_status(self):
        return {
            "module": "QueueMonitor",
            "channel_id": self.channel_id,
            "status": "active",
            "queue_count": self.queue_count,
            "counter_count": self.counter_count,
            "settings": self.settings,
            "roi_config": self.roi_config,
            "frame_count": self.frame_count,
            "tracked_persons": len(self.person_tracking),
        }
