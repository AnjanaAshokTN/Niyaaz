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

logger = logging.getLogger(__name__)

# Optional Telegram integration (configured via environment variables)
TELEGRAM_BOT_TOKEN = os.getenv("bot_token")
TELEGRAM_CHAT_ID = os.getenv("chat_id")


def send_telegram_message(text: str) -> None:
    """
    Send a message to a Telegram chat.

    Uses environment variables:
      - bot_token
      - chat_id

    If these are not set, this is a no-op.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.debug("Telegram credentials not set; skipping Telegram notification")
        return

    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
        resp = requests.post(url, data=data, timeout=5)
        if resp.status_code != 200:
            logger.warning(f"Telegram API error: {resp.text}")
    except Exception as e:
        logger.error(f"Error sending Telegram message: {e}")


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
        # Uses your custom best.pt where Person = class 12:
        # {0: 'Apron', ..., 12: 'Person'}
        self.detector = YOLODetector(
            model_path="models/yolo11n.pt",
            # model_path="models/best.pt",
            confidence_threshold=0.25,
            img_size=640,
            person_class_id=0,
        )

        # ROI configuration (normalized 0–1 coordinates)
        self.roi_config = {
            "main": [],      # Queue area
            "secondary": []  # Counter area
        }

        # Settings (overridable via set_settings / DB)
        self.settings = {
            "dwell_time_threshold": 0.0,   # **CHANGED: Count immediately, no dwell time required (matches reference)**
            "queue_alert_threshold": 3,    # V1: queue > 3
            "counter_threshold": 1,        # V3: need at least 1 at counter
            "alert_cooldown": 60.0,        # seconds between alerts
            "wait_time_threshold": 3.0,    # V2: any wait >= 3s
        }

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
        self.detection_cache = None
        self.cache_frame_count = 0
        self.cache_interval = 2  # run YOLO every 2 frames
        self.socketio_update_interval = 15
        self.db_log_interval = 600

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
                # Update settings but force dwell_time_threshold to 0.0 for instant counting (matches reference)
                self.settings.update(settings)
                # Always force instant counting (dwell_time_threshold = 0.0) to match reference implementation
                old_value = self.settings.get("dwell_time_threshold", 0.0)
                if old_value != 0.0:
                    logger.info(f"[{self.channel_id}] Overriding dwell_time_threshold from {old_value} to 0.0 for instant counting")
                self.settings["dwell_time_threshold"] = 0.0  # Always set to 0.0

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

        self.roi_cache["main"] = _build_polygon("main")
        self.roi_cache["secondary"] = _build_polygon("secondary")
        self.roi_cache_frame_size = (frame_w, frame_h)

    @staticmethod
    def _point_in_polygon_optimized(point, polygon, bbox):
        """Fast ROI check using bbox + cv2.pointPolygonTest."""
        x, y = point
        (min_x, min_y, max_x, max_y) = bbox
        if x < min_x or x > max_x or y < min_y or y > max_y:
            return False
        return cv2.pointPolygonTest(polygon, (float(x), float(y)), False) >= 0

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

            # Use bottom-right corner (x2, y2) for ROI check - matches reference implementation
            # This is more accurate for determining if person's feet are in the ROI
            point_to_check = (x2, y2)  # Bottom-right corner

            # Ensure we have center for tracking (if needed)
            if "center" not in det:
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                det["center"] = (cx, cy)
                det["bottom_center"] = (cx, int(y2))

            # Check if in queue area first (reference uses if/elif, so person can only be in one area)
            if queue_roi and self._point_in_polygon_optimized(
                point_to_check, queue_roi["polygon"], queue_roi["bbox"]
            ):
                det["area_type"] = "queue"
                queue_dets.append(det)
            # Check if in counter area (only if not in queue)
            elif counter_roi and self._point_in_polygon_optimized(
                point_to_check, counter_roi["polygon"], counter_roi["bbox"]
            ):
                det["area_type"] = "counter"
                counter_dets.append(det)
            else:
                det["area_type"] = "none"
            
            # Debug logging for first few detections
            if self.frame_count <= 10 and len(detections) > 0:
                logger.info(f"  Detection: bbox={det['bbox']}, point_checked={point_to_check}, area={det['area_type']}")

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
                first_det = detections[0]
                bottom_center = first_det.get('bottom_center', 'N/A')
                logger.info(f"  First detection bottom_center: {bottom_center}")
                if counter_roi:
                    in_counter_check = self._point_in_polygon_optimized(
                        bottom_center, counter_roi["polygon"], counter_roi["bbox"]
                    ) if isinstance(bottom_center, tuple) else False
                    logger.info(f"  First detection in counter ROI: {in_counter_check}")
            if len(counter_dets) > 0:
                logger.info(f"  Counter detections (first 3):")
                for i, det in enumerate(counter_dets[:3]):
                    logger.info(f"    Counter det {i+1}: bbox={det['bbox']}, bottom_center={det.get('bottom_center', 'N/A')}")

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
        # Force instant counting (dwell_time_threshold = 0.0) to match reference implementation
        # This ensures counter people are counted immediately
        dwell_thresh = 0.0  # Always use instant counting
        
        # If dwell time = 0 → instant count
        if dwell_thresh == 0.0:
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
                }
                self.next_track_id += 1
                self.person_tracking.append(track)
                tracks.append(track)
            else:
                best_track["center"] = c
                best_track["last_seen"] = now_ts
                best_track["matched"] = True
                # Only update entered_roi_time if track was previously out of ROI
                if not best_track.get("in_roi", False):
                    # Track was out of ROI, now back in - reset entered_roi_time
                    best_track["entered_roi_time"] = now_ts
                    best_track["in_roi"] = True
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

        # Count people whose time in ROI exceeds dwell threshold
        count = 0
        for t in tracks:
            if t["in_roi"] and t["entered_roi_time"] is not None:
                time_in = now_ts - t["entered_roi_time"]
                if time_in >= dwell_thresh:
                    t["counted"] = True
            if t.get("counted"):
                count += 1
        
        # Debug logging for counter tracking
        if area_type == "counter" and (self.frame_count <= 10 or self.frame_count % 30 == 0):
            logger.info(f"[{self.channel_id}] Counter tracking debug (Frame {self.frame_count}):")
            logger.info(f"  Detections passed to tracking: {len(detections)}")
            logger.info(f"  Active tracks in counter area: {len(tracks)}")
            logger.info(f"  Dwell time threshold: {dwell_thresh}s")
            logger.info(f"  Final count: {count}")
            for i, t in enumerate(tracks[:5]):  # Show first 5 tracks
                time_in = (now_ts - t["entered_roi_time"]) if t.get("entered_roi_time") else 0
                logger.info(f"  Track {i+1}: in_roi={t.get('in_roi')}, entered_time={t.get('entered_roi_time')}, time_in={time_in:.2f}s, counted={t.get('counted')}")

        return count

    # ------------------------------------------------------------------ #
    # Violation saving
    # ------------------------------------------------------------------ #

    def _save_violation_snapshot(self, frame, violation_type, violation_message):
        """
        Save a snapshot of a queue violation
        
        Args:
            frame: Annotated frame to save
            violation_type: Type of violation ('queue_overflow', 'wait_time_exceeded', 'no_counter_staff')
            violation_message: Human-readable violation message
            
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
            
            # Add violation text overlay
            annotated = frame.copy()
            y_offset = 30
            cv2.putText(annotated, "QUEUE VIOLATION", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            y_offset += 30
            cv2.putText(annotated, violation_message, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25
            cv2.putText(annotated, f"Queue: {self.queue_count} | Counter: {self.counter_count}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            cv2.imwrite(str(snapshot_path), annotated)
            logger.info(f"Queue violation snapshot saved: {snapshot_path}")
            
            # Return full path for database storage
            return str(snapshot_path)
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
        
        current_time = datetime.now()
        message = alert_info.get("message", "")
        logger.info(f"[{self.channel_id}] Saving violations for message: {message}")
        
        # Parse violations from message (message can contain multiple violations separated by "; ")
        violations = []
        message_parts = [part.strip() for part in message.split(";")]
        
        for part in message_parts:
            if "Queue too long" in part or (">" in part and "queue" in part.lower()):
                violations.append(("queue_overflow", part))
            elif "wait time" in part.lower() or "≥" in part or ">=" in part:
                violations.append(("wait_time_exceeded", part))
            elif "No staff at counter" in part or ("counter" in part.lower() and "staff" in part.lower()):
                violations.append(("no_counter_staff", part))
        
        if not violations:
            # Fallback: save as general violation if parsing fails
            logger.warning(f"[{self.channel_id}] Could not parse violation types from message: {message}")
            violations.append(("queue_alert", message))
        
        logger.info(f"[{self.channel_id}] Parsed {len(violations)} violation(s) from alert message: {message}")
        
        # Save each violation type separately
        for violation_type, violation_message in violations:
            try:
                # Save snapshot
                snapshot_path = self._save_violation_snapshot(frame, violation_type, violation_message)
                
                # Save to database (with proper app context handling)
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
                            snapshot_path=snapshot_path,
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
                                snapshot_path=snapshot_path,
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
                        snapshot_path=snapshot_path,
                        alert_data=alert_info
                    )
                
                if violation_id:
                    logger.info(f"[{self.channel_id}] ✅ Queue violation saved to database: ID {violation_id}, type: {violation_type}, message: {violation_message}")
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
        - V2: any person in queue waited >= wait_time_threshold seconds
        - V3: queue_count > 0 and counter_count < counter_threshold
        """
        current_time = datetime.now()
        now_ts = current_time.timestamp()

        dwell_thresh = float(self.settings.get("dwell_time_threshold", 0.0) or 0.0)
        max_queue_without_violation = int(
            self.settings.get("queue_alert_threshold", 3) or 3
        )
        counter_required = int(self.settings.get("counter_threshold", 1) or 1)
        cooldown = float(self.settings.get("alert_cooldown", 60.0) or 60.0)
        wait_threshold = float(self.settings.get("wait_time_threshold", 3.0) or 3.0)

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
                f"Queue too long: {self.queue_count} > {max_queue_without_violation}"
            )
            logger.debug(f"[{self.channel_id}] V1 violation detected: queue_count={self.queue_count} > threshold={max_queue_without_violation}")

        # V2: Wait time exceeded
        if self.queue_count > 0 and max_wait >= wait_threshold:
            violations.append(
                f"Queue wait time {int(max_wait)}s ≥ {int(wait_threshold)}s"
            )
            logger.debug(f"[{self.channel_id}] V2 violation detected: max_wait={max_wait:.1f}s >= threshold={wait_threshold}s")

        # V3: No staff at counter
        if self.queue_count > 0 and self.counter_count < counter_required:
            violations.append(
                f"No staff at counter (queue={self.queue_count}, counter={self.counter_count})"
            )
            logger.debug(f"[{self.channel_id}] V3 violation detected: queue={self.queue_count} > 0, counter={self.counter_count} < required={counter_required}")

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

        # Classify into queue/counter
        queue_dets, counter_dets = self._classify_detections(detections, w, h)

        # Draw boxes
        for det in queue_dets:
            x1, y1, x2, y2 = det["bbox"]
            cv2.rectangle(original_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        for det in counter_dets:
            x1, y1, x2, y2 = det["bbox"]
            cv2.rectangle(original_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw ROIs
        self._update_roi_cache(w, h)
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
        
        # Debug logging for counter tracking
        if self.frame_count <= 10 or self.frame_count % 30 == 0:
            logger.info(f"[{self.channel_id}] Frame {self.frame_count} tracking results:")
            logger.info(f"  Counter detections passed to tracking: {len(counter_dets)}")
            logger.info(f"  Counter count after tracking: {self.counter_count}")
            logger.info(f"  Dwell time threshold: {self.settings.get('dwell_time_threshold', 0.0)}")
            if len(counter_dets) > 0 and self.counter_count == 0:
                logger.warning(f"  ⚠️ WARNING: {len(counter_dets)} counter detections but count is 0! (dwell time filtering?)")

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
        
        # Check for alerts and violations
        alert_info = self._check_alert_conditions()
        if alert_info:
            logger.warning(f"[{self.channel_id}] 🚨 Queue alert triggered: {alert_info['message']}")
            logger.info(f"[{self.channel_id}] Alert details: queue={self.queue_count}, counter={self.counter_count}, wait={alert_info.get('max_wait_seconds', 0):.1f}s")
            
            # Save violation snapshots and database records
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
