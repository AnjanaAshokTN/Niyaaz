"""
Dress Code Monitoring Module for Sakshi.AI
Detects employee uniform compliance including:
- Uniform colors (grey, black, beige, blue, red)
- Shirt tucking status
- Black shoes requirement
- Name tag visibility
- ID badge visibility
"""

import time
import cv2
import numpy as np
import logging
import torch
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO
from modules.gif_recorder import AlertGifRecorder
from .model_manager import get_shared_model, release_shared_model

logger = logging.getLogger(__name__)


class DressCodeMonitoring:
    """Employee dress code compliance monitoring with violation alerts"""
    
    def __init__(self, channel_id, socketio, db_manager=None, app=None):
        """
        Initialize dress code monitoring module
        
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
        
        # Dress code / uniform classes (from your best.pt)
        # Base classes - Uniform_grey is now available in the model (class 13)
        self.uniform_classes = {
            "Uniform_black", "Uniform_blue", "Uniform_cream", "Uniform_grey"
        }
        
        # ROI configuration (for area-specific detection rules)
        self.counter_roi = []  # Counter area - use best.pt for uniform detection
        self.queue_roi = []    # Queue area - use YOLOv11 for person detection
        
        # Per-camera allowed uniform colors in counter area
        # Format: {'counter': ['Uniform_grey', 'Uniform_black']} for Camera 1
        #         {'counter': ['Uniform_grey']} for Camera 2
        self.allowed_uniforms_in_counter = {
            'counter': []  # Empty = no restrictions, all uniforms allowed
        }
        
        # Dual YOLO models: best.pt for counter (uniforms), YOLOv11 for queue (persons)
        from .yolo_detector import YOLODetector
        
        # Model for counter area - uniform detection using best.pt
        self.uniform_detector = get_shared_model(self.model_weight, device='auto')
        if self.uniform_detector is None:
            raise RuntimeError(f"Failed to load uniform detection model: {self.model_weight}")
        
        # Model for queue area - person detection using YOLOv11
        self.person_detector = YOLODetector(
            model_path="models/yolo11n.pt",
            confidence_threshold=0.5,
            img_size=640,
            person_class_id=0  # Person class in YOLOv11
        )
        
        logger.info("Dual YOLO models initialized: best.pt for counter (uniforms), YOLOv11 for queue (persons)")
        # Core compliance-related classes we care about for dress/PPE:
        # - Apron vs No_apron
        # - Gloves vs No_gloves
        # - Hairnet
        # (ID/name tag/shoes can be added later if your model has them)
        self.compliance_classes = {
            "Apron", "No_apron",
            "Gloves", "No_gloves",
            "Hairnet"
        }
        # Hairnet configuration (class names from your model)
        self.hairnet_positive_class = "Hairnet"
        # Your model does not have an explicit "No_hairnet" class, so leave this empty
        self.hairnet_violation_class = ""
        # Hairnet checks are enabled by default, but not strictly required
        self.enable_hairnet_check = True
        self.require_hairnet = False  # Can be overridden per camera via config
        
        # Uniform-only mode: if True, only check for uniform colors, skip all PPE checks
        self.uniform_only = False
        
        # Required compliance items (logical requirements, not direct classes).
        # These defaults can be overridden per camera via config['required_items'].
        self.required_items = {
            "apron": False,
            "gloves": False,
            "hairnet": False
        }
        
        # Alert configuration
        self.alert_cooldown = 30.0  # seconds between alerts for same violation
        self.violation_duration_threshold = 0.5  # seconds to confirm violation (reduced for faster alerts)
        
        # State tracking - track violations by uniform position (rounded to grid for stability)
        self.last_alert_time = {}  # position_key -> timestamp
        self.violation_start_time = {}  # position_key -> timestamp
        self.current_violations = {}  # position_key -> list of violations
        
        # Statistics
        self.total_checks = 0
        self.total_violations = 0
        self.total_compliant = 0
        
        # Frame counter
        self.frame_count = 0
        
        # Note: YOLO models are initialized above in ROI configuration section
        # Log available classes in the uniform model and verify expected classes
        try:
            if hasattr(self.uniform_detector, 'names'):
                available_classes = list(self.uniform_detector.names.values())
                logger.info(f"📋 Uniform model (best.pt) has {len(available_classes)} classes: {available_classes}")
                
                # Verify Uniform_grey is available (should be class 13 in new model)
                if "Uniform_grey" in available_classes:
                    if "Uniform_grey" not in self.uniform_classes:
                        self.uniform_classes.add("Uniform_grey")
                    logger.info(f"✅ Uniform_grey confirmed in model (class {available_classes.index('Uniform_grey') if 'Uniform_grey' in available_classes else 'N/A'})")
                else:
                    logger.warning(f"⚠️ Uniform_grey not found in model classes!")
                
                # Check which expected classes are missing
                hairnet_expected = {
                    self.hairnet_positive_class
                } | ({self.hairnet_violation_class} if self.hairnet_violation_class else set())
                all_expected = self.uniform_classes | self.compliance_classes | hairnet_expected
                missing = all_expected - set(available_classes)
                if missing:
                    logger.warning(f"⚠️ Uniform model is missing expected classes: {missing}")
                
                logger.info(f"📋 Final uniform_classes: {self.uniform_classes}")
        except Exception as e:
            logger.error(f"Error checking uniform model classes: {e}")
        
        # Initialize GIF recorder for violation snapshots
        self.gif_recorder = AlertGifRecorder(fps=5)
        
        # Load saved configuration (counter ROI and allowed uniforms)
        self.load_configuration()
        
        logger.info(f"Dress Code Monitoring initialized for channel {channel_id}")
    
    def set_counter_roi(self, roi_points):
        """
        Set counter area ROI for uniform compliance checking
        
        Args:
            roi_points: List of normalized points [{'x': 0.0-1.0, 'y': 0.0-1.0}, ...]
        """
        self.counter_roi = roi_points
        logger.info(f"Counter ROI set for channel {self.channel_id}: {len(roi_points)} points")
        
        # Save to database if available
        if self.db_manager:
            try:
                from flask import has_app_context
                if has_app_context():
                    self.db_manager.save_channel_config(
                        self.channel_id, 'DressCodeMonitoring', 'counter_roi', roi_points
                    )
            except Exception as e:
                logger.error(f"Failed to save counter ROI: {e}")
    
    def set_queue_roi(self, roi_points):
        """
        Set queue area ROI for person detection
        
        Args:
            roi_points: List of normalized points [{'x': 0.0-1.0, 'y': 0.0-1.0}, ...]
        """
        self.queue_roi = roi_points
        logger.info(f"Queue ROI set for channel {self.channel_id}: {len(roi_points)} points")
        
        # Save to database if available
        if self.db_manager:
            try:
                from flask import has_app_context
                if has_app_context():
                    self.db_manager.save_channel_config(
                        self.channel_id, 'DressCodeMonitoring', 'queue_roi', roi_points
                    )
            except Exception as e:
                logger.error(f"Failed to save queue ROI: {e}")
    
    def set_allowed_uniforms(self, allowed_uniforms):
        """
        Set allowed uniform colors for counter area
        
        Args:
            allowed_uniforms: Dict with area -> list of allowed uniform class names
                Example: {'counter': ['Uniform_grey', 'Uniform_black']}
        """
        self.allowed_uniforms_in_counter = allowed_uniforms
        logger.info(f"Allowed uniforms set for channel {self.channel_id}: {allowed_uniforms}")
        
        # Save to database if available
        if self.db_manager:
            try:
                from flask import has_app_context
                if has_app_context():
                    self.db_manager.save_channel_config(
                        self.channel_id, 'DressCodeMonitoring', 'allowed_uniforms', allowed_uniforms
                    )
            except Exception as e:
                logger.error(f"Failed to save allowed uniforms: {e}")
    
    def load_configuration(self):
        """Load saved configuration from database"""
        if self.db_manager:
            try:
                from flask import has_app_context
                if not has_app_context():
                    return
                
                # Load counter ROI
                counter_roi = self.db_manager.get_channel_config(
                    self.channel_id, 'DressCodeMonitoring', 'counter_roi'
                )
                if counter_roi:
                    self.counter_roi = counter_roi
                
                # Load allowed uniforms
                allowed_uniforms = self.db_manager.get_channel_config(
                    self.channel_id, 'DressCodeMonitoring', 'allowed_uniforms'
                )
                if allowed_uniforms:
                    self.allowed_uniforms_in_counter = allowed_uniforms
                
                logger.info(f"Configuration loaded for channel {self.channel_id}")
            except Exception as e:
                logger.error(f"Failed to load configuration: {e}")
    
    def point_in_polygon(self, point, polygon):
        """Check if a point is inside a polygon (using ray casting algorithm)"""
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
    
    def get_counter_roi_pixels(self, frame_width, frame_height):
        """Convert normalized counter ROI coordinates to pixel coordinates"""
        if not self.counter_roi:
            return []
        
        pixel_points = []
        for point in self.counter_roi:
            x = int(point['x'] * frame_width)
            y = int(point['y'] * frame_height)
            pixel_points.append((x, y))
        return pixel_points
    
    def get_queue_roi_pixels(self, frame_width, frame_height):
        """Convert normalized queue ROI coordinates to pixel coordinates"""
        if not self.queue_roi:
            return []
        
        pixel_points = []
        for point in self.queue_roi:
            x = int(point['x'] * frame_width)
            y = int(point['y'] * frame_height)
            pixel_points.append((x, y))
        return pixel_points
    
    def is_uniform_in_counter(self, uniform_bbox, frame_width, frame_height):
        """
        Check if a uniform detection is within the counter ROI
        
        Args:
            uniform_bbox: Bounding box [x, y, w, h]
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            
        Returns:
            bool: True if uniform is in counter area
        """
        if not self.counter_roi:
            return False  # No counter ROI configured
        
        # Convert ROI to pixel coordinates
        counter_polygon = self.get_counter_roi_pixels(frame_width, frame_height)
        if not counter_polygon:
            return False
        
        # Check if center of uniform bbox is in counter ROI
        x, y, w, h = uniform_bbox
        center_x = int(x + w / 2)
        center_y = int(y + h / 2)
        
        return self.point_in_polygon((center_x, center_y), counter_polygon)
    
    def is_person_in_queue(self, person_bbox, frame_width, frame_height):
        """
        Check if a person detection is within the queue ROI
        
        Args:
            person_bbox: Bounding box [x, y, w, h]
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            
        Returns:
            bool: True if person is in queue area
        """
        if not self.queue_roi:
            return False  # No queue ROI configured
        
        # Convert ROI to pixel coordinates
        queue_polygon = self.get_queue_roi_pixels(frame_width, frame_height)
        if not queue_polygon:
            return False
        
        # Check if center of person bbox is in queue ROI
        x, y, w, h = person_bbox
        center_x = int(x + w / 2)
        center_y = int(y + h / 2)
        
        return self.point_in_polygon((center_x, center_y), queue_polygon)
    
    def __del__(self):
        """Cleanup: release shared model reference"""
        if hasattr(self, 'uniform_detector') and self.uniform_detector:
            try:
                release_shared_model(self.model_weight)
                logger.info(f"Released uniform detection model: {self.model_weight}")
            except Exception as e:
                logger.error(f"Error releasing uniform model: {e}")
    
    def check_compliance(self, detections, uniform_bbox=None, frame_width=None, frame_height=None, in_counter=False):
        """
        Check if employee meets dress code requirements
        
        Args:
            detections: List of detected objects with class names
            uniform_bbox: Bounding box of uniform detection [x, y, w, h] (optional)
            frame_width: Frame width in pixels (optional, for ROI checking)
            frame_height: Frame height in pixels (optional, for ROI checking)
            in_counter: Whether uniform is detected in counter area (optional, auto-detected if bbox provided)
            
        Returns:
            dict: {
                'is_compliant': bool,
                'violations': list of violation strings,
                'uniform_color': str or None,
                'detected_items': dict,
                'in_counter': bool
            }
        """
        # Extract detected classes
        detected_classes = {d['class_name'] for d in detections}
        
        # Find uniform color (based on your Uniform_* classes)
        uniform_color = None
        for uniform in self.uniform_classes:
            if uniform in detected_classes:
                uniform_color = uniform
                break
        
        # Check if uniform is in counter area (if bbox provided)
        in_counter = False
        if uniform_bbox and frame_width and frame_height:
            in_counter = self.is_uniform_in_counter(uniform_bbox, frame_width, frame_height)
        
        violations = []
        
        # Check counter area uniform restrictions
        if in_counter and uniform_color:
            allowed_in_counter = self.allowed_uniforms_in_counter.get('counter', [])
            if allowed_in_counter:  # If restrictions are configured
                # Check if uniform matches any allowed uniform
                is_allowed = uniform_color in allowed_in_counter
                
                if not is_allowed:
                    violation_msg = f"Wrong uniform at counter (allowed: {', '.join(allowed_in_counter)})"
                    violations.append(violation_msg)
                    logger.warning(f"🚨 [{self.channel_id}] Uniform violation at counter: {uniform_color} not in allowed list {allowed_in_counter}")
                else:
                    if self.frame_count % 100 == 0:
                        logger.debug(f"[{self.channel_id}] Uniform {uniform_color} is allowed in counter area")
            else:
                # No restrictions configured - log warning to help user configure
                if self.frame_count % 200 == 0:
                    logger.warning(f"[{self.channel_id}] ⚠️ Uniform {uniform_color} detected in counter but no restrictions configured (all uniforms allowed) - configure allowed_uniforms to enable violation detection!")
        elif uniform_color and not in_counter:
            if self.frame_count % 100 == 0:
                logger.debug(f"[{self.channel_id}] Uniform {uniform_color} detected but not in counter area (no violation check)")
        elif in_counter and not uniform_color:
            if self.frame_count % 100 == 0:
                logger.debug(f"[{self.channel_id}] In counter area but no uniform color detected")
        
        # PPE / dress-related detections from your model
        detected_items = {
            'apron': 'Apron' in detected_classes,
            'no_apron': 'No_apron' in detected_classes,
            'gloves': 'Gloves' in detected_classes,
            'no_gloves': 'No_gloves' in detected_classes,
            'hairnet': self.hairnet_positive_class in detected_classes if self.enable_hairnet_check else False,
        }
        
        # If uniform_only mode is enabled, skip all PPE checks
        if not self.uniform_only:
            # Apron logic
            if detected_items['no_apron']:
                violations.append("Apron not worn")
            elif self.required_items.get('apron', False) and uniform_color and not detected_items['apron']:
                violations.append("Apron not detected")
            
            # Gloves logic
            if detected_items['no_gloves']:
                violations.append("Gloves not worn")
            elif self.required_items.get('gloves', False) and uniform_color and not detected_items['gloves']:
                violations.append("Gloves not detected")
            
            # Hairnet logic
            if self.enable_hairnet_check:
                # If an explicit violation class is configured and detected, honour it
                if self.hairnet_violation_class and self.hairnet_violation_class in detected_classes:
                    violations.append("Hairnet not worn")
                # Otherwise, if hairnet is required, check for positive detection
                elif self.require_hairnet and uniform_color and not detected_items['hairnet']:
                    violations.append("Hairnet not detected")
        
        is_compliant = len(violations) == 0 and uniform_color is not None
        
        return {
            'is_compliant': is_compliant,
            'violations': violations,
            'uniform_color': uniform_color,
            'detected_items': detected_items,
            'in_counter': in_counter
        }
    
    def _extract_detections(self, results):
        """Extract detection information from YOLO results"""
        detections = []
        
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                boxes = result.boxes
                
                for i in range(len(boxes)):
                    box = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls = int(boxes.cls[i].cpu().numpy())
                    class_name = result.names[cls]
                    
                    x1, y1, x2, y2 = box
                    w = x2 - x1
                    h = y2 - y1
                    
                    detections.append({
                        'bbox': [x1, y1, w, h],
                        'confidence': conf,
                        'class_id': cls,
                        'class_name': class_name
                    })
        
        return detections
    
    def _save_violation_snapshot(self, frame, violations, uniform_color, position_key):
        """Save snapshot of dress code violation"""
        try:
            timestamp = datetime.now()
            
            # Generate unique filename
            filename = f"dresscode_{self.channel_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
            filepath = Path('static/dresscode_snapshots') / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Normalize path to use forward slashes for web compatibility
            snapshot_path_str = str(filepath).replace('\\', '/')
            
            logger.info(f"[{self.channel_id}] Saving violation snapshot to: {filepath}")
            
            # Ensure violations is a list
            if isinstance(violations, str):
                violations = [violations]
            elif not isinstance(violations, list):
                violations = list(violations) if violations else []
            
            # Add violation text to frame
            annotated_frame = frame.copy()
            y_offset = 30
            cv2.putText(annotated_frame, "DRESS CODE VIOLATION", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            y_offset += 30
            if uniform_color:
                cv2.putText(annotated_frame, f"Uniform: {uniform_color}", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 25
            
            for violation in violations[:3]:  # Show max 3 violations
                cv2.putText(annotated_frame, f"- {violation}", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                y_offset += 25
            
            # Save snapshot
            success = cv2.imwrite(str(filepath), annotated_frame)
            if not success:
                logger.error(f"[{self.channel_id}] Failed to write snapshot file: {filepath}")
                return None
            
            logger.info(f"[{self.channel_id}] Snapshot file written: {filepath} (exists: {filepath.exists()})")
            
            # Store in database - always use app context
            if self.db_manager and self.app:
                try:
                    # Always use app context when saving to database
                    # Push context to ensure it's available for the entire operation
                    ctx = self.app.app_context()
                    ctx.push()
                    try:
                        alert_obj = self.db_manager.add_dresscode_alert(
                            channel_id=self.channel_id,
                            violations=', '.join(violations),
                            uniform_color=uniform_color or 'Unknown',
                            snapshot_path=snapshot_path_str,
                            employee_id=None
                        )
                        if alert_obj and hasattr(alert_obj, 'id'):
                            logger.info(f"[{self.channel_id}] ✅ Dress code violation saved to database: ID {alert_obj.id}, filename: {filename}")
                        elif alert_obj:
                            logger.info(f"[{self.channel_id}] ✅ Dress code violation saved to database: {alert_obj}, filename: {filename}")
                        else:
                            logger.warning(f"[{self.channel_id}] ⚠️ Database save returned None (alert may not be saved)")
                    finally:
                        ctx.pop()
                except Exception as e:
                    logger.error(f"[{self.channel_id}] ❌ Failed to save violation to database: {e}", exc_info=True)
                    import traceback
                    logger.error(traceback.format_exc())
            
            return snapshot_path_str
        except Exception as e:
            logger.error(f"[{self.channel_id}] Error in _save_violation_snapshot: {e}", exc_info=True)
            return None
    
    def _send_alert(self, violations, uniform_color, snapshot_path, position_key):
        """Send real-time alert via SocketIO"""
        try:
            # Ensure violations is a list
            if isinstance(violations, str):
                violations_list = [violations]
            elif isinstance(violations, list):
                violations_list = violations
            else:
                violations_list = list(violations) if violations else []
            
            if self.socketio:
                alert_data = {
                    'channel_id': self.channel_id,
                    'violations': violations_list,
                    'violations_text': ', '.join(violations_list),
                    'uniform_color': uniform_color or 'Unknown',
                    'snapshot': snapshot_path or '',
                    'snapshot_url': f"/static/dresscode_snapshots/{Path(snapshot_path).name}" if snapshot_path else '',
                    'timestamp': datetime.now().isoformat()
                }
                
                try:
                    self.socketio.emit('dresscode_violation', alert_data, namespace='/')
                    logger.info(f"[{self.channel_id}] Dress code alert sent via SocketIO")
                    logger.info(f"   Alert data: {alert_data}")
                except Exception as e:
                    logger.error(f"[{self.channel_id}] Failed to send SocketIO alert: {e}", exc_info=True)
            else:
                logger.warning(f"[{self.channel_id}] SocketIO not available, cannot send alert")
        except Exception as e:
            logger.error(f"[{self.channel_id}] Error in _send_alert: {e}", exc_info=True)
    
    def process_frame(self, frame):
        """
        Process frame for dress code compliance monitoring
        
        Args:
            frame: Input video frame
            
        Returns:
            Annotated frame with detection boxes and compliance status
        """
        if frame is None:
            return None
        
        self.frame_count += 1
        t_now = time.time()
        
        # Add frame to GIF recorder
        self.gif_recorder.add_frame(frame)
        
        # Clear GPU cache periodically
        if torch.cuda.is_available() and self.frame_count % 100 == 0:
            torch.cuda.empty_cache()
        
        # Get frame dimensions for ROI checking
        frame_height, frame_width = frame.shape[:2]
        
        # Dual detection: best.pt for counter area, YOLOv11 for queue area
        all_detections = []
        
        # 1. Detect uniforms in counter area using best.pt
        if self.counter_roi:
            try:
                counter_results = self.uniform_detector(
                    frame,
                    conf=self.conf_threshold,
                    iou=self.nms_iou,
                    verbose=False
                )
                counter_detections = self._extract_detections(counter_results)
                
                # Log raw detections for debugging
                if self.frame_count <= 5 or (self.frame_count % 100 == 0 and len(counter_detections) > 0):
                    uniform_dets = [d for d in counter_detections if d['class_name'] in self.uniform_classes]
                    if uniform_dets:
                        logger.info(f"[{self.channel_id}] Raw uniform detections from best.pt: {[d['class_name'] for d in uniform_dets]}")
                
                # Filter detections to only those in counter ROI
                for det in counter_detections:
                    if det['class_name'] in self.uniform_classes or det['class_name'] in self.compliance_classes:
                        x, y, w, h = det['bbox']
                        in_counter_roi = self.is_uniform_in_counter([x, y, w, h], frame_width, frame_height)
                        if in_counter_roi:
                            det['area'] = 'counter'  # Mark as counter detection
                            all_detections.append(det)
                        # Log if uniform detected but not in ROI (for debugging)
                        elif det['class_name'] in self.uniform_classes and (self.frame_count <= 5 or self.frame_count % 100 == 0):
                            logger.debug(f"[{self.channel_id}] Uniform {det['class_name']} detected but outside counter ROI (bbox center: {int(x+w/2)}, {int(y+h/2)})")
            except Exception as e:
                logger.error(f"Uniform detection error (best.pt): {e}", exc_info=True)
        else:
            # Log if counter ROI is not configured
            if self.frame_count <= 5 or self.frame_count % 200 == 0:
                logger.warning(f"[{self.channel_id}] Counter ROI not configured - uniform detection in counter area will not work!")
        
        # 2. Detect persons in queue area using YOLOv11
        if self.queue_roi:
            try:
                queue_persons = self.person_detector.detect_persons(frame)
                
                # Filter detections to only those in queue ROI
                for person in queue_persons:
                    x1, y1, x2, y2 = person['bbox']
                    w = x2 - x1
                    h = y2 - y1
                    if self.is_person_in_queue([x1, y1, w, h], frame_width, frame_height):
                        # Convert person detection to standard format
                        all_detections.append({
                            'bbox': [x1, y1, w, h],
                            'confidence': person.get('confidence', 0.5),
                            'class_id': 0,  # Person class
                            'class_name': 'Person',
                            'area': 'queue'  # Mark as queue detection
                        })
            except Exception as e:
                logger.error(f"Person detection error (YOLOv11): {e}")
        
        detections = all_detections
        
        # Log detection details every 50 frames for debugging
        if self.frame_count % 50 == 0 or self.frame_count <= 10:
            logger.info(f"📊 [{self.channel_id}] DressCode Frame {self.frame_count}: Total detections: {len(detections)}")
            logger.info(f"   Counter ROI configured: {len(self.counter_roi) > 0}, Queue ROI configured: {len(self.queue_roi) > 0}")
            logger.info(f"   Allowed uniforms in counter: {self.allowed_uniforms_in_counter.get('counter', [])}")
            
            if len(detections) > 0:
                class_counts = {}
                area_counts = {'counter': 0, 'queue': 0, 'unknown': 0}
                for det in detections:
                    class_name = det['class_name']
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                    area = det.get('area', 'unknown')
                    area_counts[area] = area_counts.get(area, 0) + 1
                logger.info(f"   Detected classes: {class_counts}")
                logger.info(f"   Detections by area: {area_counts}")
                
                # Specifically log uniform detections in counter
                counter_uniforms = [det for det in detections 
                                   if det.get('area') == 'counter' and det['class_name'] in self.uniform_classes]
                if counter_uniforms:
                    logger.info(f"   👔 Uniforms in counter: {[d['class_name'] for d in counter_uniforms]}")
                
                # Specifically log compliance items
                compliance_detected = [cls for cls in class_counts.keys() 
                                     if cls in self.compliance_classes]
                if compliance_detected:
                    logger.info(f"   👔 Compliance items detected: {compliance_detected}")
        
        # Draw detection boxes
        annotated_frame = frame.copy()
        frame_height, frame_width = frame.shape[:2]
        
        # Draw ROIs
        if self.counter_roi:
            counter_polygon = self.get_counter_roi_pixels(frame_width, frame_height)
            if counter_polygon:
                pts = np.array(counter_polygon, np.int32)
                cv2.polylines(annotated_frame, [pts], True, (255, 255, 0), 2)  # Cyan for counter
                cv2.putText(annotated_frame, "Counter (best.pt)", (counter_polygon[0][0], counter_polygon[0][1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        if self.queue_roi:
            queue_polygon = self.get_queue_roi_pixels(frame_width, frame_height)
            if queue_polygon:
                pts = np.array(queue_polygon, np.int32)
                cv2.polylines(annotated_frame, [pts], True, (0, 255, 255), 2)  # Yellow for queue
                cv2.putText(annotated_frame, "Queue (YOLOv11)", (queue_polygon[0][0], queue_polygon[0][1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Group detections by employee (based on proximity)
        # Only process uniform detections from counter area for compliance checking
        employee_groups = []
        for detection in detections:
            # Only check compliance for uniforms in counter area
            if detection['class_name'] in self.uniform_classes and detection.get('area') == 'counter':
                # This is an employee in counter - collect all nearby compliance items
                employee_detections = [detection]
                uniform_bbox = detection['bbox']
                
                # Find nearby compliance items (also in counter area)
                for other in detections:
                    if other['class_name'] in self.compliance_classes and other.get('area') == 'counter':
                        # Check if within reasonable distance
                        other_bbox = other['bbox']
                        if self._is_nearby(uniform_bbox, other_bbox):
                            employee_detections.append(other)
                
                employee_groups.append(employee_detections)
        
        # Check compliance for each uniform detection in counter area
        for idx, employee_detections in enumerate(employee_groups):
            # Find uniform bbox for ROI checking
            uniform_bbox = None
            uniform_detection = None
            for det in employee_detections:
                if det['class_name'] in self.uniform_classes:
                    uniform_bbox = det['bbox']
                    uniform_detection = det
                    break
            
            # Check compliance (with counter ROI checking) - need to do this first to get uniform color
            compliance = self.check_compliance(
                employee_detections,
                uniform_bbox=uniform_bbox,
                frame_width=frame_width,
                frame_height=frame_height
            )
            
            # Generate stable position key - use uniform color + channel for maximum stability
            # This prevents position_key from changing when uniform moves slightly
            if compliance.get('uniform_color'):
                # Use uniform color + channel as key (most stable approach)
                # This ensures the same uniform color on the same channel always uses the same key
                position_key = f"{self.channel_id}_{compliance['uniform_color']}"
            elif uniform_bbox:
                x, y, w, h = uniform_bbox
                center_x = int(x + w / 2)
                center_y = int(y + h / 2)
                # Use 200px grid for maximum stability
                stable_x = (center_x // 200) * 200
                stable_y = (center_y // 200) * 200
                position_key = f"{self.channel_id}_uniform_{stable_x}_{stable_y}"
            else:
                position_key = f"{self.channel_id}_detection_{idx}"
            
            # Update position_key with uniform color after compliance check for better tracking
            if uniform_bbox and compliance.get('uniform_color'):
                x, y, w, h = uniform_bbox
                center_x = int(x + w / 2)
                center_y = int(y + h / 2)
                stable_x = (center_x // 100) * 100
                stable_y = (center_y // 100) * 100
                position_key = f"{compliance['uniform_color']}_{stable_x}_{stable_y}"
            
            # Log all compliance checks for debugging (first few frames and periodically)
            if self.frame_count <= 10 or self.frame_count % 50 == 0:
                logger.info(f"[{self.channel_id}] Compliance check: Uniform={compliance['uniform_color']}, In counter={compliance.get('in_counter', False)}")
                logger.info(f"   Is compliant: {compliance['is_compliant']}, Violations: {compliance['violations']}")
                logger.info(f"   Allowed uniforms: {self.allowed_uniforms_in_counter.get('counter', [])}")
                logger.info(f"   Position key: {position_key}")
            
            # Draw bounding boxes for counter area uniforms (compliance checking)
            for detection in employee_detections:
                x, y, w, h = detection['bbox']
                x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
                
                # Color based on compliance
                if compliance['is_compliant']:
                    color = (0, 255, 0)  # Green for compliant
                    label_color = "COMPLIANT"
                else:
                    color = (0, 0, 255)  # Red for violations
                    label_color = "VIOLATION"
                
                # Draw box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"{detection['class_name']} {detection['confidence']:.2f}"
                cv2.putText(annotated_frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Handle violations
            if not compliance['is_compliant'] and compliance['uniform_color'] and compliance.get('in_counter', False):
                self.total_checks += 1
                
                # Check if this is a sustained violation
                if position_key not in self.violation_start_time:
                    # NEW violation - start tracking
                    self.violation_start_time[position_key] = t_now
                    self.current_violations[position_key] = compliance['violations']
                    logger.warning(f"🚨 [{self.channel_id}] NEW Violation detected: {compliance['violations']}")
                    logger.warning(f"   Uniform: {compliance['uniform_color']}, Position key: {position_key}")
                    logger.info(f"[{self.channel_id}] ✅ Violation tracking STARTED at {t_now:.2f}")
                else:
                    # Existing violation - check duration
                    violation_duration = t_now - self.violation_start_time[position_key]
                    
                    # Log violation progress every 5 frames for better visibility
                    if self.frame_count % 5 == 0:
                        logger.warning(f"[{self.channel_id}] ⏱️ Violation sustained: {violation_duration:.2f}s / {self.violation_duration_threshold}s")
                    
                    # If violation sustained and cooldown passed
                    if violation_duration >= self.violation_duration_threshold:
                        last_alert = self.last_alert_time.get(position_key, 0)
                        # Calculate time since last alert (if no previous alert, set to cooldown + 1 to pass)
                        if last_alert == 0:
                            time_since_last_alert = self.alert_cooldown + 1  # No previous alert, allow this one
                        else:
                            time_since_last_alert = t_now - last_alert
                        
                        # Always log alert check details when violation is sustained
                        logger.warning(f"[{self.channel_id}] 🔍 Alert check: duration={violation_duration:.2f}s (need {self.violation_duration_threshold}s) ✓, cooldown={time_since_last_alert:.2f}s (need {self.alert_cooldown}s), position_key={position_key}")
                        
                        if time_since_last_alert >= self.alert_cooldown:
                            logger.warning(f"[{self.channel_id}] ✅ Cooldown passed! Triggering alert NOW...")
                            # Trigger alert
                            logger.error(f"🚨🚨🚨 [{self.channel_id}] ⚠️ ALERT TRIGGERED: {compliance['violations']}")
                            logger.error(f"   Uniform: {compliance['uniform_color']}, Violation duration: {violation_duration:.2f}s")
                            logger.error(f"   Violations list: {compliance['violations']}")
                            logger.error(f"   Position key: {position_key}")
                            
                            self.total_violations += 1
                            
                            # Save snapshot
                            try:
                                snapshot_path = self._save_violation_snapshot(
                                    annotated_frame,
                                    compliance['violations'],
                                    compliance['uniform_color'],
                                    position_key
                                )
                                if snapshot_path:
                                    logger.info(f"[{self.channel_id}] ✅ Snapshot saved: {snapshot_path}")
                                else:
                                    logger.error(f"[{self.channel_id}] ❌ Snapshot save returned None")
                            except Exception as e:
                                logger.error(f"[{self.channel_id}] ❌ Failed to save snapshot: {e}", exc_info=True)
                                snapshot_path = None
                            
                            # Send alert
                            try:
                                self._send_alert(
                                    compliance['violations'],
                                    compliance['uniform_color'],
                                    snapshot_path,
                                    position_key
                                )
                                logger.info(f"[{self.channel_id}] ✅ Alert sent via SocketIO")
                            except Exception as e:
                                logger.error(f"[{self.channel_id}] ❌ Failed to send alert: {e}", exc_info=True)
                            
                            self.last_alert_time[position_key] = t_now
                        else:
                            # Still in cooldown
                            if self.frame_count % 5 == 0:
                                logger.warning(f"[{self.channel_id}] ⏳ Alert in cooldown: {time_since_last_alert:.2f}s / {self.alert_cooldown}s (need {self.alert_cooldown - time_since_last_alert:.2f}s more)")
                    else:
                        # Violation not yet sustained long enough
                        if self.frame_count % 10 == 0:
                            logger.warning(f"[{self.channel_id}] ⏳ Violation building: {violation_duration:.2f}s < {self.violation_duration_threshold}s")
            elif compliance['is_compliant'] and position_key in self.violation_start_time:
                # Violation cleared - reset tracking
                logger.info(f"[{self.channel_id}] ✅ Violation cleared - resetting tracking")
                del self.violation_start_time[position_key]
                if position_key in self.current_violations:
                    del self.current_violations[position_key]
        
        # Draw alert banner if there are active violations
        if self.current_violations:
            cv2.rectangle(annotated_frame, (0, 0), (frame.shape[1], 50), (0, 0, 255), -1)
            cv2.putText(annotated_frame, "DRESS CODE VIOLATION DETECTED", (10, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Return in the same format as other modules for compatibility with video streaming
        return {
            'frame': annotated_frame,
            'status': {
                'total_checks': self.total_checks,
                'total_violations': self.total_violations,
                'total_compliant': self.total_compliant,
                'active_violations': len(self.violation_start_time)
            },
            'metadata': {
                'frame_count': self.frame_count,
                'timestamp': t_now
            }
        }
    
    def _is_nearby(self, bbox1, bbox2, threshold=150):
        """Check if two bounding boxes are nearby (for grouping detections)"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate centers
        center1_x = x1 + w1 / 2
        center1_y = y1 + h1 / 2
        center2_x = x2 + w2 / 2
        center2_y = y2 + h2 / 2
        
        # Calculate distance
        distance = np.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)
        
        return distance < threshold
    
    def get_stats(self):
        """Get module statistics"""
        return {
            'total_checks': self.total_checks,
            'total_violations': self.total_violations,
            'total_compliant': self.total_compliant,
            'compliance_rate': (self.total_compliant / max(1, self.total_checks)) * 100
        }
