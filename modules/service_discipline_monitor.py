"""
Service Discipline Monitor
--------------------------

Event-based service monitoring:
    T_seated: Customer sits at a table
    T_order_start: Waiter first reaches that table and interacts with customer
    T_food_served: Waiter brings food (plate, tray, etc.) to that table

Metrics:
    Order wait time = T_order_start - T_seated
    Service wait time = T_food_served - T_order_start

Uses DeepSORT for persistent person tracking with track IDs.
"""

import cv2
import math
import logging
import numpy as np
import json
import time
import torch
from datetime import datetime
from pathlib import Path
from collections import deque

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    DEEPSORT_AVAILABLE = True
except ImportError:
    DEEPSORT_AVAILABLE = False
    logging.warning("DeepSort not available. Install with: pip install deep-sort-realtime")

from .model_manager import get_shared_model
from .yolo_detector import YOLODetector

logger = logging.getLogger(__name__)


class ServiceDisciplineMonitor:
    def __init__(self, channel_id, socketio, db_manager=None, app=None):
        self.channel_id = channel_id
        self.socketio = socketio
        self.db_manager = db_manager
        self.app = app

        # Model configuration
        # Use YOLOv11n.pt for person detection (better accuracy)
        # Use best.pt for uniform detection (has uniform classes)
        self.person_model_path = "models/yolo11n.pt"
        self.uniform_model_path = "models/best.pt"
        self.conf_threshold = 0.5
        self.nms_iou = 0.45

        # Uniform classes for server detection (from best.pt)
        self.server_uniform_classes = {
            "Uniform_black",
            "Uniform_grey",
            "Uniform_cream"
        }

        # Person class IDs
        self.person_class_id_yolo11n = 0  # Person class in YOLOv11n.pt
        self.person_class_id_best = 12    # Person class in best.pt (for reference)

        # Load both models
        # YOLOv11n for person detection (using YOLODetector for better person detection)
        self.person_detector = YOLODetector(
            model_path=self.person_model_path,
            confidence_threshold=0.25,  # Lower threshold for better person detection
            img_size=640,
            person_class_id=self.person_class_id_yolo11n
        )
        
        # best.pt for uniform detection and food/plate detection (using shared model manager)
        self.uniform_model = get_shared_model(self.uniform_model_path)
        
        logger.info(f"[{self.channel_id}] Loaded dual models: {self.person_model_path} (persons) + {self.uniform_model_path} (uniforms)")

        # Initialize DeepSORT tracker for persistent person tracking
        if DEEPSORT_AVAILABLE:
            logger.info(f"[{self.channel_id}] Initializing DeepSORT tracker")
            self.tracker = DeepSort(
                max_age=30,
                n_init=3,
                max_iou_distance=0.7,
                max_cosine_distance=0.3,
                nn_budget=50,
                embedder="mobilenet",
                embedder_gpu=True if torch.cuda.is_available() else False
            )
            self.tracking_enabled = True
        else:
            logger.warning(f"[{self.channel_id}] DeepSORT not available - using simple tracking")
            self.tracker = None
            self.tracking_enabled = False

        # Table ROI configuration
        # {table_id: {"polygon": [(x,y)], "bbox": (min_x, min_y, max_x, max_y)}}
        self.table_rois = {}

        # Settings
        self.settings = {
            "order_wait_threshold": 300.0,   # seconds - alert if order wait > threshold (default: 5 minutes)
            "service_wait_threshold": 180.0, # seconds - alert if service wait > threshold (default: 3 minutes)
            "alert_cooldown": 300.0,         # seconds between repeated alerts
            "track_timeout": 15.0,           # seconds before removing stale tracks
            "interaction_distance": 200.0,   # pixels - waiter near customer for interaction
            "interaction_duration": 2.0,     # seconds - waiter must stay near customer this long
            "food_served_gap": 10.0,         # seconds - min gap between T_order_start and T_food_served
            # Legacy setting for backward compatibility
            "wait_time_threshold": 300.0     # Alias for order_wait_threshold (5 minutes)
        }

        # Event-based tracking per person (using track_id from DeepSORT)
        # {track_id: {
        #   "type": "customer" | "waiter",
        #   "table_id": str,
        #   "center": (x, y),
        #   "bbox": [x1, y1, x2, y2],
        #   "T_seated": timestamp | None,
        #   "T_order_start": timestamp | None,
        #   "T_order_end": timestamp | None,  # When waiter leaves after taking order
        #   "T_food_served": timestamp | None,
        #   "last_seen": timestamp,
        #   "order_wait_time": float,  # T_order_start - T_seated
        #   "service_wait_time": float  # T_food_served - T_order_end
        # }}
        self.person_tracks = {}  # Global tracking across all tables

        # Table-level tracking for quick lookups
        # {table_id: {
        #   "customer_track_ids": [track_id, ...],
        #   "waiter_track_ids": [track_id, ...],
        #   "last_alert_time": timestamp | None
        # }}
        self.table_tracking = {}

        # Food/plate detection tracking
        # Track food objects near customers
        self.food_detections = []  # List of {bbox, center, timestamp}

        self.frame_count = 0
        self.total_alerts = 0
        self.last_update_time = time.time()

        self.load_configuration()

    # --- Configuration helpers ---
    def load_configuration(self):
        try:
            # Load from channels.json
            self._load_table_rois_from_config()
            
            # Also try loading from database as fallback
            if not self.table_rois and self.db_manager:
                roi_config = self.db_manager.get_channel_config(
                    self.channel_id, "ServiceDisciplineMonitor", "table_rois"
                )
                if roi_config:
                    self.table_rois = roi_config
                    logger.info(f"[{self.channel_id}] Loaded {len(self.table_rois)} table ROIs from database (fallback)")
            
            # Load settings from channels.json or database
            self._load_settings_from_config()
            if not self.settings.get('wait_time_threshold') and self.db_manager:
                settings = self.db_manager.get_channel_config(
                    self.channel_id, "ServiceDisciplineMonitor", "settings"
                )
                if settings:
                    self.settings.update(settings)
            
            logger.info(f"[{self.channel_id}] Service discipline config loaded: {len(self.table_rois)} tables configured")
        except Exception as e:
            logger.error(f"Failed to load service discipline configuration: {e}", exc_info=True)
    
    def _load_table_rois_from_config(self):
        """Load table ROIs from channels.json"""
        try:
            config_path = Path("config/channels.json")
            logger.info(f"[{self.channel_id}] 📂 Loading table ROIs from {config_path.absolute()}")
            
            if not config_path.exists():
                logger.warning(f"[{self.channel_id}] ⚠️ channels.json not found at {config_path.absolute()}")
                return
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            logger.info(f"[{self.channel_id}] 📖 Read channels.json, found {len(config.get('channels', []))} channels")
            
            # Find the channel and ServiceDisciplineMonitor module
            channel_found = False
            module_found = False
            
            for channel in config.get('channels', []):
                if channel.get('channel_id') != self.channel_id:
                    continue
                
                channel_found = True
                logger.info(f"[{self.channel_id}] ✓ Found channel in config, checking modules...")
                
                for module in channel.get('modules', []):
                    if module.get('type') != 'ServiceDisciplineMonitor':
                        continue
                    
                    module_found = True
                    logger.info(f"[{self.channel_id}] ✓ Found ServiceDisciplineMonitor module")
                    
                    module_config = module.get('config', {})
                    table_rois_config = module_config.get('table_rois', {})
                    
                    logger.info(f"[{self.channel_id}] 📊 Table ROIs in config: {len(table_rois_config)} tables")
                    
                    if table_rois_config:
                        # Convert from channels.json format to internal format
                        self.table_rois = {}
                        for table_id, roi_data in table_rois_config.items():
                            logger.info(f"[{self.channel_id}]   Processing table '{table_id}'...")
                            
                            if isinstance(roi_data, dict) and 'points' in roi_data:
                                # Format: {"points": [{"x": 0.2, "y": 0.3}, ...]}
                                points = roi_data['points']
                                logger.info(f"[{self.channel_id}]     Found {len(points)} points in config")
                                
                                polygon = []
                                for i, p in enumerate(points):
                                    if isinstance(p, dict) and 'x' in p and 'y' in p:
                                        polygon.append((float(p['x']), float(p['y'])))
                                    elif isinstance(p, (list, tuple)) and len(p) >= 2:
                                        polygon.append((float(p[0]), float(p[1])))
                                    else:
                                        logger.warning(f"[{self.channel_id}]     Invalid point {i}: {p}")
                                
                                if len(polygon) >= 3:
                                    min_x = min(p[0] for p in polygon)
                                    min_y = min(p[1] for p in polygon)
                                    max_x = max(p[0] for p in polygon)
                                    max_y = max(p[1] for p in polygon)
                                    
                                    self.table_rois[table_id] = {
                                        "polygon": polygon,
                                        "bbox": (min_x, min_y, max_x, max_y)
                                    }
                                    logger.info(f"[{self.channel_id}]     ✓ Loaded table '{table_id}' with {len(polygon)} points, bbox: ({min_x:.3f}, {min_y:.3f}, {max_x:.3f}, {max_y:.3f})")
                                else:
                                    logger.warning(f"[{self.channel_id}]     ⚠️ Table '{table_id}' has only {len(polygon)} valid points (need 3+)")
                            else:
                                logger.warning(f"[{self.channel_id}]     ⚠️ Table '{table_id}' has invalid format: {type(roi_data)}")
                        
                        logger.info(f"[{self.channel_id}] ✅ Successfully loaded {len(self.table_rois)} table ROIs from channels.json")
                        logger.info(f"[{self.channel_id}]   Table IDs: {list(self.table_rois.keys())}")
                        return
                    else:
                        logger.info(f"[{self.channel_id}]   No table_rois found in config (empty dict)")
            
            if not channel_found:
                logger.warning(f"[{self.channel_id}] ⚠️ Channel not found in channels.json")
            elif not module_found:
                logger.warning(f"[{self.channel_id}] ⚠️ ServiceDisciplineMonitor module not found for this channel")
            else:
                logger.info(f"[{self.channel_id}] ℹ️ No table ROIs configured in channels.json")
        except Exception as e:
            logger.error(f"[{self.channel_id}] ❌ Failed to load table ROIs from channels.json: {e}", exc_info=True)
    
    def _load_settings_from_config(self):
        """Load settings from channels.json"""
        try:
            config_path = Path("config/channels.json")
            if not config_path.exists():
                return
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            for channel in config.get('channels', []):
                if channel.get('channel_id') != self.channel_id:
                    continue
                
                for module in channel.get('modules', []):
                    if module.get('type') != 'ServiceDisciplineMonitor':
                        continue
                    
                    module_config = module.get('config', {})
                    settings_config = module_config.get('settings', {})
                    
                    if settings_config:
                        self.settings.update(settings_config)
                        logger.info(f"[{self.channel_id}] Loaded settings from channels.json")
                        return
        except Exception as e:
            logger.error(f"Failed to load settings from channels.json: {e}")
    
    def _save_table_rois_to_config(self):
        """Save table ROIs to channels.json"""
        try:
            config_path = Path("config/channels.json")
            logger.info(f"[{self.channel_id}] 💾 Saving to {config_path.absolute()}")
            
            if not config_path.exists():
                logger.error(f"[{self.channel_id}] ❌ channels.json not found at {config_path.absolute()}")
                return
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            logger.info(f"[{self.channel_id}] 📖 Read channels.json, found {len(config.get('channels', []))} channels")
            
            # Find the channel and ServiceDisciplineMonitor module
            channel_found = False
            module_found = False
            
            for channel in config.get('channels', []):
                if channel.get('channel_id') != self.channel_id:
                    continue
                
                channel_found = True
                logger.info(f"[{self.channel_id}] ✓ Found channel in config")
                
                modules = channel.get('modules', [])
                logger.info(f"[{self.channel_id}]   Checking {len(modules)} modules...")
                
                for module in modules:
                    if module.get('type') != 'ServiceDisciplineMonitor':
                        continue
                    
                    module_found = True
                    logger.info(f"[{self.channel_id}] ✓ Found ServiceDisciplineMonitor module")
                    
                    # Convert internal format to channels.json format
                    table_rois_config = {}
                    logger.info(f"[{self.channel_id}]   Converting {len(self.table_rois)} table ROIs to JSON format...")
                    
                    for table_id, roi_info in self.table_rois.items():
                        polygon = roi_info.get('polygon', [])
                        logger.info(f"[{self.channel_id}]     Converting table '{table_id}' ({len(polygon)} points)...")
                        
                        points = []
                        for i, p in enumerate(polygon):
                            if isinstance(p, (list, tuple)) and len(p) >= 2:
                                points.append({"x": float(p[0]), "y": float(p[1])})
                            elif isinstance(p, dict) and 'x' in p and 'y' in p:
                                points.append({"x": float(p['x']), "y": float(p['y'])})
                            else:
                                logger.warning(f"[{self.channel_id}]       Invalid point {i}: {p}")
                        
                        if len(points) >= 3:
                            table_rois_config[table_id] = {"points": points}
                            logger.info(f"[{self.channel_id}]       ✓ Converted table '{table_id}' with {len(points)} points")
                        else:
                            logger.warning(f"[{self.channel_id}]       ⚠️ Table '{table_id}' has only {len(points)} valid points")
                    
                    # Update the module config
                    if 'config' not in module:
                        module['config'] = {}
                        logger.info(f"[{self.channel_id}]   Created 'config' section in module")
                    
                    old_count = len(module['config'].get('table_rois', {}))
                    module['config']['table_rois'] = table_rois_config
                    logger.info(f"[{self.channel_id}]   Updated table_rois: {old_count} → {len(table_rois_config)} tables")
                    break
                
                if module_found:
                    break
            
            if not channel_found:
                logger.error(f"[{self.channel_id}] ❌ Channel not found in channels.json")
                return
            elif not module_found:
                logger.error(f"[{self.channel_id}] ❌ ServiceDisciplineMonitor module not found for channel {self.channel_id} in channels.json")
                return
            
            # Write back to file
            logger.info(f"[{self.channel_id}] 💾 Writing updated config to {config_path.absolute()}...")
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            logger.info(f"[{self.channel_id}] ✅ Successfully saved {len(self.table_rois)} table ROIs to channels.json")
            logger.info(f"[{self.channel_id}]   Saved tables: {list(self.table_rois.keys())}")
        except Exception as e:
            logger.error(f"[{self.channel_id}] ❌ Failed to save table ROIs to channels.json: {e}", exc_info=True)
            raise

    def set_table_roi(self, table_id, polygon_points):
        if not polygon_points or len(polygon_points) < 3:
            logger.warning(f"Invalid polygon for table {table_id}")
            return
        
        # Normalize point format - handle both dict {x, y} and list/tuple [x, y] formats
        normalized_points = []
        for p in polygon_points:
            if isinstance(p, dict):
                # Handle {x, y} format
                if 'x' in p and 'y' in p:
                    normalized_points.append((float(p['x']), float(p['y'])))
                else:
                    logger.warning(f"Invalid point format (dict without x/y): {p}")
                    continue
            elif isinstance(p, (list, tuple)) and len(p) >= 2:
                # Handle [x, y] or (x, y) format
                normalized_points.append((float(p[0]), float(p[1])))
            else:
                logger.warning(f"Invalid point format: {p}")
                continue
        
        if len(normalized_points) < 3:
            logger.warning(f"Not enough valid points for table {table_id} (got {len(normalized_points)}, need 3)")
            return
        
        min_x = min(p[0] for p in normalized_points)
        min_y = min(p[1] for p in normalized_points)
        max_x = max(p[0] for p in normalized_points)
        max_y = max(p[1] for p in normalized_points)

        self.table_rois[table_id] = {
            "polygon": normalized_points,
            "bbox": (min_x, min_y, max_x, max_y)
        }

        # Ensure tracking structure exists for this table
        self._ensure_table_tracking(table_id)

        # Save to channels.json
        try:
            logger.info(f"[{self.channel_id}] 💾 Saving table ROI for '{table_id}' to channels.json...")
            logger.info(f"[{self.channel_id}]   Table ROI data: {len(normalized_points)} points")
            logger.info(f"[{self.channel_id}]   Points: {normalized_points[:3]}..." if len(normalized_points) > 3 else f"[{self.channel_id}]   Points: {normalized_points}")
            self._save_table_rois_to_config()
            logger.info(f"[{self.channel_id}] ✅ Successfully saved table ROI for '{table_id}' to channels.json")
        except Exception as e:
            logger.error(f"[{self.channel_id}] ❌ Failed to save service discipline ROI to channels.json: {e}", exc_info=True)

        logger.info(f"[{self.channel_id}] ✓ Set service discipline ROI for table {table_id} (total tables: {len(self.table_rois)})")

    def _ensure_table_tracking(self, table_id):
        """
        Ensure table_tracking entry exists with required keys.
        """
        if table_id not in self.table_tracking:
            self.table_tracking[table_id] = {}
        self.table_tracking[table_id].setdefault("customer_track_ids", [])
        self.table_tracking[table_id].setdefault("waiter_track_ids", [])
        self.table_tracking[table_id].setdefault("last_alert_time", None)
        # Legacy keys for compatibility
        self.table_tracking[table_id].setdefault("customer_tracks", [])
        self.table_tracking[table_id].setdefault("server_tracks", [])

    # --- Geometry helpers ---
    def _point_in_polygon(self, point, polygon, bbox):
        x, y = point
        min_x, min_y, max_x, max_y = bbox
        if x < min_x or x > max_x or y < min_y or y > max_y:
            return False
        inside = False
        p1x, p1y = polygon[0]
        for i in range(1, len(polygon) + 1):
            p2x, p2y = polygon[i % len(polygon)]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    # --- Processing helpers ---
    def _classify_detections(self, detections, frame_shape):
        h, w = frame_shape[:2]
        table_detections = {tid: {"customers": [], "servers": []} for tid in self.table_rois.keys()}

        for det in detections:
            class_name = det.get("class_name", "")
            class_id = det.get("class_id", -1)
            bbox = det.get("bbox", [])
            if len(bbox) != 4:
                continue
            x1, y1, x2, y2 = bbox
            point_to_check = (x2 / w, y2 / h)  # bottom-right normalized (feet position)

            for table_id, roi_info in self.table_rois.items():
                polygon = roi_info["polygon"]
                bbox_norm = roi_info["bbox"]
                if self._point_in_polygon(point_to_check, polygon, bbox_norm):
                    # Check if this is a person detection (from YOLOv11n, class_id=0)
                    if class_id == self.person_class_id_yolo11n or class_name == "Person":
                        # Check nearby uniform to exclude servers
                        has_uniform = False
                        for other_det in detections:
                            other_class = other_det.get("class_name", "")
                            if other_class in self.server_uniform_classes:
                                ob = other_det.get("bbox", [])
                                if len(ob) == 4:
                                    ocx = (ob[0] + ob[2]) / 2
                                    ocy = (ob[1] + ob[3]) / 2
                                    dcx = (x1 + x2) / 2
                                    dcy = (y1 + y2) / 2
                                    distance = math.hypot(ocx - dcx, ocy - dcy)
                                    if distance < 100:
                                        has_uniform = True
                                        logger.debug(f"[{self.channel_id}] Table {table_id}: Person has nearby uniform (dist={distance:.1f}px), excluding from customers")
                                        break
                        if not has_uniform:
                            table_detections[table_id]["customers"].append(det)
                            logger.debug(f"[{self.channel_id}] Table {table_id}: Classified as customer (Person without nearby uniform)")
                    elif class_name in self.server_uniform_classes:
                        table_detections[table_id]["servers"].append(det)
                        logger.debug(f"[{self.channel_id}] Table {table_id}: Classified as server (Uniform: {class_name})")
                    break
        return table_detections

    def _update_table_tracking(self, table_id, customers, servers, current_time, frame=None):
        if table_id not in self.table_tracking:
            self.table_tracking[table_id] = {
                "customer_tracks": [],
                "server_tracks": [],
                "last_alert_time": None
            }
        tracking = self.table_tracking[table_id]
        now_ts = current_time.timestamp()

        # Reset matched flags
        for track in tracking["customer_tracks"]:
            track["matched"] = False
        for track in tracking["server_tracks"]:
            track["matched"] = False

        # Customers
        customer_centers = [
            [(d["bbox"][0] + d["bbox"][2]) / 2, (d["bbox"][1] + d["bbox"][3]) / 2]
            for d in customers if len(d.get("bbox", [])) == 4
        ]
        
        if customer_centers and self.frame_count % 30 == 0:  # Log every 30 frames
            logger.debug(f"[{self.channel_id}] Table {table_id}: {len(customer_centers)} customers detected")
        
        for center in customer_centers:
            matched = False
            for track in tracking["customer_tracks"]:
                if track.get("matched"):
                    continue
                distance = math.hypot(center[0] - track["center"][0], center[1] - track["center"][1])
                if distance < 100:
                    track["center"] = center
                    track["last_seen"] = now_ts
                    track["matched"] = True
                    matched = True
                    break
            if not matched:
                new_track = {
                    "id": len(tracking["customer_tracks"]),
                    "center": center,
                    "start_time": now_ts,
                    "last_seen": now_ts,
                    "attend_time": None,
                    "matched": True
                }
                tracking["customer_tracks"].append(new_track)
                logger.info(f"[{self.channel_id}] 🪑 Table {table_id}: New customer detected (track {new_track['id']}) at {center}")

        # Servers
        server_centers = [
            [(d["bbox"][0] + d["bbox"][2]) / 2, (d["bbox"][1] + d["bbox"][3]) / 2]
            for d in servers if len(d.get("bbox", [])) == 4
        ]
        
        if server_centers and self.frame_count % 30 == 0:
            logger.debug(f"[{self.channel_id}] Table {table_id}: {len(server_centers)} servers detected")
        
        for center in server_centers:
            matched = False
            for track in tracking["server_tracks"]:
                if track.get("matched"):
                    continue
                distance = math.hypot(center[0] - track["center"][0], center[1] - track["center"][1])
                if distance < 100:
                    track["center"] = center
                    track["last_seen"] = now_ts
                    track["matched"] = True
                    matched = True
                    break
            if not matched:
                s_track = {
                    "id": len(tracking["server_tracks"]),
                    "center": center,
                    "last_seen": now_ts,
                    "matched": True
                }
                tracking["server_tracks"].append(s_track)
                logger.info(f"[{self.channel_id}] 👔 Table {table_id}: New server detected (track {s_track['id']}) at {center}")
                
                # Mark attend time for waiting customers nearby
                for cust in tracking["customer_tracks"]:
                    if cust.get("attend_time") is None:
                        distance_to_customer = math.hypot(center[0] - cust["center"][0], center[1] - cust["center"][1])
                        if distance_to_customer < 200:
                            cust["attend_time"] = now_ts
                            wait_time = now_ts - cust.get("start_time", now_ts)
                            logger.info(f"[{self.channel_id}] ✅ Table {table_id}: Server attended customer (track {cust['id']}), wait time: {wait_time:.1f}s")

        # Remove stale tracks
        before_customer_count = len(tracking["customer_tracks"])
        before_server_count = len(tracking["server_tracks"])
        
        tracking["customer_tracks"] = [
            t for t in tracking["customer_tracks"]
            if (now_ts - t["last_seen"]) < self.settings["track_timeout"]
        ]
        tracking["server_tracks"] = [
            t for t in tracking["server_tracks"]
            if (now_ts - t["last_seen"]) < self.settings["track_timeout"]
        ]
        
        if len(tracking["customer_tracks"]) < before_customer_count:
            logger.debug(f"[{self.channel_id}] Table {table_id}: Removed {before_customer_count - len(tracking['customer_tracks'])} stale customer tracks")
        if len(tracking["server_tracks"]) < before_server_count:
            logger.debug(f"[{self.channel_id}] Table {table_id}: Removed {before_server_count - len(tracking['server_tracks'])} stale server tracks")

        self._check_violations(table_id, tracking, current_time, frame)

    def _check_violations(self, table_id, tracking, current_time, frame=None):
        now_ts = current_time.timestamp()
        wait_threshold = self.settings["wait_time_threshold"]
        cooldown = self.settings["alert_cooldown"]

        for cust in tracking["customer_tracks"]:
            start_time = cust.get("start_time")
            attend_time = cust.get("attend_time")
            if start_time is None:
                continue
            
            # Calculate waiting time
            if attend_time is None:
                # Customer is still waiting
                waiting_time = now_ts - start_time
                is_waiting = True
            else:
                # Customer was already attended - use the time until they were attended
                waiting_time = attend_time - start_time
                is_waiting = False
            
            # Only alert if customer is STILL waiting AND exceeds threshold
            if is_waiting and waiting_time > wait_threshold:
                last_alert = tracking.get("last_alert_time")
                time_since_last_alert = (now_ts - last_alert) if last_alert else float('inf')
                
                if last_alert is None or time_since_last_alert > cooldown:
                    logger.warning(
                        f"[{self.channel_id}] ⚠️ Table {table_id}: Customer waiting {waiting_time:.1f}s "
                        f"(threshold: {wait_threshold}s, cooldown: {cooldown}s, last alert: {time_since_last_alert:.1f}s ago)"
                    )
                    self._trigger_violation_alert(table_id, cust, waiting_time, current_time, frame)
                    tracking["last_alert_time"] = now_ts
                else:
                    logger.debug(
                        f"[{self.channel_id}] Table {table_id}: Violation detected but in cooldown "
                        f"({time_since_last_alert:.1f}s / {cooldown}s)"
                    )
            elif self.frame_count % 60 == 0:  # Log status every 60 frames
                if is_waiting:
                    logger.debug(
                        f"[{self.channel_id}] Table {table_id}: Customer waiting {waiting_time:.1f}s "
                        f"(threshold: {wait_threshold}s)"
                    )
                else:
                    logger.debug(
                        f"[{self.channel_id}] Table {table_id}: Customer was attended after {waiting_time:.1f}s"
                    )

    def _trigger_violation_alert(self, table_id, customer_track, waiting_time, current_time, frame=None):
        logger.warning(
            f"[{self.channel_id}] Service discipline violation: Table {table_id} "
            f"waiting {waiting_time:.1f}s (threshold {self.settings['wait_time_threshold']}s)"
        )

        snapshot_path = self._save_snapshot(table_id, customer_track, waiting_time, current_time, frame)

        if self.socketio:
            self.socketio.emit("service_discipline_alert", {
                "channel_id": self.channel_id,
                "table_id": table_id,
                "waiting_time": round(waiting_time, 1),
                "threshold": self.settings["wait_time_threshold"],
                "timestamp": current_time.isoformat(),
                "snapshot_path": snapshot_path
            })

        if self.db_manager:
            try:
                # Save to table_service_violations table
                # Wrap in app context to avoid "Working outside of application context" error
                if self.app:
                    with self.app.app_context():
                        self.db_manager.add_table_service_violation(
                            channel_id=self.channel_id,
                            table_id=table_id,
                            waiting_time=waiting_time,
                            snapshot_path=snapshot_path,
                            timestamp=current_time,
                            alert_data={"violation_type": "service_discipline", "waiting_time": waiting_time}
                        )
                else:
                    logger.warning(f"[{self.channel_id}] ⚠️ Cannot save violation: Flask app context not available")
                
                # Also log to general alerts table for consistency with other modules
                alert_message = f"Service discipline violation: Table {table_id} waiting {waiting_time:.1f}s"
                if self.app:
                    with self.app.app_context():
                        self.db_manager.log_alert(
                            self.channel_id,
                            'service_discipline_alert',
                            alert_message,
                            alert_data={
                                "violation_type": "service_discipline",
                                "table_id": table_id,
                                "waiting_time": waiting_time,
                                "threshold": self.settings.get("wait_time_threshold", 300.0),
                                "snapshot_path": snapshot_path
                            }
                        )
                else:
                    self.db_manager.log_alert(
                        self.channel_id,
                        'service_discipline_alert',
                        alert_message,
                        alert_data={
                            "violation_type": "service_discipline",
                            "table_id": table_id,
                            "waiting_time": waiting_time,
                            "threshold": self.settings.get("wait_time_threshold", 300.0),
                            "snapshot_path": snapshot_path
                        }
                    )
                logger.info(f"Service discipline alert logged to general alerts table: {alert_message}")
            except Exception as e:
                logger.error(f"Failed to save service discipline violation: {e}")

    def _save_snapshot(self, table_id, customer_track, waiting_time, current_time, frame=None):
        try:
            snapshot_dir = Path("static/service_discipline")
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            ts = current_time.strftime("%Y%m%d_%H%M%S")
            filename = f"service_table_{table_id}_{self.channel_id}_{ts}.jpg"
            snapshot_path = snapshot_dir / filename
            if frame is not None:
                annotated = frame.copy()
                center = customer_track.get("center", [0, 0])
                cv2.circle(annotated, (int(center[0]), int(center[1])), 20, (0, 0, 255), -1)
                cv2.putText(annotated, f"Table {table_id}: {waiting_time:.1f}s",
                            (int(center[0]) - 120, int(center[1]) - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imwrite(str(snapshot_path), annotated)
                logger.info(f"Service discipline snapshot saved: {snapshot_path}")
            return str(snapshot_path.relative_to("static"))
        except Exception as e:
            logger.error(f"Failed to save service discipline snapshot: {e}")
            return None

    # --- Main frame processing ---
    def process_frame(self, frame):
        self.frame_count += 1
        current_time = datetime.now()
        now_ts = current_time.timestamp()
        
        if frame is None or frame.size == 0:
            return frame

        try:
            # Save a clean copy of the original frame for snapshots
            # This ensures snapshots don't include annotations from other modules (e.g., queue monitor text)
            clean_frame = frame.copy()
            h, w = frame.shape[:2]
            
            # 1. Detect persons using YOLOv11n.pt
            person_detections = self.person_detector.detect_persons(frame)
            
            # 2. Detect uniforms using best.pt
            uniform_results = self.uniform_model(frame, conf=self.conf_threshold, iou=self.nms_iou, verbose=False)
            
            # Prepare detections for DeepSORT tracking
            ds_inputs = []
            person_detections_list = []
            uniform_detections_list = []
            
            # Process person detections
            for det in person_detections:
                bbox = det.get("bbox", [])
                if len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    # DeepSORT format: ([x, y, w, h], confidence, class_name)
                    ds_inputs.append(([int(x1), int(y1), int(x2-x1), int(y2-y1)], det.get("confidence", 0.5), "person"))
                    person_detections_list.append({
                        "bbox": bbox,
                        "confidence": det.get("confidence", 0.5),
                        "center": [(x1 + x2) / 2, (y1 + y2) / 2]
                    })
            
            # Process uniform detections (waiter detection)
            if len(uniform_results) > 0 and uniform_results[0].boxes is not None:
                boxes = uniform_results[0].boxes
                class_names = uniform_results[0].names
                for box in boxes:
                    class_id = int(box.cls[0])
                    class_name = class_names[class_id]
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    if class_name in self.server_uniform_classes:
                        # Add uniform detections as "person" for tracking (we'll classify later)
                        ds_inputs.append(([int(x1), int(y1), int(x2-x1), int(y2-y1)], float(conf), "person"))
                        uniform_detections_list.append({
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "confidence": conf,
                            "class_name": class_name,
                            "center": [(x1 + x2) / 2, (y1 + y2) / 2]
                        })
            
            # Update DeepSORT tracker
            if self.tracking_enabled and self.tracker and ds_inputs:
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    tracks = self.tracker.update_tracks(ds_inputs, frame=frame)
                except Exception as e:
                    logger.error(f"DeepSORT tracking error: {e}")
                    tracks = []
            else:
                tracks = []
            
            # Process tracked persons and classify them
            self._process_tracked_persons(tracks, person_detections_list, uniform_detections_list, current_time, frame)
            
            # Detect interactions and update events
            self._detect_interactions(current_time)
            
            # Check for violations - use clean_frame for snapshots to avoid other modules' annotations
            self._check_violations(current_time, clean_frame)
            
            # Clean up stale tracks
            self._cleanup_stale_tracks(now_ts)
            
            # Send real-time status updates (every 1 second)
            if now_ts - self.last_update_time >= 1.0:
                self._send_realtime_update()
                self.last_update_time = now_ts
            
            # Draw annotations
            annotated = self._draw_annotations_new(frame, current_time)
            return annotated
            
        except Exception as e:
            logger.error(f"ServiceDiscipline processing error: {e}", exc_info=True)
            return frame

    # --- New event-based tracking methods ---
    
    def _process_tracked_persons(self, tracks, person_detections, uniform_detections, current_time, frame):
        """Process DeepSORT tracks and classify as customer/waiter, detect T_seated event"""
        h, w = frame.shape[:2]
        now_ts = current_time.timestamp()
        
        # Map detections to tracks by position
        track_to_detection = {}
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_tlbr()  # left, top, right, bottom
            track_bbox = [int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])]
            track_center = [(track_bbox[0] + track_bbox[2]) / 2, (track_bbox[1] + track_bbox[3]) / 2]
            
            # Check if this track has a uniform detection nearby
            is_waiter = False
            for uni_det in uniform_detections:
                uni_center = uni_det["center"]
                distance = math.hypot(track_center[0] - uni_center[0], track_center[1] - uni_center[1])
                if distance < 50:  # Uniform detection is very close to person
                    is_waiter = True
                    break
            
            # Determine which table this person is in
            table_id = None
            point_to_check = (track_center[0] / w, track_center[1] / h)  # Normalized
            
            for tid, roi_info in self.table_rois.items():
                polygon = roi_info["polygon"]
                bbox_norm = roi_info["bbox"]
                if self._point_in_polygon(point_to_check, polygon, bbox_norm):
                    table_id = tid
                    break
            
            # Ensure table tracking structure exists
            if table_id:
                self._ensure_table_tracking(table_id)
            
            # Initialize or update track
            if track_id not in self.person_tracks:
                # New track - initialize
                person_type = "waiter" if is_waiter else "customer"
                self.person_tracks[track_id] = {
                    "type": person_type,
                    "table_id": table_id,
                    "center": track_center,
                    "bbox": track_bbox,
                    "T_seated": None,
                    "T_order_start": None,
                    "T_order_end": None,  # When waiter leaves after taking order
                    "T_food_served": None,
                    "last_seen": now_ts,
                    "order_wait_time": None,
                    "service_wait_time": None,
                    "interaction_history": [],
                    "last_snapshot_time": None,
                    "waiter_at_table": False  # Track if waiter is currently at table
                }
                
                # If customer and in table ROI, set T_seated
                if person_type == "customer" and table_id:
                    self.person_tracks[track_id]["T_seated"] = now_ts
                    logger.info(f"[{self.channel_id}] 🪑 T_seated: Customer track {track_id} at table {table_id}")
                    
                    # Update table tracking
                    if table_id not in self.table_tracking:
                        self.table_tracking[table_id] = {
                            "customer_track_ids": [],
                            "waiter_track_ids": [],
                            "last_alert_time": None
                        }
                    if track_id not in self.table_tracking[table_id]["customer_track_ids"]:
                        self.table_tracking[table_id]["customer_track_ids"].append(track_id)
            else:
                # Update existing track
                self.person_tracks[track_id]["center"] = track_center
                self.person_tracks[track_id]["bbox"] = track_bbox
                self.person_tracks[track_id]["last_seen"] = now_ts
                
                # Update type if we detect uniform
                if is_waiter:
                    self.person_tracks[track_id]["type"] = "waiter"
                    if table_id and table_id not in self.table_tracking:
                        self.table_tracking[table_id] = {
                            "customer_track_ids": [],
                            "waiter_track_ids": [],
                            "last_alert_time": None
                        }
                    if table_id and track_id not in self.table_tracking[table_id]["waiter_track_ids"]:
                        self.table_tracking[table_id]["waiter_track_ids"].append(track_id)
                
                # Update table_id if changed
                if table_id and self.person_tracks[track_id]["table_id"] != table_id:
                    # Remove from old table
                    old_table = self.person_tracks[track_id]["table_id"]
                    if old_table and old_table in self.table_tracking:
                        if self.person_tracks[track_id]["type"] == "customer":
                            if track_id in self.table_tracking[old_table]["customer_track_ids"]:
                                self.table_tracking[old_table]["customer_track_ids"].remove(track_id)
                        else:
                            if track_id in self.table_tracking[old_table]["waiter_track_ids"]:
                                self.table_tracking[old_table]["waiter_track_ids"].remove(track_id)
                    
                    # Add to new table
                    self.person_tracks[track_id]["table_id"] = table_id
                    if table_id not in self.table_tracking:
                        self.table_tracking[table_id] = {
                            "customer_track_ids": [],
                            "waiter_track_ids": [],
                            "last_alert_time": None
                        }
                    if self.person_tracks[track_id]["type"] == "customer":
                        if track_id not in self.table_tracking[table_id]["customer_track_ids"]:
                            self.table_tracking[table_id]["customer_track_ids"].append(track_id)
                    else:
                        if track_id not in self.table_tracking[table_id]["waiter_track_ids"]:
                            self.table_tracking[table_id]["waiter_track_ids"].append(track_id)
                
                # If customer just entered table ROI, set T_seated
                if (self.person_tracks[track_id]["type"] == "customer" and 
                    table_id and 
                    self.person_tracks[track_id]["T_seated"] is None):
                    self.person_tracks[track_id]["T_seated"] = now_ts
                    logger.info(f"[{self.channel_id}] 🪑 T_seated: Customer track {track_id} at table {table_id}")
    
    def _detect_interactions(self, current_time):
        """Detect T_order_start, T_order_end, and T_food_served events based on waiter-customer proximity"""
        now_ts = current_time.timestamp()
        interaction_distance = self.settings["interaction_distance"]
        interaction_duration = self.settings.get("interaction_duration", 2.0)
        food_served_gap = self.settings.get("food_served_gap", 10.0)
        
        # Track ongoing interactions
        ongoing_interactions = {}  # {(customer_track_id, waiter_track_id): start_time}
        
        for table_id, table_info in self.table_tracking.items():
            customer_ids = table_info.get("customer_track_ids", [])
            waiter_ids = table_info.get("waiter_track_ids", [])
            
            for customer_id in customer_ids:
                if customer_id not in self.person_tracks:
                    continue
                customer = self.person_tracks[customer_id]
                if customer["type"] != "customer" or customer["T_seated"] is None:
                    continue
                
                customer_center = customer["center"]
                waiter_nearby = False  # Track if any waiter is currently near
                
                # Check for nearby waiters
                for waiter_id in waiter_ids:
                    if waiter_id not in self.person_tracks:
                        continue
                    waiter = self.person_tracks[waiter_id]
                    if waiter["type"] != "waiter":
                        continue
                    
                    waiter_center = waiter["center"]
                    distance = math.hypot(
                        customer_center[0] - waiter_center[0],
                        customer_center[1] - waiter_center[1]
                    )
                    
                    interaction_key = (customer_id, waiter_id)
                    
                    if distance < interaction_distance:
                        waiter_nearby = True
                        # Waiter is near customer
                        if interaction_key not in ongoing_interactions:
                            # Start new interaction
                            ongoing_interactions[interaction_key] = now_ts
                        else:
                            # Interaction ongoing - check duration
                            interaction_start = ongoing_interactions[interaction_key]
                            interaction_duration_actual = now_ts - interaction_start
                            
                            if interaction_duration_actual >= interaction_duration:
                                # Valid interaction detected
                                
                                # Check if this is T_order_start (first interaction)
                                if customer["T_order_start"] is None:
                                    customer["T_order_start"] = now_ts
                                    customer["order_wait_time"] = now_ts - customer["T_seated"]
                                    customer["waiter_at_table"] = True
                                    customer["interaction_history"].append((waiter_id, now_ts, "order"))
                                    logger.info(
                                        f"[{self.channel_id}] 📝 T_order_start: Waiter {waiter_id} interacted with "
                                        f"customer {customer_id} at table {table_id} "
                                        f"(order wait: {customer['order_wait_time']:.1f}s)"
                                    )
                                
                                # Check if this is T_food_served (second interaction after gap)
                                elif (customer["T_food_served"] is None and 
                                      customer["T_order_start"] is not None and
                                      customer["T_order_end"] is not None and
                                      (now_ts - customer["T_order_end"]) >= food_served_gap):
                                    # Check if this waiter interacted before
                                    has_previous_interaction = any(
                                        w_id == waiter_id and event_type == "order"
                                        for w_id, ts, event_type in customer["interaction_history"]
                                    )
                                    
                                    if not has_previous_interaction or (now_ts - customer["T_order_end"]) >= food_served_gap:
                                        customer["T_food_served"] = now_ts
                                        # Service wait time = time from when waiter left to when food is served
                                        if customer["T_order_end"]:
                                            customer["service_wait_time"] = now_ts - customer["T_order_end"]
                                        else:
                                            # Fallback: use T_order_start if T_order_end not set
                                            customer["service_wait_time"] = now_ts - customer["T_order_start"]
                                        customer["interaction_history"].append((waiter_id, now_ts, "food"))
                                        logger.info(
                                            f"[{self.channel_id}] 🍽️ T_food_served: Waiter {waiter_id} served food to "
                                            f"customer {customer_id} at table {table_id} "
                                            f"(service wait: {customer['service_wait_time']:.1f}s from order end)"
                                        )
                                        
                                        # Save completed order to database (all orders, not just violations)
                                        if self.db_manager:
                                            try:
                                                current_dt = datetime.fromtimestamp(now_ts)
                                                
                                                if self.app:
                                                    with self.app.app_context():
                                                        self.db_manager.add_table_service_order(
                                                            channel_id=self.channel_id,
                                                            table_id=table_id,
                                                            order_wait_time=customer.get("order_wait_time"),
                                                            service_wait_time=customer.get("service_wait_time"),
                                                            timestamp=current_dt,
                                                            alert_data={
                                                                "T_seated": customer.get("T_seated"),
                                                                "T_order_start": customer.get("T_order_start"),
                                                                "T_order_end": customer.get("T_order_end"),
                                                                "T_food_served": customer.get("T_food_served"),
                                                                "order_wait_time": customer.get("order_wait_time"),
                                                                "service_wait_time": customer.get("service_wait_time")
                                                            }
                                                        )
                                                else:
                                                    logger.warning(f"[{self.channel_id}] ⚠️ Cannot save completed order: Flask app context not available")
                                            except Exception as e:
                                                logger.error(f"[{self.channel_id}] ❌ Failed to save completed order: {e}", exc_info=True)
                    else:
                        # Waiter moved away - clear interaction
                        if interaction_key in ongoing_interactions:
                            del ongoing_interactions[interaction_key]
                
                # Detect when waiter leaves after taking order (T_order_end)
                if not waiter_nearby and customer["T_order_start"] is not None and customer["T_order_end"] is None:
                    # Waiter was at table (taking order) but now left
                    if customer.get("waiter_at_table", False):
                        customer["T_order_end"] = now_ts
                        customer["waiter_at_table"] = False
                        logger.info(
                            f"[{self.channel_id}] 👋 T_order_end: Waiter left table {table_id} after taking order "
                            f"from customer {customer_id} (order duration: {now_ts - customer['T_order_start']:.1f}s)"
                        )
                elif waiter_nearby and customer["T_order_start"] is not None:
                    # Waiter is at table
                    customer["waiter_at_table"] = True
    
    def _check_violations(self, current_time, frame=None):
        """Check for violations based on order wait time and service wait time"""
        now_ts = current_time.timestamp()
        order_threshold = self.settings["order_wait_threshold"]
        service_threshold = self.settings["service_wait_threshold"]
        cooldown = self.settings["alert_cooldown"]
        
        # Diagnostic logging (every 300 frames ~10 seconds at 30fps)
        if self.frame_count % 300 == 0:
            total_customers = sum(len(info.get("customer_track_ids", [])) for info in self.table_tracking.values())
            total_tables = len(self.table_tracking)
            logger.debug(
                f"[{self.channel_id}] 📊 Violation check: {total_tables} tables, {total_customers} customers tracked, "
                f"thresholds: order={order_threshold}s, service={service_threshold}s"
            )
            if total_tables == 0:
                logger.warning(f"[{self.channel_id}] ⚠️ No tables being tracked! Check table ROIs configuration.")
            if total_customers == 0:
                logger.debug(f"[{self.channel_id}] No customers currently tracked (may be normal if no customers present)")
        
        for table_id, table_info in self.table_tracking.items():
            customer_ids = table_info.get("customer_track_ids", [])
            last_alert_time = table_info.get("last_alert_time")
            
            for customer_id in customer_ids:
                if customer_id not in self.person_tracks:
                    continue
                customer = self.person_tracks[customer_id]
                
                if customer["T_seated"] is None:
                    continue
                
                # Check order wait time violation
                if customer["T_order_start"] is None:
                    # Still waiting for order
                    order_wait = now_ts - customer["T_seated"]
                    
                    # Only save to database when violation threshold is exceeded
                    if order_wait > order_threshold:
                        time_since_last_alert = (now_ts - last_alert_time) if last_alert_time else float('inf')
                        if last_alert_time is None or time_since_last_alert > cooldown:
                            logger.warning(
                                f"[{self.channel_id}] ⚠️ Order wait violation: Table {table_id}, "
                                f"customer {customer_id} waiting {order_wait:.1f}s (threshold: {order_threshold}s)"
                            )
                            logger.info(
                                f"[{self.channel_id}] 📊 Violation details: T_seated={customer.get('T_seated')}, "
                                f"T_order_start={customer.get('T_order_start')}, order_wait={order_wait:.1f}s"
                            )
                            self._trigger_violation_alert_new(
                                table_id, customer_id, "order_wait", order_wait, current_time, frame
                            )
                            table_info["last_alert_time"] = now_ts
                            # Update last snapshot time to prevent duplicate
                            customer["last_snapshot_time"] = now_ts
                        else:
                            logger.debug(
                                f"[{self.channel_id}] Order wait violation detected but in cooldown: "
                                f"{time_since_last_alert:.1f}s / {cooldown}s"
                            )
                    elif self.frame_count % 300 == 0:  # Log status every 300 frames (~10 seconds at 30fps)
                        logger.debug(
                            f"[{self.channel_id}] Table {table_id}, customer {customer_id}: "
                            f"Order wait {order_wait:.1f}s (threshold: {order_threshold}s) - OK"
                        )
                
                # Check service wait time violation
                elif customer["T_food_served"] is None and customer["T_order_start"] is not None:
                    # Order taken but food not served
                    # Use T_order_end if available, otherwise fallback to T_order_start
                    if customer.get("T_order_end"):
                        service_wait = now_ts - customer["T_order_end"]
                    else:
                        # Waiter still at table or hasn't left yet - don't count service wait
                        service_wait = 0
                    
                    # Only track service wait if waiter has left (T_order_end is set)
                    # Only save to database when violation threshold is exceeded
                    if service_wait > service_threshold:
                        time_since_last_alert = (now_ts - last_alert_time) if last_alert_time else float('inf')
                        if last_alert_time is None or time_since_last_alert > cooldown:
                            logger.warning(
                                f"[{self.channel_id}] ⚠️ Service wait violation: Table {table_id}, "
                                f"customer {customer_id} waiting {service_wait:.1f}s for food "
                                f"(threshold: {service_threshold}s)"
                            )
                            logger.info(
                                f"[{self.channel_id}] 📊 Violation details: T_order_start={customer.get('T_order_start')}, "
                                f"T_order_end={customer.get('T_order_end')}, T_food_served={customer.get('T_food_served')}, "
                                f"service_wait={service_wait:.1f}s"
                            )
                            self._trigger_violation_alert_new(
                                table_id, customer_id, "service_wait", service_wait, current_time, frame
                            )
                            table_info["last_alert_time"] = now_ts
                            # Update last snapshot time to prevent duplicate
                            customer["last_snapshot_time"] = now_ts
                        else:
                            logger.debug(
                                f"[{self.channel_id}] Service wait violation detected but in cooldown: "
                                f"{time_since_last_alert:.1f}s / {cooldown}s"
                            )
                    elif self.frame_count % 300 == 0:  # Log status every 300 frames (~10 seconds at 30fps)
                        logger.debug(
                            f"[{self.channel_id}] Table {table_id}, customer {customer_id}: "
                            f"Service wait {service_wait:.1f}s (threshold: {service_threshold}s) - OK"
                        )
    
    def _cleanup_stale_tracks(self, now_ts):
        """Remove tracks that haven't been seen recently"""
        timeout = self.settings["track_timeout"]
        tracks_to_remove = []
        
        for track_id, track in self.person_tracks.items():
            if (now_ts - track["last_seen"]) > timeout:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            track = self.person_tracks[track_id]
            table_id = track.get("table_id")
            
            # Remove from table tracking
            if table_id and table_id in self.table_tracking:
                if track["type"] == "customer":
                    if track_id in self.table_tracking[table_id]["customer_track_ids"]:
                        self.table_tracking[table_id]["customer_track_ids"].remove(track_id)
                else:
                    if track_id in self.table_tracking[table_id]["waiter_track_ids"]:
                        self.table_tracking[table_id]["waiter_track_ids"].remove(track_id)
            
            del self.person_tracks[track_id]
    
    def _trigger_violation_alert_new(self, table_id, customer_track_id, violation_type, wait_time, current_time, frame=None):
        """Trigger alert for new event-based violations"""
        customer = self.person_tracks.get(customer_track_id)
        if not customer:
            return
        
        snapshot_path = self._save_snapshot_new(table_id, customer, violation_type, wait_time, current_time, frame)
        
        alert_data = {
            "violation_type": violation_type,
            "wait_time": wait_time,
            "T_seated": customer.get("T_seated"),
            "T_order_start": customer.get("T_order_start"),
            "T_order_end": customer.get("T_order_end"),
            "T_food_served": customer.get("T_food_served"),
            "order_wait_time": customer.get("order_wait_time"),
            "service_wait_time": customer.get("service_wait_time")
        }
        
        self.total_alerts += 1
        
        if self.socketio:
            self.socketio.emit("service_discipline_alert", {
                "channel_id": self.channel_id,
                "table_id": table_id,
                "customer_track_id": customer_track_id,
                "violation_type": violation_type,
                "wait_time": round(wait_time, 1),
                "timestamp": current_time.isoformat(),
                "snapshot_path": snapshot_path,
                "alert_data": alert_data
            })
        
        if self.db_manager:
            try:
                logger.info(f"[{self.channel_id}] 💾 Saving violation to database: Table {table_id}, {violation_type} = {wait_time:.1f}s")
                
                # Save to table_service_violations table
                # Wrap in app context to avoid "Working outside of application context" error
                if self.app:
                    with self.app.app_context():
                        result = self.db_manager.add_table_service_violation(
                            channel_id=self.channel_id,
                            table_id=table_id,
                            waiting_time=wait_time,
                            snapshot_path=snapshot_path,
                            timestamp=current_time,
                            alert_data={
                                "violation_type": violation_type,
                                "wait_time": wait_time,
                                "T_seated": customer.get("T_seated"),
                                "T_order_start": customer.get("T_order_start"),
                                "T_order_end": customer.get("T_order_end"),
                                "T_food_served": customer.get("T_food_served"),
                                "order_wait_time": customer.get("order_wait_time"),
                                "service_wait_time": customer.get("service_wait_time")
                            }
                        )
                        logger.info(f"[{self.channel_id}] ✅ Violation saved to table_service_violations: {result}")
                else:
                    logger.warning(f"[{self.channel_id}] ⚠️ Cannot save violation: Flask app context not available")
                    result = None
                
                # Also log to general alerts table for consistency with other modules
                # Determine the correct threshold based on violation type
                if violation_type == "order_wait":
                    threshold = self.settings.get("order_wait_threshold", 120.0)
                elif violation_type == "service_wait":
                    threshold = self.settings.get("service_wait_threshold", 300.0)
                else:
                    threshold = self.settings.get("wait_time_threshold", 120.0)
                
                alert_message = f"Service discipline violation: Table {table_id} {violation_type} = {wait_time:.1f}s (threshold: {threshold}s)"
                
                try:
                    if self.app:
                        with self.app.app_context():
                            self.db_manager.log_alert(
                                self.channel_id,
                                'service_discipline_alert',
                                alert_message,
                                alert_data={
                                    "violation_type": violation_type,
                                    "table_id": table_id,
                                    "wait_time": wait_time,
                                    "threshold": threshold,
                                    "snapshot_path": snapshot_path,
                                    "T_seated": customer.get("T_seated"),
                                    "T_order_start": customer.get("T_order_start"),
                                    "T_order_end": customer.get("T_order_end"),
                                    "T_food_served": customer.get("T_food_served"),
                                    "order_wait_time": customer.get("order_wait_time"),
                                    "service_wait_time": customer.get("service_wait_time")
                                }
                            )
                    else:
                        self.db_manager.log_alert(
                            self.channel_id,
                            'service_discipline_alert',
                            alert_message,
                            alert_data={
                                "violation_type": violation_type,
                                "table_id": table_id,
                                "wait_time": wait_time,
                                "threshold": threshold,
                                "snapshot_path": snapshot_path,
                                "T_seated": customer.get("T_seated"),
                                "T_order_start": customer.get("T_order_start"),
                                "T_order_end": customer.get("T_order_end"),
                                "T_food_served": customer.get("T_food_served"),
                                "order_wait_time": customer.get("order_wait_time"),
                                "service_wait_time": customer.get("service_wait_time")
                            }
                        )
                    logger.info(f"[{self.channel_id}] ✅ Alert logged to general alerts table: {alert_message}")
                except Exception as e2:
                    logger.error(f"[{self.channel_id}] ❌ Failed to log to general alerts table: {e2}", exc_info=True)
                    
            except Exception as e:
                logger.error(f"[{self.channel_id}] ❌ Failed to save service discipline violation: {e}", exc_info=True)
        else:
            logger.warning(f"[{self.channel_id}] ⚠️ db_manager is None - cannot save violation to database")
    
    def _save_snapshot_new(self, table_id, customer, violation_type, wait_time, current_time, frame=None):
        """Save snapshot with event information"""
        try:
            snapshot_dir = Path("static/service_discipline")
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            ts = current_time.strftime("%Y%m%d_%H%M%S")
            filename = f"service_{violation_type}_{table_id}_{self.channel_id}_{ts}.jpg"
            snapshot_path = snapshot_dir / filename
            
            if frame is not None:
                annotated = frame.copy()
                center = customer.get("center", [0, 0])
                cv2.circle(annotated, (int(center[0]), int(center[1])), 20, (0, 0, 255), -1)
                
                # Draw event timeline
                info_y = 30
                cv2.putText(annotated, f"Table {table_id}: {violation_type}", (10, info_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                info_y += 30
                cv2.putText(annotated, f"Wait time: {wait_time:.1f}s", (10, info_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                if customer.get("T_seated"):
                    info_y += 25
                    cv2.putText(annotated, f"T_seated: {customer['T_seated']:.1f}s ago", (10, info_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                if customer.get("T_order_start"):
                    info_y += 25
                    cv2.putText(annotated, f"T_order_start: {customer['T_order_start']:.1f}s ago", (10, info_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                if customer.get("T_order_end"):
                    info_y += 25
                    cv2.putText(annotated, f"T_order_end: {customer['T_order_end']:.1f}s ago", (10, info_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                if customer.get("T_food_served"):
                    info_y += 25
                    cv2.putText(annotated, f"T_food_served: {customer['T_food_served']:.1f}s ago", (10, info_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                # Show both wait times
                if customer.get("order_wait_time"):
                    info_y += 25
                    cv2.putText(annotated, f"Order wait: {customer['order_wait_time']:.1f}s", (10, info_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                if customer.get("service_wait_time"):
                    info_y += 25
                    cv2.putText(annotated, f"Service wait: {customer['service_wait_time']:.1f}s", (10, info_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                cv2.imwrite(str(snapshot_path), annotated)
                logger.info(f"Service discipline snapshot saved: {snapshot_path}")
            
            return str(snapshot_path.relative_to("static"))
        except Exception as e:
            logger.error(f"Failed to save service discipline snapshot: {e}")
            return None
    
    def _save_wait_time_snapshot(self, table_id, customer_id, wait_type, wait_time, current_time, frame=None, is_violation=False):
        """
        Save snapshot for wait times - ONLY CALLED FOR VIOLATIONS
        This function is kept for backward compatibility but should only be called with is_violation=True
        """
        # Only save to database if it's a violation
        if not is_violation:
            logger.warning(f"[{self.channel_id}] _save_wait_time_snapshot called with is_violation=False - this should not happen")
            return None
        
        try:
            customer = self.person_tracks.get(customer_id)
            if not customer:
                return None
            
            snapshot_path = self._save_snapshot_new(table_id, customer, wait_type, wait_time, current_time, frame)
            if not snapshot_path:
                return None
            
            # Prepare alert data - always a violation when this function is called
            alert_data = {
                "violation_type": "service_discipline",
                "wait_type": wait_type,
                "wait_time": wait_time,
                "is_violation": True,
                "T_seated": customer.get("T_seated"),
                "T_order_start": customer.get("T_order_start"),
                "T_order_end": customer.get("T_order_end"),
                "T_food_served": customer.get("T_food_served"),
                "order_wait_time": customer.get("order_wait_time"),
                "service_wait_time": customer.get("service_wait_time")
            }
            
            # Save to database - only violations are saved
            if self.db_manager and self.app:
                try:
                    with self.app.app_context():
                        self.db_manager.add_table_service_violation(
                            channel_id=self.channel_id,
                            table_id=table_id,
                            waiting_time=wait_time,
                            snapshot_path=snapshot_path,
                            timestamp=current_time,
                            alert_data=alert_data
                        )
                        logger.info(f"[{self.channel_id}] ✅ Violation saved: Table {table_id}, {wait_type} = {wait_time:.1f}s")
                except Exception as e:
                    logger.error(f"Error saving violation to database: {e}")
            
            return snapshot_path
        except Exception as e:
            logger.error(f"Failed to save violation snapshot: {e}")
            return None
    
    def _draw_annotations_new(self, frame, current_time):
        """Draw annotations with event information"""
        h, w = frame.shape[:2]
        annotated = frame.copy()
        
        if not self.table_rois:
            cv2.putText(annotated, "No table ROIs configured", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return annotated
        
        now_ts = current_time.timestamp()
        
        # Draw table ROIs and tracked persons
        for table_id, roi_info in self.table_rois.items():
            polygon = roi_info.get("polygon", [])
            if not polygon or len(polygon) < 3:
                continue
            
            # Draw ROI polygon
            try:
                polygon_pixels = []
                for p in polygon:
                    if isinstance(p, (list, tuple)) and len(p) >= 2:
                        px = int(float(p[0]) * w)
                        py = int(float(p[1]) * h)
                        polygon_pixels.append((px, py))
                    elif isinstance(p, dict) and 'x' in p and 'y' in p:
                        px = int(float(p['x']) * w)
                        py = int(float(p['y']) * h)
                        polygon_pixels.append((px, py))
                
                if len(polygon_pixels) >= 3:
                    cv2.polylines(annotated, [np.array(polygon_pixels, np.int32)], True, (0, 255, 255), 3)
                    if polygon_pixels:
                        label_pos = polygon_pixels[0]
                        cv2.putText(annotated, f"Table {table_id}", label_pos,
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            except Exception as e:
                logger.error(f"Error drawing ROI for table {table_id}: {e}")
                continue
            
            # Draw tracked persons for this table
            if table_id in self.table_tracking:
                table_info = self.table_tracking[table_id]
                
                # Table-level waiting time summary (max current waits)
                order_waits = []
                service_waits = []
                for customer_id in table_info.get("customer_track_ids", []):
                    cust = self.person_tracks.get(customer_id)
                    if not cust:
                        continue
                    if cust.get("T_seated"):
                        order_waits.append((cust.get("T_order_start") or now_ts) - cust["T_seated"])
                    if cust.get("T_order_start"):
                        service_waits.append((cust.get("T_food_served") or now_ts) - cust["T_order_start"])
                if polygon_pixels:
                    text_y = label_pos[1] + 20
                    if order_waits:
                        max_order = max(order_waits)
                        cv2.putText(annotated, f"Order wait: {max_order:.1f}s", (label_pos[0], text_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        text_y += 18
                    if service_waits:
                        max_service = max(service_waits)
                        cv2.putText(annotated, f"Service wait: {max_service:.1f}s", (label_pos[0], text_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        text_y += 18
                
                # Draw customers
                for customer_id in table_info.get("customer_track_ids", []):
                    if customer_id not in self.person_tracks:
                        continue
                    customer = self.person_tracks[customer_id]
                    center = customer.get("center", [0, 0])
                    bbox = customer.get("bbox", [])
                    
                    if len(bbox) == 4:
                        x1, y1, x2, y2 = bbox
                        # Color based on status
                        if customer.get("T_food_served"):
                            color = (0, 255, 0)  # Green - served
                        elif customer.get("T_order_start"):
                            color = (0, 165, 255)  # Orange - order taken
                        else:
                            color = (0, 0, 255)  # Red - waiting
                        
                        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        
                        # Draw event timeline
                        info_text = f"ID:{customer_id}"
                        if customer.get("T_seated"):
                            order_wait = (customer.get("T_order_start") or now_ts) - customer["T_seated"]
                            info_text += f" | Order:{order_wait:.1f}s"
                        if customer.get("T_order_start"):
                            service_wait = (customer.get("T_food_served") or now_ts) - customer["T_order_start"]
                            info_text += f" | Service:{service_wait:.1f}s"
                        
                        cv2.putText(annotated, info_text, (int(x1), int(y1) - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw waiters
                for waiter_id in table_info.get("waiter_track_ids", []):
                    if waiter_id not in self.person_tracks:
                        continue
                    waiter = self.person_tracks[waiter_id]
                    bbox = waiter.get("bbox", [])
                    
                    if len(bbox) == 4:
                        x1, y1, x2, y2 = bbox
                        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                        cv2.putText(annotated, f"Waiter ID:{waiter_id}", (int(x1), int(y1) - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        return annotated

    def _draw_annotations(self, frame, table_dets, current_time):
        h, w = frame.shape[:2]
        annotated = frame.copy()
        
        # Debug: Log if no ROIs configured
        if not self.table_rois:
            cv2.putText(annotated, "No table ROIs configured", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return annotated
        
        # Draw table ROIs
        for table_id, roi_info in self.table_rois.items():
            polygon = roi_info.get("polygon", [])
            if not polygon or len(polygon) < 3:
                logger.warning(f"Table {table_id} has invalid polygon: {len(polygon)} points")
                continue
            
            # Handle both tuple/list format from DB and dict format
            try:
                polygon_pixels = []
                for p in polygon:
                    if isinstance(p, (list, tuple)) and len(p) >= 2:
                        # Handle [x, y] or (x, y) format (from JSON/database)
                        px = int(float(p[0]) * w)
                        py = int(float(p[1]) * h)
                        polygon_pixels.append((px, py))
                    elif isinstance(p, dict) and 'x' in p and 'y' in p:
                        # Handle {x, y} format
                        px = int(float(p['x']) * w)
                        py = int(float(p['y']) * h)
                        polygon_pixels.append((px, py))
                    else:
                        logger.warning(f"Invalid polygon point format for table {table_id}: {p} (type: {type(p)})")
                        continue
                
                if len(polygon_pixels) >= 3:
                    # Draw thick cyan border for visibility
                    cv2.polylines(annotated, [np.array(polygon_pixels, np.int32)], True, (0, 255, 255), 3)
                    # Draw table label
                    if polygon_pixels:
                        label_pos = polygon_pixels[0]
                        cv2.putText(annotated, f"Table {table_id}", label_pos,
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                else:
                    logger.warning(f"Table {table_id}: Only {len(polygon_pixels)} valid pixels after conversion")
            except Exception as e:
                logger.error(f"Error drawing ROI for table {table_id}: {e}", exc_info=True)
                continue

            # Draw customer tracks with waiting times
            if table_id in self.table_tracking:
                tracking = self.table_tracking[table_id]
                now_ts = current_time.timestamp()
                wait_threshold = self.settings["wait_time_threshold"]
                
                for cust in tracking["customer_tracks"]:
                    center = cust.get("center", [0, 0])
                    start_time = cust.get("start_time")
                    attend_time = cust.get("attend_time")
                    
                    if start_time:
                        # Calculate waiting time
                        if attend_time:
                            waiting_time = attend_time - start_time
                            is_waiting = False
                        else:
                            waiting_time = now_ts - start_time
                            is_waiting = True
                        
                        # Determine color based on waiting time
                        if waiting_time > wait_threshold:
                            # Red for violations (>120s)
                            box_color = (0, 0, 255)  # Red
                            text_color = (0, 0, 255)  # Red
                            bg_color = (0, 0, 0)  # Black background for text
                        elif waiting_time > wait_threshold * 0.8:
                            # Orange for approaching threshold (80-100% of threshold)
                            box_color = (0, 165, 255)  # Orange
                            text_color = (0, 165, 255)  # Orange
                            bg_color = (0, 0, 0)  # Black background
                        else:
                            # Green for normal waiting
                            box_color = (0, 255, 0)  # Green
                            text_color = (0, 255, 0)  # Green
                            bg_color = (0, 0, 0)  # Black background
                        
                        # Draw bounding box around customer (estimate from center)
                        box_size = 40
                        x1 = int(center[0] - box_size)
                        y1 = int(center[1] - box_size)
                        x2 = int(center[0] + box_size)
                        y2 = int(center[1] + box_size)
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 3)
                        
                        # Format waiting time text
                        if is_waiting:
                            wait_text = f"Waiting: {waiting_time:.1f}s"
                            if waiting_time > wait_threshold:
                                wait_text = f"⚠️ {wait_text} (ALERT!)"
                        else:
                            wait_text = f"Attended: {waiting_time:.1f}s"
                        
                        # Draw text with background for better visibility
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.7
                        thickness = 2
                        (text_width, text_height), baseline = cv2.getTextSize(wait_text, font, font_scale, thickness)
                        
                        # Draw background rectangle for text
                        text_x = int(center[0] - text_width // 2)
                        text_y = int(center[1] - box_size - 10)
                        cv2.rectangle(annotated, 
                                     (text_x - 5, text_y - text_height - 5),
                                     (text_x + text_width + 5, text_y + baseline + 5),
                                     bg_color, -1)
                        
                        # Draw waiting time text
                        cv2.putText(annotated, wait_text, (text_x, text_y),
                                   font, font_scale, text_color, thickness)
                        
                        # Also show table ID and customer ID for reference
                        info_text = f"Table {table_id} | Customer #{cust.get('id', 0)}"
                        info_font_scale = 0.5
                        (info_width, info_height), info_baseline = cv2.getTextSize(info_text, font, info_font_scale, 1)
                        info_x = int(center[0] - info_width // 2)
                        info_y = text_y - text_height - 15
                        cv2.rectangle(annotated,
                                     (info_x - 5, info_y - info_height - 5),
                                     (info_x + info_width + 5, info_y + info_baseline + 5),
                                     (0, 0, 0), -1)
                        cv2.putText(annotated, info_text, (info_x, info_y),
                                   font, info_font_scale, (255, 255, 255), 1)

            # Draw current frame detections (customers)
            for det in table_dets[table_id]["customers"]:
                bbox = det.get("bbox", [])
                if len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    # Draw light green box for detected customers
                    cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(annotated, "Customer", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Servers
            for det in table_dets[table_id]["servers"]:
                bbox = det.get("bbox", [])
                if len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                    cv2.putText(annotated, "Waiter", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Show table summary at top of ROI
            if table_id in self.table_tracking:
                tracking = self.table_tracking[table_id]
                waiting_customers = [c for c in tracking.get("customer_tracks", []) if c.get("attend_time") is None]
                if waiting_customers:
                    now_ts = current_time.timestamp()
                    max_wait = max([(now_ts - c.get("start_time", now_ts)) for c in waiting_customers], default=0)
                    summary_text = f"Table {table_id}: {len(waiting_customers)} waiting (max: {max_wait:.1f}s)"
                    if max_wait > wait_threshold:
                        summary_color = (0, 0, 255)  # Red
                    else:
                        summary_color = (0, 255, 255)  # Cyan
                    
                    if polygon_pixels:
                        summary_y = max(0, polygon_pixels[0][1] - 30)
                        summary_x = polygon_pixels[0][0]
                        cv2.putText(annotated, summary_text, (summary_x, summary_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, summary_color, 2)

        return annotated

    def get_status(self):
        """Get current status for debugging and monitoring"""
        status = {
            "channel_id": self.channel_id,
            "tables_configured": len(self.table_rois),
            "table_ids": list(self.table_rois.keys()),
            "frame_count": self.frame_count,
            "settings": self.settings.copy(),
            "table_status": {}
        }
        
        for table_id, tracking in self.table_tracking.items():
            customers_waiting = []
            customers_attended = []
            
            for cust in tracking.get("customer_tracks", []):
                start_time = cust.get("start_time")
                attend_time = cust.get("attend_time")
                if start_time:
                    if attend_time is None:
                        waiting_time = datetime.now().timestamp() - start_time
                        customers_waiting.append({
                            "track_id": cust.get("id"),
                            "waiting_time": round(waiting_time, 1),
                            "center": cust.get("center")
                        })
                    else:
                        waiting_time = attend_time - start_time
                        customers_attended.append({
                            "track_id": cust.get("id"),
                            "wait_time": round(waiting_time, 1),
                            "center": cust.get("center")
                        })
            
            status["table_status"][table_id] = {
                "customers_waiting": len(customers_waiting),
                "customers_attended": len(customers_attended),
                "servers_present": len(tracking.get("server_tracks", [])),
                "waiting_details": customers_waiting,
                "last_alert_time": tracking.get("last_alert_time")
            }
        
        return status

    def _send_realtime_update(self):
        """Send real-time status update via Socket.IO"""
        if not self.socketio:
            return
        
        # Count tables with waiting customers
        waiting_tables = 0
        for table_id, table_info in self.table_tracking.items():
            customer_ids = table_info.get("customer_track_ids", [])
            for customer_id in customer_ids:
                customer = self.person_tracks.get(customer_id)
                if customer and customer.get("T_seated") and customer.get("T_order_start") is None:
                    waiting_tables += 1
                    break  # Count each table only once
        
        try:
            self.socketio.emit('service_discipline_update', {
                'channel_id': self.channel_id,
                'waiting_tables': waiting_tables,
                'total_alerts': self.total_alerts,
                'tables_configured': len(self.table_rois),
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Error sending service discipline status update: {e}")
    
    def get_current_status(self):
        """Get simplified current status for dashboard"""
        # Count tables with waiting customers
        waiting_tables = 0
        for table_id, table_info in self.table_tracking.items():
            customer_ids = table_info.get("customer_track_ids", [])
            for customer_id in customer_ids:
                customer = self.person_tracks.get(customer_id)
                if customer and customer.get("T_seated") and customer.get("T_order_start") is None:
                    waiting_tables += 1
                    break  # Count each table only once
        
        return {
            "waiting_tables": waiting_tables,
            "total_alerts": self.total_alerts,
            "tables_configured": len(self.table_rois)
        }


