"""
Crowd Detection Module
Monitors crowd gathering in parking space using ROI-based person detection
"""
import cv2
import numpy as np
import logging
import torch
import os
import math
import time
from datetime import datetime
from collections import deque
from .yolo_detector import YOLODetector
from .gif_recorder import AlertGifRecorder

logger = logging.getLogger(__name__)

class CrowdDetection:
    def __init__(self, channel_id, socketio, db_manager=None, app=None):
        """
        Initialize Crowd Detection
        
        Args:
            channel_id: Unique identifier for the video channel
            socketio: SocketIO instance for real-time updates
            db_manager: Database manager for storing analytics
            app: Flask application instance for database context
        """
        self.channel_id = channel_id
        self.socketio = socketio
        self.db_manager = db_manager
        self.app = app
        
        # Initialize YOLO detector with best.pt for person detection
        self.detector = YOLODetector(
            model_path='models/best.pt',
            confidence_threshold=0.4,
            img_size=640,
            person_class_id=12  # Person class in best.pt model
        )
        
        # Initialize GIF recorder for alerts
        self.gif_recorder = AlertGifRecorder(
            buffer_size=90,  # 3 seconds at 30fps
            gif_duration=4.0,  # 4 second GIFs
            fps=30
        )
        
        # ROI configuration (normalized coordinates 0-1)
        self.roi_config = {
            'main': []  # Parking space area
        }
        
        # Crowd detection settings
        self.settings = {
            'crowd_threshold': 1,  # Alert when this many people detected
            'alert_cooldown': 60,  # Cooldown between alerts in seconds
            'dwell_time_threshold': 60.0,  # Person must stay in ROI for this long (60s for long-stay)
            'cluster_min_people': 3,  # Minimum people in a cluster to be considered gathering
            'cluster_max_dist': 100,  # Max distance for clustering (pixels)
            'gathering_hold_seconds': 20,  # How long gathering must persist before alert
            'min_movement_threshold': 20,  # Pixels - if people move less, they're "stationary"
            'history_seconds': 70,  # How far back to look for long-stay detection
            'history_max_frames': 20,  # Max history frames to remember
            'stay_dist_pixels': 40,  # How close positions must be to count as "same place"
            'min_frames_in_window': 10  # How many frames inside window => considered "long stay"
        }
        
        # Tracking variables
        self.person_tracking = {}  # {track_id: {'position': (x,y), 'first_seen': timestamp, 'last_seen': timestamp, 'active': bool}}
        self.center_history = deque(maxlen=self.settings['history_max_frames'])  # History of (timestamp, centers_list)
        self.crowd_count = 0
        self.raw_crowd_now = 0
        self.last_alert_time = None
        self.alert_condition_start_time = None
        self.alert_condition_sustained_duration = 5.0  # Must sustain for 5 seconds before alerting
        self.last_cluster_start_time = None
        self.active_cluster_size = 0
        self.previous_centers = []
        self.next_track_id = 0
        self.max_track_dist = 50  # Max distance for track association
        
        # Performance optimization
        self.detection_cache = None
        self.cache_frame_count = 0
        self.cache_interval = 4  # Run detection every 4 frames (matching user's FRAME_SKIP)
        
        # ROI cache for faster polygon checks
        self.roi_cache = {'main': None}
        self.roi_cache_frame_size = None
        
        # Frame counter
        self.frame_count = 0
        
        logger.info(f"Crowd Detection initialized for channel {channel_id}")
    
    def set_roi(self, roi_points):
        """
        Set ROI configuration
        
        Args:
            roi_points: Dictionary with 'main' polygon points
        """
        if isinstance(roi_points, dict) and 'main' in roi_points:
            self.roi_config = roi_points
        elif isinstance(roi_points, list):
            self.roi_config = {'main': roi_points}
        else:
            self.roi_config = {'main': []}
        
        logger.info(f"ROI updated for channel {self.channel_id}")
        
        # Save to database if available
        if self.db_manager:
            try:
                from flask import has_app_context
                if has_app_context():
                    self.db_manager.save_channel_config(
                        self.channel_id,
                        'CrowdDetection',
                        'roi',
                        self.roi_config
                    )
            except Exception as e:
                logger.error(f"Failed to save ROI config: {e}")
    
    def get_roi(self):
        """Get current ROI configuration"""
        return self.roi_config.copy()
    
    def set_settings(self, settings):
        """Update crowd detection settings"""
        self.settings.update(settings)
        logger.info(f"Settings updated for channel {self.channel_id}")
        
        # Save to database if available
        if self.db_manager:
            try:
                from flask import has_app_context
                if has_app_context():
                    self.db_manager.save_channel_config(
                        self.channel_id,
                        'CrowdDetection',
                        'settings',
                        self.settings
                    )
                else:
                    logger.debug(f"Skipping DB save for settings (no app context available)")
            except Exception as e:
                logger.error(f"Failed to save settings: {e}")
    
    def load_configuration(self):
        """Load saved configuration from database"""
        if self.db_manager:
            try:
                from flask import has_app_context
                if not has_app_context():
                    logger.debug(f"Skipping config load from DB (no app context available)")
                    return
                
                # Load ROI configuration
                roi_config = self.db_manager.get_channel_config(
                    self.channel_id, 'CrowdDetection', 'roi'
                )
                if roi_config:
                    self.roi_config = roi_config
                    logger.info(f"Loaded ROI configuration from database for {self.channel_id}")
                
                # Load settings
                settings = self.db_manager.get_channel_config(
                    self.channel_id, 'CrowdDetection', 'settings'
                )
                if settings:
                    self.settings.update(settings)
                    logger.info(f"Loaded settings from database for {self.channel_id}")
                
                logger.info(f"Configuration loaded for channel {self.channel_id}")
            except Exception as e:
                logger.error(f"Failed to load configuration: {e}")
    
    def euclidean(self, p1, p2):
        """Calculate Euclidean distance between two points"""
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])
    
    def cluster_points(self, points, max_dist):
        """Simple single-linkage clustering"""
        n = len(points)
        if n == 0:
            return []
        
        visited = [False] * n
        clusters = []
        
        for i in range(n):
            if visited[i]:
                continue
            cluster = [i]
            visited[i] = True
            queue = [i]
            
            while queue:
                cur = queue.pop(0)
                for j in range(n):
                    if visited[j]:
                        continue
                    if self.euclidean(points[cur], points[j]) <= max_dist:
                        visited[j] = True
                        cluster.append(j)
                        queue.append(j)
            
            clusters.append(cluster)
        
        return clusters
    
    def get_long_stay_centers(self, center_history, now_seconds):
        """
        Get centers that have been seen around the same spot for at least DWELL_TIME_SECONDS
        """
        if not center_history:
            return []
        
        # Latest entry is the current frame
        current_ts, current_centers = center_history[-1]
        
        # Convert datetime to seconds if needed
        def to_seconds(ts):
            if hasattr(ts, 'timestamp'):
                return ts.timestamp()
            elif isinstance(ts, (int, float)):
                return float(ts)
            else:
                return time.mktime(ts.timetuple()) if hasattr(ts, 'timetuple') else 0
        
        current_ts_seconds = to_seconds(current_ts)
        
        # Consider only frames within time window
        valid_history = [
            (ts, centers)
            for (ts, centers) in center_history
            if (now_seconds - to_seconds(ts)) <= self.settings['history_seconds']
        ]
        
        long_stay_centers = []
        
        for (cx, cy) in current_centers:
            count = 0
            for ts, centers in valid_history:
                # Look if any center in that frame is near (cx, cy)
                for (px, py) in centers:
                    if self.euclidean((cx, cy), (px, py)) <= self.settings['stay_dist_pixels']:
                        count += 1
                        break  # Found match in this frame, go to next frame
            
            if count >= self.settings['min_frames_in_window']:
                long_stay_centers.append((cx, cy))
        
        return long_stay_centers
    
    def calculate_cluster_movement(self, current_centers, previous_centers, cluster_indices):
        """Calculate average movement of people in a cluster"""
        if not previous_centers or len(previous_centers) != len(current_centers):
            return float('inf')  # Can't compare, assume movement
        
        movements = []
        for idx in cluster_indices:
            if idx < len(current_centers) and idx < len(previous_centers):
                dist = self.euclidean(current_centers[idx], previous_centers[idx])
                movements.append(dist)
        
        return np.mean(movements) if movements else float('inf')
    
    def point_in_polygon_optimized(self, point, polygon, polygon_bbox=None):
        """Optimized point-in-polygon check"""
        if len(polygon) < 3:
            return False
        
        x, y = point
        
        # Fast bounding box check first
        if polygon_bbox:
            min_x, min_y, max_x, max_y = polygon_bbox
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
    
    def compute_polygon_bbox(self, polygon):
        """Compute bounding box for a polygon"""
        if not polygon:
            return None
        
        x_coords = [p[0] for p in polygon]
        y_coords = [p[1] for p in polygon]
        
        return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
    
    def get_roi_pixels(self, frame_width, frame_height):
        """Convert normalized ROI coordinates to pixel coordinates"""
        if not self.roi_config.get('main'):
            return []
        
        pixel_points = []
        for point in self.roi_config['main']:
            x = int(point['x'] * frame_width)
            y = int(point['y'] * frame_height)
            pixel_points.append((x, y))
        
        return pixel_points
    
    def update_roi_cache(self, frame_width, frame_height):
        """Update cached ROI data for faster processing"""
        current_frame_size = (frame_width, frame_height)
        if self.roi_cache_frame_size != current_frame_size:
            self.roi_cache_frame_size = current_frame_size
            
            roi_polygon = self.get_roi_pixels(frame_width, frame_height)
            if roi_polygon:
                self.roi_cache['main'] = {
                    'polygon': roi_polygon,
                    'bbox': self.compute_polygon_bbox(roi_polygon)
                }
            else:
                self.roi_cache['main'] = None
    
    def classify_detections(self, detections, frame_width, frame_height):
        """Classify detections into ROI area"""
        roi_detections = []
        
        # Update ROI cache if needed
        self.update_roi_cache(frame_width, frame_height)
        
        # Get cached ROI data
        roi = self.roi_cache['main']
        
        if not roi:
            return roi_detections
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            # Use center point for ROI check
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            point_to_check = (center_x, center_y)
            
            # Check if in ROI
            if self.point_in_polygon_optimized(point_to_check, roi['polygon'], roi['bbox']):
                detection['area_type'] = 'roi'
                roi_detections.append(detection)
        
        return roi_detections
    
    def update_person_tracking(self, centers, current_time):
        """
        Update person tracking using SimpleTracker approach
        Returns: list of (track_id, (x, y), dwell_time)
        """
        assigned_ids = set()
        id_for_center = [-1] * len(centers)
        
        # Try to match existing tracks to new centers
        for track_id, track in self.person_tracking.items():
            if not track.get("active", True):
                continue
            best_idx = -1
            best_dist = float("inf")
            (tx, ty) = track["position"]
            
            for i, (cx, cy) in enumerate(centers):
                if id_for_center[i] != -1:
                    continue
                dist = self.euclidean((tx, ty), (cx, cy))
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
            
            if best_idx != -1 and best_dist <= self.max_track_dist:
                # Match found
                id_for_center[best_idx] = track_id
                assigned_ids.add(track_id)
                track["position"] = centers[best_idx]
                track["last_seen"] = current_time
        
        # Create new tracks for unmatched centers
        for i, (cx, cy) in enumerate(centers):
            if id_for_center[i] != -1:
                continue
            track_id = self.next_track_id
            self.next_track_id += 1
            self.person_tracking[track_id] = {
                "position": (cx, cy),
                "first_seen": current_time,
                "last_seen": current_time,
                "active": True
            }
            id_for_center[i] = track_id
            assigned_ids.add(track_id)
        
        # Deactivate tracks that weren't matched
        for track_id, track in self.person_tracking.items():
            if track_id not in assigned_ids:
                track["active"] = False
        
        # Clean up old inactive tracks
        timeout = 3.0
        to_remove = [
            tid for tid, data in self.person_tracking.items()
            if not data.get("active", True) and 
            (current_time - data["last_seen"]).total_seconds() > timeout
        ]
        for track_id in to_remove:
            del self.person_tracking[track_id]
        
        # Build result list with dwell times
        result = []
        for i, (cx, cy) in enumerate(centers):
            tid = id_for_center[i]
            if tid in self.person_tracking:
                track = self.person_tracking[tid]
                dwell = (track["last_seen"] - track["first_seen"]).total_seconds()
                result.append((tid, (cx, cy), dwell))
        
        return result
    
    def check_alert_conditions(self):
        """Check if crowd threshold is exceeded"""
        current_time = datetime.now()
        
        # Check if crowd threshold is exceeded
        alert_condition_met = self.crowd_count >= self.settings['crowd_threshold']
        
        if alert_condition_met:
            if self.alert_condition_start_time is None:
                self.alert_condition_start_time = current_time
                return None
            
            sustained_duration = (current_time - self.alert_condition_start_time).total_seconds()
            
            if sustained_duration < self.alert_condition_sustained_duration:
                return None
            
            # Check cooldown
            if (self.last_alert_time and
                (current_time - self.last_alert_time).total_seconds() < self.settings['alert_cooldown']):
                return None
            
            alert_info = {
                'type': 'crowd_alert',
                'message': f"Crowd gathering detected: {self.crowd_count} people in parking space",
                'crowd_count': self.crowd_count,
                'timestamp': current_time.isoformat(),
                'channel_id': self.channel_id
            }
            
            self.last_alert_time = current_time
            self.alert_condition_start_time = None
            return alert_info
        else:
            if self.alert_condition_start_time is not None:
                self.alert_condition_start_time = None
            return None
    
    def process_frame(self, frame):
        """Process a single frame for crowd detection with clustering and long-stay detection"""
        self.frame_count += 1
        original_frame = frame.copy()
        current_time = datetime.now()
        now_seconds = current_time.timestamp()
        
        # Clear GPU cache periodically
        if torch.cuda.is_available() and self.frame_count % 100 == 0:
            torch.cuda.empty_cache()
        
        # Add frame to GIF recorder
        self.gif_recorder.add_frame(original_frame)
        
        # Get frame dimensions
        frame_height, frame_width = frame.shape[:2]
        
        # Initialize variables
        centers = []
        bboxes = []
        gathering_clusters = []
        long_stay_centers = []
        is_stationary_gathering = False
        
        # Run YOLO detection every N frames
        if self.frame_count % self.cache_interval == 0:
            detections = self.detector.detect_persons(frame)
            
            # Classify detections into ROI
            roi_detections = self.classify_detections(detections, frame_width, frame_height)
            
            # Extract centers and bboxes from ROI detections
            for detection in roi_detections:
                x1, y1, x2, y2 = detection['bbox']
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                centers.append((cx, cy))
                bboxes.append((x1, y1, x2, y2))
            
            # Push current centers to history with timestamp
            self.center_history.append((current_time, centers))
            
            # Get only long-stay centers (people who've been in same area for DWELL_TIME)
            long_stay_centers = self.get_long_stay_centers(self.center_history, now_seconds)
            
            # Cluster only long-stay people
            clusters = self.cluster_points(long_stay_centers, self.settings['cluster_max_dist'])
            gathering_clusters = [c for c in clusters if len(c) >= self.settings['cluster_min_people']]
            
            # Store previous long-stay centers for movement calculation
            # (will be used in next frame)
            self.previous_centers = long_stay_centers.copy() if long_stay_centers else []
            
            # Cache for next frames
            self.detection_cache = {
                'centers': centers,
                'bboxes': bboxes,
                'long_stay_centers': long_stay_centers,
                'gathering_clusters': gathering_clusters
            }
            self.cache_frame_count = self.frame_count
        else:
            # Use cached data
            if self.detection_cache:
                centers = self.detection_cache.get('centers', [])
                bboxes = self.detection_cache.get('bboxes', [])
                long_stay_centers = self.detection_cache.get('long_stay_centers', [])
                gathering_clusters = self.detection_cache.get('gathering_clusters', [])
        
        # Update raw count (instantaneous)
        self.raw_crowd_now = len(centers)
        
        # Count long-stay people
        self.crowd_count = len(long_stay_centers)
        
        # Handle gathering clusters
        alert_info = None
        if gathering_clusters:
            largest_cluster = max(gathering_clusters, key=len)
            cluster_size = len(largest_cluster)
            self.active_cluster_size = cluster_size
            
            # Calculate movement to determine if stationary
            avg_movement = float('inf')
            if self.previous_centers and len(long_stay_centers) > 0:
                # Calculate movement of cluster members
                avg_movement = self.calculate_cluster_movement(
                    long_stay_centers, self.previous_centers, largest_cluster
                )
                is_stationary_gathering = avg_movement < self.settings['min_movement_threshold']
            
            # Track cluster duration
            if self.last_cluster_start_time is None:
                self.last_cluster_start_time = current_time
            
            duration = (current_time - self.last_cluster_start_time).total_seconds()
            
            # Check if should alert (stationary gathering for GATHERING_HOLD_SECONDS)
            should_alert = (
                duration >= self.settings['gathering_hold_seconds'] and
                is_stationary_gathering and
                (not self.last_alert_time or 
                 (current_time - self.last_alert_time).total_seconds() > self.settings['alert_cooldown'])
            )
            
            if should_alert:
                status_label = "STATIONARY" if is_stationary_gathering else "MOBILE"
                alert_message = (
                    f"{status_label} gathering of {cluster_size} people detected "
                    f"for {duration:.0f} seconds in {self.channel_id}"
                )
                
                alert_info = {
                    'type': 'crowd_alert',
                    'message': alert_message,
                    'crowd_count': cluster_size,
                    'long_stay_count': self.crowd_count,
                    'duration': duration,
                    'is_stationary': is_stationary_gathering,
                    'timestamp': current_time.isoformat(),
                    'channel_id': self.channel_id
                }
                
                logger.warning(f"Crowd alert triggered: {alert_message}")
                
                # Start GIF recording
                self.gif_recorder.start_alert_recording(alert_info)
                self.gif_recorder.add_alert_frame(original_frame)
                
                # Log alert to database
                if self.db_manager:
                    try:
                        if self.app:
                            with self.app.app_context():
                                self.db_manager.log_alert(
                                    self.channel_id,
                                    'crowd_alert',
                                    alert_message,
                                    alert_data={
                                        'crowd_count': cluster_size,
                                        'long_stay_count': self.crowd_count,
                                        'duration': duration,
                                        'is_stationary': is_stationary_gathering,
                                        'threshold': self.settings['crowd_threshold']
                                    }
                                )
                        else:
                            self.db_manager.log_alert(
                                self.channel_id,
                                'crowd_alert',
                                alert_message,
                                alert_data={
                                    'crowd_count': cluster_size,
                                    'long_stay_count': self.crowd_count,
                                    'duration': duration,
                                    'is_stationary': is_stationary_gathering,
                                    'threshold': self.settings['crowd_threshold']
                                }
                            )
                    except Exception as e:
                        logger.error(f"Database logging error for crowd alert: {e}")
                
                # Send real-time alert
                self.socketio.emit('crowd_alert', alert_info)
                self.last_alert_time = current_time
        else:
            self.active_cluster_size = 0
            self.last_cluster_start_time = None
        
        # Continue recording if alert is in progress
        if self.gif_recorder.is_recording_alert:
            self.gif_recorder.add_alert_frame(original_frame)
            
            # Check if recording just ended
            if hasattr(self.gif_recorder, 'alert_end_time') and self.gif_recorder.alert_end_time:
                # Save GIF to database
                if self.db_manager:
                    try:
                        if self.app:
                            with self.app.app_context():
                                gif_filename = f"crowd_alert_{self.channel_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.gif"
                                gif_path = os.path.join('static', 'alerts', gif_filename)
                                
                                # Save the GIF file
                                saved_path = self.gif_recorder.save_gif(gif_path) if hasattr(self.gif_recorder, 'save_gif') else None
                                if saved_path:
                                    # Update the alert entry with GIF info
                                    self.db_manager.save_alert_gif(
                                        self.channel_id,
                                        'crowd_alert',
                                        {'filename': gif_filename, 'path': gif_path, 'gif_path': saved_path},
                                        alert_message=alert_info['message'] if alert_info else "Crowd detected",
                                        alert_data={'crowd_count': self.crowd_count}
                                    )
                    except Exception as e:
                        logger.error(f"Error saving alert GIF: {e}")
        
        # Annotate frame with improved visualization
        annotated_frame = self._annotate_frame_improved(
            original_frame, centers, bboxes, long_stay_centers, 
            gathering_clusters, is_stationary_gathering, frame_width, frame_height
        )
        
        # Send real-time update
        if self.frame_count % 15 == 0:  # Every 15 frames
            self.socketio.emit('crowd_update', {
                'channel_id': self.channel_id,
                'crowd_count': self.crowd_count,
                'raw_crowd_now': self.raw_crowd_now,
                'active_cluster_size': self.active_cluster_size,
                'threshold': self.settings['crowd_threshold']
            })
        
        return annotated_frame
    
    def _annotate_frame_improved(self, frame, centers, bboxes, long_stay_centers, 
                                gathering_clusters, is_stationary, frame_width, frame_height):
        """Annotate frame with ROI, detections, clusters, and gathering boxes"""
        annotated = frame.copy()
        
        # Draw ROI
        roi_polygon = self.get_roi_pixels(frame_width, frame_height)
        if roi_polygon:
            pts = np.array(roi_polygon, np.int32)
            cv2.polylines(annotated, [pts], True, (0, 255, 255), 2)  # Yellow
        
        # Draw all person detections
        for (x1, y1, x2, y2) in bboxes:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)
        
        # Draw gathering clusters with bounding boxes
        if gathering_clusters:
            largest_cluster = max(gathering_clusters, key=len)
            cluster_size = len(largest_cluster)
            
            # Get bounding box for cluster
            if long_stay_centers and largest_cluster:
                xs = [long_stay_centers[idx][0] for idx in largest_cluster if idx < len(long_stay_centers)]
                ys = [long_stay_centers[idx][1] for idx in largest_cluster if idx < len(long_stay_centers)]
                
                if xs and ys:
                    min_x, max_x = min(xs), max(xs)
                    min_y, max_y = min(ys), max(ys)
                    pad = 20
                    min_x = max(0, min_x - pad)
                    max_x = min(frame_width - 1, max_x + pad)
                    min_y = max(0, min_y - pad)
                    max_y = min(frame_height - 1, max_y + pad)
                    
                    # Color: red for stationary, orange for mobile
                    color = (0, 0, 255) if is_stationary else (0, 165, 255)
                    cv2.rectangle(annotated, (min_x, min_y), (max_x, max_y), color, 2)
                    
                    status = "STATIONARY" if is_stationary else "MOBILE"
                    cv2.putText(annotated, f"{status}: {cluster_size}",
                               (min_x, max(min_y - 10, 20)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw text overlays
        people_count = len(centers)  # Total current
        long_stay_count = len(long_stay_centers)  # Only long-stay
        
        cv2.putText(annotated, f"People: {people_count} | Long-stay: {long_stay_count}",
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        cv2.putText(annotated, f"Channel: {self.channel_id}",
                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        if self.active_cluster_size >= self.settings['cluster_min_people']:
            cv2.putText(annotated, f"Gathering: {self.active_cluster_size}",
                       (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return annotated
    
    def _annotate_frame(self, frame, detections, frame_width, frame_height):
        """Legacy annotation method - kept for compatibility"""
        return self._annotate_frame_improved(
            frame, [], [], [], [], False, frame_width, frame_height
        )
    
    def get_status(self):
        """Get module status information"""
        return {
            'module': 'CrowdDetection',
            'channel_id': self.channel_id,
            'status': 'active',
            'crowd_count': self.crowd_count,
            'raw_crowd_now': self.raw_crowd_now,
            'active_cluster_size': self.active_cluster_size,
            'threshold': self.settings['crowd_threshold'],
            'roi_config': self.roi_config,
            'frame_count': self.frame_count,
            'tracked_persons': len(self.person_tracking)
        }

