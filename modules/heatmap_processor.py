"""
Heatmap Processor Module
Generates crowd density heatmaps by tracking person presence over time
"""
import cv2
import numpy as np
import logging
from datetime import datetime
from collections import defaultdict
import time

logger = logging.getLogger(__name__)

# Heatmap Configuration
HEATMAP_GRID_SIZE = 30  # Grid cell size in pixels
HEATMAP_TIME_THRESHOLD_SEC = 3600.0  # Time window to consider for heatmap (1 hour = 60 * 60)
HEATMAP_PEOPLE_THRESHOLD = 5  # Minimum people count to consider as hotspot
HEATMAP_RADIUS = 20  # Radius for heatmap circles
HEATMAP_BLUR_KERNEL = (91, 91)  # Gaussian blur kernel size

class HeatmapProcessor:
    def __init__(self, channel_id, socketio, db_manager=None, app=None):
        """
        Initialize Heatmap Processor
        
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
        
        # Initialize YOLO detector
        from .yolo_detector import YOLODetector
        self.detector = YOLODetector(confidence_threshold=0.6)
        
        # Heatmap data storage
        self.heatmap_data = defaultdict(lambda: {'timestamps': []})
        self.hotspots = []
        self.last_logic_update = 0
        
        # Enhanced metrics tracking
        self.total_detections = 0
        self.peak_hotspot_count = 0
        self.peak_person_count = 0
        
        # Frame storage
        self.latest_frame = None
        self.frame_count = 0
        
        # Snapshot settings
        self.snapshot_interval = 3600  # Save snapshot every 1 hour (60 * 60 = 3600 seconds)
        self.last_snapshot_time = time.time()
        
        logger.info(f"Heatmap Processor initialized for channel {channel_id}")
    
    def _update_heatmap_logic(self):
        """Update heatmap data by removing old timestamps and identifying hotspots"""
        current_time = time.time()
        new_hotspots = []
        
        for key in list(self.heatmap_data.keys()):
            cell = self.heatmap_data[key]
            # Remove timestamps older than threshold
            cell['timestamps'] = [
                ts for ts in cell['timestamps'] 
                if current_time - ts <= HEATMAP_TIME_THRESHOLD_SEC
            ]
            
            person_count = len(cell['timestamps'])
            if person_count >= HEATMAP_PEOPLE_THRESHOLD:
                heat_level = person_count // HEATMAP_PEOPLE_THRESHOLD
                col, row = map(int, key.split(','))
                new_hotspots.append({
                    'col': col, 
                    'row': row, 
                    'heatLevel': heat_level,
                    'personCount': person_count
                })
            elif person_count == 0:
                # Remove empty cells
                del self.heatmap_data[key]
        
        self.hotspots = new_hotspots
    
    def _apply_heatmap_overlay(self, frame, hotspots):
        """
        Apply heatmap overlay to the frame
        
        Args:
            frame: Input video frame
            hotspots: List of hotspot locations and intensities
            
        Returns:
            Frame with heatmap overlay
        """
        if not hotspots:
            return frame
        
        # Create heatmap canvas
        heatmap_canvas = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        
        # Draw circles for each hotspot
        for spot in hotspots:
            col, row, heat_level = spot['col'], spot['row'], spot['heatLevel']
            center_x = int((col + 0.5) * HEATMAP_GRID_SIZE)
            center_y = int((row + 0.5) * HEATMAP_GRID_SIZE)
            
            # Calculate intensity (capped at 255)
            intensity = min(heat_level * 50, 255)
            cv2.circle(heatmap_canvas, (center_x, center_y), HEATMAP_RADIUS, int(intensity), -1)
        
        # Apply Gaussian blur for smooth heatmap
        blurred_heatmap = cv2.GaussianBlur(heatmap_canvas, HEATMAP_BLUR_KERNEL, 0)
        
        # Apply color map (JET colormap: blue->green->yellow->red)
        colored_heatmap = cv2.applyColorMap(blurred_heatmap, cv2.COLORMAP_JET)
        
        # Blend with original frame
        result = cv2.addWeighted(frame, 0.6, colored_heatmap, 0.4, 0)
        
        return result
    
    def process_frame(self, frame):
        """
        Process a single frame to update heatmap data
        
        Args:
            frame: Input video frame
            
        Returns:
            Annotated frame with heatmap overlay
        """
        self.frame_count += 1
        self.latest_frame = frame.copy()
        
        # Detect persons in frame
        detections = self.detector.detect_persons(frame)
        
        # Update metrics
        self.total_detections += len(detections)
        if len(detections) > self.peak_person_count:
            self.peak_person_count = len(detections)
        
        # Update heatmap data with detections
        current_time = time.time()
        for detection in detections:
            # Use bottom center of bounding box for position
            x_center = detection['bottom_center'][0]
            y_bottom = detection['bottom_center'][1]
            
            # Calculate grid cell
            col = int(x_center // HEATMAP_GRID_SIZE)
            row = int(y_bottom // HEATMAP_GRID_SIZE)
            
            # Add timestamp to this cell
            self.heatmap_data[f"{col},{row}"]['timestamps'].append(current_time)
        
        # Update heatmap logic every second
        if current_time - self.last_logic_update > 1.0:
            self._update_heatmap_logic()
            self.last_logic_update = current_time
            
            # Track peak hotspot count
            if len(self.hotspots) > self.peak_hotspot_count:
                self.peak_hotspot_count = len(self.hotspots)
            
            # Send real-time hotspot updates
            self.socketio.emit('heatmap_update', {
                'channel_id': self.channel_id,
                'hotspot_count': len(self.hotspots),
                'hotspots': self.hotspots,
                'total_detections': len(detections),
                'peak_hotspot_count': self.peak_hotspot_count,
                'active_cells': len(self.heatmap_data)
            })
        
        # Save periodic snapshots
        if current_time - self.last_snapshot_time >= self.snapshot_interval:
            self._save_snapshot()
            self.last_snapshot_time = current_time
        
        # Apply heatmap overlay to frame
        annotated_frame = self._apply_heatmap_overlay(frame.copy(), self.hotspots)
        
        # Draw detection boxes
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw statistics
        stats_text = [
            f"Detections: {len(detections)}",
            f"Hotspots: {len(self.hotspots)}",
            f"Active Cells: {len(self.heatmap_data)}"
        ]
        
        y_offset = 30
        for text in stats_text:
            cv2.putText(annotated_frame, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            y_offset += 30
        
        # Return structured result format consistent with other modules
        return {
            'frame': annotated_frame,
            'status': {
                'detections': len(detections),
                'hotspot_count': len(self.hotspots),
                'active_cells': len(self.heatmap_data),
                'hotspots': self.hotspots
            },
            'metadata': {
                'frame_count': self.frame_count,
                'timestamp': current_time,
                'channel_id': self.channel_id
            }
        }
    
    def _save_snapshot(self):
        """Save periodic heatmap snapshot to database"""
        if self.latest_frame is None or not self.db_manager:
            return
        
        try:
            # Create snapshot with heatmap overlay
            snapshot = self._apply_heatmap_overlay(
                self.latest_frame.copy(), 
                self.hotspots
            )
            
            # Save to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot_filename = f"heatmap_{self.channel_id}_{timestamp}.jpg"
            snapshot_path = f"static/heatmaps/{snapshot_filename}"
            
            # Ensure directory exists
            import os
            os.makedirs("static/heatmaps", exist_ok=True)
            
            cv2.imwrite(snapshot_path, snapshot)
            
            # Save to database
            if self.app:
                with self.app.app_context():
                    self.db_manager.save_heatmap_snapshot(
                        channel_id=self.channel_id,
                        snapshot_path=snapshot_path,
                        hotspot_count=len(self.hotspots),
                        hotspots=self.hotspots
                    )
            else:
                self.db_manager.save_heatmap_snapshot(
                    channel_id=self.channel_id,
                    snapshot_path=snapshot_path,
                    hotspot_count=len(self.hotspots),
                    hotspots=self.hotspots
                )
            
            logger.info(f"Heatmap snapshot saved: {snapshot_filename}")
            
        except Exception as e:
            logger.error(f"Error saving heatmap snapshot: {e}")
    
    def get_snapshot_frame(self):
        """
        Get current frame with heatmap overlay for snapshot
        
        Returns:
            Frame with heatmap overlay or None
        """
        if self.latest_frame is None:
            return None
        
        return self._apply_heatmap_overlay(self.latest_frame.copy(), self.hotspots)
    
    def get_current_status(self):
        """Get current heatmap status"""
        return {
            'hotspot_count': len(self.hotspots),
            'active_cells': len(self.heatmap_data),
            'hotspots': self.hotspots,
            'peak_hotspot_count': self.peak_hotspot_count,
            'peak_person_count': self.peak_person_count,
            'total_detections': self.total_detections
        }
    
    def get_status(self):
        """Get module status information"""
        return {
            'module': 'HeatmapProcessor',
            'channel_id': self.channel_id,
            'status': 'active',
            'hotspot_count': len(self.hotspots),
            'active_cells': len(self.heatmap_data),
            'frame_count': self.frame_count,
            'peak_hotspot_count': self.peak_hotspot_count,
            'peak_person_count': self.peak_person_count
        }
