"""
GIF Recorder Module
Records video frames as GIF for alert scenarios
"""
import cv2
import numpy as np
import logging
from datetime import datetime
import os
from collections import deque
import threading
from PIL import Image
import io

logger = logging.getLogger(__name__)

class AlertGifRecorder:
    def __init__(self, buffer_size=90, gif_duration=3.0, fps=30):
        """
        Initialize GIF recorder for alerts
        
        Args:
            buffer_size: Number of frames to keep in circular buffer (default: 90 frames = 3 seconds at 30fps)
            gif_duration: Duration of GIF in seconds
            fps: Target FPS for recording
        """
        self.buffer_size = buffer_size
        self.gif_duration = gif_duration
        self.fps = fps
        
        # Circular buffer to store recent frames
        self.frame_buffer = deque(maxlen=buffer_size)
        self.buffer_lock = threading.Lock()
        
        # Recording state
        self.is_recording_alert = False
        self.alert_frames = []
        self.alert_start_time = None
        self.alert_end_time = None
        self.last_gif_info = None  # Store last created GIF info (frame_count, duration, etc.)
        
        # GIF settings
        self.gif_width = 480  # Reduced size for faster processing
        self.gif_height = 360
        self.gif_quality = 85
        
        # Storage
        self.alerts_dir = "static/alerts"
        os.makedirs(self.alerts_dir, exist_ok=True)
        
        logger.info(f"GIF Recorder initialized: buffer={buffer_size} frames, duration={gif_duration}s")
    
    def add_frame(self, frame):
        """
        Add frame to circular buffer for potential GIF creation
        
        Args:
            frame: OpenCV frame (BGR format)
        """
        if frame is None or frame.size == 0:
            return
        
        # Resize frame for efficient storage
        resized_frame = cv2.resize(frame, (self.gif_width, self.gif_height))
        
        with self.buffer_lock:
            # Convert BGR to RGB for PIL
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            self.frame_buffer.append({
                'frame': rgb_frame,
                'timestamp': datetime.now()
            })
    
    def start_alert_recording(self, alert_info):
        """
        Start recording alert scenario
        
        Args:
            alert_info: Dictionary containing alert information
        """
        if self.is_recording_alert:
            logger.warning("Alert recording already in progress")
            return
        
        self.is_recording_alert = True
        self.alert_start_time = datetime.now()
        self.alert_frames = []
        
        # Copy recent frames from buffer (pre-alert context)
        with self.buffer_lock:
            # Get frames from buffer (last 1 second before alert)
            pre_alert_frames = list(self.frame_buffer)[-30:]  # 1 second at 30fps
            self.alert_frames.extend([f['frame'] for f in pre_alert_frames])
        
        logger.info(f"Started alert GIF recording for: {alert_info.get('message', 'Unknown alert')}")
    
    def add_alert_frame(self, frame):
        """
        Add frame during alert recording
        
        Args:
            frame: OpenCV frame during alert
        """
        if not self.is_recording_alert:
            return
        
        if frame is None or frame.size == 0:
            return
        
        # Resize and convert to RGB
        resized_frame = cv2.resize(frame, (self.gif_width, self.gif_height))
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        
        self.alert_frames.append(rgb_frame)
        
        # Stop recording after specified duration
        if self.alert_start_time:
            elapsed_time = (datetime.now() - self.alert_start_time).total_seconds()
            if elapsed_time >= self.gif_duration:
                self.stop_alert_recording()
    
    def stop_alert_recording(self):
        """Stop alert recording and create GIF"""
        if not self.is_recording_alert:
            return None
        
        self.is_recording_alert = False
        self.alert_end_time = datetime.now()
        
        if not self.alert_frames:
            logger.warning("No frames recorded for alert GIF")
            return None
        
        try:
            # Create GIF filename with timestamp
            timestamp = self.alert_start_time.strftime("%Y%m%d_%H%M%S")
            gif_filename = f"alert_{timestamp}.gif"
            gif_path = os.path.join(self.alerts_dir, gif_filename)
            
            # Store frame count before clearing
            frame_count = len(self.alert_frames)
            duration = (self.alert_end_time - self.alert_start_time).total_seconds()
            
            # Create GIF
            self._create_gif(self.alert_frames, gif_path)
            
            logger.info(f"Alert GIF created: {gif_path} ({frame_count} frames)")
            
            # Create return info before resetting state
            gif_info = {
                'gif_path': gif_path,
                'gif_filename': gif_filename,
                'frame_count': frame_count,
                'duration': duration,
                'alert_time': self.alert_start_time.isoformat()
            }
            
            # Store gif_info for later retrieval
            self.last_gif_info = gif_info
            
            # Reset recording state
            self.alert_frames = []
            
            return gif_info
            
        except Exception as e:
            logger.error(f"Error creating alert GIF: {e}")
            self.alert_frames = []
            self.last_gif_info = None
            return None
    
    def get_last_gif_info(self):
        """Get the last created GIF info (frame_count, duration, etc.)"""
        return self.last_gif_info
    
    def _create_gif(self, frames, output_path):
        """
        Create GIF from frames using PIL
        
        Args:
            frames: List of RGB frames (numpy arrays)
            output_path: Output file path for GIF
        """
        if not frames:
            raise ValueError("No frames provided for GIF creation")
        
        # Convert numpy arrays to PIL Images
        pil_images = []
        for frame in frames:
            if frame.shape[-1] == 3:  # RGB
                pil_image = Image.fromarray(frame)
                pil_images.append(pil_image)
        
        if not pil_images:
            raise ValueError("No valid frames for GIF creation")
        
        # Calculate frame duration in milliseconds
        frame_duration = int(1000 / self.fps)  # milliseconds per frame
        
        # Save as GIF
        pil_images[0].save(
            output_path,
            save_all=True,
            append_images=pil_images[1:],
            duration=frame_duration,
            loop=0,  # Infinite loop
            optimize=True,
            quality=self.gif_quality
        )
        
        logger.info(f"GIF saved: {output_path} ({len(pil_images)} frames, {frame_duration}ms/frame)")
    
    def create_manual_gif(self, frames, alert_info, output_filename=None):
        """
        Create GIF manually from provided frames
        
        Args:
            frames: List of OpenCV frames
            alert_info: Alert information dictionary
            output_filename: Optional custom filename
            
        Returns:
            Dictionary with GIF information or None if failed
        """
        if not frames:
            logger.warning("No frames provided for manual GIF creation")
            return None
        
        try:
            # Generate filename
            if not output_filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"alert_{timestamp}.gif"
            
            gif_path = os.path.join(self.alerts_dir, output_filename)
            
            # Process frames
            processed_frames = []
            for frame in frames:
                if frame is not None and frame.size > 0:
                    resized = cv2.resize(frame, (self.gif_width, self.gif_height))
                    rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                    processed_frames.append(rgb_frame)
            
            if not processed_frames:
                logger.warning("No valid frames after processing")
                return None
            
            # Create GIF
            self._create_gif(processed_frames, gif_path)
            
            return {
                'gif_path': gif_path,
                'gif_filename': output_filename,
                'frame_count': len(processed_frames),
                'alert_info': alert_info,
                'creation_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error creating manual GIF: {e}")
            return None
    
    def get_recent_gifs(self, limit=10):
        """
        Get list of recent alert GIFs
        
        Args:
            limit: Maximum number of GIFs to return
            
        Returns:
            List of GIF file information
        """
        try:
            gif_files = []
            
            if os.path.exists(self.alerts_dir):
                for filename in os.listdir(self.alerts_dir):
                    if filename.endswith('.gif'):
                        file_path = os.path.join(self.alerts_dir, filename)
                        file_stat = os.stat(file_path)
                        
                        gif_files.append({
                            'filename': filename,
                            'path': file_path,
                            'size': file_stat.st_size,
                            'created': datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
                            'modified': datetime.fromtimestamp(file_stat.st_mtime).isoformat()
                        })
            
            # Sort by creation time (newest first)
            gif_files.sort(key=lambda x: x['created'], reverse=True)
            
            return gif_files[:limit]
            
        except Exception as e:
            logger.error(f"Error getting recent GIFs: {e}")
            return []
    
    def cleanup_old_gifs(self, max_age_days=7, max_count=100):
        """
        Clean up old GIF files to save storage space
        
        Args:
            max_age_days: Maximum age of GIFs to keep (in days)
            max_count: Maximum number of GIFs to keep
        """
        try:
            if not os.path.exists(self.alerts_dir):
                return
            
            gif_files = []
            cutoff_time = datetime.now().timestamp() - (max_age_days * 24 * 3600)
            
            # Get all GIF files with timestamps
            for filename in os.listdir(self.alerts_dir):
                if filename.endswith('.gif'):
                    file_path = os.path.join(self.alerts_dir, filename)
                    file_stat = os.stat(file_path)
                    
                    gif_files.append({
                        'path': file_path,
                        'filename': filename,
                        'created': file_stat.st_ctime
                    })
            
            # Sort by creation time (oldest first)
            gif_files.sort(key=lambda x: x['created'])
            
            removed_count = 0
            
            # Remove files older than max_age_days
            for gif_file in gif_files:
                if gif_file['created'] < cutoff_time:
                    os.remove(gif_file['path'])
                    removed_count += 1
                    logger.info(f"Removed old GIF: {gif_file['filename']}")
            
            # Remove excess files if more than max_count
            remaining_files = [f for f in gif_files if f['created'] >= cutoff_time]
            if len(remaining_files) > max_count:
                excess_files = remaining_files[:-max_count]  # Keep newest max_count files
                for gif_file in excess_files:
                    os.remove(gif_file['path'])
                    removed_count += 1
                    logger.info(f"Removed excess GIF: {gif_file['filename']}")
            
            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} old GIF files")
            
        except Exception as e:
            logger.error(f"Error cleaning up GIFs: {e}")
    
    def get_stats(self):
        """Get recorder statistics"""
        try:
            gif_count = 0
            total_size = 0
            
            if os.path.exists(self.alerts_dir):
                for filename in os.listdir(self.alerts_dir):
                    if filename.endswith('.gif'):
                        file_path = os.path.join(self.alerts_dir, filename)
                        file_size = os.path.getsize(file_path)
                        gif_count += 1
                        total_size += file_size
            
            return {
                'buffer_size': self.buffer_size,
                'gif_duration': self.gif_duration,
                'gif_resolution': f"{self.gif_width}x{self.gif_height}",
                'stored_gifs': gif_count,
                'total_storage': total_size,
                'storage_mb': round(total_size / (1024 * 1024), 2),
                'is_recording': self.is_recording_alert,
                'buffer_frames': len(self.frame_buffer)
            }
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}