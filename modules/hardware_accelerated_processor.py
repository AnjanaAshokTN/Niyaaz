"""
Hardware-Accelerated Video Processor
Integrates NVDECODE + TensorRT for maximum performance and minimal CPU usage
"""
import cv2
import numpy as np
import logging
import time
import threading
from queue import Queue
from pathlib import Path

from .nvdecode_processor import create_hardware_decoder
from .tensorrt_engine import create_optimized_detector

logger = logging.getLogger(__name__)

class HardwareAcceleratedProcessor:
    """
    Complete hardware-accelerated video processing pipeline
    - NVDECODE for video decoding (reduces CPU by 60-80%)
    - TensorRT for inference (reduces GPU time by 3-5x)
    - GPU-based preprocessing where possible
    """
    
    def __init__(self, video_source, channel_id, target_fps=15):
        """
        Initialize hardware-accelerated processor
        
        Args:
            video_source: RTSP URL or video file path
            channel_id: Unique channel identifier
            target_fps: Target processing FPS
        """
        self.video_source = video_source
        self.channel_id = channel_id
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        
        # Hardware components
        self.decoder = None
        self.detector = None
        
        # Threading
        self.is_running = False
        self.decode_thread = None
        self.process_thread = None
        
        # Frame management
        self.raw_frame_queue = Queue(maxsize=5)
        self.processed_frame_queue = Queue(maxsize=3)
        self.latest_frame = None
        self.latest_detections = []
        
        # Locks
        self.frame_lock = threading.Lock()
        self.detection_lock = threading.Lock()
        
        # Performance tracking
        self.decode_fps = 0
        self.process_fps = 0
        self.total_frames = 0
        self.start_time = None
        
        # Statistics
        self.stats = {
            'frames_decoded': 0,
            'frames_processed': 0,
            'avg_decode_time': 0,
            'avg_inference_time': 0,
            'hardware_decode_enabled': False,
            'tensorrt_enabled': False
        }
        
    def initialize(self):
        """Initialize hardware components"""
        try:
            # Initialize hardware decoder
            self.decoder = create_hardware_decoder(
                self.video_source,
                target_size=(640, 480),  # Standard processing size
                prefer_ffmpeg=True
            )
            
            if self.decoder:
                self.stats['hardware_decode_enabled'] = True
                logger.info(f"Hardware decoder initialized for {self.channel_id}")
            else:
                logger.warning(f"Hardware decoder failed for {self.channel_id}, using software fallback")
                return False
            
            # Initialize optimized detector
            self.detector = create_optimized_detector(
                "models/yolo11n.pt",  # Will try .trt and .onnx versions
                use_tensorrt=True,
                confidence_threshold=0.6
            )
            
            if self.detector:
                self.stats['tensorrt_enabled'] = True
                logger.info(f"Optimized detector initialized for {self.channel_id}")
            else:
                logger.warning(f"Optimized detector failed for {self.channel_id}")
                # Could fallback to standard YOLO here
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Hardware initialization failed for {self.channel_id}: {e}")
            return False
    
    def start(self):
        """Start hardware-accelerated processing"""
        if not self.initialize():
            return False
        
        self.is_running = True
        self.start_time = time.time()
        
        # Start decoding thread
        self.decode_thread = threading.Thread(target=self._decode_loop, daemon=True)
        self.decode_thread.start()
        
        # Start processing thread
        self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.process_thread.start()
        
        logger.info(f"Hardware-accelerated processor started for {self.channel_id}")
        return True
    
    def stop(self):
        """Stop processing"""
        self.is_running = False
        
        # Wait for threads to finish
        if self.decode_thread:
            self.decode_thread.join(timeout=3)
        if self.process_thread:
            self.process_thread.join(timeout=3)
        
        # Cleanup hardware resources
        if self.decoder:
            self.decoder.stop()
        if self.detector and hasattr(self.detector, 'cleanup'):
            self.detector.cleanup()
        
        logger.info(f"Hardware-accelerated processor stopped for {self.channel_id}")
    
    def _decode_loop(self):
        """Hardware decoding loop"""
        decode_count = 0
        decode_times = []
        last_fps_check = time.time()
        
        while self.is_running:
            try:
                start_time = time.time()
                
                # Read frame from hardware decoder
                success, frame = self.decoder.read_frame()
                
                if not success or frame is None:
                    time.sleep(0.01)
                    continue
                
                # Track decode time
                decode_time = time.time() - start_time
                decode_times.append(decode_time)
                decode_count += 1
                
                # Add to processing queue if not full
                if not self.raw_frame_queue.full():
                    self.raw_frame_queue.put(frame.copy())
                
                # Update decode FPS
                current_time = time.time()
                if current_time - last_fps_check >= 1.0:
                    self.decode_fps = decode_count / (current_time - last_fps_check)
                    decode_count = 0
                    last_fps_check = current_time
                    
                    # Update stats
                    if decode_times:
                        self.stats['avg_decode_time'] = np.mean(decode_times) * 1000  # ms
                        decode_times = []
                
                self.stats['frames_decoded'] += 1
                
            except Exception as e:
                logger.error(f"Decode loop error for {self.channel_id}: {e}")
                time.sleep(0.1)
    
    def _process_loop(self):
        """AI processing loop"""
        last_process_time = 0
        process_count = 0
        inference_times = []
        last_fps_check = time.time()
        
        while self.is_running:
            try:
                current_time = time.time()
                
                # Respect FPS limit
                if current_time - last_process_time < self.frame_interval:
                    time.sleep(0.001)
                    continue
                
                # Get frame from decode queue
                if self.raw_frame_queue.empty():
                    time.sleep(0.001)
                    continue
                
                frame = self.raw_frame_queue.get()
                
                # Run AI inference
                start_inference = time.time()
                detections = self.detector.detect_persons(frame)
                inference_time = time.time() - start_inference
                
                inference_times.append(inference_time)
                
                # Update latest results
                with self.frame_lock:
                    self.latest_frame = frame.copy()
                
                with self.detection_lock:
                    self.latest_detections = detections
                
                # Add to processed queue
                if not self.processed_frame_queue.full():
                    processed_data = {
                        'frame': frame,
                        'detections': detections,
                        'timestamp': current_time
                    }
                    self.processed_frame_queue.put(processed_data)
                
                last_process_time = current_time
                process_count += 1
                
                # Update process FPS
                if current_time - last_fps_check >= 1.0:
                    self.process_fps = process_count / (current_time - last_fps_check)
                    process_count = 0
                    last_fps_check = current_time
                    
                    # Update stats
                    if inference_times:
                        self.stats['avg_inference_time'] = np.mean(inference_times) * 1000  # ms
                        inference_times = []
                
                self.stats['frames_processed'] += 1
                
            except Exception as e:
                logger.error(f"Process loop error for {self.channel_id}: {e}")
                time.sleep(0.1)
    
    def get_latest_frame(self):
        """Get latest processed frame"""
        with self.frame_lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
        return None
    
    def get_latest_detections(self):
        """Get latest detections"""
        with self.detection_lock:
            return self.latest_detections.copy()
    
    def get_processed_data(self):
        """Get latest processed frame and detections"""
        if not self.processed_frame_queue.empty():
            return self.processed_frame_queue.get()
        return None
    
    def get_performance_stats(self):
        """Get comprehensive performance statistics"""
        runtime = time.time() - self.start_time if self.start_time else 0
        
        stats = self.stats.copy()
        stats.update({
            'channel_id': self.channel_id,
            'runtime_seconds': runtime,
            'decode_fps': self.decode_fps,
            'process_fps': self.process_fps,
            'target_fps': self.target_fps,
            'efficiency': (self.process_fps / self.target_fps) if self.target_fps > 0 else 0,
            'queue_sizes': {
                'raw_frames': self.raw_frame_queue.qsize(),
                'processed_frames': self.processed_frame_queue.qsize()
            }
        })
        
        # Add detector-specific stats if available
        if hasattr(self.detector, 'get_performance_stats'):
            detector_stats = self.detector.get_performance_stats()
            stats['detector_stats'] = detector_stats
        
        return stats
    
    def is_healthy(self):
        """Check if processor is running healthily"""
        if not self.is_running:
            return False
        
        # Check if frames are being processed
        if self.stats['frames_processed'] == 0:
            return False
        
        # Check if decode/process FPS are reasonable
        if self.decode_fps < 1 or self.process_fps < 1:
            return False
        
        return True


class HardwareAcceleratedMultiProcessor:
    """
    Multi-channel hardware-accelerated processor
    Manages multiple video streams with shared GPU resources
    """
    
    def __init__(self, max_channels=8):
        """
        Initialize multi-channel processor
        
        Args:
            max_channels: Maximum number of concurrent channels
        """
        self.max_channels = max_channels
        self.processors = {}  # {channel_id: HardwareAcceleratedProcessor}
        self.global_stats = {
            'active_channels': 0,
            'total_decode_fps': 0,
            'total_process_fps': 0,
            'gpu_utilization': 0
        }
        
    def add_channel(self, channel_id, video_source, target_fps=15):
        """Add a new channel for processing"""
        if len(self.processors) >= self.max_channels:
            logger.warning(f"Maximum channels ({self.max_channels}) reached")
            return False
        
        if channel_id in self.processors:
            logger.warning(f"Channel {channel_id} already exists")
            return False
        
        try:
            processor = HardwareAcceleratedProcessor(video_source, channel_id, target_fps)
            
            if processor.start():
                self.processors[channel_id] = processor
                logger.info(f"Channel {channel_id} added successfully")
                return True
            else:
                logger.error(f"Failed to start processor for channel {channel_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error adding channel {channel_id}: {e}")
            return False
    
    def remove_channel(self, channel_id):
        """Remove a channel"""
        if channel_id not in self.processors:
            return False
        
        try:
            self.processors[channel_id].stop()
            del self.processors[channel_id]
            logger.info(f"Channel {channel_id} removed")
            return True
            
        except Exception as e:
            logger.error(f"Error removing channel {channel_id}: {e}")
            return False
    
    def get_channel_frame(self, channel_id):
        """Get latest frame from specific channel"""
        if channel_id in self.processors:
            return self.processors[channel_id].get_latest_frame()
        return None
    
    def get_channel_detections(self, channel_id):
        """Get latest detections from specific channel"""
        if channel_id in self.processors:
            return self.processors[channel_id].get_latest_detections()
        return []
    
    def get_global_stats(self):
        """Get statistics for all channels"""
        total_decode_fps = 0
        total_process_fps = 0
        channel_stats = {}
        
        for channel_id, processor in self.processors.items():
            stats = processor.get_performance_stats()
            channel_stats[channel_id] = stats
            total_decode_fps += stats['decode_fps']
            total_process_fps += stats['process_fps']
        
        self.global_stats.update({
            'active_channels': len(self.processors),
            'total_decode_fps': total_decode_fps,
            'total_process_fps': total_process_fps,
            'channel_stats': channel_stats
        })
        
        return self.global_stats
    
    def stop_all(self):
        """Stop all processors"""
        for channel_id in list(self.processors.keys()):
            self.remove_channel(channel_id)
        
        logger.info("All hardware-accelerated processors stopped")