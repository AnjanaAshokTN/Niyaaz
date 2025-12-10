"""
Multi-Module Video Processor
Handles multiple analysis modules on the same video feed for efficiency
"""
import cv2
import threading
import time
import logging
from queue import Queue
import numpy as np
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class MultiModuleVideoProcessor:
    def __init__(self, video_source, channel_id, fps_limit=30):
        """
        Initialize Multi-Module Video Processor
        
        Args:
            video_source: Path to video file or camera index
            channel_id: Unique identifier for the video channel
            fps_limit: Maximum FPS for processing
        """
        self.video_source = video_source
        self.channel_id = channel_id
        self.fps_limit = fps_limit
        self.frame_interval = 1.0 / fps_limit
        
        # Processing modules
        self.modules = {}  # {module_name: module_instance}
        self.module_results = {}  # {module_name: latest_result}
        
        # Threading and control
        self.is_running = False
        self.processing_thread = None
        
        # Video capture
        self.cap = None
        
        # Shared resources
        self.latest_raw_frame = None
        self.latest_annotated_frame = None
        self.frame_lock = threading.Lock()
        
        # Shared YOLO detections cache (optimization for multiple modules)
        self.shared_detections = None
        self.detections_lock = threading.Lock()
        self.detections_frame_id = 0
        
        # Performance optimization flags
        self.frame_skip_counter = 0
        self.adaptive_skip = 1  # Process every Nth frame (1 = no skip, 2 = skip every other frame)
        self.target_fps = fps_limit
        self.actual_fps = 0
        
        # Statistics
        self.frames_processed = 0
        self.start_time = None
        
        logger.info(f"Multi-module processor initialized for channel {channel_id}")
    

        # GPU Optimization imports
        from modules.batch_processor import batch_processor
        
        # Start batch processor if not already running
        if not batch_processor.is_running:
            batch_processor.start()
        
        # GPU optimization flags in __init__
        self.gpu_optimized = True
        self.frame_skip_dynamic = True
        self.last_gpu_check = 0
        self.current_scale_settings = {}

    def add_module(self, module_name: str, module_instance):
        """
        Add an analysis module to this video processor
        
        Args:
            module_name: Name of the module (e.g., 'PeopleCounter', 'QueueMonitor')
            module_instance: Instance of the analysis module
        """
        self.modules[module_name] = module_instance
        self.module_results[module_name] = None
        logger.info(f"Added module '{module_name}' to channel {self.channel_id}")
    
    def remove_module(self, module_name: str):
        """Remove an analysis module"""
        if module_name in self.modules:
            del self.modules[module_name]
            del self.module_results[module_name]
            logger.info(f"Removed module '{module_name}' from channel {self.channel_id}")
    
    def get_active_modules(self) -> List[str]:
        """Get list of active module names"""
        return list(self.modules.keys())
    
    def initialize_capture(self):
        """Initialize video capture"""
        try:
            # Determine source type
            if isinstance(self.video_source, str):
                if self.video_source.startswith(('rtsp://', 'rtmp://', 'http://', 'https://')):
                    # RTSP/network stream - OPTIMIZED for low latency
                    self.cap = cv2.VideoCapture(self.video_source, cv2.CAP_FFMPEG)
                    
                    # Aggressive optimization for real-time RTSP
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffering
                    self.cap.set(cv2.CAP_PROP_FPS, 15)  # Limit to 15 FPS for RTSP (reduces load)
                    
                    # Set transport protocol to TCP for reliability (optional)
                    self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H','2','6','4'))
                    
                    logger.info(f"Initializing RTSP stream with optimizations: {self.video_source}")
                    
                elif self.video_source.isdigit():
                    # Camera index
                    self.cap = cv2.VideoCapture(int(self.video_source))
                    logger.info(f"Initializing camera: {self.video_source}")
                    
                else:
                    # File path
                    self.cap = cv2.VideoCapture(self.video_source)
                    logger.info(f"Initializing video file: {self.video_source}")
            else:
                # Assume numeric camera index
                self.cap = cv2.VideoCapture(self.video_source)
                logger.info(f"Initializing camera index: {self.video_source}")
            
            if not self.cap.isOpened():
                raise ValueError(f"Cannot open video source: {self.video_source}")
            
            # Additional RTSP optimization settings
            if isinstance(self.video_source, str) and self.video_source.startswith(('rtsp://', 'rtmp://')):
                # RTSP-specific optimizations
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                # Try to set a reasonable timeout
                self.cap.set(cv2.CAP_PROP_POS_MSEC, 1000)
            
            # Get video properties
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            # Validate frame dimensions
            if self.frame_width <= 0 or self.frame_height <= 0:
                # Set default dimensions for RTSP streams that don't report correctly
                self.frame_width = 640
                self.frame_height = 480
                logger.warning(f"Invalid frame dimensions detected, using defaults: {self.frame_width}x{self.frame_height}")
            
            logger.info(f"Video capture initialized: {self.frame_width}x{self.frame_height} @ {self.video_fps} FPS")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize video capture: {e}")
            return False
    
    def start(self):
        """Start video processing"""
        if self.is_running:
            logger.warning("Multi-module processor is already running")
            return False
        
        if not self.initialize_capture():
            return False
        
        self.is_running = True
        self.start_time = time.time()
        
        # Load configuration for all modules
        for module in self.modules.values():
            if hasattr(module, 'load_configuration'):
                module.load_configuration()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        logger.info(f"Multi-module processing started for channel {self.channel_id}")
        return True
    
    def stop(self):
        """Stop video processing"""
        self.is_running = False
        
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        
        if self.cap:
            self.cap.release()
        
        logger.info(f"Multi-module processing stopped for channel {self.channel_id}")
    
    def _processing_loop(self):
        """Main processing loop with performance optimizations"""
        last_frame_time = 0
        consecutive_failures = 0
        max_failures = 30  # Allow up to 30 consecutive failures before giving up
        
        # FPS monitoring for adaptive performance
        fps_samples = []
        last_fps_check = time.time()
        
        # Determine if this is a live stream
        is_live_stream = isinstance(self.video_source, str) and (
            self.video_source.startswith(('rtsp://', 'rtmp://', 'http://', 'https://')) or
            self.video_source.isdigit()
        )
        
        while self.is_running:
            try:
                current_time = time.time()
                
                # Respect FPS limit
                if current_time - last_frame_time < self.frame_interval:
                    time.sleep(0.001)
                    continue
                
                # Read frame
                ret, frame = self.cap.read()
                
                if not ret:
                    # For video files, loop back to beginning (normal end of video)
                    if isinstance(self.video_source, str) and not self.video_source.isdigit():
                        logger.debug(f"Video file reached end, looping back to start for channel {self.channel_id}")
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    
                    # For live streams or other sources, this is an actual failure
                    consecutive_failures += 1
                    logger.warning(f"Failed to read frame (attempt {consecutive_failures}/{max_failures})")
                    
                    if consecutive_failures >= max_failures:
                        logger.error("Too many consecutive frame read failures. Stopping processing.")
                        break
                    
                    # For live streams, try to reconnect
                    if is_live_stream:
                        logger.info("Attempting to reconnect to stream...")
                        self.cap.release()
                        time.sleep(2)  # Wait before reconnecting
                        if not self.initialize_capture():
                            logger.error("Failed to reconnect to stream")
                            break
                        continue
                    else:
                        logger.error("Failed to read frame from video source")
                        break
                
                # Reset failure counter on successful read
                consecutive_failures = 0
                
                # Validate frame
                if frame is None or frame.size == 0:
                    logger.warning("Received empty frame")
                    continue
                
                # Resize frame if too large for better performance
                if frame.shape[1] > 1280:
                    scale_factor = 1280 / frame.shape[1]
                    new_width = 1280
                    new_height = int(frame.shape[0] * scale_factor)
                    frame = cv2.resize(frame, (new_width, new_height))
                
                # **OPTIMIZATION: Frame skipping for multiple modules**
                # When multiple modules are active, skip frames to maintain FPS
                self.frame_skip_counter += 1
                should_process = (self.frame_skip_counter % self.adaptive_skip) == 0
                
                # Adjust adaptive skip based on module count (more aggressive for queue monitor)
                num_modules = len(self.modules)
                if num_modules == 1:
                    self.adaptive_skip = 1  # Process every frame for single module
                elif num_modules == 2:
                    self.adaptive_skip = 1  # Still process every frame for 2 modules
                elif num_modules <= 3:
                    self.adaptive_skip = 2  # Process every 2nd frame for 3 modules
                elif num_modules <= 5:
                    self.adaptive_skip = 3  # Process every 3rd frame for 4-5 modules
                else:
                    self.adaptive_skip = 4  # Process every 4th frame for 6+ modules
                
                # Store raw frame (always update for streaming)
                with self.frame_lock:
                    self.latest_raw_frame = frame.copy()
                
                # Process frame through all modules (with skipping)
                if should_process:
                    annotated_frame = self._process_through_modules(frame)
                    
                    # Update latest annotated frame
                    with self.frame_lock:
                        self.latest_annotated_frame = annotated_frame
                    
                    self.frames_processed += 1
                
                last_frame_time = current_time
                
                # **PERFORMANCE MONITORING**
                # Calculate actual FPS
                fps_samples.append(current_time)
                if len(fps_samples) > 30:
                    fps_samples.pop(0)
                
                if current_time - last_fps_check > 2.0:  # Check every 2 seconds
                    if len(fps_samples) > 1:
                        time_span = fps_samples[-1] - fps_samples[0]
                        self.actual_fps = (len(fps_samples) - 1) / time_span if time_span > 0 else 0
                    last_fps_check = current_time
                
                # Log performance statistics periodically (reduced frequency)
                if self.frames_processed % 600 == 0:  # Every 600 frames (~20 seconds at 30fps)
                    elapsed_time = current_time - self.start_time
                    avg_fps = self.frames_processed / elapsed_time
                    logger.info(f"Channel {self.channel_id}: {self.frames_processed} frames, "
                              f"FPS: {avg_fps:.1f}, modules: {len(self.modules)}")
                
            except Exception as e:
                logger.error(f"Error in multi-module processing loop: {e}")
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    logger.error("Too many consecutive processing errors. Stopping.")
                    break
                if not self.is_running:
                    break
                time.sleep(0.1)
        
        logger.info(f"Multi-module processing loop ended for channel {self.channel_id}")
        
        # Cleanup
        if self.cap:
            self.cap.release()
    
    def _process_through_modules(self, frame):
        """
        Process frame through all active modules and combine annotations
        **OPTIMIZED: Runs YOLO detection once and shares results across all modules**
        
        Args:
            frame: Input video frame
            
        Returns:
            Annotated frame with all module outputs combined
        """
        if not self.modules:
            return frame
        
        # **OPTIMIZATION: Run YOLO detection ONCE for all modules**
        # This is the key optimization - detect persons once, share with all modules
        shared_detections = None
        self.detections_frame_id += 1
        
        for module_name, module in self.modules.items():
            try:
                # Check if module has a YOLO detector that can be shared
                if hasattr(module, 'detector') and shared_detections is None:
                    # First module runs detection
                    shared_detections = module.detector.detect_persons(frame)
                    
                    # Cache detections for other modules
                    with self.detections_lock:
                        self.shared_detections = shared_detections
                
                # For modules with detectors, inject shared detections
                if hasattr(module, 'detector') and shared_detections is not None:
                    # Process frame with pre-computed detections
                    if hasattr(module, 'process_frame_with_detections'):
                        module_result = module.process_frame_with_detections(frame.copy(), shared_detections)
                    else:
                        # Fallback: module doesn't support shared detections
                        module_result = module.process_frame(frame.copy())
                else:
                    # Module doesn't use YOLO, process normally
                    module_result = module.process_frame(frame.copy())
                
                # Debug: Log module result type
                if self.frames_processed % 30 == 0:
                    result_type = type(module_result).__name__
                    is_dict = isinstance(module_result, dict)
                    has_frame = 'frame' in module_result if is_dict else False
                    logger.info(f"Module {module_name}: result_type={result_type}, is_dict={is_dict}, has_frame={has_frame}")
                
                # Store module result - handle both dict and direct frame returns
                if isinstance(module_result, dict) and 'frame' in module_result:
                    # Module returns structured result {'frame': ..., 'status': ..., 'metadata': ...}
                    self.module_results[module_name] = {
                        'frame': module_result['frame'],
                        'timestamp': time.time(),
                        'status': module_result.get('status', {}),
                        'metadata': module_result.get('metadata', {})
                    }
                    if self.frames_processed % 30 == 0:
                        logger.info(f"{module_name} result type: <class 'dict'>")
                        logger.info(f"{module_name} keys: {module_result.keys()}")
                        logger.info(f"{module_name} status: {module_result.get('status', {})}")
                else:
                    # Module returns frame directly (legacy modules)
                    self.module_results[module_name] = {
                        'frame': module_result,
                        'timestamp': time.time(),
                        'status': module.get_status() if hasattr(module, 'get_status') else {}
                    }
                
            except Exception as e:
                logger.error(f"Error processing frame through module {module_name}: {e}")
                self.module_results[module_name] = {
                    'frame': frame.copy(),
                    'timestamp': time.time(),
                    'status': {'error': str(e)}
                }
        
        # Combine annotations from all modules
        combined_frame = self._combine_module_annotations(frame, self.module_results)
        
        return combined_frame
    
    def _combine_module_annotations(self, original_frame, module_results):
        """
        Combine visual annotations from multiple modules
        
        Args:
            original_frame: Original video frame
            module_results: Dictionary of module processing results
            
        Returns:
            Frame with combined annotations from all modules
        """
        frame_height, frame_width = original_frame.shape[:2]
        
        # **SOLUTION: Use the annotated frame from modules, not the original**
        # If we have module results, use them; otherwise use original
        combined_frame = original_frame.copy()
        
        # Debug logging
        if self.frames_processed % 30 == 0:
            logger.info(f"=== Frame Combination Debug ===")
            logger.info(f"Module results count: {len(module_results)}")
            for mod_name, mod_res in module_results.items():
                has_frame = 'frame' in mod_res if isinstance(mod_res, dict) else False
                frame_type = type(mod_res.get('frame')).__name__ if has_frame else 'N/A'
                logger.info(f"  {mod_name}: has_frame={has_frame}, frame_type={frame_type}")
        
        # For single module, use its annotated frame directly
        if len(module_results) == 1:
            for module_name, result in module_results.items():
                if result and 'frame' in result:
                    module_frame = result['frame']
                    if module_frame is not None and isinstance(module_frame, np.ndarray):
                        combined_frame = module_frame
                        if self.frames_processed % 30 == 0:
                            logger.info(f"✓ Using annotated frame from {module_name}")
        else:
            # For multiple modules, we need to layer them properly
            # Priority order: QueueMonitor, PeopleCounter, Heatmap, others
            priority_order = ['QueueMonitor', 'PeopleCounter', 'HeatmapProcessor', 
                            'BagDetection', 'CashDetection', 'FallDetection', 'GroomingDetection']
            
            # Use the first module's frame as base (it has all detections drawn)
            for priority_module in priority_order:
                if priority_module in module_results:
                    result = module_results[priority_module]
                    if result and 'frame' in result:
                        module_frame = result['frame']
                        if module_frame is not None and isinstance(module_frame, np.ndarray):
                            combined_frame = module_frame
                            if self.frames_processed % 30 == 0:
                                logger.info(f"✓ Using annotated frame from {priority_module} (multi-module)")
                            break
        
        # Draw a header showing active modules
        header_height = 40
        header_frame = np.zeros((header_height, frame_width, 3), dtype=np.uint8)
        
        # List active modules
        module_names = list(module_results.keys())
        if module_names:
            header_text = f"Active: {', '.join(module_names)}"
            cv2.putText(header_frame, header_text, (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Combine header with main frame
        final_frame = np.vstack([header_frame, combined_frame])
        
        return final_frame
    
    def _overlay_people_counter_info(self, frame, result, y_pos):
        """Overlay people counter information"""
        if 'status' in result:
            status = result['status']
            in_count = status.get('in_count', 0)
            out_count = status.get('out_count', 0)
            
            # Draw counts
            text = f"People Counter - IN: {in_count} | OUT: {out_count}"
            cv2.putText(frame, text, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    def _overlay_queue_monitor_info(self, frame, result, y_pos):
        """Overlay queue monitor information"""
        if 'status' in result:
            status = result['status']
            queue_count = status.get('queue_count', 0)
            counter_count = status.get('counter_count', 0)
            
            # Draw counts
            text = f"Queue Monitor - Queue: {queue_count} | Counter: {counter_count}"
            color = (0, 255, 255) if queue_count <= 2 else (0, 0, 255)  # Yellow if normal, red if crowded
            cv2.putText(frame, text, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    def get_latest_frame(self, module_name=None):
        """
        Get the latest processed frame (optimized - returns reference, not copy)
        
        Args:
            module_name: If specified, get frame from specific module, otherwise combined frame
            
        Returns:
            Latest frame (combined or module-specific)
        """
        with self.frame_lock:
            if module_name and module_name in self.module_results:
                result = self.module_results[module_name]
                if result and 'frame' in result:
                    frame = result['frame']
                    # Debug: Log frame retrieval
                    if self.frames_processed % 60 == 0:
                        logger.info(f"get_latest_frame: Returning {module_name} frame, shape={frame.shape if frame is not None else 'None'}")
                    return frame
            
            # Return combined frame if available
            if self.latest_annotated_frame is not None:
                if self.frames_processed % 60 == 0:
                    logger.info(f"get_latest_frame: Returning combined annotated frame")
                return self.latest_annotated_frame
            elif self.latest_raw_frame is not None:
                if self.frames_processed % 60 == 0:
                    logger.info(f"get_latest_frame: Returning raw frame (no annotations)")
                return self.latest_raw_frame
            else:
                # Return blank frame if nothing available
                return np.zeros((480, 640, 3), dtype=np.uint8)
    
    def get_module_result(self, module_name):
        """Get the latest result from a specific module"""
        return self.module_results.get(module_name)
    
    def get_all_module_results(self):
        """Get results from all modules"""
        return self.module_results.copy()
    
    def get_status(self):
        """Get processor status including all modules"""
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        avg_fps = self.frames_processed / elapsed_time if elapsed_time > 0 else 0
        
        module_statuses = {}
        for module_name, module in self.modules.items():
            if hasattr(module, 'get_status'):
                module_statuses[module_name] = module.get_status()
        
        return {
            'channel_id': self.channel_id,
            'is_running': self.is_running,
            'video_source': self.video_source,
            'frames_processed': self.frames_processed,
            'elapsed_time': elapsed_time,
            'average_fps': avg_fps,
            'actual_fps': getattr(self, 'actual_fps', 0),
            'target_fps': self.fps_limit,
            'adaptive_skip': self.adaptive_skip,
            'frame_size': f"{getattr(self, 'frame_width', 0)}x{getattr(self, 'frame_height', 0)}",
            'active_modules': list(self.modules.keys()),
            'num_modules': len(self.modules),
            'module_statuses': module_statuses,
            'performance_mode': 'optimized' if len(self.modules) > 1 else 'normal'
        }