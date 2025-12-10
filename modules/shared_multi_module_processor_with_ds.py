"""
Enhanced Multi-Module Video Processor with RTSP Connection Pooling
Handles multiple analysis modules on the same video feed with shared RTSP connections
Supports DeepStream SDK for hardware-accelerated processing
"""
import cv2
import threading
import time
import logging
from queue import Queue
import numpy as np
from typing import List, Dict, Any, Optional

from .rtsp_connection_pool import rtsp_pool

# Try to import DeepStream processor
DEEPSTREAM_AVAILABLE = False
try:
    from .deepstream_processor import DeepStreamProcessor, check_deepstream_availability
    DEEPSTREAM_AVAILABLE = check_deepstream_availability()
    if DEEPSTREAM_AVAILABLE:
        logger.info("✅ DeepStream SDK available for hardware acceleration")
except ImportError:
    logger.info("ℹ️ DeepStream SDK not available, using OpenCV fallback")

logger = logging.getLogger(__name__)

def safe_array_check(obj, check_type="non_empty"):
    """
    Safe array validation that avoids ambiguous truth values
    
    Args:
        obj: Object to check (could be array, None, etc.)
        check_type: Type of check ('non_empty', 'valid', 'not_none')
    
    Returns:
        bool: True if object passes check, False otherwise
    """
    try:
        if obj is None:
            return False
        
        if not hasattr(obj, 'size'):
            return obj is not None  # For non-array objects
        
        if check_type == "non_empty":
            return obj.size > 0
        elif check_type == "valid":
            return hasattr(obj, 'shape') and obj.size > 0
        elif check_type == "not_none":
            return obj is not None
        else:
            return obj.size > 0
            
    except Exception as e:
        logger.error(f"Error in safe_array_check: {e}")
        return False

class SharedMultiModuleVideoProcessor:
    """
    Enhanced multi-module processor that uses shared RTSP connections
    Supports DeepStream SDK for hardware-accelerated processing
    """
    
    def __init__(self, video_source, channel_id, fps_limit=30, use_deepstream=True):
        """
        Initialize Shared Multi-Module Video Processor
        
        Args:
            video_source: Path to video file, camera index, or RTSP URL
            channel_id: Unique identifier for the video channel
            fps_limit: Maximum FPS for processing
            use_deepstream: Enable DeepStream SDK if available (default: True)
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
        
        # Frame management
        self.frame_queue = Queue(maxsize=5)
        self.latest_raw_frame = None
        self.latest_annotated_frame = None
        self.frame_lock = threading.Lock()
        
        # RTSP connection management
        self.rtsp_connection = None
        self.is_rtsp_stream = self._is_rtsp_url(video_source)
        
        # DeepStream processor (if enabled and available)
        self.deepstream_processor = None
        self.use_deepstream = use_deepstream and DEEPSTREAM_AVAILABLE and self.is_rtsp_stream
        self.processing_mode = None  # Will be set during initialization
        
        # Local video capture (for non-RTSP sources or fallback)
        self.cap = None
        
        # Shared YOLO detections cache
        self.shared_detections = None
        self.detections_lock = threading.Lock()
        self.detections_frame_id = 0
        
        # Performance tracking
        self.frames_processed = 0
        self.start_time = None
        self.actual_fps = 0
        
        logger.info(f"Shared multi-module processor initialized for channel {channel_id}")
        logger.info(f"Source type: {'RTSP' if self.is_rtsp_stream else 'Local'}")
        logger.info(f"DeepStream enabled: {self.use_deepstream}")
    
    def _is_rtsp_url(self, source: str) -> bool:
        """Check if source is an RTSP URL"""
        if isinstance(source, str):
            return source.startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        return False
    
    def _on_rtsp_frame(self, frame: np.ndarray):
        """
        Callback function for receiving frames from RTSP pool
        
        Args:
            frame: New frame from RTSP stream
        """
        try:
            # Validate frame using safe check
            if not safe_array_check(frame, "valid"):
                return
            
            # Add frame to processing queue
            if not self.frame_queue.full():
                self.frame_queue.put(frame, block=False)
            
            # Update latest raw frame
            with self.frame_lock:
                self.latest_raw_frame = frame
                
        except Exception as e:
            logger.error(f"Error handling RTSP frame for channel {self.channel_id}: {e}")
    
    def add_module(self, module_name: str, module_instance):
        """
        Add an analysis module to this video processor
        
        Args:
            module_name: Name of the module
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
        """
        Initialize video capture with DeepStream or fallback methods
        Priority: DeepStream (RTSP only) -> RTSP Pool -> Local OpenCV
        """
        try:
            # Attempt DeepStream for RTSP streams
            if self.use_deepstream and self.is_rtsp_stream:
                try:
                    logger.info(f"🚀 Initializing DeepStream pipeline for: {self.video_source}")
                    
                    self.deepstream_processor = DeepStreamProcessor(
                        rtsp_url=self.video_source,
                        channel_id=self.channel_id,
                        model_path="models/yolo11n.engine",  # TensorRT engine
                        tracker_config="config/deepstream_tracker.txt",
                        input_size=(640, 480),
                        fps_limit=self.fps_limit,
                        confidence_threshold=0.5
                    )
                    
                    # Register callbacks
                    self.deepstream_processor.register_frame_callback(self._on_deepstream_frame)
                    self.deepstream_processor.register_detection_callback(self._on_deepstream_detections)
                    
                    # Start DeepStream pipeline
                    if self.deepstream_processor.start():
                        self.processing_mode = "DeepStream"
                        logger.info(f"✅ DeepStream pipeline started successfully")
                        logger.info(f"📊 Mode: RTSP Ingest -> NVDEC -> Preprocessing -> TensorRT -> NvTracker")
                        
                        # Wait for first frame
                        timeout = 10
                        start_wait = time.time()
                        while self.latest_raw_frame is None and (time.time() - start_wait) < timeout:
                            time.sleep(0.1)
                        
                        if self.latest_raw_frame is not None:
                            return True
                        else:
                            logger.warning("DeepStream started but no frames received, falling back...")
                    else:
                        logger.warning("DeepStream failed to start, falling back to RTSP pool")
                
                except Exception as e:
                    logger.warning(f"DeepStream initialization failed: {e}")
                    logger.info("Falling back to RTSP connection pool")
                    self.deepstream_processor = None
            
            # Fallback to RTSP connection pool for RTSP streams
            if self.is_rtsp_stream and self.deepstream_processor is None:
                logger.info(f"Connecting to shared RTSP stream: {self.video_source}")
                self.rtsp_connection = rtsp_pool.get_connection(
                    rtsp_url=self.video_source,
                    channel_id=self.channel_id,
                    frame_callback=self._on_rtsp_frame
                )
                # Set pool reference for broadcasting
                self.rtsp_connection.set_pool(rtsp_pool)
                self.processing_mode = "RTSP Pool (GPU-accelerated)"
                
                # Wait a moment for first frame
                timeout = 5
                start_wait = time.time()
                while self.latest_raw_frame is None and (time.time() - start_wait) < timeout:
                    time.sleep(0.1)
                
                if self.latest_raw_frame is None:
                    raise ValueError(f"No frames received from RTSP stream: {self.video_source}")
                
                logger.info(f"Successfully connected to shared RTSP: {self.video_source}")
                return True
            
            # Local video capture for non-RTSP sources
            elif not self.is_rtsp_stream:
                if isinstance(self.video_source, str) and self.video_source.isdigit():
                    self.cap = cv2.VideoCapture(int(self.video_source))
                else:
                    self.cap = cv2.VideoCapture(self.video_source)
                
                if not self.cap.isOpened():
                    raise ValueError(f"Cannot open video source: {self.video_source}")
                
                # Optimize local capture settings
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
                self.cap.set(cv2.CAP_PROP_FPS, self.fps_limit)
                self.processing_mode = "Local OpenCV"
                
                logger.info(f"Local video capture initialized: {self.video_source}")
                return True
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize capture for channel {self.channel_id}: {e}")
            return False
    
    def _on_deepstream_frame(self, frame: np.ndarray):
        """Callback for frames from DeepStream pipeline"""
        try:
            if not safe_array_check(frame, "valid"):
                return
            
            # Add frame to processing queue
            if not self.frame_queue.full():
                self.frame_queue.put(frame, block=False)
            
            # Update latest raw frame
            with self.frame_lock:
                self.latest_raw_frame = frame
                
        except Exception as e:
            logger.error(f"Error handling DeepStream frame for channel {self.channel_id}: {e}")
    
    def _on_deepstream_detections(self, detections: List[Dict]):
        """Callback for detections from DeepStream inference"""
        try:
            # Cache detections for modules to use
            with self.detections_lock:
                # Convert DeepStream detections to standard format
                self.shared_detections = []
                for det in detections:
                    bbox = det['bbox']
                    self.shared_detections.append({
                        'class_id': det['class_id'],
                        'confidence': det['confidence'],
                        'bbox': [
                            bbox['left'],
                            bbox['top'],
                            bbox['left'] + bbox['width'],
                            bbox['top'] + bbox['height']
                        ],
                        'tracker_id': det.get('tracker_id', -1)
                    })
                self.detections_frame_id += 1
                
        except Exception as e:
            logger.error(f"Error handling DeepStream detections for channel {self.channel_id}: {e}")
    
    def start(self):
        """Start video processing"""
        if self.is_running:
            logger.warning(f"Channel {self.channel_id} is already running")
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
        
        logger.info(f"Shared multi-module processing started for channel {self.channel_id}")
        return True
    
    def stop(self):
        """Stop video processing and cleanup resources"""
        self.is_running = False
        
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        
        # Cleanup DeepStream pipeline if active
        if self.deepstream_processor:
            logger.info(f"Stopping DeepStream pipeline for channel {self.channel_id}")
            self.deepstream_processor.stop()
            self.deepstream_processor = None
        
        # Cleanup RTSP connection
        if self.is_rtsp_stream and self.rtsp_connection:
            rtsp_pool.unsubscribe(self.video_source, self.channel_id)
            self.rtsp_connection = None
        
        # Cleanup local capture
        if self.cap:
            self.cap.release()
        
        logger.info(f"Shared multi-module processing stopped for channel {self.channel_id}")
        logger.info(f"Processing mode was: {self.processing_mode}")
    
    def _processing_loop(self):
        """Main processing loop"""
        last_frame_time = 0
        consecutive_failures = 0
        max_failures = 30
        
        while self.is_running:
            try:
                current_time = time.time()
                
                # Respect FPS limit
                if current_time - last_frame_time < self.frame_interval:
                    time.sleep(0.001)
                    continue
                
                frame = None
                
                # Get frame based on processing mode
                if self.deepstream_processor:
                    # DeepStream mode - frames come from callback, detections already available
                    try:
                        frame = self.frame_queue.get(timeout=0.1)
                    except:
                        continue  # No frame available
                    
                    # Detections are already cached from DeepStream callback
                    # Modules can use self.shared_detections
                    
                elif self.is_rtsp_stream:
                    # RTSP Pool mode - get frame from queue (fed by RTSP callback)
                    try:
                        frame = self.frame_queue.get(timeout=0.1)
                    except:
                        continue  # No frame available
                        
                else:
                    # Local capture mode - read from OpenCV
                    ret, frame = self.cap.read()
                    if not ret:
                        consecutive_failures += 1
                        if consecutive_failures >= max_failures:
                            logger.error(f"Too many failures reading from {self.video_source}")
                            break
                        time.sleep(0.1)
                        continue
                
                if not safe_array_check(frame, "valid"):
                    continue
                
                # Reset failure counter
                consecutive_failures = 0
                
                # Resize frame if too large (not needed for DeepStream - already preprocessed)
                if not self.deepstream_processor and frame.shape[1] > 1280:
                    scale_factor = 1280 / frame.shape[1]
                    new_width = 1280
                    new_height = int(frame.shape[0] * scale_factor)
                    frame = cv2.resize(frame, (new_width, new_height))
                
                # Process frame through all modules
                module_results = {}
                
                for module_name, module in self.modules.items():
                    try:
                        # Validate frame before passing to module using safe check
                        if not safe_array_check(frame, "valid"):
                            logger.warning(f"Invalid frame for module {module_name}, skipping")
                            continue
                            
                        result = module.process_frame(frame)
                        
                        # Debug logging for people counter
                        if module_name == 'PeopleCounter' and self.frames_processed % 60 == 0:
                            logger.info(f"PeopleCounter result type: {type(result)}")
                            if isinstance(result, dict):
                                logger.info(f"PeopleCounter keys: {result.keys()}")
                                if 'status' in result:
                                    logger.info(f"PeopleCounter status: {result['status']}")
                        
                        # Validate result
                        if result is not None:
                            module_results[module_name] = result
                            self.module_results[module_name] = result
                        else:
                            logger.debug(f"Module {module_name} returned None result")
                            
                    except Exception as e:
                        logger.error(f"Error processing frame in module {module_name}: {e}")
                        import traceback
                        logger.error(f"Traceback: {traceback.format_exc()}")
                        continue
                
                # Create combined annotated frame
                try:
                    annotated_frame = self._create_combined_frame(frame, module_results)
                except Exception as e:
                    logger.error(f"Error creating combined frame for channel {self.channel_id}: {e}")
                    annotated_frame = frame.copy()  # Fallback to original frame
                
                # Update latest frames
                with self.frame_lock:
                    self.latest_raw_frame = frame
                    self.latest_annotated_frame = annotated_frame
                
                # Update statistics
                self.frames_processed += 1
                last_frame_time = current_time
                
                # Calculate FPS
                if self.frames_processed % 30 == 0:
                    elapsed_time = current_time - self.start_time
                    self.actual_fps = self.frames_processed / elapsed_time if elapsed_time > 0 else 0
                    
                    if self.frames_processed % 300 == 0:
                        logger.info(f"Channel {self.channel_id}: {self.frames_processed} frames, "
                                  f"FPS: {self.actual_fps:.2f}")
                
            except Exception as e:
                logger.error(f"Error in processing loop for channel {self.channel_id}: {e}")
                consecutive_failures += 1
                
                if consecutive_failures >= max_failures:
                    break
                
                time.sleep(0.1)
        
        logger.info(f"Processing loop ended for channel {self.channel_id}")
    
    def _create_combined_frame(self, original_frame: np.ndarray, 
                             module_results: Dict[str, Any]) -> np.ndarray:
        """
        Create combined frame with annotations from all modules
        
        Args:
            original_frame: Original video frame
            module_results: Dictionary of module processing results
            
        Returns:
            Frame with combined annotations from all modules
        """
        # Start with the original frame
        combined_frame = original_frame.copy()
        frame_height, frame_width = combined_frame.shape[:2]
        
        # Use the annotated frame from the first module that has one
        # Priority: PeopleCounter, QueueMonitor, others
        module_priority = ['PeopleCounter', 'QueueMonitor']
        
        for module_name in module_priority:
            if module_name in module_results:
                result = module_results[module_name]
                if result is not None and isinstance(result, dict) and 'frame' in result:
                    module_frame = result['frame']
                    if safe_array_check(module_frame, "valid"):
                        combined_frame = module_frame.copy()
                        logger.debug(f"Using annotated frame from {module_name}")
                        break
        
        # If no priority module found, use any available annotated frame
        if np.array_equal(combined_frame, original_frame):
            for module_name, result in module_results.items():
                if result is not None and isinstance(result, dict) and 'frame' in result:
                    module_frame = result['frame']
                    if safe_array_check(module_frame, "valid"):
                        combined_frame = module_frame.copy()
                        logger.debug(f"Using annotated frame from {module_name}")
                        break
        
        # Draw header showing active modules
        header_height = 40
        header_frame = np.zeros((header_height, frame_width, 3), dtype=np.uint8)
        
        # List active modules
        module_names = list(module_results.keys())
        if module_names:
            header_text = f"Shared RTSP - Active: {', '.join(module_names)}"
            cv2.putText(header_frame, header_text, (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        
        # Add RTSP sharing indicator
        if self.is_rtsp_stream:
            pool_stats = rtsp_pool.get_connection_stats()
            if self.video_source in pool_stats:
                subscriber_count = pool_stats[self.video_source]['subscribers']
                sharing_text = f"Shared with {subscriber_count} channel(s)"
                cv2.putText(header_frame, sharing_text, (frame_width - 300, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Add additional module status overlays (without duplicating the main annotations)
        y_offset = 60
        for i, (module_name, result) in enumerate(module_results.items()):
            if result is not None and isinstance(result, dict) and 'status' in result:
                if module_name == 'PeopleCounter':
                    self._overlay_people_counter_status(combined_frame, result, y_offset + i * 25)
                elif module_name == 'QueueMonitor':
                    self._overlay_queue_monitor_status(combined_frame, result, y_offset + i * 25)
        
        # Combine header with main frame
        final_frame = np.vstack([header_frame, combined_frame])
        return final_frame
    
    def _overlay_people_counter_status(self, frame, result, y_pos):
        """Overlay people counter status information (small overlay)"""
        if result is not None and isinstance(result, dict) and 'status' in result:
            status = result['status']
            if status is not None and isinstance(status, dict):
                in_count = status.get('in_count', 0)
                out_count = status.get('out_count', 0)
                detections = status.get('detections', 0)
                
                # Small status overlay in top-right corner
                status_text = f"PC: IN:{in_count} OUT:{out_count} DET:{detections}"
                cv2.putText(frame, status_text, (frame.shape[1] - 300, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    def _overlay_queue_monitor_status(self, frame, result, y_pos):
        """Overlay queue monitor status information (small overlay)"""
        if result is not None and isinstance(result, dict) and 'status' in result:
            status = result['status']
            if status is not None and isinstance(status, dict):
                queue_count = status.get('queue_count', 0)
                counter_count = status.get('counter_count', 0)
                
                # Small status overlay in top-right corner
                status_text = f"QM: Q:{queue_count} C:{counter_count}"
                color = (0, 255, 255) if queue_count <= 2 else (0, 0, 255)
                cv2.putText(frame, status_text, (frame.shape[1] - 200, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def get_latest_frame(self, module_name: Optional[str] = None) -> np.ndarray:
        """
        Get the latest processed frame
        
        Args:
            module_name: If specified, get frame from specific module
            
        Returns:
            Latest frame (combined or module-specific)
        """
        with self.frame_lock:
            if module_name and module_name in self.module_results:
                result = self.module_results[module_name]
                if result is not None and isinstance(result, dict) and 'frame' in result:
                    frame = result['frame']
                    if safe_array_check(frame, "valid"):
                        return frame
            
            # Return combined frame if available
            if safe_array_check(self.latest_annotated_frame, "valid"):
                return self.latest_annotated_frame
            elif safe_array_check(self.latest_raw_frame, "valid"):
                return self.latest_raw_frame
            else:
                return np.zeros((480, 640, 3), dtype=np.uint8)
    
    def get_module_result(self, module_name: str):
        """Get the latest result from a specific module"""
        return self.module_results.get(module_name)
    
    def get_all_module_results(self):
        """Get results from all modules"""
        return self.module_results.copy()
    
    def get_status(self):
        """Get processor status including DeepStream and RTSP sharing info"""
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        avg_fps = self.frames_processed / elapsed_time if elapsed_time > 0 else 0
        
        module_statuses = {}
        for module_name, module in self.modules.items():
            if hasattr(module, 'get_status'):
                module_statuses[module_name] = module.get_status()
        
        # Get RTSP sharing information
        rtsp_info = {}
        if self.is_rtsp_stream and self.rtsp_connection:
            pool_stats = rtsp_pool.get_connection_stats()
            if self.video_source in pool_stats:
                rtsp_info = pool_stats[self.video_source]
        
        # Get DeepStream statistics if active
        deepstream_stats = {}
        if self.deepstream_processor:
            deepstream_stats = self.deepstream_processor.get_statistics()
        
        return {
            'channel_id': self.channel_id,
            'is_running': self.is_running,
            'video_source': self.video_source,
            'processing_mode': self.processing_mode or 'Unknown',
            'source_type': 'RTSP (Shared)' if self.is_rtsp_stream else 'Local',
            'deepstream_enabled': self.deepstream_processor is not None,
            'hardware_accelerated': self.deepstream_processor is not None or (self.is_rtsp_stream and self.rtsp_connection),
            'frames_processed': self.frames_processed,
            'elapsed_time': elapsed_time,
            'average_fps': avg_fps,
            'actual_fps': self.actual_fps,
            'target_fps': self.fps_limit,
            'active_modules': list(self.modules.keys()),
            'num_modules': len(self.modules),
            'module_statuses': module_statuses,
            'rtsp_sharing': rtsp_info,
            'deepstream_stats': deepstream_stats,
            'performance_mode': self.processing_mode or 'Unknown'
        }