"""
Video Processing Pipeline
Handles video input, frame processing, and streaming
"""
import cv2
import threading
import time
import logging
from queue import Queue
import numpy as np

logger = logging.getLogger(__name__)

class VideoProcessor:
    def optimize_capture_properties(self):
        """Optimize capture properties for multiple channels"""
        if self.cap and self.cap.isOpened():
            # Reduce buffer to minimum for real-time processing
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Set optimal frame rate
            self.cap.set(cv2.CAP_PROP_FPS, 15)  # Reduced for multiple channels
            
            # Set reasonable resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Optimize codec settings for streaming
            if hasattr(cv2, 'CAP_PROP_FOURCC'):
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))

    def __init__(self, video_source, processing_module, fps_limit=30):
        """
        Initialize Video Processor
        
        Args:
            video_source: Path to video file or camera index
            processing_module: Module to process frames (PeopleCounter, QueueMonitor, etc.)
            fps_limit: Maximum FPS for processing
        """
        self.video_source = video_source
        self.processing_module = processing_module
        self.fps_limit = fps_limit
        self.frame_interval = 1.0 / fps_limit
        
        # Threading and control
        self.is_running = False
        self.processing_thread = None
        self.frame_queue = Queue(maxsize=10)
        
        # Video capture
        self.cap = None
        
        # Latest processed frame for streaming
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        
        # Statistics
        self.frames_processed = 0
        self.start_time = None
        
        logger.info(f"Video processor initialized for source: {video_source}")
        self.optimize_capture_properties()
    
    def initialize_capture(self):
        """Initialize video capture"""
        try:
            # Try to open as file first, then as camera index
            if isinstance(self.video_source, str) and self.video_source.isdigit():
                self.cap = cv2.VideoCapture(int(self.video_source))
            else:
                self.cap = cv2.VideoCapture(self.video_source)
            
            if not self.cap.isOpened():
                raise ValueError(f"Cannot open video source: {self.video_source}")
            
            # Set capture properties for better performance
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps_limit)
            
            # Get video properties
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Video capture initialized: {self.frame_width}x{self.frame_height} @ {self.video_fps} FPS")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize video capture: {e}")
            return False
    
    def start(self):
        """Start video processing"""
        if self.is_running:
            logger.warning("Video processor is already running")
            return False
        
        if not self.initialize_capture():
            return False
        
        self.is_running = True
        self.start_time = time.time()
        
        # Load configuration for processing module
        if hasattr(self.processing_module, 'load_configuration'):
            self.processing_module.load_configuration()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        logger.info("Video processing started")
        return True
    
    def stop(self):
        """Stop video processing"""
        self.is_running = False
        
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        
        if self.cap:
            self.cap.release()
        
        logger.info("Video processing stopped")
    
    def _processing_loop(self):
        """Main processing loop"""
        last_frame_time = 0
        
        while self.is_running:
            try:
                current_time = time.time()
                
                # Respect FPS limit
                if current_time - last_frame_time < self.frame_interval:
                    time.sleep(0.001)  # Small sleep to prevent busy waiting
                    continue
                
                # Read frame
                ret, frame = self.cap.read()
                
                if not ret:
                    # If reading from file, loop back to beginning
                    if isinstance(self.video_source, str) and not self.video_source.isdigit():
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        logger.error("Failed to read frame from video source")
                        break
                
                # Resize frame if too large for better performance
                if frame.shape[1] > 1280:
                    scale_factor = 1280 / frame.shape[1]
                    new_width = 1280
                    new_height = int(frame.shape[0] * scale_factor)
                    frame = cv2.resize(frame, (new_width, new_height))
                
                # Process frame through the module
                processed_frame = self.processing_module.process_frame(frame)
                
                # Update latest frame for streaming
                with self.frame_lock:
                    self.latest_frame = processed_frame.copy()
                
                self.frames_processed += 1
                last_frame_time = current_time
                
                # Log performance statistics periodically
                if self.frames_processed % 300 == 0:  # Every 300 frames
                    elapsed_time = current_time - self.start_time
                    avg_fps = self.frames_processed / elapsed_time
                    logger.info(f"Processing stats: {self.frames_processed} frames, "
                              f"avg FPS: {avg_fps:.2f}")
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                if not self.is_running:
                    break
                time.sleep(0.1)  # Brief pause before retrying
        
        logger.info("Processing loop ended")
    
    def get_latest_frame(self):
        """Get the latest processed frame for streaming"""
        with self.frame_lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            else:
                # Return a blank frame if no frame is available
                return np.zeros((480, 640, 3), dtype=np.uint8)
    
    def get_status(self):
        """Get processor status"""
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        avg_fps = self.frames_processed / elapsed_time if elapsed_time > 0 else 0
        
        return {
            'is_running': self.is_running,
            'video_source': self.video_source,
            'frames_processed': self.frames_processed,
            'elapsed_time': elapsed_time,
            'average_fps': avg_fps,
            'target_fps': self.fps_limit,
            'frame_size': f"{getattr(self, 'frame_width', 0)}x{getattr(self, 'frame_height', 0)}",
            'module_status': self.processing_module.get_status() if hasattr(self.processing_module, 'get_status') else {}
        }

class StreamingVideoProcessor:
    """
    Enhanced video processor with streaming capabilities
    Optimized for real-time web streaming
    """
    
    def __init__(self, video_source, processing_module, stream_fps=15, process_fps=30):
        """
        Initialize Streaming Video Processor
        
        Args:
            video_source: Path to video file or camera index
            processing_module: Module to process frames
            stream_fps: FPS for web streaming (lower for bandwidth)
            process_fps: FPS for AI processing (higher for accuracy)
        """
        self.video_source = video_source
        self.processing_module = processing_module
        self.stream_fps = stream_fps
        self.process_fps = process_fps
        
        self.stream_interval = 1.0 / stream_fps
        self.process_interval = 1.0 / process_fps
        
        # Threading
        self.is_running = False
        self.capture_thread = None
        self.processing_thread = None
        self.streaming_thread = None
        
        # Frame queues
        self.raw_frame_queue = Queue(maxsize=5)
        self.processed_frame_queue = Queue(maxsize=3)
        
        # Latest frames
        self.latest_raw_frame = None
        self.latest_processed_frame = None
        self.stream_frame = None
        
        # Locks
        self.raw_frame_lock = threading.Lock()
        self.processed_frame_lock = threading.Lock()
        self.stream_frame_lock = threading.Lock()
        
        # Video capture
        self.cap = None
        
        # Statistics
        self.capture_fps = 0
        self.processing_fps = 0
        self.streaming_fps = 0
        
    def start(self):
        """Start all processing threads"""
        if self.is_running:
            return False
        
        # Initialize video capture
        try:
            if isinstance(self.video_source, str) and self.video_source.isdigit():
                self.cap = cv2.VideoCapture(int(self.video_source))
            else:
                self.cap = cv2.VideoCapture(self.video_source)
            
            if not self.cap.isOpened():
                raise ValueError(f"Cannot open video source: {self.video_source}")
            
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
            
        except Exception as e:
            logger.error(f"Failed to initialize capture: {e}")
            return False
        
        self.is_running = True
        
        # Start threads
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.streaming_thread = threading.Thread(target=self._streaming_loop)
        
        self.capture_thread.daemon = True
        self.processing_thread.daemon = True
        self.streaming_thread.daemon = True
        
        self.capture_thread.start()
        self.processing_thread.start()
        self.streaming_thread.start()
        
        logger.info("Streaming video processor started")
        return True
    
    def stop(self):
        """Stop all threads"""
        self.is_running = False
        
        # Wait for threads to finish
        for thread in [self.capture_thread, self.processing_thread, self.streaming_thread]:
            if thread:
                thread.join(timeout=2)
        
        if self.cap:
            self.cap.release()
        
        logger.info("Streaming video processor stopped")
    
    def _capture_loop(self):
        """Capture frames from video source"""
        while self.is_running:
            try:
                ret, frame = self.cap.read()
                
                if not ret:
                    if isinstance(self.video_source, str) and not self.video_source.isdigit():
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        break
                
                # Resize for performance
                if frame.shape[1] > 1280:
                    scale_factor = 1280 / frame.shape[1]
                    new_width = 1280
                    new_height = int(frame.shape[0] * scale_factor)
                    frame = cv2.resize(frame, (new_width, new_height))
                
                with self.raw_frame_lock:
                    self.latest_raw_frame = frame.copy()
                
                # Add to processing queue
                if not self.raw_frame_queue.full():
                    self.raw_frame_queue.put(frame.copy())
                
                time.sleep(0.01)  # Small delay
                
            except Exception as e:
                logger.error(f"Capture loop error: {e}")
                break
    
    def _processing_loop(self):
        """Process frames through AI module"""
        last_process_time = 0
        
        while self.is_running:
            try:
                current_time = time.time()
                
                # Respect processing FPS limit
                if current_time - last_process_time < self.process_interval:
                    time.sleep(0.001)
                    continue
                
                # Get frame from queue
                if self.raw_frame_queue.empty():
                    time.sleep(0.001)
                    continue
                
                frame = self.raw_frame_queue.get()
                
                # Process through module
                processed_frame = self.processing_module.process_frame(frame)
                
                with self.processed_frame_lock:
                    self.latest_processed_frame = processed_frame.copy()
                
                last_process_time = current_time
                
            except Exception as e:
                logger.error(f"Processing loop error: {e}")
                time.sleep(0.1)
    
    def _streaming_loop(self):
        """Prepare frames for streaming"""
        last_stream_time = 0
        
        while self.is_running:
            try:
                current_time = time.time()
                
                # Respect streaming FPS limit
                if current_time - last_stream_time < self.stream_interval:
                    time.sleep(0.001)
                    continue
                
                # Use processed frame if available, otherwise raw frame
                frame_to_stream = None
                
                with self.processed_frame_lock:
                    if self.latest_processed_frame is not None:
                        frame_to_stream = self.latest_processed_frame.copy()
                
                if frame_to_stream is None:
                    with self.raw_frame_lock:
                        if self.latest_raw_frame is not None:
                            frame_to_stream = self.latest_raw_frame.copy()
                
                if frame_to_stream is not None:
                    # Resize for streaming if needed
                    if frame_to_stream.shape[1] > 800:
                        scale_factor = 800 / frame_to_stream.shape[1]
                        new_width = 800
                        new_height = int(frame_to_stream.shape[0] * scale_factor)
                        frame_to_stream = cv2.resize(frame_to_stream, (new_width, new_height))
                    
                    with self.stream_frame_lock:
                        self.stream_frame = frame_to_stream
                
                last_stream_time = current_time
                
            except Exception as e:
                logger.error(f"Streaming loop error: {e}")
                time.sleep(0.1)
    
    def get_latest_frame(self):
        """Get latest frame for streaming"""
        with self.stream_frame_lock:
            if self.stream_frame is not None:
                return self.stream_frame.copy()
            else:
                return np.zeros((480, 640, 3), dtype=np.uint8)