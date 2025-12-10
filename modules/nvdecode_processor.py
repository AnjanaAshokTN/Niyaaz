"""
NVDECODE Hardware Video Processor
Uses NVIDIA GPU hardware decoding to reduce CPU load
"""
import cv2
import numpy as np
import logging
import time
from pathlib import Path
import subprocess

logger = logging.getLogger(__name__)

class NVDecodeProcessor:
    """
    Hardware-accelerated video processor using NVDECODE
    Reduces CPU utilization by offloading video decoding to GPU
    """
    
    def __init__(self, video_source, target_size=(640, 480)):
        """
        Initialize NVDECODE processor
        
        Args:
            video_source: RTSP URL or video file path
            target_size: Target resolution for processing (width, height)
        """
        self.video_source = video_source
        self.target_size = target_size
        self.cap = None
        self.is_running = False
        self.frame_count = 0
        self.last_fps_check = time.time()
        self.fps = 0
        
        # Try to initialize hardware decoder
        self.hardware_available = self._check_hardware_support()
        
    def _check_hardware_support(self):
        """Check if NVDECODE is available"""
        try:
            # Check if OpenCV was built with CUDA support
            if not cv2.cuda.getCudaEnabledDeviceCount() > 0:
                logger.warning("CUDA not available in OpenCV")
                return False
            
            # Check for NVIDIA GPU
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
                if result.returncode != 0:
                    logger.warning("NVIDIA GPU not detected")
                    return False
            except FileNotFoundError:
                logger.warning("nvidia-smi not found")
                return False
            
            logger.info("Hardware decoding support detected")
            return True
            
        except Exception as e:
            logger.warning(f"Hardware support check failed: {e}")
            return False
    
    def initialize_decoder(self):
        """Initialize video decoder with hardware acceleration"""
        try:
            if self.hardware_available and isinstance(self.video_source, str):
                # For RTSP streams, try hardware decoding
                if self.video_source.startswith(('rtsp://', 'rtmp://', 'http://')):
                    self.cap = self._init_hardware_rtsp_decoder()
                else:
                    self.cap = self._init_hardware_file_decoder()
            
            # Fallback to software decoder
            if self.cap is None:
                logger.info("Using software decoder as fallback")
                self.cap = cv2.VideoCapture(self.video_source)
            
            if not self.cap.isOpened():
                raise ValueError(f"Cannot open video source: {self.video_source}")
            
            # Configure decoder properties
            self._configure_decoder()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize decoder: {e}")
            return False
    
    def _init_hardware_rtsp_decoder(self):
        """Initialize hardware RTSP decoder using FFmpeg backend"""
        try:
            # Use CAP_FFMPEG with hardware acceleration
            cap = cv2.VideoCapture(self.video_source, cv2.CAP_FFMPEG)
            
            # Set hardware decoder properties
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
            
            # Test if it works
            ret, frame = cap.read()
            if ret and frame is not None:
                logger.info("Hardware RTSP decoder initialized successfully")
                return cap
            else:
                cap.release()
                return None
                
        except Exception as e:
            logger.warning(f"Hardware RTSP decoder failed: {e}")
            return None
    
    def _init_hardware_file_decoder(self):
        """Initialize hardware file decoder"""
        try:
            # For video files, use default OpenCV with optimizations
            cap = cv2.VideoCapture(self.video_source)
            
            # Configure for best performance
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
            
            return cap
            
        except Exception as e:
            logger.warning(f"Hardware file decoder failed: {e}")
            return None
    
    def _configure_decoder(self):
        """Configure decoder for optimal performance"""
        try:
            # Get video properties
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            logger.info(f"Video properties: {width}x{height} @ {fps} FPS")
            
            # Set buffer size for real-time processing
            if isinstance(self.video_source, str) and self.video_source.startswith(('rtsp://', 'rtmp://')):
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer for live streams
            else:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Small buffer for files
            
        except Exception as e:
            logger.warning(f"Decoder configuration failed: {e}")
    
    def read_frame(self):
        """
        Read and preprocess frame with hardware acceleration
        
        Returns:
            tuple: (success, processed_frame)
        """
        if not self.cap:
            return False, None
        
        try:
            ret, frame = self.cap.read()
            
            if not ret:
                return False, None
            
            # Hardware-accelerated preprocessing
            processed_frame = self._preprocess_frame(frame)
            
            # Update FPS counter
            self._update_fps_counter()
            
            return True, processed_frame
            
        except Exception as e:
            logger.error(f"Frame reading error: {e}")
            return False, None
    
    def _preprocess_frame(self, frame):
        """Preprocess frame with potential GPU acceleration"""
        try:
            # Resize frame if needed
            if frame.shape[1] != self.target_size[0] or frame.shape[0] != self.target_size[1]:
                # Use GPU for resizing if available
                if self.hardware_available:
                    processed_frame = self._gpu_resize(frame)
                else:
                    processed_frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_LINEAR)
            else:
                processed_frame = frame
            
            return processed_frame
            
        except Exception as e:
            logger.warning(f"Frame preprocessing failed: {e}")
            return frame
    
    def _gpu_resize(self, frame):
        """GPU-accelerated frame resizing using OpenCV CUDA"""
        try:
            # Upload frame to GPU
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)
            
            # Resize on GPU
            gpu_resized = cv2.cuda.resize(gpu_frame, self.target_size)
            
            # Download back to CPU
            resized_frame = gpu_resized.download()
            
            return resized_frame
            
        except Exception as e:
            logger.warning(f"GPU resize failed, using CPU: {e}")
            return cv2.resize(frame, self.target_size, interpolation=cv2.INTER_LINEAR)
    
    def _update_fps_counter(self):
        """Update FPS counter"""
        self.frame_count += 1
        current_time = time.time()
        
        if current_time - self.last_fps_check >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_fps_check)
            self.frame_count = 0
            self.last_fps_check = current_time
    
    def start(self):
        """Start the decoder"""
        if self.initialize_decoder():
            self.is_running = True
            logger.info(f"NVDECODE processor started for: {self.video_source}")
            return True
        return False
    
    def stop(self):
        """Stop the decoder"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        logger.info("NVDECODE processor stopped")
    
    def get_stats(self):
        """Get performance statistics"""
        return {
            'fps': self.fps,
            'hardware_enabled': self.hardware_available,
            'target_size': self.target_size,
            'is_running': self.is_running
        }


class FFmpegNVDecoder:
    """
    Alternative decoder using FFmpeg subprocess with NVDECODE
    For maximum hardware acceleration
    """
    
    def __init__(self, video_source, target_size=(640, 480)):
        self.video_source = video_source
        self.target_size = target_size
        self.process = None
        self.is_running = False
        
    def _build_ffmpeg_command(self):
        """Build FFmpeg command with NVDECODE"""
        cmd = [
            'ffmpeg',
            '-hwaccel', 'cuda',  # Use CUDA hardware acceleration
            '-hwaccel_device', '0',  # Use first GPU
            '-i', self.video_source,  # Input source
            '-vf', f'scale_cuda={self.target_size[0]}:{self.target_size[1]}',  # GPU scaling
            '-f', 'rawvideo',  # Output raw frames
            '-pix_fmt', 'bgr24',  # OpenCV compatible format
            '-'  # Output to stdout
        ]
        return cmd
    
    def start_decoder(self):
        """Start FFmpeg decoder process"""
        try:
            cmd = self._build_ffmpeg_command()
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10**8
            )
            self.is_running = True
            logger.info("FFmpeg NVDECODE process started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start FFmpeg decoder: {e}")
            return False
    
    def read_frame(self):
        """Read frame from FFmpeg process"""
        if not self.process or not self.is_running:
            return False, None
        
        try:
            # Calculate frame size
            frame_size = self.target_size[0] * self.target_size[1] * 3  # BGR = 3 channels
            
            # Read frame data
            raw_frame = self.process.stdout.read(frame_size)
            
            if len(raw_frame) != frame_size:
                return False, None
            
            # Convert to numpy array
            frame = np.frombuffer(raw_frame, dtype=np.uint8)
            frame = frame.reshape((self.target_size[1], self.target_size[0], 3))
            
            return True, frame
            
        except Exception as e:
            logger.error(f"Frame reading error: {e}")
            return False, None
    
    def stop(self):
        """Stop FFmpeg process"""
        self.is_running = False
        if self.process:
            self.process.terminate()
            self.process.wait()
        logger.info("FFmpeg NVDECODE process stopped")


# Factory function to create appropriate decoder
def create_hardware_decoder(video_source, target_size=(640, 480), prefer_ffmpeg=False):
    """
    Create hardware decoder based on availability and preferences
    
    Args:
        video_source: Video source (RTSP URL or file path)
        target_size: Target resolution
        prefer_ffmpeg: Whether to prefer FFmpeg over OpenCV
        
    Returns:
        Decoder instance
    """
    if prefer_ffmpeg:
        decoder = FFmpegNVDecoder(video_source, target_size)
        if decoder.start_decoder():
            return decoder
    
    # Fallback to OpenCV-based decoder
    decoder = NVDecodeProcessor(video_source, target_size)
    if decoder.start():
        return decoder
    
    logger.warning("Hardware decoding not available, consider software fallback")
    return None