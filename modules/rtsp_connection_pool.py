"""
RTSP Connection Pool Manager
Efficiently manages shared RTSP connections across multiple channels
Supports GPU-accelerated video decoding with automatic fallback
"""
import cv2
import threading
import time
import logging
import subprocess as sp
from queue import Queue, Empty
from typing import Dict, List, Callable, Optional, Tuple
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

# Import Telegram notification function
try:
    from modules.queue_monitor import send_telegram_message
    import os
    # Check if credentials are actually set
    bot_token = os.getenv("bot_token") or os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("chat_id") or os.getenv("TELEGRAM_CHAT_ID")
    if bot_token and chat_id:
        TELEGRAM_AVAILABLE = True
        logger.info("✅ Telegram notification function imported successfully and credentials are configured")
    else:
        TELEGRAM_AVAILABLE = False
        logger.warning("⚠️ Telegram notification function imported but credentials NOT SET (bot_token/chat_id or TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID)")
        logger.warning("   RTSP connection alerts will NOT be sent to Telegram")
except ImportError as e:
    # Fallback if queue_monitor is not available
    TELEGRAM_AVAILABLE = False
    logger.warning(f"Failed to import send_telegram_message: {e}")
    def send_telegram_message(text: str) -> bool:
        logger.warning("Telegram notification not available (fallback function)")
        return False

# Try to import GStreamer NVDEC decoder
try:
    from modules.gstreamer_nvdec import GStreamerNVDECDecoder, test_nvdec_availability
    # Temporarily disable GStreamer NVDEC due to stability issues
    # TODO: Re-enable after resolving nvv4l2decoder pipeline issues
    GSTREAMER_NVDEC_AVAILABLE = False  # Set to True when fixed
    logger.info("ℹ️ GStreamer NVDEC module available but disabled (stability)")
except ImportError as e:
    GSTREAMER_NVDEC_AVAILABLE = False
    logger.warning(f"GStreamer NVDEC module not available: {e}")


class FFmpegRTSPReader:
    """
    FFmpeg-based RTSP reader using subprocess for reliable RTSP streaming
    Similar to the user's working example
    """
    
    def __init__(self, rtsp_url: str, width: int = 960, height: int = 540):
        """
        Initialize FFmpeg RTSP reader
        Matches the reference script's resolution (960x540 scaled from 1920x1080)
        
        Args:
            rtsp_url: RTSP stream URL
            width: Target frame width (default 960 to match reference)
            height: Target frame height (default 540 to match reference)
        """
        self.rtsp_url = rtsp_url
        self.width = width
        self.height = height
        self.frame_size = width * height * 3  # BGR24 format
        
        self.proc = None
        self.is_running = False
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.error_thread = None
        
    def start(self) -> bool:
        """Start ffmpeg subprocess"""
        try:
            cmd = [
                "ffmpeg",
                "-rtsp_transport", "tcp",
                "-rtsp_flags", "prefer_tcp",  # Match reference script
                "-i", self.rtsp_url,
                "-an",  # No audio
                "-vf", f"scale={self.width}:{self.height}",
                "-r", "8",  # Limit to 8 fps to match reference script (reduces load)
                "-f", "rawvideo",
                "-pix_fmt", "bgr24",
                "-loglevel", "error",  # Reduce log noise
                "pipe:1",
            ]
            
            logger.info(f"Starting ffmpeg subprocess for: {self.rtsp_url}")
            self.proc = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.PIPE, bufsize=10**8)
            
            # Start error logging thread
            self.error_thread = threading.Thread(target=self._log_errors, daemon=True)
            self.error_thread.start()
            
            # Test read to verify it's working
            test_frame = self.proc.stdout.read(self.frame_size)
            if len(test_frame) == self.frame_size:
                self.is_running = True
                logger.info(f"✅ FFmpeg subprocess started successfully for: {self.rtsp_url}")
                return True
            else:
                logger.error(f"FFmpeg subprocess test read failed: got {len(test_frame)} bytes, expected {self.frame_size}")
                self.proc.terminate()
                self.proc = None
                return False
                
        except Exception as e:
            logger.error(f"Failed to start ffmpeg subprocess: {e}")
            if self.proc:
                self.proc.terminate()
                self.proc = None
            return False
    
    def _log_errors(self):
        """Log ffmpeg stderr output"""
        if self.proc and self.proc.stderr:
            try:
                for line in iter(self.proc.stderr.readline, b''):
                    if line:
                        logger.debug(f"[FFMPEG] {line.decode(errors='ignore').strip()}")
            except Exception as e:
                logger.debug(f"Error logging thread ended: {e}")
    
    def _read_exact(self, n):
        """
        Read exactly n bytes from stdout (matches reference script approach)
        More robust than single read() call
        """
        buf = b""
        while len(buf) < n:
            chunk = self.proc.stdout.read(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        return buf
    
    def read(self):
        """
        Read a frame from ffmpeg subprocess
        Uses _read_exact for robust frame reading (matches reference script)
        
        Returns:
            (ret, frame): Tuple of success status and frame (numpy array)
        """
        if not self.is_running or not self.proc:
            return False, None
        
        try:
            raw_frame = self._read_exact(self.frame_size)
            
            if raw_frame is None or len(raw_frame) != self.frame_size:
                return False, None
            
            # Convert raw bytes to numpy array
            frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((self.height, self.width, 3)).copy()
            
            with self.frame_lock:
                self.latest_frame = frame
            
            return True, frame
            
        except Exception as e:
            logger.error(f"Error reading frame from ffmpeg: {e}")
            return False, None
    
    def release(self):
        """Stop and cleanup ffmpeg subprocess"""
        self.is_running = False
        
        if self.proc:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=5)
            except:
                try:
                    self.proc.kill()
                except:
                    pass
            self.proc = None
        
        logger.info(f"FFmpeg subprocess stopped for: {self.rtsp_url}")
    
    def isOpened(self):
        """Check if reader is open and running"""
        return self.is_running and self.proc is not None


class GPUDecoderManager:
    """Manages GPU video decoder availability and creation"""
    
    @staticmethod
    def is_cuda_available() -> bool:
        """Check if CUDA is available for video decoding"""
        try:
            return cv2.cuda.getCudaEnabledDeviceCount() > 0
        except Exception:
            return False
    
    @staticmethod
    def create_gpu_video_reader(rtsp_url: str, target_size=(640, 640)) -> Tuple[Optional[object], bool, str]:
        """
        Create GPU-accelerated video reader with multiple fallback options
        
        Priority:
        1. GStreamer NVDEC (Hardware decoder via GStreamer)
        2. OpenCV cudacodec (if available)
        3. CPU OpenCV (fallback)
        
        Args:
            rtsp_url: RTSP stream URL
            target_size: Target frame size for GStreamer decoder
            
        Returns:
            (video_reader, is_gpu, decoder_type): Tuple of reader, GPU flag, and decoder type
        """
        # Try GStreamer NVDEC first (BEST option for DeepStream)
        if GSTREAMER_NVDEC_AVAILABLE:
            try:
                logger.info(f"Attempting GStreamer NVDEC decoder for: {rtsp_url}")
                from modules.gstreamer_nvdec import create_nvdec_decoder
                decoder = create_nvdec_decoder(rtsp_url, target_size=target_size)
                if decoder is not None:
                    logger.info(f"✅ GStreamer NVDEC decoder successfully created for: {rtsp_url}")
                    return decoder, True, "GStreamer NVDEC (Hardware)"
                else:
                    logger.warning(f"GStreamer NVDEC decoder creation returned None: {rtsp_url}")
            except Exception as e:
                logger.warning(f"GStreamer NVDEC decoder failed for {rtsp_url}: {e}", exc_info=False)
        
        # Try OpenCV CUDA decoder (SECOND option)
        if GPUDecoderManager.is_cuda_available():
            try:
                logger.info(f"Attempting OpenCV CUDA decoder for: {rtsp_url}")
                gpu_reader = cv2.cudacodec.createVideoReader(rtsp_url)
                
                if gpu_reader is not None:
                    # Test if GPU reader can actually read frames
                    test_ret, test_frame = gpu_reader.nextFrame()
                    if test_ret and test_frame is not None:
                        logger.info(f"✅ OpenCV CUDA decoder successfully created for: {rtsp_url}")
                        return gpu_reader, True, "OpenCV CUDA (Hardware)"
                    else:
                        logger.warning(f"OpenCV CUDA decoder created but cannot read frames: {rtsp_url}")
                else:
                    logger.warning(f"OpenCV CUDA decoder creation returned None: {rtsp_url}")
                    
            except Exception as e:
                logger.warning(f"OpenCV CUDA decoder failed for {rtsp_url}: {e}")
        
        # Fallback to CPU decoder (THIRD option) - Try OpenCV with short timeout
        logger.info(f"Using CPU decoder for: {rtsp_url}")
        
        # Use threading with SHORT timeout to fail fast for fallback
        try:
            result_queue = Queue()
            exception_queue = Queue()
            
            def create_capture():
                try:
                    logger.debug(f"Attempting RTSP connection with URL: {rtsp_url}")
                    cpu_reader = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                    if cpu_reader.isOpened():
                        ret, test_frame = cpu_reader.read()
                        if ret and test_frame is not None:
                            result_queue.put(('success', cpu_reader))
                        else:
                            cpu_reader.release()
                            result_queue.put(('error', 'Failed to read frame'))
                    else:
                        result_queue.put(('error', 'Failed to open'))
                except Exception as e:
                    exception_queue.put(e)
            
            # Use SHORT timeout (5 seconds) to fail fast and allow fallback
            thread = threading.Thread(target=create_capture, daemon=True)
            thread.start()
            thread.join(timeout=5)  # 5 second timeout for fast fallback
            
            if thread.is_alive():
                logger.warning(f"OpenCV CPU decoder timeout after 5s for: {rtsp_url} - will fail for fallback")
                return None, False, "None"  # Return None immediately for fast fallback
            else:
                if not exception_queue.empty():
                    exc = exception_queue.get()
                    logger.warning(f"OpenCV CPU decoder exception: {exc}")
                elif not result_queue.empty():
                    status, result = result_queue.get()
                    if status == 'success':
                        logger.info(f"CPU decoder connection verified for: {rtsp_url}")
                        return result, False, "CPU (OpenCV)"
                    else:
                        logger.warning(f"OpenCV CPU decoder failed: {result}")
        
        except Exception as e:
            logger.warning(f"OpenCV CPU decoder threading error: {e}")
        
        # Final fallback: Use ffmpeg subprocess (FOURTH option)
        # Use 960x540 resolution to match reference script (scaled from 1920x1080)
        # Check if ffmpeg is available first
        import shutil
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path:
            logger.info(f"Trying ffmpeg subprocess for: {rtsp_url} (found at {ffmpeg_path})")
            try:
                ffmpeg_reader = FFmpegRTSPReader(rtsp_url, width=960, height=540)
                if ffmpeg_reader.start():
                    logger.info(f"FFmpeg subprocess decoder started successfully for: {rtsp_url} (960x540 @ 8fps)")
                    return ffmpeg_reader, False, "FFmpeg (Subprocess)"
                else:
                    logger.warning(f"FFmpeg subprocess failed to start for: {rtsp_url}")
            except Exception as e:
                logger.warning(f"FFmpeg subprocess exception for {rtsp_url}: {e}")
        else:
            logger.warning(f"FFmpeg not found in PATH, skipping ffmpeg subprocess fallback for: {rtsp_url}")
            logger.warning(f"   To use ffmpeg fallback, install ffmpeg and add it to your system PATH")
        
        # If we get here, all decoders failed - return None quickly for video file fallback
        logger.warning(f"⚠ RTSP connection failed for: {rtsp_url} (will use video file fallback if configured)")
        return None, False, "None"

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

class RTSPConnectionPool:
    """
    Manages shared RTSP connections to avoid duplicate streams
    """
    
    def __init__(self):
        """Initialize RTSP connection pool"""
        self._connections = {}  # {rtsp_url: RTSPConnection}
        self._lock = threading.RLock()
        self._subscribers = {}  # {rtsp_url: {channel_id: callback}}
        
        # Alert aggregation for Telegram messages
        self._pending_alerts = set()  # Set of alert types that are pending
        self._alert_lock = threading.Lock()
        self._alert_timer = None  # Timer for debouncing alerts
    
    def _send_aggregated_alert(self, alert_type: str, immediate: bool = False):
        """
        Send aggregated Telegram alert for connection issues
        Args:
            alert_type: Type of alert ('lost', 'restored', 'issues')
            immediate: If True, send immediately without waiting for aggregation timer
        """
        with self._alert_lock:
            # Add alert to pending set
            self._pending_alerts.add(alert_type)
            
            # Cancel existing timer if any
            if self._alert_timer is not None:
                self._alert_timer.cancel()
                self._alert_timer = None
            
            # Function to send the message
            def send_message():
                with self._alert_lock:
                    if not self._pending_alerts:
                        return
                    
                    # Determine message based on alert types
                    if 'lost' in self._pending_alerts:
                        message = "🔴 RTSP Connection Lost"
                    elif 'restored' in self._pending_alerts:
                        message = "✅ RTSP Connection Restored"
                    elif 'issues' in self._pending_alerts:
                        message = "⚠️ RTSP Connection Issues"
                    else:
                        message = "⚠️ RTSP Connection Alert"
                    
                    # Send single aggregated message
                    if TELEGRAM_AVAILABLE and self._pending_alerts:
                        try:
                            success = send_telegram_message(message)
                            if success:
                                logger.info(f"✅ Sent aggregated Telegram alert: {message}")
                            else:
                                logger.warning(f"⚠️ Failed to send aggregated Telegram alert: {message}")
                        except Exception as e:
                            logger.error(f"Failed to send aggregated Telegram alert: {e}")
                    
                    # Clear pending alerts
                    self._pending_alerts.clear()
                    self._alert_timer = None
            
            # Send immediately if requested, otherwise use timer
            if immediate:
                send_message()
            else:
                self._alert_timer = threading.Timer(2.0, send_message)
                self._alert_timer.daemon = True
                self._alert_timer.start()
        
    def get_connection(self, rtsp_url: str, channel_id: str, frame_callback: Callable):
        """
        Get or create RTSP connection and subscribe to frames
        
        Args:
            rtsp_url: RTSP stream URL
            channel_id: Unique channel identifier
            frame_callback: Function to call with new frames
            
        Returns:
            RTSPConnection instance
        """
        with self._lock:
            # Create connection if it doesn't exist
            if rtsp_url not in self._connections:
                logger.info(f"Creating new RTSP connection for: {rtsp_url}")
                connection = RTSPConnection(rtsp_url)
                connection.set_pool(self)  # Set pool reference for broadcasting
                
                # Add subscriber first so we can get affected channels if connection fails
                if rtsp_url not in self._subscribers:
                    self._subscribers[rtsp_url] = {}
                self._subscribers[rtsp_url][channel_id] = frame_callback
                
                if connection.start():
                    self._connections[rtsp_url] = connection
                else:
                    # Connection failed to start - send simplified aggregated alert immediately
                    logger.warning(f"RTSP connection failed to start for: {rtsp_url}")
                    self._send_aggregated_alert('lost', immediate=True)
                    
                    # Remove the subscriber since connection failed
                    self._subscribers[rtsp_url].pop(channel_id, None)
                    if not self._subscribers[rtsp_url]:
                        del self._subscribers[rtsp_url]
                    
                    raise ValueError(f"Failed to create RTSP connection: {rtsp_url}")
            else:
                # Subscribe channel to existing connection
                if rtsp_url not in self._subscribers:
                    self._subscribers[rtsp_url] = {}
                self._subscribers[rtsp_url][channel_id] = frame_callback
            
            connection = self._connections[rtsp_url]
            
            logger.info(f"Channel {channel_id} subscribed to RTSP: {rtsp_url}")
            logger.info(f"Total subscribers for {rtsp_url}: {len(self._subscribers[rtsp_url])}")
            
            return connection
    
    def unsubscribe(self, rtsp_url: str, channel_id: str):
        """
        Unsubscribe channel from RTSP connection
        
        Args:
            rtsp_url: RTSP stream URL
            channel_id: Channel to unsubscribe
        """
        with self._lock:
            if rtsp_url in self._subscribers:
                self._subscribers[rtsp_url].pop(channel_id, None)
                
                # If no more subscribers, stop the connection
                if not self._subscribers[rtsp_url]:
                    logger.info(f"No more subscribers for {rtsp_url}, stopping connection")
                    if rtsp_url in self._connections:
                        self._connections[rtsp_url].stop()
                        del self._connections[rtsp_url]
                    del self._subscribers[rtsp_url]
                else:
                    logger.info(f"Channel {channel_id} unsubscribed from {rtsp_url}")
                    logger.info(f"Remaining subscribers: {len(self._subscribers[rtsp_url])}")
    
    def broadcast_frame(self, rtsp_url: str, frame: np.ndarray):
        """
        Broadcast frame to all subscribers of an RTSP stream
        
        Args:
            rtsp_url: RTSP stream URL
            frame: Video frame to broadcast
        """
        # Validate frame before broadcasting using safe check
        if not safe_array_check(frame, "valid"):
            return
            
        if rtsp_url in self._subscribers:
            for channel_id, callback in self._subscribers[rtsp_url].items():
                try:
                    callback(frame.copy())  # Send copy to each subscriber
                except Exception as e:
                    logger.error(f"Error broadcasting frame to {channel_id}: {e}")
    
    def get_affected_channels(self, rtsp_url: str) -> List[str]:
        """
        Get list of channel IDs affected by an RTSP connection
        
        Args:
            rtsp_url: RTSP stream URL
            
        Returns:
            List of channel IDs subscribed to this RTSP stream
        """
        with self._lock:
            if rtsp_url in self._subscribers:
                return list(self._subscribers[rtsp_url].keys())
            return []
    
    def get_connection_stats(self) -> Dict:
        """Get comprehensive statistics about RTSP connections including GPU performance"""
        with self._lock:
            stats = {}
            total_gpu_connections = 0
            total_cpu_connections = 0
            
            for rtsp_url, connection in self._connections.items():
                subscriber_count = len(self._subscribers.get(rtsp_url, {}))
                decoder_stats = connection.get_decoder_stats()
                
                if decoder_stats['is_gpu']:
                    total_gpu_connections += 1
                else:
                    total_cpu_connections += 1
                
                stats[rtsp_url] = {
                    'subscribers': subscriber_count,
                    'is_active': connection.is_running,
                    'fps': connection.current_fps,
                    'frames_read': connection.frames_read,
                    'decoder_type': decoder_stats['decoder_type'],
                    'is_gpu': decoder_stats['is_gpu'],
                    'gpu_frames_processed': decoder_stats['gpu_frames_processed'],
                    'cpu_fallback_count': decoder_stats['cpu_fallback_count']
                }
            
            # Add summary statistics
            stats['_summary'] = {
                'total_connections': len(self._connections),
                'gpu_connections': total_gpu_connections,
                'cpu_connections': total_cpu_connections,
                'gpu_acceleration_ratio': total_gpu_connections / max(len(self._connections), 1),
                'cuda_available': GPUDecoderManager.is_cuda_available()
            }
            
            return stats


class RTSPConnection:
    """
    Individual RTSP connection handler with GPU-accelerated frame decoding
    """
    
    def __init__(self, rtsp_url: str, fps_limit: int = 30):
        """
        Initialize RTSP connection with GPU support
        
        Args:
            rtsp_url: RTSP stream URL
            fps_limit: Maximum FPS for reading frames
        """
        self.rtsp_url = rtsp_url
        self.fps_limit = fps_limit
        self.frame_interval = 1.0 / fps_limit
        
        # Video capture (GPU or CPU)
        self.cap = None
        self.is_gpu_decoder = False
        self.decoder_type = "Unknown"
        
        # Threading
        self.is_running = False
        self.capture_thread = None
        
        # Frame management
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        
        # Statistics
        self.frames_read = 0
        self.start_time = None
        self.current_fps = 0
        self.gpu_frames_processed = 0
        self.cpu_fallback_count = 0
        
        # Connection pool reference
        self.pool = None
    
    def set_pool(self, pool: RTSPConnectionPool):
        """Set reference to connection pool for broadcasting"""
        self.pool = pool
    
    def start(self) -> bool:
        """
        Start RTSP connection with GPU-accelerated decoding
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create GPU or CPU video reader (try GStreamer NVDEC first)
            self.cap, self.is_gpu_decoder, self.decoder_type = GPUDecoderManager.create_gpu_video_reader(
                self.rtsp_url, target_size=(640, 640)
            )
            
            if self.cap is None:
                logger.error(f"Cannot create video reader for: {self.rtsp_url}")
                
                # Send simplified aggregated alert immediately
                if self.pool:
                    self.pool._send_aggregated_alert('lost', immediate=True)
                
                return False
            
            logger.info(f"Using {self.decoder_type} decoder for: {self.rtsp_url}")
            
            # Configure video capture (CPU only settings, skip for FFmpeg)
            if not self.is_gpu_decoder and isinstance(self.cap, cv2.VideoCapture):
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
                self.cap.set(cv2.CAP_PROP_FPS, self.fps_limit)
                
                # Try to set codec for better performance
                try:
                    self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))
                except:
                    pass  # Ignore if not supported
                
                # Get stream properties for CPU decoder
                try:
                    frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    stream_fps = self.cap.get(cv2.CAP_PROP_FPS)
                    logger.info(f"RTSP stream opened: {frame_width}x{frame_height} @ {stream_fps} FPS ({self.decoder_type})")
                except:
                    logger.info(f"RTSP stream opened with {self.decoder_type} (properties unavailable)")
            elif isinstance(self.cap, FFmpegRTSPReader):
                logger.info(f"FFmpeg RTSP stream opened: {self.cap.width}x{self.cap.height} ({self.decoder_type})")
            else:
                logger.info(f"GPU-accelerated RTSP stream opened: {self.rtsp_url} ({self.decoder_type})")
            
            # Start capture thread
            self.is_running = True
            self.start_time = time.time()
            
            self.capture_thread = threading.Thread(target=self._capture_loop)
            self.capture_thread.daemon = True
            self.capture_thread.start()
            
            logger.info(f"RTSP connection started: {self.rtsp_url} ({self.decoder_type})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start RTSP connection {self.rtsp_url}: {e}")
            return False
    
    def stop(self):
        """Stop RTSP connection"""
        self.is_running = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=5)
        
        if self.cap:
            if self.is_gpu_decoder:
                # GPU decoder cleanup (no explicit release needed)
                self.cap = None
            elif isinstance(self.cap, FFmpegRTSPReader):
                # FFmpeg subprocess cleanup
                self.cap.release()
            else:
                # CPU decoder cleanup (OpenCV)
                self.cap.release()
        
        # Log final statistics
        if self.is_gpu_decoder:
            logger.info(f"RTSP connection stopped: {self.rtsp_url} "
                       f"(GPU frames: {self.gpu_frames_processed}, "
                       f"CPU fallbacks: {self.cpu_fallback_count})")
        else:
            logger.info(f"RTSP connection stopped: {self.rtsp_url} (CPU decoder)")
    
    def _read_frame_gpu(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read frame using GPU decoder (supports both OpenCV CUDA and GStreamer NVDEC)
        
        Returns:
            (success, frame): Tuple of success status and frame data
        """
        try:
            # Check if it's GStreamer NVDEC decoder (has .read() method like OpenCV)
            if hasattr(self.cap, 'read') and 'GStreamer' in self.decoder_type:
                # GStreamer NVDEC decoder
                ret, frame = self.cap.read()
                
                if ret and safe_array_check(frame, "valid"):
                    self.gpu_frames_processed += 1
                    return True, frame
                else:
                    return False, None
            else:
                # OpenCV CUDA decoder
                ret, gpu_frame = self.cap.nextFrame()
                
                if ret and gpu_frame is not None:
                    # Download frame from GPU memory to CPU memory
                    cpu_frame = gpu_frame.download()
                    
                    # Validate frame
                    if safe_array_check(cpu_frame, "valid"):
                        self.gpu_frames_processed += 1
                        return True, cpu_frame
                    else:
                        logger.warning(f"GPU decoder returned invalid frame: {self.rtsp_url}")
                        return False, None
                else:
                    return False, None
                
        except Exception as e:
            logger.error(f"GPU frame reading error for {self.rtsp_url}: {e}")
            return False, None
    
    def _read_frame_cpu(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read frame using CPU decoder
        
        Returns:
            (success, frame): Tuple of success status and frame data
        """
        try:
            ret, frame = self.cap.read()
            
            if ret and safe_array_check(frame, "valid"):
                return True, frame
            else:
                return False, None
                
        except Exception as e:
            logger.error(f"CPU frame reading error for {self.rtsp_url}: {e}")
            return False, None
    
    def _capture_loop(self):
        """Main capture loop with GPU/CPU frame reading and automatic reconnection"""
        last_frame_time = 0
        consecutive_failures = 0
        max_failures = 30
        gpu_fallback_attempted = False
        last_warning_time = 0
        warning_interval = 10  # Only log warnings every 10 seconds
        reconnect_attempts = 0
        max_reconnect_attempts = 5
        early_warning_sent = False  # Track if early warning was sent in current failure sequence
        
        while self.is_running:
            try:
                current_time = time.time()
                
                # Respect FPS limit
                if current_time - last_frame_time < self.frame_interval:
                    time.sleep(0.001)
                    continue
                
                # Read frame based on decoder type
                if self.is_gpu_decoder:
                    ret, frame = self._read_frame_gpu()
                    
                    # If GPU reading fails repeatedly, try CPU fallback
                    if not ret and consecutive_failures > 10 and not gpu_fallback_attempted:
                        logger.warning(f"GPU decoder struggling, attempting CPU fallback for: {self.rtsp_url}")
                        try:
                            # Create CPU fallback with FFmpeg backend
                            cpu_cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
                            if cpu_cap.isOpened():
                                self.cap = cpu_cap
                                self.is_gpu_decoder = False
                                self.decoder_type = "CPU (Fallback)"
                                self.cpu_fallback_count += 1
                                gpu_fallback_attempted = True
                                logger.info(f"Successfully switched to CPU fallback for: {self.rtsp_url}")
                                continue
                        except Exception as e:
                            logger.error(f"CPU fallback failed for {self.rtsp_url}: {e}")
                else:
                    ret, frame = self._read_frame_cpu()
                
                if not ret or not safe_array_check(frame, "valid"):
                    consecutive_failures += 1
                    
                    # Send early warning alert when failures start accumulating (once per failure sequence)
                    if consecutive_failures >= 10 and not early_warning_sent:
                        early_warning_sent = True  # Mark as sent to avoid duplicates
                        if self.pool:
                            self.pool._send_aggregated_alert('issues')
                    
                    # Only log warnings periodically to reduce spam
                    if current_time - last_warning_time >= warning_interval:
                        logger.warning(f"Failed to read frame from {self.rtsp_url} "
                                     f"(failure {consecutive_failures}/{max_failures}) [{self.decoder_type}]")
                        last_warning_time = current_time
                    
                    # Attempt reconnection if failures persist
                    if consecutive_failures >= 15 and reconnect_attempts < max_reconnect_attempts:
                        reconnect_attempts += 1
                        logger.info(f"Attempting to reconnect to {self.rtsp_url} (attempt {reconnect_attempts}/{max_reconnect_attempts})")
                        
                        # Release current connection
                        try:
                            if isinstance(self.cap, cv2.VideoCapture):
                                self.cap.release()
                            elif isinstance(self.cap, FFmpegRTSPReader):
                                self.cap.release()
                        except:
                            pass
                        
                        # Wait before reconnecting
                        time.sleep(2)
                        
                        # Try to reconnect
                        reconnect_successful = False
                        try:
                            if isinstance(self.cap, cv2.VideoCapture) or self.cap is None:
                                new_cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
                                if new_cap.isOpened():
                                    # Test read
                                    test_ret, test_frame = new_cap.read()
                                    if test_ret and test_frame is not None:
                                        # Store reconnection attempt count before resetting
                                        successful_reconnect_attempt = reconnect_attempts
                                        
                                        self.cap = new_cap
                                        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                                        self.cap.set(cv2.CAP_PROP_FPS, self.fps_limit)
                                        consecutive_failures = 0
                                        reconnect_attempts = 0
                                        reconnect_successful = True
                                        early_warning_sent = False  # Reset early warning flag on successful reconnection
                                        logger.info(f"Successfully reconnected to {self.rtsp_url}")
                                        
                                        # Force immediate frame read and broadcast to notify subscribers
                                        try:
                                            test_ret, test_frame = self.cap.read()
                                            if test_ret and safe_array_check(test_frame, "valid"):
                                                # Update latest frame immediately
                                                with self.frame_lock:
                                                    self.latest_frame = test_frame.copy()
                                                
                                                # Broadcast immediately to notify all subscribers
                                                if self.pool:
                                                    self.pool.broadcast_frame(self.rtsp_url, test_frame)
                                                logger.info(f"Immediate frame broadcast after reconnection: {self.rtsp_url}")
                                        except Exception as e:
                                            logger.debug(f"Could not immediately broadcast frame after reconnection: {e}")
                                        
                                        # Send Telegram alert for successful reconnection
                                        if self.pool:
                                            self.pool._send_aggregated_alert('restored')
                                        continue
                                    else:
                                        new_cap.release()
                                else:
                                    logger.warning(f"Reconnection attempt {reconnect_attempts} failed: could not open stream")
                        except Exception as e:
                            logger.warning(f"Reconnection attempt {reconnect_attempts} failed: {e}")
                        
                        # Send Telegram alert if all reconnection attempts failed
                        if not reconnect_successful and reconnect_attempts >= max_reconnect_attempts:
                            if self.pool:
                                self.pool._send_aggregated_alert('lost')
                    
                    if consecutive_failures >= max_failures:
                        logger.error(f"Too many consecutive failures for {self.rtsp_url}, stopping")
                        
                        # Send Telegram alert for connection loss
                        if self.pool:
                            self.pool._send_aggregated_alert('lost')
                        break
                    
                    time.sleep(0.1)
                    continue
                
                # Reset failure counter and reconnection attempts on successful read
                consecutive_failures = 0
                reconnect_attempts = 0
                early_warning_sent = False  # Reset early warning flag on successful read
                
                # Resize frame if too large (optimize for performance)
                if frame.shape[1] > 1280:
                    scale_factor = 1280 / frame.shape[1]
                    new_width = 1280
                    new_height = int(frame.shape[0] * scale_factor)
                    frame = cv2.resize(frame, (new_width, new_height))
                
                # Update latest frame
                with self.frame_lock:
                    self.latest_frame = frame.copy()
                
                # Broadcast to all subscribers via pool
                if self.pool:
                    self.pool.broadcast_frame(self.rtsp_url, frame)
                
                # Update statistics
                self.frames_read += 1
                last_frame_time = current_time
                
                # Calculate FPS every 30 frames
                if self.frames_read % 30 == 0:
                    elapsed_time = current_time - self.start_time
                    self.current_fps = self.frames_read / elapsed_time if elapsed_time > 0 else 0
                    
                    if self.frames_read % 300 == 0:  # Log every 300 frames
                        gpu_info = f", GPU frames: {self.gpu_frames_processed}" if self.gpu_frames_processed > 0 else ""
                        fallback_info = f", CPU fallbacks: {self.cpu_fallback_count}" if self.cpu_fallback_count > 0 else ""
                        
                        logger.info(f"RTSP {self.rtsp_url}: {self.frames_read} frames, "
                                  f"FPS: {self.current_fps:.2f} [{self.decoder_type}]{gpu_info}{fallback_info}")
                
            except Exception as e:
                logger.error(f"Error in RTSP capture loop {self.rtsp_url}: {e}")
                consecutive_failures += 1
                
                if consecutive_failures >= max_failures:
                    break
                
                time.sleep(0.1)
        
        logger.info(f"RTSP capture loop ended for: {self.rtsp_url}")
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """
        Get the latest frame from this RTSP connection
        
        Returns:
            Latest frame or None if not available
        """
        with self.frame_lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            return None
    
    def get_decoder_stats(self) -> Dict:
        """
        Get detailed decoder statistics
        
        Returns:
            Dictionary with decoder performance stats
        """
        return {
            'decoder_type': self.decoder_type,
            'is_gpu': self.is_gpu_decoder,
            'frames_read': self.frames_read,
            'current_fps': self.current_fps,
            'gpu_frames_processed': self.gpu_frames_processed,
            'cpu_fallback_count': self.cpu_fallback_count,
            'rtsp_url': self.rtsp_url
        }


# Global RTSP connection pool instance
rtsp_pool = RTSPConnectionPool()