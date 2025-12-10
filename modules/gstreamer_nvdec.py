"""
GStreamer NVDEC Video Decoder
Hardware-accelerated video decoding using GStreamer with nvdec plugin
"""
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import numpy as np
import cv2
import threading
import queue
import logging
import time

logger = logging.getLogger(__name__)

# Initialize GStreamer
Gst.init(None)

class GStreamerNVDECDecoder:
    """
    Hardware-accelerated video decoder using GStreamer with nvdec
    Pipeline: rtspsrc → rtph264depay → nvdec → nvvidconv → video/x-raw → appsink
    """
    
    def __init__(self, rtsp_url, target_size=(640, 640), buffer_size=5):
        """
        Initialize GStreamer NVDEC decoder
        
        Args:
            rtsp_url: RTSP stream URL
            target_size: Target frame size (width, height)
            buffer_size: Frame buffer size
        """
        self.rtsp_url = rtsp_url
        self.target_size = target_size
        self.buffer_size = buffer_size
        
        # Frame buffer
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        
        # Pipeline components
        self.pipeline = None
        self.bus = None
        self.loop = None
        
        # Status
        self.is_running = False
        self.thread = None
        self.frames_decoded = 0
        self.last_fps_time = time.time()
        self.current_fps = 0
        
        # Create pipeline
        self._create_pipeline()
    
    def _create_pipeline(self):
        """Create GStreamer pipeline with nvdec"""
        try:
            # Build pipeline string
            width, height = self.target_size
            
            # DeepStream 6.4 compatible pipeline
            # Using nvv4l2decoder instead of deprecated nvdec
            pipeline_str = (
                f"rtspsrc location={self.rtsp_url} latency=0 ! "
                "rtph264depay ! h264parse ! "
                "nvv4l2decoder ! "  # NVDEC hardware decoder
                f"nvvideoconvert ! "  # NVIDIA video converter (DeepStream 6.x)
                f"video/x-raw(memory:NVMM),format=RGBA,width={width},height={height} ! "
                "nvvideoconvert ! video/x-raw,format=BGRx ! "
                "videoconvert ! video/x-raw,format=BGR ! "
                "appsink name=sink emit-signals=True sync=False max-buffers=2 drop=True"
            )
            
            logger.info(f"Creating GStreamer pipeline: {pipeline_str[:100]}...")
            
            # Create pipeline
            self.pipeline = Gst.parse_launch(pipeline_str)
            
            # Get appsink
            self.appsink = self.pipeline.get_by_name('sink')
            if self.appsink:
                self.appsink.connect('new-sample', self._on_new_sample)
            
            # Get bus for messages
            self.bus = self.pipeline.get_bus()
            self.bus.add_signal_watch()
            self.bus.connect('message', self._on_bus_message)
            
            logger.info(f"✅ GStreamer NVDEC pipeline created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create GStreamer pipeline: {e}")
            return False
    
    def _on_new_sample(self, sink):
        """Callback for new frame from appsink"""
        try:
            sample = sink.emit('pull-sample')
            if sample is None:
                return Gst.FlowReturn.ERROR
            
            # Get buffer
            buf = sample.get_buffer()
            caps = sample.get_caps()
            
            # Get frame dimensions
            structure = caps.get_structure(0)
            width = structure.get_value('width')
            height = structure.get_value('height')
            
            # Extract frame data
            success, map_info = buf.map(Gst.MapFlags.READ)
            if not success:
                return Gst.FlowReturn.ERROR
            
            # Convert to numpy array
            frame = np.ndarray(
                shape=(height, width, 3),
                dtype=np.uint8,
                buffer=map_info.data
            )
            
            # Make a copy since buffer will be unmapped
            frame = frame.copy()
            
            buf.unmap(map_info)
            
            # Update latest frame
            with self.frame_lock:
                self.latest_frame = frame
                self.frames_decoded += 1
                
                # Calculate FPS
                current_time = time.time()
                if current_time - self.last_fps_time >= 1.0:
                    elapsed = current_time - self.last_fps_time
                    self.current_fps = self.frames_decoded / elapsed
                    self.frames_decoded = 0
                    self.last_fps_time = current_time
            
            # Add to queue (non-blocking)
            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                # Drop frame if queue is full
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait(frame)
                except:
                    pass
            
            return Gst.FlowReturn.OK
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return Gst.FlowReturn.ERROR
    
    def _on_bus_message(self, bus, message):
        """Handle GStreamer bus messages"""
        t = message.type
        
        if t == Gst.MessageType.EOS:
            logger.warning("End of stream")
            self.stop()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            logger.error(f"GStreamer error: {err}, {debug}")
            self.stop()
        elif t == Gst.MessageType.WARNING:
            warn, debug = message.parse_warning()
            logger.warning(f"GStreamer warning: {warn}")
        elif t == Gst.MessageType.STATE_CHANGED:
            if message.src == self.pipeline:
                old_state, new_state, pending_state = message.parse_state_changed()
                logger.debug(f"Pipeline state changed: {old_state.value_nick} -> {new_state.value_nick}")
    
    def start(self):
        """Start decoding"""
        if self.is_running:
            logger.warning("Decoder already running")
            return False
        
        if self.pipeline is None:
            logger.error("Pipeline not created")
            return False
        
        try:
            # Start pipeline
            ret = self.pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                logger.error("Failed to start pipeline")
                return False
            
            # Wait for pipeline to stabilize (important for NVDEC)
            state_ret = self.pipeline.get_state(timeout=5 * Gst.SECOND)
            if state_ret[0] == Gst.StateChangeReturn.FAILURE:
                logger.error("Pipeline failed to reach PLAYING state")
                self.pipeline.set_state(Gst.State.NULL)
                return False
            
            self.is_running = True
            logger.info(f"✅ GStreamer NVDEC decoder started: {self.rtsp_url}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start decoder: {e}")
            try:
                if self.pipeline:
                    self.pipeline.set_state(Gst.State.NULL)
            except:
                pass
            return False
    
    def stop(self):
        """Stop decoding"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
        
        logger.info("GStreamer NVDEC decoder stopped")
    
    def read(self):
        """
        Read a frame (OpenCV VideoCapture compatible interface)
        
        Returns:
            (success, frame): Tuple of success flag and frame
        """
        if not self.is_running:
            return False, None
        
        with self.frame_lock:
            if self.latest_frame is not None:
                return True, self.latest_frame.copy()
        
        return False, None
    
    def get_frame(self, timeout=1.0):
        """
        Get frame from queue with timeout
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            frame or None
        """
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def isOpened(self):
        """Check if decoder is running (OpenCV compatible)"""
        return self.is_running
    
    def release(self):
        """Release decoder (OpenCV compatible)"""
        self.stop()
    
    def get_fps(self):
        """Get current FPS"""
        return self.current_fps
    
    def get_stats(self):
        """Get decoder statistics"""
        return {
            'is_running': self.is_running,
            'fps': self.current_fps,
            'queue_size': self.frame_queue.qsize(),
            'decoder_type': 'GStreamer NVDEC (Hardware)'
        }


def test_nvdec_availability():
    """
    Test if nvdec (nvv4l2decoder) is available in GStreamer
    
    Returns:
        bool: True if nvdec is available
    """
    try:
        registry = Gst.Registry.get()
        
        # Check for nvv4l2decoder (DeepStream NVDEC plugin)
        nvdec = registry.find_feature("nvv4l2decoder", Gst.ElementFactory)
        if nvdec:
            logger.info("✅ nvv4l2decoder plugin found")
        
        # Check for nvvideoconvert (DeepStream 6.x)
        nvvideoconvert = registry.find_feature("nvvideoconvert", Gst.ElementFactory)
        if nvvideoconvert:
            logger.info("✅ nvvideoconvert plugin found")
        
        if nvdec and nvvideoconvert:
            return True
        
        logger.warning(f"❌ NVDEC plugins not found (nvv4l2decoder: {bool(nvdec)}, nvvideoconvert: {bool(nvvideoconvert)})")
        return False
        
    except Exception as e:
        logger.error(f"Error checking NVDEC availability: {e}")
        return False


def create_nvdec_decoder(rtsp_url, target_size=(640, 640)):
    """
    Factory function to create NVDEC decoder with fallback
    
    Args:
        rtsp_url: RTSP stream URL
        target_size: Target frame size
        
    Returns:
        GStreamerNVDECDecoder or None
    """
    try:
        # Check if NVDEC is available
        if not test_nvdec_availability():
            logger.warning("NVDEC not available, decoder creation skipped")
            return None
        
        # Create decoder
        decoder = GStreamerNVDECDecoder(rtsp_url, target_size)
        if decoder.pipeline is None:
            logger.warning("Failed to create pipeline")
            return None
        
        # Test start (with timeout protection)
        if decoder.start():
            # Wait a moment to see if it works
            time.sleep(2)
            success, frame = decoder.read()
            if success and frame is not None:
                logger.info(f"✅ NVDEC decoder successfully created and tested")
                return decoder
            else:
                logger.warning("NVDEC decoder created but not producing frames yet")
                # Give it one more chance
                time.sleep(1)
                success, frame = decoder.read()
                if success and frame is not None:
                    logger.info(f"✅ NVDEC decoder working (after retry)")
                    return decoder
                else:
                    logger.warning("NVDEC decoder not working, stopping")
                    try:
                        decoder.stop()
                    except:
                        pass
                    return None
        else:
            logger.warning("Failed to start NVDEC decoder")
            return None
            
    except Exception as e:
        logger.error(f"Error creating NVDEC decoder: {e}", exc_info=True)
        return None


if __name__ == "__main__":
    # Test script
    logging.basicConfig(level=logging.INFO)
    
    print("Testing NVDEC availability...")
    if test_nvdec_availability():
        print("✅ NVDEC is available!")
    else:
        print("❌ NVDEC is NOT available")
