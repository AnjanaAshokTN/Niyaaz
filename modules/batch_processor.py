
"""
Batch Frame Processor for GPU Efficiency
Processes multiple channel frames together to reduce GPU calls
"""
import cv2
import numpy as np
import threading
import time
import queue
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class BatchFrameProcessor:
    """
    Processes frames from multiple channels in batches to optimize GPU usage
    """
    
    def __init__(self, batch_size=4, max_wait_time=0.1):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        
        # Thread-safe queues
        self.frame_queue = queue.Queue(maxsize=50)
        self.result_queue = queue.Queue(maxsize=50)
        
        # Processing control
        self.is_running = False
        self.processing_thread = None
        
        # Statistics
        self.frames_processed = 0
        self.batches_processed = 0
        
    def start(self):
        """Start batch processing"""
        if self.is_running:
            return
        
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._batch_processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        logger.info("🚀 Batch frame processor started")
    
    def stop(self):
        """Stop batch processing"""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        
        logger.info("⏹️ Batch frame processor stopped")
    
    def add_frame(self, channel_id, frame, callback):
        """Add frame for batch processing"""
        try:
            self.frame_queue.put({
                'channel_id': channel_id,
                'frame': frame,
                'callback': callback,
                'timestamp': time.time()
            }, timeout=0.01)
        except queue.Full:
            logger.warning("Frame queue full, dropping frame")
    
    def _batch_processing_loop(self):
        """Main batch processing loop"""
        from modules.optimized_model_manager import optimized_model_manager
        
        # Load optimized model
        model = optimized_model_manager.load_optimized_model('models/yolov8n.pt')
        if not model:
            logger.error("Failed to load model for batch processing")
            return
        
        frame_batch = []
        callback_batch = []
        
        while self.is_running:
            try:
                # Collect frames for batch
                start_time = time.time()
                
                while (len(frame_batch) < self.batch_size and 
                       time.time() - start_time < self.max_wait_time):
                    
                    try:
                        frame_data = self.frame_queue.get(timeout=0.01)
                        frame_batch.append(frame_data['frame'])
                        callback_batch.append({
                            'channel_id': frame_data['channel_id'],
                            'callback': frame_data['callback']
                        })
                    except queue.Empty:
                        break
                
                # Process batch if we have frames
                if frame_batch:
                    results = optimized_model_manager.process_batch(frame_batch, model)
                    
                    # Send results back to respective channels
                    for i, result in enumerate(results):
                        if i < len(callback_batch):
                            callback_info = callback_batch[i]
                            callback_info['callback'](result)
                    
                    self.frames_processed += len(frame_batch)
                    self.batches_processed += 1
                    
                    # Clear batch
                    frame_batch = []
                    callback_batch = []
                
                # Small delay to prevent busy waiting
                time.sleep(0.001)
                
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                frame_batch = []
                callback_batch = []

# Global batch processor
batch_processor = BatchFrameProcessor()
