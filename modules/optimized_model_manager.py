
"""
Optimized Model Manager with GPU Efficiency Improvements
"""
import torch
import logging
from modules.gpu_optimizer import optimize_gpu_settings

logger = logging.getLogger(__name__)

class OptimizedModelManager:
    """Enhanced model manager with GPU optimization"""
    
    def __init__(self):
        # Apply GPU optimizations
        optimize_gpu_settings()
        
        # Model loading optimizations
        self.model_cache = {}
        self.batch_size = 4  # Process multiple frames together
        self.use_half_precision = True  # Use FP16 for speed
        self.model_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def load_optimized_model(self, model_path):
        """Load model with optimizations"""
        if model_path in self.model_cache:
            return self.model_cache[model_path]
        
        try:
            from ultralytics import YOLO
            
            # Load model
            model = YOLO(model_path)
            
            # Apply optimizations
            if torch.cuda.is_available():
                model.to(self.model_device)
                
                # Enable half precision for 2x speed improvement
                if self.use_half_precision:
                    model.model.half()
                
                # Warm up model
                dummy_input = torch.randn(1, 3, 640, 640).to(self.model_device)
                if self.use_half_precision:
                    dummy_input = dummy_input.half()
                
                with torch.no_grad():
                    _ = model.model(dummy_input)
                
                logger.info(f"Model optimized and warmed up: {model_path}")
            
            self.model_cache[model_path] = model
            return model
            
        except Exception as e:
            logger.error(f"Failed to load optimized model: {e}")
            return None
    
    def process_batch(self, frames, model):
        """Process multiple frames in a batch for efficiency"""
        if not frames:
            return []
        
        try:
            # Batch processing reduces GPU overhead
            results = model.predict(frames, verbose=False, device=self.model_device)
            return results
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            return []

# Global optimized model manager
optimized_model_manager = OptimizedModelManager()
