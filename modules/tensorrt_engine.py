"""
TensorRT Inference Engine
High-performance inference using optimized TensorRT models
"""
import numpy as np
import cv2
import time
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class TensorRTInferenceEngine:
    """
    High-performance inference engine using TensorRT optimized models
    Replaces standard YOLO detector for maximum GPU utilization
    """
    
    def __init__(self, model_path, confidence_threshold=0.5, nms_threshold=0.4):
        """
        Initialize TensorRT inference engine
        
        Args:
            model_path: Path to TensorRT engine file (.trt)
            confidence_threshold: Minimum confidence for detections
            nms_threshold: NMS threshold for duplicate removal
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        
        # TensorRT components
        self.engine = None
        self.context = None
        self.stream = None
        self.bindings = []
        self.input_shape = None
        self.output_shape = None
        
        # Device memory
        self.d_input = None
        self.d_output = None
        self.h_output = None
        
        # Performance tracking
        self.inference_times = []
        self.avg_inference_time = 0
        
        # Initialize engine
        self._load_engine()
        self._setup_bindings()
    
    def _load_engine(self):
        """Load TensorRT engine from file"""
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
            
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            
            # Load serialized engine
            with open(self.model_path, 'rb') as f:
                engine_data = f.read()
            
            runtime = trt.Runtime(TRT_LOGGER)
            self.engine = runtime.deserialize_cuda_engine(engine_data)
            
            if self.engine is None:
                raise RuntimeError("Failed to load TensorRT engine")
            
            # Create execution context
            self.context = self.engine.create_execution_context()
            
            # Create CUDA stream
            self.stream = cuda.Stream()
            
            logger.info(f"TensorRT engine loaded successfully: {self.model_path}")
            
        except ImportError:
            raise ImportError("TensorRT and PyCUDA required. Install with: pip install tensorrt pycuda")
        except Exception as e:
            raise RuntimeError(f"Failed to load TensorRT engine: {e}")
    
    def _setup_bindings(self):
        """Setup input/output bindings and allocate GPU memory"""
        try:
            import pycuda.driver as cuda
            
            # Get binding information
            input_binding = self.engine.get_binding_index("input")
            output_binding = self.engine.get_binding_index("output")
            
            # Get shapes
            self.input_shape = self.engine.get_binding_shape(input_binding)
            self.output_shape = self.engine.get_binding_shape(output_binding)
            
            logger.info(f"Input shape: {self.input_shape}")
            logger.info(f"Output shape: {self.output_shape}")
            
            # Calculate sizes
            input_size = np.prod(self.input_shape) * np.dtype(np.float32).itemsize
            output_size = np.prod(self.output_shape) * np.dtype(np.float32).itemsize
            
            # Allocate device memory
            self.d_input = cuda.mem_alloc(input_size)
            self.d_output = cuda.mem_alloc(output_size)
            
            # Create output host buffer
            self.h_output = np.empty(self.output_shape, dtype=np.float32)
            
            # Setup bindings
            self.bindings = [int(self.d_input), int(self.d_output)]
            
            logger.info("Memory bindings setup complete")
            
        except Exception as e:
            raise RuntimeError(f"Failed to setup bindings: {e}")
    
    def preprocess_frame(self, frame):
        """
        Preprocess frame for TensorRT inference
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Preprocessed tensor ready for inference
        """
        try:
            # Get target size from input shape (assuming NCHW format)
            target_h, target_w = self.input_shape[2], self.input_shape[3]
            
            # Resize frame
            resized = cv2.resize(frame, (target_w, target_h))
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            # Normalize to [0, 1]
            normalized = rgb_frame.astype(np.float32) / 255.0
            
            # Transpose HWC to CHW
            chw_frame = np.transpose(normalized, (2, 0, 1))
            
            # Add batch dimension
            batch_frame = np.expand_dims(chw_frame, axis=0)
            
            # Ensure contiguous memory layout
            return np.ascontiguousarray(batch_frame, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Frame preprocessing failed: {e}")
            return None
    
    def inference(self, preprocessed_frame):
        """
        Run TensorRT inference
        
        Args:
            preprocessed_frame: Preprocessed input tensor
            
        Returns:
            Raw model output
        """
        try:
            import pycuda.driver as cuda
            
            start_time = time.time()
            
            # Copy input to device
            cuda.memcpy_htod_async(self.d_input, preprocessed_frame, self.stream)
            
            # Set input shape for dynamic shapes
            if self.engine.has_implicit_batch_dimension:
                self.context.execute_async(bindings=self.bindings, stream_handle=self.stream.handle)
            else:
                self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            
            # Copy output from device
            cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
            
            # Synchronize stream
            self.stream.synchronize()
            
            # Track inference time
            inference_time = (time.time() - start_time) * 1000  # Convert to ms
            self.inference_times.append(inference_time)
            
            # Keep only last 100 measurements
            if len(self.inference_times) > 100:
                self.inference_times.pop(0)
            
            self.avg_inference_time = np.mean(self.inference_times)
            
            return self.h_output.copy()
            
        except Exception as e:
            logger.error(f"TensorRT inference failed: {e}")
            return None
    
    def postprocess_output(self, raw_output, original_shape):
        """
        Post-process TensorRT output to get detections
        
        Args:
            raw_output: Raw model output
            original_shape: Original frame shape (H, W, C)
            
        Returns:
            List of detection dictionaries
        """
        try:
            detections = []
            
            # Parse output (assuming YOLO format)
            # Output shape is typically (batch, num_detections, 85) for COCO
            # First 4 values are bbox (x, y, w, h), 5th is confidence, rest are class scores
            
            batch_output = raw_output[0]  # Remove batch dimension
            
            for detection in batch_output:
                # Extract bbox and confidence
                x, y, w, h = detection[:4]
                confidence = detection[4]
                
                # Skip low confidence detections
                if confidence < self.confidence_threshold:
                    continue
                
                # Get class scores and find best class
                class_scores = detection[5:]
                class_id = np.argmax(class_scores)
                class_confidence = class_scores[class_id]
                
                # Final confidence is object confidence * class confidence
                final_confidence = confidence * class_confidence
                
                if final_confidence < self.confidence_threshold:
                    continue
                
                # Convert to pixel coordinates
                orig_h, orig_w = original_shape[:2]
                input_h, input_w = self.input_shape[2], self.input_shape[3]
                
                # Scale factors
                scale_x = orig_w / input_w
                scale_y = orig_h / input_h
                
                # Convert center coordinates to corner coordinates
                x1 = int((x - w/2) * scale_x)
                y1 = int((y - h/2) * scale_y)
                x2 = int((x + w/2) * scale_x)
                y2 = int((y + h/2) * scale_y)
                
                # Clamp to image boundaries
                x1 = max(0, min(x1, orig_w))
                y1 = max(0, min(y1, orig_h))
                x2 = max(0, min(x2, orig_w))
                y2 = max(0, min(y2, orig_h))
                
                # Create detection dictionary
                detection_dict = {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(final_confidence),
                    'class_id': int(class_id),
                    'class_name': 'person' if class_id == 0 else f'class_{class_id}',
                    'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                    'bottom_center': ((x1 + x2) // 2, y2),
                    'area': (x2 - x1) * (y2 - y1)
                }
                
                detections.append(detection_dict)
            
            # Apply Non-Maximum Suppression
            if len(detections) > 1:
                detections = self._apply_nms(detections)
            
            return detections
            
        except Exception as e:
            logger.error(f"Post-processing failed: {e}")
            return []
    
    def _apply_nms(self, detections):
        """Apply Non-Maximum Suppression to remove duplicate detections"""
        try:
            if len(detections) <= 1:
                return detections
            
            # Extract bounding boxes and scores
            boxes = np.array([det['bbox'] for det in detections])
            scores = np.array([det['confidence'] for det in detections])
            
            # Apply OpenCV NMS
            indices = cv2.dnn.NMSBoxes(
                boxes.tolist(),
                scores.tolist(),
                self.confidence_threshold,
                self.nms_threshold
            )
            
            # Return filtered detections
            if len(indices) > 0:
                return [detections[i] for i in indices.flatten()]
            else:
                return []
                
        except Exception as e:
            logger.warning(f"NMS failed: {e}")
            return detections
    
    def detect_persons(self, frame):
        """
        Main detection method - compatible with existing YOLO detector interface
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            List of person detections
        """
        try:
            # Preprocess frame
            preprocessed = self.preprocess_frame(frame)
            if preprocessed is None:
                return []
            
            # Run inference
            raw_output = self.inference(preprocessed)
            if raw_output is None:
                return []
            
            # Post-process to get detections
            all_detections = self.postprocess_output(raw_output, frame.shape)
            
            # Filter for persons only (class_id = 0 for COCO)
            person_detections = [det for det in all_detections if det['class_id'] == 0]
            
            return person_detections
            
        except Exception as e:
            logger.error(f"Person detection failed: {e}")
            return []
    
    def get_performance_stats(self):
        """Get performance statistics"""
        return {
            'avg_inference_time_ms': self.avg_inference_time,
            'total_inferences': len(self.inference_times),
            'model_path': self.model_path,
            'input_shape': self.input_shape,
            'output_shape': self.output_shape
        }
    
    def cleanup(self):
        """Clean up GPU resources"""
        try:
            import pycuda.driver as cuda
            
            if self.d_input:
                self.d_input.free()
            if self.d_output:
                self.d_output.free()
            
            logger.info("TensorRT resources cleaned up")
            
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")


class ONNXInferenceEngine:
    """
    ONNX Runtime inference engine as fallback option
    Better performance than PyTorch, but not as fast as TensorRT
    """
    
    def __init__(self, model_path, confidence_threshold=0.5):
        """Initialize ONNX inference engine"""
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        
        try:
            import onnxruntime
            
            # Create ONNX Runtime session
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.session = onnxruntime.InferenceSession(model_path, providers=providers)
            
            # Get input/output info
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            logger.info(f"ONNX engine loaded: {model_path}")
            logger.info(f"Using providers: {self.session.get_providers()}")
            
        except ImportError:
            raise ImportError("ONNX Runtime required. Install with: pip install onnxruntime-gpu")
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model: {e}")
    
    def detect_persons(self, frame):
        """Person detection using ONNX Runtime"""
        try:
            # Preprocess (similar to TensorRT)
            target_h, target_w = 640, 640  # Standard YOLO input size
            resized = cv2.resize(frame, (target_w, target_h))
            rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            normalized = rgb_frame.astype(np.float32) / 255.0
            chw_frame = np.transpose(normalized, (2, 0, 1))
            batch_frame = np.expand_dims(chw_frame, axis=0)
            
            # Run inference
            outputs = self.session.run(self.output_names, {self.input_name: batch_frame})
            
            # Post-process (simplified version)
            detections = []
            raw_output = outputs[0][0]  # Remove batch dimension
            
            for detection in raw_output:
                confidence = detection[4]
                if confidence > self.confidence_threshold:
                    x, y, w, h = detection[:4]
                    class_scores = detection[5:]
                    class_id = np.argmax(class_scores)
                    
                    if class_id == 0:  # Person class
                        # Convert to pixel coordinates
                        orig_h, orig_w = frame.shape[:2]
                        scale_x = orig_w / target_w
                        scale_y = orig_h / target_h
                        
                        x1 = int((x - w/2) * scale_x)
                        y1 = int((y - h/2) * scale_y)
                        x2 = int((x + w/2) * scale_x)
                        y2 = int((y + h/2) * scale_y)
                        
                        detection_dict = {
                            'bbox': [x1, y1, x2, y2],
                            'confidence': float(confidence),
                            'class_id': 0,
                            'class_name': 'person',
                            'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                            'bottom_center': ((x1 + x2) // 2, y2),
                            'area': (x2 - x1) * (y2 - y1)
                        }
                        detections.append(detection_dict)
            
            return detections
            
        except Exception as e:
            logger.error(f"ONNX detection failed: {e}")
            return []


def create_optimized_detector(model_path, use_tensorrt=True, confidence_threshold=0.5):
    """
    Factory function to create optimized detector
    
    Args:
        model_path: Path to model file (.trt, .onnx, or .pt)
        use_tensorrt: Prefer TensorRT if available
        confidence_threshold: Detection confidence threshold
        
    Returns:
        Optimized detector instance
    """
    model_path = Path(model_path)
    
    # Try TensorRT first if requested and available
    if use_tensorrt:
        trt_path = model_path.with_suffix('.trt')
        if trt_path.exists():
            try:
                return TensorRTInferenceEngine(str(trt_path), confidence_threshold)
            except Exception as e:
                logger.warning(f"TensorRT failed, trying ONNX: {e}")
    
    # Try ONNX as fallback
    onnx_path = model_path.with_suffix('.onnx')
    if onnx_path.exists():
        try:
            return ONNXInferenceEngine(str(onnx_path), confidence_threshold)
        except Exception as e:
            logger.warning(f"ONNX failed: {e}")
    
    # Return None if no optimized version available
    logger.warning("No optimized model found, use PyTorch fallback")
    return None