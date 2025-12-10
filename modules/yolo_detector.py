"""
YOLO Detection Service for Person Detection
Uses YOLOv8 for real-time person detection
"""
import cv2
import numpy as np
import logging
import os
import sys

# Configure PyTorch environment before any imports
os.environ['PYTORCH_WEIGHTS_ONLY'] = 'False'
os.environ['TORCH_DISABLE_WEIGHTS_ONLY'] = '1'

# Set up PyTorch to accept YOLO models
import torch

# Disable weights_only globally by patching the default
if hasattr(torch.serialization, '_get_default_pickle_version'):
    original_get_default = torch.serialization._get_default_pickle_version
    def patched_get_default():
        return 2
    torch.serialization._get_default_pickle_version = patched_get_default

# Monkey patch torch.load to always set weights_only=False
original_load = torch.load
def safe_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = safe_load

# Import ultralytics after patching
try:
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel
    # Add DetectionModel to safe globals for newer PyTorch versions
    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals([DetectionModel])
except ImportError as e:
    print(f"Failed to import ultralytics: {e}")
    YOLO = None

logger = logging.getLogger(__name__)

class YOLODetector:
    def __init__(self, model_path='models/best.pt', confidence_threshold=0.5, device='auto', img_size=640, person_class_id=None):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to YOLO model file (TensorRT .engine or PyTorch .pt)
            confidence_threshold: Minimum confidence for detections
            device: Device to run inference on ('auto', 'cpu', 'cuda')
            img_size: Input image size (must match TensorRT engine build size - default 640)
            person_class_id: Class ID for person detection (0 for COCO/YOLOv11, 12 for custom best.pt)
                            If None, auto-detects based on model path
        """
        if YOLO is None:
            raise ImportError("Ultralytics YOLO could not be imported")
            
        self.confidence_threshold = confidence_threshold
        self.device = self._select_device(device)
        self.img_size = img_size  # Must match TensorRT engine build size (640)
        self.model_path = model_path
        
        # Log which model is being loaded
        logger.info(f"Initializing YOLODetector with model: {model_path}")
        
        # Check if model is TensorRT engine or PyTorch model
        self.is_engine = model_path.endswith('.engine') or model_path.endswith('.trt')
        
        try:
            # TensorRT engines need special handling
            if self.is_engine:
                # Load TensorRT engine - specify task explicitly
                self.model = YOLO(model_path, task='detect')
                logger.info(f"TensorRT engine loaded successfully: {model_path}")
            else:
                # PyTorch model - use normal loading with safety patches
                with torch.serialization.safe_globals([
                    'ultralytics.nn.tasks.DetectionModel',
                    'ultralytics.nn.modules.conv.Conv',
                    'ultralytics.nn.modules.block.C2f',
                    'ultralytics.nn.modules.block.SPPF',
                    'ultralytics.nn.modules.head.Detect'
                ]):
                    self.model = YOLO(model_path)
                self.model.to(self.device)
                
                # Enable PyTorch optimizations for faster inference
                if self.device == 'cuda':
                    # Enable CUDA optimizations (without FP16 to avoid dtype errors)
                    torch.backends.cudnn.benchmark = True
                    logger.info(f"YOLO model loaded with CUDA optimizations on {self.device}")
                else:
                    logger.info(f"YOLO model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Safe context loading failed: {e}")
            try:
                if self.is_engine:
                    # Retry engine with explicit parameters
                    self.model = YOLO(model_path)
                    logger.info(f"TensorRT engine loaded (retry): {model_path}")
                else:
                    # Force disable weights_only for this specific load
                    original_env = os.environ.get('PYTORCH_DISABLE_WEIGHTS_ONLY', '0')
                    os.environ['PYTORCH_DISABLE_WEIGHTS_ONLY'] = '1'
                    
                    # Temporary patch for this specific model load
                    def force_unsafe_load(*args, **kwargs):
                        kwargs.pop('weights_only', None)  # Remove if present
                        return original_load(*args, weights_only=False, **kwargs)
                    
                    torch.load = force_unsafe_load
                    self.model = YOLO(model_path)
                    torch.load = safe_load  # Restore our patched version
                    
                    self.model.to(self.device)
                    logger.info(f"YOLO model loaded with forced unsafe loading on {self.device}")
                    
                    # Restore environment
                    os.environ['PYTORCH_DISABLE_WEIGHTS_ONLY'] = original_env
                
            except Exception as e2:
                logger.error(f"All YOLO loading attempts failed: {e2}")
                raise Exception(f"Could not load YOLO model. Error: {e2}")
        
        # Person class ID: 
        # - 0 for COCO dataset (YOLOv11, YOLOv8, etc.)
        # - 12 for custom best.pt model
        if person_class_id is not None:
            self.person_class_id = person_class_id
        elif 'yolov11' in model_path.lower() or 'yolov8' in model_path.lower() or 'yolov5' in model_path.lower():
            # Standard COCO models use class 0 for person
            self.person_class_id = 0
            logger.info(f"Auto-detected person_class_id=0 for COCO model: {model_path}")
        else:
            # Default to 12 for custom best.pt model
            self.person_class_id = 12
            logger.info(f"Using default person_class_id=12 for custom model: {model_path}")
    
    def _select_device(self, device):
        """Select the best available device"""
        if device == 'auto':
            if torch.cuda.is_available():
                return 'cuda'
            else:
                return 'cpu'
        return device
    
    def detect_persons(self, frame):
        """
        Detect persons in frame with optimized settings
        
        Args:
            frame: Input image frame
            
        Returns:
            List of detection dictionaries with bbox, confidence, and class info
        """
        try:
            # Check if frame is valid
            if frame is None or frame.size == 0:
                logger.warning("detect_persons: Frame is None or empty")
                return []
            
            # **DEBUG: First try without class filtering to see all detections**
            # This helps diagnose if the model is detecting anything at all
            results_all = self.model(
                frame, 
                verbose=False,
                imgsz=self.img_size,
                conf=0.25,  # Lower threshold to see more detections
                iou=0.45,
                max_det=50,
                device=self.device
            )
            
            # Log all detections for debugging
            all_detections_count = 0
            all_classes_found = set()
            for result in results_all:
                boxes = result.boxes
                if boxes is not None:
                    cls = boxes.cls.cpu().numpy()
                    conf = boxes.conf.cpu().numpy()
                    all_detections_count += len(boxes)
                    for class_id, confidence in zip(cls, conf):
                        all_classes_found.add((int(class_id), float(confidence)))
            
            # Now run with person class filtering
            # Use the confidence threshold directly (no reduction) for yolov8n.pt
            # YOLOv8 models work well with standard thresholds
            results = self.model(
                frame, 
                verbose=False,
                imgsz=self.img_size,      # Configurable image size
                conf=self.confidence_threshold,  # Use threshold directly
                iou=0.45,                  # NMS IoU threshold (default 0.45)
                classes=[self.person_class_id],   # Only detect persons
                max_det=50,                # Reduced for performance
                device=self.device         # Explicit device
            )
            
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    # Extract detection data
                    xyxy = boxes.xyxy.cpu().numpy()  # Bounding boxes
                    conf = boxes.conf.cpu().numpy()  # Confidence scores
                    cls = boxes.cls.cpu().numpy()   # Class IDs
                    
                    # Filter for person detections with sufficient confidence
                    for i, (box, confidence, class_id) in enumerate(zip(xyxy, conf, cls)):
                        class_id_int = int(class_id)
                        if class_id_int == self.person_class_id and confidence >= self.confidence_threshold:
                            x1, y1, x2, y2 = map(int, box)
                            
                            # Calculate center point and bottom center (for line crossing)
                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2
                            bottom_center_x = center_x
                            bottom_center_y = y2
                            
                            detection = {
                                'bbox': [x1, y1, x2, y2],
                                'confidence': float(confidence),
                                'class_id': class_id_int,
                                'class_name': 'person',
                                'center': (center_x, center_y),
                                'bottom_center': (bottom_center_x, bottom_center_y),
                                'area': (x2 - x1) * (y2 - y1)
                            }
                            
                            detections.append(detection)
            
            # **DEBUG LOGGING** - Log every 30 calls to avoid spam, but always log first 10 calls
            if not hasattr(self, '_detect_call_count'):
                self._detect_call_count = 0
            self._detect_call_count += 1
            
            should_log = (self._detect_call_count <= 10) or (self._detect_call_count % 30 == 0)
            
            if should_log:
                logger.info(f"🔍 YOLO Detection Debug (call #{self._detect_call_count}):")
                logger.info(f"   - Model: {self.model_path}")
                logger.info(f"   - Person class ID expected: {self.person_class_id}")
                logger.info(f"   - Confidence threshold: {self.confidence_threshold}")
                logger.info(f"   - All detections (conf>=0.25): {all_detections_count}")
                if all_classes_found:
                    logger.info(f"   - All classes found: {sorted(all_classes_found)}")
                else:
                    logger.warning(f"   - ⚠️ No detections found at all (conf>=0.25)!")
                logger.info(f"   - Person detections (class={self.person_class_id}, conf>={self.confidence_threshold}): {len(detections)}")
                if len(detections) > 0:
                    logger.info(f"   - Sample person detection: bbox={detections[0]['bbox']}, conf={detections[0]['confidence']:.2f}, center={detections[0]['center']}")
                elif all_detections_count > 0:
                    logger.warning(f"   - ⚠️ Found {all_detections_count} detections but NONE are person class {self.person_class_id}!")
                else:
                    logger.warning(f"   - ⚠️ No detections found - model may not be detecting anything in this frame")
            
            return detections
            
        except Exception as e:
            logger.error(f"Error during person detection: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def detect_all_classes(self, frame, target_classes=None):
        """
        Detect all classes (or specific classes) in the frame
        Used by modules that need to detect uniforms and other classes beyond just 'person'
        
        Args:
            frame: Input frame (numpy array)
            target_classes: Optional list of class names to filter (e.g., ['person', 'uniform_beige'])
                          If None, returns all detected classes
        
        Returns:
            List of detection dictionaries with keys: bbox, confidence, class_id, class_name
        """
        if frame is None or frame.size == 0:
            logger.warning("detect_all_classes: Frame is None or empty")
            return []
        
        try:
            # Run YOLO detection without class filtering
            results = self.model.predict(
                frame,
                verbose=False,             # Silent mode
                imgsz=self.img_size,      # Configurable image size
                conf=self.confidence_threshold,  # Filter at inference time
                iou=0.45,                  # NMS IoU threshold
                max_det=50,                # Max detections
                device=self.device         # Explicit device
                # No classes filter - detect all classes
            )
            
            detections = []
            total_before_filter = 0
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    total_before_filter = len(boxes)
                    # Extract detection data
                    xyxy = boxes.xyxy.cpu().numpy()  # Bounding boxes
                    conf = boxes.conf.cpu().numpy()  # Confidence scores
                    cls = boxes.cls.cpu().numpy()   # Class IDs
                    
                    # Get class names from model
                    names = result.names  # Dict mapping class_id to class_name
                    
                    # Process each detection
                    for i, (box, confidence, class_id) in enumerate(zip(xyxy, conf, cls)):
                        if confidence >= self.confidence_threshold:
                            class_id_int = int(class_id)
                            class_name = names.get(class_id_int, f'class_{class_id_int}')
                            
                            # Filter by target classes if specified
                            if target_classes and class_name not in target_classes:
                                # Log what class was detected but filtered out
                                if i == 0:  # Only log first filtered detection to avoid spam
                                    logger.info(f"🔍 Detected '{class_name}' (conf={confidence:.2f}) but not in target classes")
                                continue
                            
                            x1, y1, x2, y2 = map(int, box)
                            
                            # Calculate center point
                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2
                            bottom_center_x = center_x
                            bottom_center_y = y2
                            
                            detection = {
                                'bbox': [x1, y1, x2, y2],
                                'confidence': float(confidence),
                                'class_id': class_id_int,
                                'class_name': class_name,
                                'center': (center_x, center_y),
                                'bottom_center': (bottom_center_x, bottom_center_y),
                                'area': (x2 - x1) * (y2 - y1)
                            }
                            
                            detections.append(detection)
            
            # Debug logging
            if total_before_filter > 0 and len(detections) == 0 and target_classes:
                logger.warning(f"🔍 detect_all_classes: {total_before_filter} raw detections, but 0 after filtering for classes: {target_classes}")
            
            return detections
            
        except Exception as e:
            logger.error(f"Error during all-class detection: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def draw_detections(self, frame, detections, draw_confidence=True, color=(0, 255, 0)):
        """
        Draw detection boxes on frame
        
        Args:
            frame: Input frame
            detections: List of detection dictionaries
            draw_confidence: Whether to draw confidence scores
            color: Color for bounding boxes (BGR)
            
        Returns:
            Frame with drawn detections
        """
        annotated_frame = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw confidence score
            if draw_confidence:
                label = f"Person {confidence:.2f}"
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated_frame, (x1, y1 - text_height - 5), (x1 + text_width, y1), color, -1)
                cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Draw center point
            center_x, center_y = detection['center']
            cv2.circle(annotated_frame, (center_x, center_y), 3, (0, 0, 255), -1)
        
        return annotated_frame
    
    def get_model_info(self):
        """Get model information"""
        return {
            'model_name': self.model.model_name if hasattr(self.model, 'model_name') else 'YOLOv8',
            'device': self.device,
            'confidence_threshold': self.confidence_threshold,
            'input_size': self.img_size,
            'optimizations': {
                'half_precision': torch.cuda.is_available(),
                'max_detections': 50,
                'person_only': True
            }
        }

class PersonTracker:
    """Simple person tracker for counting applications"""
    
    def __init__(self, max_disappeared=30, max_distance=100):
        """
        Initialize person tracker
        
        Args:
            max_disappeared: Maximum frames a person can be missing before removal
            max_distance: Maximum distance for person association
        """
        self.next_id = 0
        self.objects = {}  # {id: {'center': (x, y), 'disappeared': count}}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
    
    def register_person(self, center):
        """Register a new person"""
        self.objects[self.next_id] = {
            'center': center,
            'disappeared': 0
        }
        self.next_id += 1
        return self.next_id - 1
    
    def deregister_person(self, person_id):
        """Remove a person from tracking"""
        if person_id in self.objects:
            del self.objects[person_id]
    
    def update(self, detections):
        """
        Update tracker with new detections
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            Dictionary of tracked objects {id: center}
        """
        if len(detections) == 0:
            # Mark all existing objects as disappeared
            for person_id in list(self.objects.keys()):
                self.objects[person_id]['disappeared'] += 1
                
                # Remove if disappeared for too long
                if self.objects[person_id]['disappeared'] > self.max_disappeared:
                    self.deregister_person(person_id)
            
            return {}
        
        # Extract centers from detections
        input_centers = [det['center'] for det in detections]
        
        # If no existing objects, register all as new
        if len(self.objects) == 0:
            for center in input_centers:
                self.register_person(center)
        else:
            # Match existing objects to new detections
            object_ids = list(self.objects.keys())
            object_centers = [self.objects[obj_id]['center'] for obj_id in object_ids]
            
            # Compute distance matrix
            D = np.linalg.norm(np.array(object_centers)[:, np.newaxis] - np.array(input_centers), axis=2)
            
            # Find the minimum values and sort by distance
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_row_idxs = set()
            used_col_idxs = set()
            
            # Update existing objects
            for (row, col) in zip(rows, cols):
                if row in used_row_idxs or col in used_col_idxs:
                    continue
                
                if D[row, col] <= self.max_distance:
                    object_id = object_ids[row]
                    self.objects[object_id]['center'] = input_centers[col]
                    self.objects[object_id]['disappeared'] = 0
                    
                    used_row_idxs.add(row)
                    used_col_idxs.add(col)
            
            # Handle unmatched detections and objects
            unused_row_idxs = set(range(0, D.shape[0])).difference(used_row_idxs)
            unused_col_idxs = set(range(0, D.shape[1])).difference(used_col_idxs)
            
            # Mark unmatched objects as disappeared
            if D.shape[0] >= D.shape[1]:
                for row in unused_row_idxs:
                    object_id = object_ids[row]
                    self.objects[object_id]['disappeared'] += 1
                    
                    if self.objects[object_id]['disappeared'] > self.max_disappeared:
                        self.deregister_person(object_id)
            
            # Register new objects
            else:
                for col in unused_col_idxs:
                    self.register_person(input_centers[col])
        
        # Return current tracked objects
        return {obj_id: obj_data['center'] for obj_id, obj_data in self.objects.items()}