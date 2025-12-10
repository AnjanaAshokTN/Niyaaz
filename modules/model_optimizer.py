"""
Model Conversion and Optimization Pipeline
Converts PyTorch YOLO models to ONNX and TensorRT for maximum performance
"""
import torch
import onnx
import onnxruntime
import numpy as np
import logging
import os
from pathlib import Path
import time

logger = logging.getLogger(__name__)

class ModelConverter:
    """
    Converts and optimizes models for different inference backends
    PyTorch → ONNX → TensorRT pipeline
    """
    
    def __init__(self, model_path, output_dir="optimized_models"):
        """
        Initialize model converter
        
        Args:
            model_path: Path to original PyTorch model
            output_dir: Directory to save optimized models
        """
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Model info
        self.model_name = Path(model_path).stem
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Paths for converted models
        self.onnx_path = self.output_dir / f"{self.model_name}.onnx"
        self.tensorrt_path = self.output_dir / f"{self.model_name}.trt"
        
        logger.info(f"Model converter initialized for: {model_path}")
    
    def convert_to_onnx(self, input_size=(1, 3, 640, 640), opset_version=11):
        """
        Convert PyTorch model to ONNX format
        
        Args:
            input_size: Input tensor size (batch, channels, height, width)
            opset_version: ONNX opset version
            
        Returns:
            bool: Success status
        """
        try:
            logger.info(f"Converting {self.model_path} to ONNX...")
            
            # Load PyTorch model
            from ultralytics import YOLO
            model = YOLO(self.model_path)
            
            # Create dummy input
            dummy_input = torch.randn(*input_size).to(self.device)
            
            # Export to ONNX
            torch.onnx.export(
                model.model,  # PyTorch model
                dummy_input,  # Dummy input
                str(self.onnx_path),  # Output path
                export_params=True,  # Export parameters
                opset_version=opset_version,  # ONNX opset version
                do_constant_folding=True,  # Optimize constants
                input_names=['input'],  # Input tensor name
                output_names=['output'],  # Output tensor name
                dynamic_axes={
                    'input': {0: 'batch_size'},  # Dynamic batch size
                    'output': {0: 'batch_size'}
                }
            )
            
            # Verify ONNX model
            if self._verify_onnx_model():
                logger.info(f"ONNX conversion successful: {self.onnx_path}")
                return True
            else:
                logger.error("ONNX model verification failed")
                return False
                
        except Exception as e:
            logger.error(f"ONNX conversion failed: {e}")
            return False
    
    def _verify_onnx_model(self):
        """Verify ONNX model validity"""
        try:
            # Load and check ONNX model
            onnx_model = onnx.load(str(self.onnx_path))
            onnx.checker.check_model(onnx_model)
            
            # Test inference
            session = onnxruntime.InferenceSession(str(self.onnx_path))
            input_name = session.get_inputs()[0].name
            
            # Create test input
            test_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
            
            # Run inference
            outputs = session.run(None, {input_name: test_input})
            
            logger.info("ONNX model verification passed")
            return True
            
        except Exception as e:
            logger.error(f"ONNX verification failed: {e}")
            return False
    
    def convert_to_tensorrt(self, max_batch_size=8, fp16=True, workspace_size=1024):
        """
        Convert ONNX model to TensorRT for maximum performance
        
        Args:
            max_batch_size: Maximum batch size for optimization
            fp16: Use FP16 precision for faster inference
            workspace_size: TensorRT workspace size in MB
            
        Returns:
            bool: Success status
        """
        try:
            # Check if TensorRT is available
            try:
                import tensorrt as trt
            except ImportError:
                logger.error("TensorRT not installed. Install with: pip install tensorrt")
                return False
            
            if not self.onnx_path.exists():
                logger.error("ONNX model not found. Convert to ONNX first.")
                return False
            
            logger.info(f"Converting ONNX to TensorRT...")
            
            # Create TensorRT builder
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(TRT_LOGGER)
            
            # Create network and parser
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, TRT_LOGGER)
            
            # Parse ONNX model
            with open(self.onnx_path, 'rb') as model_file:
                if not parser.parse(model_file.read()):
                    logger.error("Failed to parse ONNX model")
                    return False
            
            # Create builder config
            config = builder.create_builder_config()
            config.max_workspace_size = workspace_size * 1024 * 1024  # Convert MB to bytes
            
            # Enable FP16 if requested and supported
            if fp16 and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                logger.info("FP16 optimization enabled")
            
            # Set optimization profiles for dynamic shapes
            profile = builder.create_optimization_profile()
            
            # Input shape ranges (min, opt, max)
            profile.set_shape("input", 
                            (1, 3, 320, 320),    # Min
                            (max_batch_size//2, 3, 640, 640),  # Optimal
                            (max_batch_size, 3, 640, 640))     # Max
            
            config.add_optimization_profile(profile)
            
            # Build TensorRT engine
            logger.info("Building TensorRT engine (this may take several minutes)...")
            engine = builder.build_engine(network, config)
            
            if engine is None:
                logger.error("Failed to build TensorRT engine")
                return False
            
            # Serialize and save engine
            with open(self.tensorrt_path, 'wb') as f:
                f.write(engine.serialize())
            
            logger.info(f"TensorRT conversion successful: {self.tensorrt_path}")
            return True
            
        except Exception as e:
            logger.error(f"TensorRT conversion failed: {e}")
            return False
    
    def benchmark_models(self, num_iterations=100):
        """
        Benchmark different model formats
        
        Args:
            num_iterations: Number of inference iterations
            
        Returns:
            dict: Benchmark results
        """
        results = {}
        test_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
        
        # Benchmark PyTorch
        if self._benchmark_pytorch(test_input, num_iterations):
            results['pytorch'] = self.pytorch_time
        
        # Benchmark ONNX
        if self.onnx_path.exists():
            if self._benchmark_onnx(test_input, num_iterations):
                results['onnx'] = self.onnx_time
        
        # Benchmark TensorRT
        if self.tensorrt_path.exists():
            if self._benchmark_tensorrt(test_input, num_iterations):
                results['tensorrt'] = self.tensorrt_time
        
        return results
    
    def _benchmark_pytorch(self, test_input, num_iterations):
        """Benchmark PyTorch model"""
        try:
            from ultralytics import YOLO
            model = YOLO(self.model_path)
            model.model.eval()
            
            # Warmup
            for _ in range(10):
                with torch.no_grad():
                    _ = model.model(torch.from_numpy(test_input).to(self.device))
            
            # Benchmark
            start_time = time.time()
            for _ in range(num_iterations):
                with torch.no_grad():
                    _ = model.model(torch.from_numpy(test_input).to(self.device))
            
            if self.device == 'cuda':
                torch.cuda.synchronize()
            
            total_time = time.time() - start_time
            self.pytorch_time = total_time / num_iterations * 1000  # ms per inference
            
            logger.info(f"PyTorch: {self.pytorch_time:.2f} ms per inference")
            return True
            
        except Exception as e:
            logger.error(f"PyTorch benchmark failed: {e}")
            return False
    
    def _benchmark_onnx(self, test_input, num_iterations):
        """Benchmark ONNX model"""
        try:
            session = onnxruntime.InferenceSession(str(self.onnx_path))
            input_name = session.get_inputs()[0].name
            
            # Warmup
            for _ in range(10):
                _ = session.run(None, {input_name: test_input})
            
            # Benchmark
            start_time = time.time()
            for _ in range(num_iterations):
                _ = session.run(None, {input_name: test_input})
            
            total_time = time.time() - start_time
            self.onnx_time = total_time / num_iterations * 1000  # ms per inference
            
            logger.info(f"ONNX: {self.onnx_time:.2f} ms per inference")
            return True
            
        except Exception as e:
            logger.error(f"ONNX benchmark failed: {e}")
            return False
    
    def _benchmark_tensorrt(self, test_input, num_iterations):
        """Benchmark TensorRT model"""
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
            
            # Load TensorRT engine
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            with open(self.tensorrt_path, 'rb') as f:
                engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(f.read())
            
            context = engine.create_execution_context()
            
            # Allocate buffers
            input_binding = engine.get_binding_index("input")
            output_binding = engine.get_binding_index("output")
            
            input_size = test_input.nbytes
            output_size = trt.volume(engine.get_binding_shape(output_binding)) * engine.max_batch_size * np.dtype(np.float32).itemsize
            
            # Allocate device memory
            d_input = cuda.mem_alloc(input_size)
            d_output = cuda.mem_alloc(output_size)
            
            bindings = [int(d_input), int(d_output)]
            stream = cuda.Stream()
            
            # Warmup
            for _ in range(10):
                cuda.memcpy_htod_async(d_input, test_input, stream)
                context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
                stream.synchronize()
            
            # Benchmark
            start_time = time.time()
            for _ in range(num_iterations):
                cuda.memcpy_htod_async(d_input, test_input, stream)
                context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
                stream.synchronize()
            
            total_time = time.time() - start_time
            self.tensorrt_time = total_time / num_iterations * 1000  # ms per inference
            
            logger.info(f"TensorRT: {self.tensorrt_time:.2f} ms per inference")
            return True
            
        except Exception as e:
            logger.error(f"TensorRT benchmark failed: {e}")
            return False
    
    def create_optimization_report(self):
        """Create detailed optimization report"""
        report = {
            'model_name': self.model_name,
            'original_path': str(self.model_path),
            'optimized_models': {},
            'benchmarks': {},
            'recommendations': []
        }
        
        # Check available optimized models
        if self.onnx_path.exists():
            report['optimized_models']['onnx'] = str(self.onnx_path)
        
        if self.tensorrt_path.exists():
            report['optimized_models']['tensorrt'] = str(self.tensorrt_path)
        
        # Run benchmarks
        report['benchmarks'] = self.benchmark_models()
        
        # Generate recommendations
        if 'tensorrt' in report['benchmarks']:
            speedup = report['benchmarks'].get('pytorch', 0) / report['benchmarks']['tensorrt']
            report['recommendations'].append(f"TensorRT provides {speedup:.1f}x speedup")
        
        if 'onnx' in report['benchmarks']:
            speedup = report['benchmarks'].get('pytorch', 0) / report['benchmarks']['onnx']
            report['recommendations'].append(f"ONNX provides {speedup:.1f}x speedup")
        
        return report


def convert_all_models(models_dir="models", output_dir="optimized_models"):
    """
    Convert all YOLO models in a directory
    
    Args:
        models_dir: Directory containing original models
        output_dir: Directory for optimized models
    """
    models_dir = Path(models_dir)
    results = {}
    
    for model_path in models_dir.glob("*.pt"):
        logger.info(f"Processing {model_path.name}...")
        
        converter = ModelConverter(str(model_path), output_dir)
        
        # Convert to ONNX
        onnx_success = converter.convert_to_onnx()
        
        # Convert to TensorRT if ONNX succeeded
        tensorrt_success = False
        if onnx_success:
            tensorrt_success = converter.convert_to_tensorrt()
        
        # Generate report
        report = converter.create_optimization_report()
        results[model_path.name] = report
        
        logger.info(f"Completed {model_path.name}: ONNX={onnx_success}, TensorRT={tensorrt_success}")
    
    return results


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Convert specific model
    converter = ModelConverter("models/yolo11n.pt")
    
    # Convert to ONNX
    if converter.convert_to_onnx():
        print("ONNX conversion successful")
    
    # Convert to TensorRT
    if converter.convert_to_tensorrt():
        print("TensorRT conversion successful")
    
    # Benchmark all formats
    results = converter.benchmark_models()
    print("Benchmark results:", results)