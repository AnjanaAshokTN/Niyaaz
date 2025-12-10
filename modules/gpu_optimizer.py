
# GPU Memory and Performance Optimizations
import torch
import os

# Set PyTorch optimizations
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,roundup_power2_divisions:16'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Enable asynchronous CUDA operations
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

# Memory management
torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
torch.backends.cuda.matmul.allow_tf32 = True  # Use TF32 for faster computation
torch.backends.cudnn.allow_tf32 = True

# Mixed precision settings
if torch.cuda.is_available():
    # Enable memory efficient attention
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

def optimize_gpu_settings():
    """Apply GPU optimization settings"""
    if torch.cuda.is_available():
        # Set memory fraction to prevent full GPU usage
        torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% max GPU memory
        
        # Clear cache
        torch.cuda.empty_cache()
        
        # Enable cuda graphs for repetitive operations
        torch.cuda.set_device(0)
        
        print("🔧 GPU optimizations applied")
    else:
        print("⚠️ CUDA not available, using CPU optimizations")

# Call optimization function
optimize_gpu_settings()
