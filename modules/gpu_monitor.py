
"""
GPU Performance Monitor and Auto-Adjustment System
"""
import time
import threading
import logging
import json
from datetime import datetime
import GPUtil
import psutil

logger = logging.getLogger(__name__)

class GPUMonitor:
    """
    Monitors GPU performance and automatically adjusts system parameters
    """
    
    def __init__(self):
        self.is_monitoring = False
        self.monitor_thread = None
        self.monitor_interval = 5.0  # Check every 5 seconds
        
        # Performance metrics
        self.gpu_history = []
        self.performance_stats = {
            'avg_gpu_usage': 0,
            'peak_gpu_usage': 0,
            'total_adjustments': 0,
            'start_time': time.time()
        }
        
        # Alert thresholds
        self.high_usage_threshold = 95
        self.sustained_high_usage_count = 0
        self.max_sustained_count = 6  # 30 seconds of high usage
    
    def start_monitoring(self):
        """Start GPU monitoring"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        logger.info("🔍 GPU monitoring started")
    
    def stop_monitoring(self):
        """Stop GPU monitoring"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        logger.info("⏹️ GPU monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Get current GPU usage
                gpu_usage = self._get_gpu_usage()
                cpu_usage = psutil.cpu_percent()
                memory_usage = psutil.virtual_memory().percent
                
                # Record metrics
                self.gpu_history.append({
                    'timestamp': time.time(),
                    'gpu_usage': gpu_usage,
                    'cpu_usage': cpu_usage,
                    'memory_usage': memory_usage
                })
                
                # Keep only last 100 readings
                if len(self.gpu_history) > 100:
                    self.gpu_history.pop(0)
                
                # Update stats
                self.performance_stats['avg_gpu_usage'] = sum([h['gpu_usage'] for h in self.gpu_history]) / len(self.gpu_history)
                self.performance_stats['peak_gpu_usage'] = max([h['gpu_usage'] for h in self.gpu_history])
                
                # Check for sustained high usage
                if gpu_usage > self.high_usage_threshold:
                    self.sustained_high_usage_count += 1
                else:
                    self.sustained_high_usage_count = 0
                
                # Auto-adjust if needed
                if self.sustained_high_usage_count >= self.max_sustained_count:
                    self._emergency_adjustment()
                    self.sustained_high_usage_count = 0
                
                # Log periodic status
                if len(self.gpu_history) % 12 == 0:  # Every minute
                    logger.info(f"📊 GPU: {gpu_usage:.1f}%, CPU: {cpu_usage:.1f}%, RAM: {memory_usage:.1f}%")
                
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(self.monitor_interval)
    
    def _get_gpu_usage(self) -> float:
        """Get current GPU usage percentage"""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                return gpus[0].load * 100
            return 0
        except Exception:
            return 0
    
    def _emergency_adjustment(self):
        """Apply emergency performance adjustments"""
        logger.warning("🚨 Applying emergency GPU optimization...")
        
        # Force dynamic scaler to emergency mode
        from modules.dynamic_scaler import dynamic_scaler
        dynamic_scaler.current_preset = 'emergency_mode'
        
        # Additional emergency measures could be added here
        self.performance_stats['total_adjustments'] += 1
        
        logger.info("✅ Emergency optimization applied")
    
    def get_performance_report(self) -> dict:
        """Get comprehensive performance report"""
        runtime = time.time() - self.performance_stats['start_time']
        
        return {
            'runtime_minutes': runtime / 60,
            'avg_gpu_usage': self.performance_stats['avg_gpu_usage'],
            'peak_gpu_usage': self.performance_stats['peak_gpu_usage'],
            'total_adjustments': self.performance_stats['total_adjustments'],
            'current_gpu_usage': self._get_gpu_usage(),
            'recent_history': self.gpu_history[-10:] if self.gpu_history else []
        }

# Global GPU monitor
gpu_monitor = GPUMonitor()
