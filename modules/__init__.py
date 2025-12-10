"""
Sakshi.AI Modules Package
Core video analytics modules for people counting and queue monitoring
"""

__version__ = "1.0.0"
__author__ = "ThinkNeural.AI"

# Import main modules for easy access
from .yolo_detector import YOLODetector, PersonTracker
from .people_counter import PeopleCounter  
from .queue_monitor import QueueMonitor
from .video_processor import VideoProcessor, StreamingVideoProcessor
from .database import DatabaseManager
from .mopping_detection import MoppingDetection
from .smoking_detection import SmokingDetection
from .phone_usage_detection import PhoneUsageDetection
from .restricted_area_monitor import RestrictedAreaMonitor

__all__ = [
    'YOLODetector',
    'PersonTracker', 
    'PeopleCounter',
    'QueueMonitor',
    'VideoProcessor',
    'StreamingVideoProcessor',
    'DatabaseManager',
    'MoppingDetection',
    'SmokingDetection',
    'PhoneUsageDetection',
    'RestrictedAreaMonitor'
]