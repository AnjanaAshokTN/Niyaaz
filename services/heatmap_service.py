#!/usr/bin/env python3
"""
Heatmap Processor Microservice
Generates customer engagement heatmaps
"""

import os
import sys
import time
import logging
import hashlib

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processors.heatmap_processor import HeatmapProcessor
import torch
from ultralytics import YOLO

# Configuration
CONFIG_FOLDER = '/app/config'
MODELS_FOLDER = '/app/models'
RTSP_LINKS_FILE = os.path.join(CONFIG_FOLDER, 'rtsp_links.txt')
MODEL_PATH = os.path.join(MODELS_FOLDER, 'yolov8n.pt')

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [HEATMAP] - %(levelname)s - %(message)s'
)

def get_stable_channel_id(link):
    """Generate stable channel ID from RTSP link"""
    return f"cam_{hashlib.md5(link.encode()).hexdigest()[:10]}"

def load_model(model_path):
    """Load YOLO model"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not os.path.exists(model_path):
        logging.error(f"Model file not found: {model_path}")
        return None
    try:
        model = YOLO(model_path)
        model.to(device)
        logging.info(f"Loaded model '{os.path.basename(model_path)}' on '{device}'")
        return model
    except Exception as e:
        logging.error(f"Failed to load model '{model_path}': {e}")
        return None

def main():
    """Main service function"""
    logging.info("🚀 Starting Heatmap Processor Microservice")
    
    # Load model
    model = load_model(MODEL_PATH)
    if not model:
        logging.error("Failed to load model")
        sys.exit(1)

    # Parse RTSP links and start processors
    if not os.path.exists(RTSP_LINKS_FILE):
        logging.error(f"Configuration file not found: {RTSP_LINKS_FILE}")
        sys.exit(1)

    processors = []
    
    with open(RTSP_LINKS_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 3:
                link, channel_name = parts[0], parts[1]
                app_names = set(parts[2:])
                
                if 'Heatmap' in app_names:
                    channel_id = get_stable_channel_id(link)
                    processor = HeatmapProcessor(
                        link, channel_id, channel_name, model
                    )
                    processor.start()
                    processors.append(processor)
                    logging.info(f"Started heatmap processor for {channel_name}")

    if not processors:
        logging.warning("No heatmap processors configured")
        sys.exit(0)
    
    # Start video server
    from video_server_mixin import start_video_server_for_processors
    start_video_server_for_processors(processors, 5013)
    
    # Keep service running
    try:
        while True:
            time.sleep(10)
            for p in processors:
                if not p.is_alive():
                    logging.error(f"Processor {p.name} died")
    except KeyboardInterrupt:
        logging.info("Shutting down...")
        for p in processors:
            if hasattr(p, 'stop'):
                p.stop()
        sys.exit(0)

if __name__ == "__main__":
    main()

