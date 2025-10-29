#!/usr/bin/env python3
"""
People Counter Processor Microservice
Handles bidirectional footfall tracking
"""

import os
import sys
import time
import logging
import hashlib
import requests

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processors.people_counter_processor import PeopleCounterProcessor, DailyFootfall, HourlyFootfall
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import torch
from ultralytics import YOLO

# Configuration
DATABASE_URL = os.environ.get('DATABASE_URL', 'postgresql+psycopg2://postgres:Tneural01@postgres:5432/sakshi')
MAIN_APP_URL = os.environ.get('MAIN_APP_URL', 'http://main-app:5001')
CONFIG_FOLDER = '/app/config'
MODELS_FOLDER = '/app/models'
RTSP_LINKS_FILE = os.path.join(CONFIG_FOLDER, 'rtsp_links.txt')
MODEL_PATH = os.path.join(MODELS_FOLDER, 'yolov8n.pt')

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [PEOPLE_COUNTER] - %(levelname)s - %(message)s'
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

# Mock SocketIO for standalone operation
class MockSocketIO:
    def emit(self, event, data):
        try:
            requests.post(
                f"{MAIN_APP_URL}/api/socketio_event",
                json={'event': event, 'data': data},
                timeout=2
            )
        except Exception as e:
            logging.debug(f"Failed to emit event: {e}")

def main():
    """Main service function"""
    logging.info("🚀 Starting People Counter Processor Microservice")
    
    # Initialize database
    try:
        engine = create_engine(DATABASE_URL, pool_pre_ping=True)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        PeopleCounterProcessor.initialize_tables(engine)
        logging.info("✅ Database connection established")
    except Exception as e:
        logging.error(f"❌ Database initialization failed: {e}")
        sys.exit(1)

    # Wait for main app to be ready
    logging.info("Waiting for main app...")
    for _ in range(30):
        try:
            requests.get(f"{MAIN_APP_URL}/health", timeout=2)
            break
        except:
            time.sleep(2)
    
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
    socketio = MockSocketIO()
    
    with open(RTSP_LINKS_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 3:
                link, channel_name = parts[0], parts[1]
                app_names = set(parts[2:])
                
                if 'PeopleCounter' in app_names:
                    channel_id = get_stable_channel_id(link)
                    processor = PeopleCounterProcessor(
                        link, channel_id, channel_name,
                        model, socketio, SessionLocal
                    )
                    processor.start()
                    processors.append(processor)
                    logging.info(f"Started people counter for {channel_name}")

    if not processors:
        logging.warning("No people counter processors configured")
        sys.exit(0)
    
    # Start video server
    from video_server_mixin import start_video_server_for_processors
    start_video_server_for_processors(processors, 5010)
    
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
            p.stop()
        sys.exit(0)

if __name__ == "__main__":
    main()

