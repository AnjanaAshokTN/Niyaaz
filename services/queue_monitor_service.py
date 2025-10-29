#!/usr/bin/env python3
"""
Queue Monitor Processor Microservice
Monitors queue length and wait times
"""

import os
import sys
import time
import logging
import hashlib
import requests
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processors.queue_monitor_processor import QueueMonitorProcessor, QueueLog
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import sessionmaker, declarative_base
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
    format='%(asctime)s - [QUEUE_MONITOR] - %(levelname)s - %(message)s'
)

# ROI Config Table
Base = declarative_base()

class RoiConfig(Base):
    __tablename__ = "roi_configs"
    id = Column(Integer, primary_key=True, index=True)
    channel_id = Column(String, index=True)
    app_name = Column(String, index=True)
    roi_points = Column(Text)

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

def handle_detection(app_name, channel_id, frames, message, is_gif=False):
    """Forward detection to main app"""
    try:
        requests.post(
            f"{MAIN_APP_URL}/api/handle_detection",
            json={
                'app_name': app_name,
                'channel_id': channel_id,
                'message': message,
                'is_gif': is_gif
            },
            timeout=5
        )
    except Exception as e:
        logging.warning(f"Failed to forward detection: {e}")

def send_telegram_notification(message):
    """Send telegram via main app"""
    try:
        requests.post(
            f"{MAIN_APP_URL}/api/telegram_notification",
            json={'message': message},
            timeout=5
        )
    except Exception as e:
        logging.warning(f"Failed to send telegram: {e}")

# Mock SocketIO
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
    logging.info("🚀 Starting Queue Monitor Processor Microservice")
    
    # Initialize database
    try:
        engine = create_engine(DATABASE_URL, pool_pre_ping=True)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        QueueMonitorProcessor.initialize_tables(engine)
        Base.metadata.create_all(bind=engine)
        logging.info("✅ Database connection established")
    except Exception as e:
        logging.error(f"❌ Database initialization failed: {e}")
        sys.exit(1)

    # Wait for main app
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
                
                if 'QueueMonitor' in app_names:
                    channel_id = get_stable_channel_id(link)
                    processor = QueueMonitorProcessor(
                        link, channel_id, channel_name, model,
                        socketio, SessionLocal,
                        handle_detection, send_telegram_notification
                    )
                    
                    # Load ROI configuration if exists
                    with SessionLocal() as db:
                        roi_record = db.query(RoiConfig).filter_by(
                            channel_id=channel_id, app_name='QueueMonitor'
                        ).first()
                        if roi_record:
                            try:
                                processor.update_roi(json.loads(roi_record.roi_points))
                                logging.info(f"Loaded ROI for {channel_name}")
                            except json.JSONDecodeError:
                                logging.error(f"Could not parse ROI for {channel_name}")
                    
                    processor.start()
                    processors.append(processor)
                    logging.info(f"Started queue monitor for {channel_name}")

    if not processors:
        logging.warning("No queue monitor processors configured")
        sys.exit(0)
    
    # Start video server
    from video_server_mixin import start_video_server_for_processors
    start_video_server_for_processors(processors, 5011)
    
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

