#!/usr/bin/env python3
"""
Detection Processor Microservice
Handles Shoplifting, QPOS, and Generic detection tasks
"""

import os
import sys
import time
import logging
import hashlib
import requests
import cv2
import imageio
from datetime import datetime
import pytz

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processors.detection_processor import DetectionProcessor
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text
from sqlalchemy.orm import sessionmaker, declarative_base
import torch
from ultralytics import YOLO

# Configuration
IST = pytz.timezone('Asia/Kolkata')
DATABASE_URL = os.environ.get('DATABASE_URL', 'postgresql+psycopg2://postgres:Tneural01@postgres:5432/sakshi')
MAIN_APP_URL = os.environ.get('MAIN_APP_URL', 'http://main-app:5001')
CONFIG_FOLDER = '/app/config'
MODELS_FOLDER = '/app/models'
STATIC_FOLDER = '/app/static'
DETECTIONS_SUBFOLDER = 'detections'
RTSP_LINKS_FILE = os.path.join(CONFIG_FOLDER, 'rtsp_links.txt')

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [DETECTION] - %(levelname)s - %(message)s'
)

# Database Setup
Base = declarative_base()

class Detection(Base):
    __tablename__ = "detections"
    id = Column(Integer, primary_key=True, index=True)
    app_name = Column(String, index=True)
    channel_id = Column(String, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(IST))
    message = Column(Text)
    media_path = Column(String, unique=True)

# App Configuration
APP_TASKS_CONFIG = {
    'Shoplifting': {
        'model_path': os.path.join(MODELS_FOLDER, 'best_shoplift.pt'),
        'target_class_id': 1,
        'confidence': 0.8,
        'is_gif': True
    },
    'QPOS': {
        'model_path': os.path.join(MODELS_FOLDER, 'best_qpos.pt'),
        'target_class_id': 0,
        'confidence': 0.87,
        'is_gif': False
    },
    'Generic': {
        'model_path': os.path.join(MODELS_FOLDER, 'best_generic.pt'),
        'target_class_id': list(range(1, 8)),
        'confidence': 0.6,
        'is_gif': True
    },
}

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
    """Handle detection and save to database"""
    timestamp = datetime.now(IST)
    ts_string = timestamp.strftime("%Y%m%d_%H%M%S")
    filename = f"{app_name}_{channel_id}_{ts_string}.{'gif' if is_gif else 'jpg'}"
    media_path = os.path.join(DETECTIONS_SUBFOLDER, filename)
    full_path = os.path.join(STATIC_FOLDER, media_path)
    
    try:
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        if is_gif and isinstance(frames, list) and len(frames) > 1:
            rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
            imageio.mimsave(full_path, rgb_frames, fps=10)
        else:
            frame_to_save = frames[0] if isinstance(frames, list) else frames
            cv2.imwrite(full_path, frame_to_save)
        logging.info(f"Saved detection: {filename}")
    except Exception as e:
        logging.error(f"Failed to save media file '{full_path}': {e}")
        return None

    # Save to database
    try:
        with SessionLocal() as db:
            db.add(Detection(
                app_name=app_name,
                channel_id=channel_id,
                timestamp=timestamp,
                message=message,
                media_path=media_path
            ))
            db.commit()
    except Exception as e:
        logging.error(f"Failed to save detection to DB: {e}")

    # Notify main app via API
    try:
        media_url = f"/{STATIC_FOLDER}/{media_path}".replace('\\', '/')
        requests.post(
            f"{MAIN_APP_URL}/api/detection_event",
            json={
                'app_name': app_name,
                'channel_id': channel_id,
                'timestamp': timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                'message': message,
                'media_url': media_url
            },
            timeout=5
        )
    except Exception as e:
        logging.warning(f"Failed to notify main app: {e}")

    return media_path

def main():
    """Main service function"""
    logging.info("🚀 Starting Detection Processor Microservice")
    
    # Initialize database
    try:
        engine = create_engine(DATABASE_URL, pool_pre_ping=True)
        global SessionLocal
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        Base.metadata.create_all(bind=engine)
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
    
    # Parse RTSP links and start processors
    if not os.path.exists(RTSP_LINKS_FILE):
        logging.error(f"Configuration file not found: {RTSP_LINKS_FILE}")
        sys.exit(1)

    processors = []
    detection_apps = {'Shoplifting', 'QPOS', 'Generic'}
    
    with open(RTSP_LINKS_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 3:
                link, channel_name = parts[0], parts[1]
                app_names = set(parts[2:])
                
                # Check if this line has any detection apps
                if app_names.intersection(detection_apps):
                    channel_id = get_stable_channel_id(link)
                    
                    # Load models for detection tasks
                    detection_tasks = []
                    for app_name in app_names.intersection(detection_apps):
                        config = APP_TASKS_CONFIG.get(app_name, {})
                        if 'model_path' in config:
                            model = load_model(config['model_path'])
                            if model:
                                detection_tasks.append({
                                    'app_name': app_name,
                                    'model': model,
                                    **config
                                })
                    
                    if detection_tasks:
                        processor = DetectionProcessor(
                            link, channel_id, channel_name,
                            detection_tasks, handle_detection
                        )
                        processor.start()
                        processors.append(processor)
                        logging.info(f"Started detection processor for {channel_name}")

    if not processors:
        logging.warning("No detection processors configured")
        sys.exit(0)
    
    # Start video server
    from video_server_mixin import start_video_server_for_processors
    start_video_server_for_processors(processors, 5016)
    
    # Keep service running
    try:
        while True:
            time.sleep(10)
            # Check if all processors are alive
            for p in processors:
                if not p.is_alive():
                    logging.error(f"Processor {p.name} died, restarting...")
                    # You could implement restart logic here
    except KeyboardInterrupt:
        logging.info("Shutting down...")
        for p in processors:
            p.stop()
        sys.exit(0)

if __name__ == "__main__":
    main()

