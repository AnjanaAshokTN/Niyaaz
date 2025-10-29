import cv2
import torch
from ultralytics import YOLO
import threading
import time
import json
from datetime import datetime, date, timedelta
from collections import defaultdict
import os
import requests
import imageio
from flask import Flask, Response, render_template, jsonify, url_for, request
from flask_socketio import SocketIO
from sqlalchemy import create_engine, Column, Integer, String, Date, DateTime, Text, UniqueConstraint, text
from sqlalchemy.orm import sessionmaker, declarative_base
import logging
import pytz
import hashlib

# --- Modular Processor Imports (Standardized Names) ---
from processors.shutter_monitor_processor006 import ShutterMonitorProcessor
from processors.security_monitor_1 import SecurityProcessor
from processors.kitchen_compliance_monitor import KitchenComplianceProcessor, KitchenViolation
from processors.people_counter_processor import PeopleCounterProcessor, DailyFootfall, HourlyFootfall
from processors.queue_monitor_processor import QueueMonitorProcessor, QueueLog
from processors.detection_processor import DetectionProcessor
from processors.heatmap_processor import HeatmapProcessor

# --- Master Configuration ---
IST = pytz.timezone('Asia/Kolkata')
DATABASE_URL = os.environ.get('DATABASE_URL', "postgresql+psycopg2://postgres:ajmal123@localhost:5432/sakshi")
CONFIG_FOLDER = 'config'
MODELS_FOLDER = 'models'
RTSP_LINKS_FILE = os.path.join(CONFIG_FOLDER, 'rtsp_links.txt')
STATIC_FOLDER = 'static'
DETECTIONS_SUBFOLDER = 'detections'
TELEGRAM_BOT_TOKEN = "7843300957:AAGVv866cPiDPVD0Wrk_wwEEHDSD64Pgaqs" # Replace with your token
TELEGRAM_CHAT_ID = "-4835836048" # Replace with your chat ID

# --- App Task Configuration ---
APP_TASKS_CONFIG = {
    'Shoplifting': {'model_path': os.path.join(MODELS_FOLDER, 'best_shoplift.pt'), 'target_class_id': 1, 'confidence': 0.8, 'is_gif': True},
    'QPOS': {'model_path': os.path.join(MODELS_FOLDER, 'best_qpos.pt'), 'target_class_id': 0, 'confidence': 0.87, 'is_gif': False},
    'Generic': {'model_path': os.path.join(MODELS_FOLDER, 'best_generic.pt'), 'target_class_id': list(range(1, 8)), 'confidence': 0.6, 'is_gif': True},
    'PeopleCounter': {'model_path': os.path.join(MODELS_FOLDER, 'yolov8n.pt')},
    'Heatmap': {'model_path': os.path.join(MODELS_FOLDER, 'yolov8n.pt')},
    'QueueMonitor': {'model_path': os.path.join(MODELS_FOLDER, 'yolov8n.pt')},
    'ShutterMonitor': {'model_path': os.path.join(MODELS_FOLDER, 'shutter_model.pt')},
    'Security': {}, 'KitchenCompliance': {}
}

# --- Flask and SocketIO Setup ---
app = Flask(__name__, static_folder=STATIC_FOLDER, template_folder='templates')
app.config['SECRET_KEY'] = 'a-very-secret-key-for-sakshi-ai'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# --- Global State ---
stream_processors = {}
shutdown_event = threading.Event()

# --- Database Setup ---
Base = declarative_base()
engine, SessionLocal = None, None

# --- Main App Tables ---
class Detection(Base):
    __tablename__ = "detections"
    id = Column(Integer, primary_key=True, index=True)
    app_name = Column(String, index=True)
    channel_id = Column(String, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(IST))
    message = Column(Text)
    media_path = Column(String, unique=True)

class ShutterLog(Base):
    __tablename__ = "shutter_logs"
    id = Column(Integer, primary_key=True, index=True)
    channel_id = Column(String, index=True)
    report_date = Column(Date, index=True)
    first_open_time = Column(DateTime(timezone=True))
    first_close_time = Column(DateTime(timezone=True))
    total_open_duration_seconds = Column(Integer, default=0)
    first_open_video_path = Column(String, nullable=True)
    first_close_video_path = Column(String, nullable=True)
    __table_args__ = (UniqueConstraint('channel_id', 'report_date', name='_shutter_channel_date_uc'),)

class SecurityViolation(Base):
    __tablename__ = "security_violations"
    id = Column(Integer, primary_key=True, index=True)
    channel_id = Column(String, index=True)
    channel_name = Column(String)
    timestamp = Column(DateTime, default=lambda: datetime.now(IST))
    message = Column(String)
    details = Column(String)

class RoiConfig(Base):
    __tablename__ = "roi_configs"
    id = Column(Integer, primary_key=True, index=True)
    channel_id = Column(String, index=True)
    app_name = Column(String, index=True)
    roi_points = Column(Text)
    __table_args__ = (UniqueConstraint('channel_id', 'app_name', name='_roi_uc'),)


# --- Core Application Functions ---
def graceful_shutdown(signum=None, frame=None):
    if shutdown_event.is_set(): return
    logging.info("Initiating graceful shutdown...")
    shutdown_event.set()
    for channel_id, processors in list(stream_processors.items()):
        for processor in processors:
            if processor.is_alive():
                logging.warning(f"Stopping {processor.name}...")
                if hasattr(processor, 'shutdown'): processor.shutdown()
                elif hasattr(processor, 'stop'): processor.stop()
    time.sleep(2)

def initialize_database():
    global engine, SessionLocal
    try:
        engine = create_engine(DATABASE_URL, pool_pre_ping=True)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        with engine.connect() as conn: conn.execute(text("SELECT 1"))
        
        Base.metadata.create_all(bind=engine)
        KitchenComplianceProcessor.initialize_tables(engine)
        PeopleCounterProcessor.initialize_tables(engine)
        QueueMonitorProcessor.initialize_tables(engine)
        
        logging.info("✅ PostgreSQL database connection successful.")
        return True
    except Exception as e:
        logging.error(f"❌ Database initialization failed: {e}")
        return False

def get_stable_channel_id(link):
    return f"cam_{hashlib.md5(link.encode()).hexdigest()[:10]}"

def load_model(model_path):
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

def start_streams():
    if not os.path.exists(RTSP_LINKS_FILE):
        logging.error(f"'{RTSP_LINKS_FILE}' not found. No streams will be started.")
        return

    stream_assignments = defaultdict(lambda: {'apps': set(), 'name': ''})
    with open(RTSP_LINKS_FILE, 'r') as f:
        for line in [l.strip() for l in f if l.strip() and not l.startswith('#')]:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 3:
                stream_assignments[parts[0]]['apps'].update(parts[2:])
                stream_assignments[parts[0]]['name'] = parts[1]

    for link, assignment in stream_assignments.items():
        channel_id, channel_name, app_names = get_stable_channel_id(link), assignment['name'], assignment['apps']
        stream_processors[channel_id] = []
        processors_to_add = []

        if 'PeopleCounter' in app_names:
            if model := load_model(APP_TASKS_CONFIG['PeopleCounter']['model_path']):
                processors_to_add.append(PeopleCounterProcessor(link, channel_id, channel_name, model, socketio, SessionLocal))
        if 'QueueMonitor' in app_names:
            if model := load_model(APP_TASKS_CONFIG['QueueMonitor']['model_path']):
                p = QueueMonitorProcessor(link, channel_id, channel_name, model, socketio, SessionLocal, handle_detection, send_telegram_notification)
                with SessionLocal() as db:
                    if roi_record := db.query(RoiConfig).filter_by(channel_id=channel_id, app_name='QueueMonitor').first():
                        try: p.update_roi(json.loads(roi_record.roi_points))
                        except json.JSONDecodeError: logging.error(f"Could not parse ROI for {channel_name}")
                processors_to_add.append(p)
        if 'ShutterMonitor' in app_names:
            if model := load_model(APP_TASKS_CONFIG['ShutterMonitor']['model_path']):
                processors_to_add.append(ShutterMonitorProcessor(link, channel_id, channel_name, model, socketio, send_telegram_notification, SessionLocal))
        if 'Security' in app_names:
            processors_to_add.append(SecurityProcessor(link, channel_id, channel_name, SessionLocal, socketio, SecurityViolation))
        if 'KitchenCompliance' in app_names:
            processors_to_add.append(KitchenComplianceProcessor(link, channel_id, channel_name, SessionLocal, socketio, send_telegram_notification, handle_detection))
        if 'Heatmap' in app_names:
            if model := load_model(APP_TASKS_CONFIG['Heatmap']['model_path']):
                processors_to_add.append(HeatmapProcessor(link, channel_id, channel_name, model))

        detection_tasks = []
        for app_name in app_names.intersection({'Shoplifting', 'QPOS', 'Generic'}):
            config = APP_TASKS_CONFIG.get(app_name, {})
            if 'model_path' in config and (model := load_model(config['model_path'])):
                detection_tasks.append({'app_name': app_name, 'model': model, **config})
        if detection_tasks:
            processors_to_add.append(DetectionProcessor(link, channel_id, channel_name, detection_tasks, handle_detection))

        for p in processors_to_add:
            stream_processors[channel_id].append(p)
            p.start()
            logging.info(f"Started processor '{p.name}' for channel {channel_id}.")

def send_telegram_notification(message):
    if not TELEGRAM_BOT_TOKEN or "YOUR_TELEGRAM" in TELEGRAM_BOT_TOKEN:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'Markdown'}
    try:
        requests.post(url, json=payload, timeout=5)
    except Exception as e:
        logging.error(f"Error sending Telegram notification: {e}")

def handle_detection(app_name, channel_id, frames, message, is_gif=False):
    timestamp = datetime.now(IST)
    ts_string = timestamp.strftime("%Y%m%d_%H%M%S")
    filename = f"{app_name}_{channel_id}_{ts_string}.{'gif' if is_gif else 'jpg'}"
    media_path = os.path.join(DETECTIONS_SUBFOLDER, filename)
    full_path = os.path.join(STATIC_FOLDER, media_path)
    
    try:
        if is_gif and isinstance(frames, list) and len(frames) > 1:
            rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
            imageio.mimsave(full_path, rgb_frames, fps=10)
        else:
            frame_to_save = frames[0] if isinstance(frames, list) else frames
            cv2.imwrite(full_path, frame_to_save)
    except Exception as e:
        logging.error(f"Failed to save media file '{full_path}': {e}")
        return None

    if SessionLocal:
        with SessionLocal() as db:
            try:
                db.add(Detection(app_name=app_name, channel_id=channel_id, timestamp=timestamp, message=message, media_path=media_path))
                db.commit()
            except Exception as e:
                logging.error(f"Failed to save detection to DB: {e}")
                db.rollback()

    media_url = f"/{STATIC_FOLDER}/{media_path}".replace('\\', '/')
    socketio.emit('new_detection', {
        'app_name': app_name, 'channel_id': channel_id,
        'timestamp': timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        'message': message, 'media_url': media_url
    })
    return media_path

# --- Flask Routes ---
@app.route('/')
def landing_page(): return render_template('landing.html')

@app.route('/dashboard')
def dashboard(): return render_template('dashboard.html', app_configs=get_app_configs())

@app.route('/video_feed/<app_name>/<channel_id>')
def video_feed(app_name, channel_id):
    # Docker mode: proxy video from processor microservices
    is_docker_mode = os.environ.get('DOCKER_MODE', 'false').lower() == 'true'
    
    if is_docker_mode:
        # Map app names to processor service ports
        processor_ports = {
            'PeopleCounter': 5010,
            'QueueMonitor': 5011,
            'Security': 5012,
            'Heatmap': 5013,
            'ShutterMonitor': 5014,
            'KitchenCompliance': 5015,
            'Detection': 5016,
            'Shoplifting': 5016,  # Detection processor
            'QPOS': 5016,  # Detection processor
            'Generic': 5016  # Detection processor
        }
        
        port = processor_ports.get(app_name)
        if not port:
            return (f"Unknown app: {app_name}", 404)
        
        # Map app names to processor service names
        processor_services = {
            'PeopleCounter': 'sakshi-people-counter',
            'QueueMonitor': 'sakshi-queue-monitor',
            'Security': 'sakshi-security-monitor',
            'Heatmap': 'sakshi-heatmap',
            'ShutterMonitor': 'sakshi-shutter-monitor',
            'KitchenCompliance': 'sakshi-kitchen-compliance',
            'Detection': 'sakshi-detection-processor',
            'Shoplifting': 'sakshi-detection-processor',
            'QPOS': 'sakshi-detection-processor',
            'Generic': 'sakshi-detection-processor'
        }
        service_name = processor_services.get(app_name, 'localhost')
        processor_url = f"http://{service_name}:{port}/video_feed/{channel_id}"
        
        def proxy_stream():
            try:
                resp = requests.get(processor_url, stream=True, timeout=5)
                for chunk in resp.iter_content(chunk_size=1024):
                    yield chunk
            except Exception as e:
                logging.error(f"Error proxying stream: {e}")
                # Return placeholder
                import numpy as np
                placeholder = np.full((480, 640, 3), (22, 27, 34), dtype=np.uint8)
                import cv2
                cv2.putText(placeholder, 'Stream Unavailable', (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (201, 209, 217), 2)
                _, jpeg = cv2.imencode('.jpg', placeholder)
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        
        return Response(proxy_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')
    
    # Traditional mode: direct access to processors
    processors = stream_processors.get(channel_id)
    if not processors: 
        logging.warning(f"No processors found for channel {channel_id}")
        return ("Stream not found", 404)
    
    target_class_map = {
        'PeopleCounter': PeopleCounterProcessor, 'QueueMonitor': QueueMonitorProcessor,
        'Security': SecurityProcessor, 'Heatmap': HeatmapProcessor,
        'ShutterMonitor': ShutterMonitorProcessor, 'KitchenCompliance': KitchenComplianceProcessor,
        'Detection': DetectionProcessor
    }
    target_class = target_class_map.get(app_name)
    
    if target_class:
        target_processor = next((p for p in processors if isinstance(p, target_class)), None)
        if target_processor:
            if not target_processor.is_alive():
                logging.warning(f"Processor {app_name} for channel {channel_id} is not alive")
                return (f"{app_name} stream not running for this channel", 503)
            
            def gen_feed():
                try:
                    while not shutdown_event.is_set():
                        frame_bytes = target_processor.get_frame()
                        if frame_bytes:
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                        time.sleep(0.03)  # ~30 FPS
                except Exception as e:
                    logging.error(f"Error in video feed generator for {app_name}/{channel_id}: {e}")
            
            response = Response(gen_feed(), 
                              mimetype='multipart/x-mixed-replace; boundary=frame',
                              headers={'Cache-Control': 'no-cache, no-store, must-revalidate',
                                      'Pragma': 'no-cache',
                                      'Expires': '0'})
            return response
        else:
            logging.warning(f"No processor instance found for {app_name} on channel {channel_id}")
            
    return (f"{app_name} stream not running for this channel", 404)

@app.route('/history/<app_name>')
def get_history(app_name):
    page = request.args.get('page', 1, type=int)
    limit = request.args.get('limit', 10, type=int)
    channel_id = request.args.get('channel_id')
    start_date_str = request.args.get('start_date')
    end_date_str = request.args.get('end_date')
    
    with SessionLocal() as db:
        query = db.query(Detection).filter(Detection.app_name == app_name)
        if channel_id and channel_id != 'null':
            query = query.filter(Detection.channel_id == channel_id)
        if start_date_str and end_date_str:
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
            query = query.filter(Detection.timestamp >= start_date, Detection.timestamp < (end_date + timedelta(days=1)))
        
        total = query.count()
        detections = query.order_by(Detection.timestamp.desc()).offset((page - 1) * limit).limit(limit).all()
        
        return jsonify({
            'detections': [{
                'timestamp': d.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                'message': d.message,
                'media_url': f"/{d.media_path}".replace('\\', '/')
            } for d in detections],
            'total': total, 'page': page, 'limit': limit
        })

@app.route('/api/set_roi', methods=['POST'])
def set_roi():
    data = request.json
    channel_id, app_name, roi_points = data.get('channel_id'), data.get('app_name'), data.get('roi_points')
    if not all([channel_id, app_name, isinstance(roi_points, dict)]):
        return jsonify({"error": "Missing or invalid data"}), 400
    
    with SessionLocal() as db:
        try:
            stmt = text("""
                INSERT INTO roi_configs (channel_id, app_name, roi_points) VALUES (:cid, :an, :rp)
                ON CONFLICT (channel_id, app_name) DO UPDATE SET roi_points = EXCLUDED.roi_points;
            """)
            db.execute(stmt, {'cid': channel_id, 'an': app_name, 'rp': json.dumps(roi_points)})
            db.commit()
            
            processors = stream_processors.get(channel_id, [])
            if app_name == 'QueueMonitor':
                for p in processors:
                    if isinstance(p, QueueMonitorProcessor):
                        p.update_roi(roi_points)
                        break
            
            return jsonify({"success": True})
        except Exception as e:
            db.rollback()
            logging.error(f"Error saving ROI: {e}")
            return jsonify({"error": "Database error"}), 500

@app.route('/report/<channel_id>/<date_str>')
def get_report(channel_id, date_str):
    try: report_date = datetime.strptime(date_str, '%Y-%m-%d').date()
    except ValueError: return jsonify({"error": "Invalid date format"}), 400
    
    hourly_data = {h: {'in': 0, 'out': 0} for h in range(24)}
    with SessionLocal() as db:
        for r in db.query(HourlyFootfall).filter_by(channel_id=channel_id, report_date=report_date).all():
            hourly_data[r.hour] = {'in': r.in_count, 'out': r.out_count}

    return jsonify({'hourly_data': hourly_data})

@app.route('/generate_report/<channel_id>')
def generate_report(channel_id):
    period = request.args.get('period', '7days')
    end_date = date.today()
    start_date = end_date - timedelta(days=6 if period == '7days' else 29)
    
    with SessionLocal() as db:
        daily_records = db.query(DailyFootfall).filter(DailyFootfall.channel_id == channel_id, DailyFootfall.report_date.between(start_date, end_date)).all()
        hourly_records = db.query(HourlyFootfall).filter(HourlyFootfall.channel_id == channel_id, HourlyFootfall.report_date.between(start_date, end_date)).all()
        if not daily_records: return jsonify({"error": f"No data for the last {period.replace('days', '')} days."})

        labels = [(start_date + timedelta(days=i)).strftime("%b %d") for i in range((end_date - start_date).days + 1)]
        daily_totals = {label: 0 for label in labels}
        for r in daily_records: daily_totals[r.report_date.strftime("%b %d")] = r.in_count + r.out_count
        
        hourly_totals = defaultdict(list)
        for r in hourly_records: hourly_totals[r.hour].append(r.in_count + r.out_count)
        
        total_traffic = sum(daily_totals.values())
        busiest_day_label = max(daily_totals, key=daily_totals.get) if total_traffic > 0 else "N/A"
        peak_hour_label = "N/A"
        if hourly_totals:
            avg_hourly = {h: sum(c)/len(c) for h,c in hourly_totals.items()}
            peak_hour_24 = max(avg_hourly, key=avg_hourly.get)
            peak_hour_label = datetime.strptime(str(peak_hour_24), '%H').strftime('%I %p')
        
        summary = {"total_footfall": total_traffic, "busiest_day": busiest_day_label, "peak_hour": peak_hour_label}
        
        return jsonify({"labels": list(daily_totals.keys()), "data": list(daily_totals.values()), "summary": summary})

@app.route('/queue_report/<channel_id>')
def get_queue_report(channel_id):
    now = datetime.now(IST)
    period, start_str, end_str = request.args.get('period'), request.args.get('start_date'), request.args.get('end_date')
    if start_str and end_str:
        start_dt, end_dt = IST.localize(datetime.strptime(start_str, '%Y-%m-%d')), IST.localize(datetime.combine(datetime.strptime(end_str, '%Y-%m-%d'), datetime.max.time()))
    elif period == 'today': start_dt, end_dt = now.replace(hour=0, minute=0, second=0), now
    elif period == 'yesterday':
        yesterday = now - timedelta(days=1)
        start_dt, end_dt = yesterday.replace(hour=0, minute=0, second=0), yesterday.replace(hour=23, minute=59, second=59)
    else: start_dt, end_dt = now - timedelta(days=7), now

    with SessionLocal() as db:
        records = db.query(QueueLog).filter(QueueLog.channel_id == channel_id, QueueLog.timestamp.between(start_dt, end_dt)).order_by(QueueLog.timestamp).all()
        if not records: return jsonify({"error": "No data found for the selected period."})
        
        labels = [r.timestamp.strftime('%H:%M' if period in ['today', 'yesterday'] else '%d %b %H:%M') for r in records]
        data = [r.queue_count for r in records]
        
        summary = { 'max_queue_length': max(data), 'avg_queue_length': round(sum(data)/len(data),1), 'peak_hour': 'N/A' }
        return jsonify({'labels': labels, 'data': data, 'summary': summary})

@app.route('/shutter_report/<channel_id>')
def get_shutter_report(channel_id):
    start_str, end_str = request.args.get('start_date'), request.args.get('end_date')
    try: start_date, end_date = datetime.strptime(start_str, '%Y-%m-%d').date(), datetime.strptime(end_str, '%Y-%m-%d').date()
    except: return jsonify({"error": "Invalid date format or missing params"}), 400
    
    with SessionLocal() as db:
        records = db.query(ShutterLog).filter(ShutterLog.channel_id == channel_id, ShutterLog.report_date.between(start_date, end_date)).order_by(ShutterLog.report_date).all()
        report_data = [{
            'date': r.report_date.strftime('%Y-%m-%d'),
            'first_open_time': r.first_open_time.astimezone(IST).strftime('%I:%M %p') if r.first_open_time else 'N/A',
            'first_close_time': r.first_close_time.astimezone(IST).strftime('%I:%M %p') if r.first_close_time else 'N/A',
            'first_open_video_url': f"/{r.first_open_video_path}".replace('\\', '/') if r.first_open_video_path else None,
            'first_close_video_url': f"/{r.first_close_video_path}".replace('\\', '/') if r.first_close_video_path else None
        } for r in records]
        return jsonify({'report_data': report_data})

@app.route('/reports/security/<channel_id>')
def get_security_reports(channel_id):
    with SessionLocal() as db:
        violations = db.query(SecurityViolation).filter_by(channel_id=channel_id).order_by(SecurityViolation.timestamp.desc()).limit(15).all()
        return jsonify([{'timestamp': v.timestamp.strftime("%Y-%m-%d %H:%M:%S"), 'message': v.message, 'details': v.details} for v in violations])

# --- Microservice API Endpoints ---
@app.route('/health')
def health_check():
    """Health check endpoint for containers"""
    return jsonify({"status": "healthy", "timestamp": datetime.now(IST).isoformat()})

@app.route('/api/detection_event', methods=['POST'])
def detection_event():
    """Receive detection events from processor microservices"""
    try:
        data = request.json
        socketio.emit('new_detection', data)
        return jsonify({"success": True}), 200
    except Exception as e:
        logging.error(f"Error handling detection event: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/socketio_event', methods=['POST'])
def socketio_event():
    """Forward SocketIO events from processor microservices"""
    try:
        data = request.json
        event = data.get('event')
        event_data = data.get('data')
        socketio.emit(event, event_data)
        return jsonify({"success": True}), 200
    except Exception as e:
        logging.error(f"Error forwarding socketio event: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/telegram_notification', methods=['POST'])
def telegram_notification_api():
    """Forward Telegram notifications from processor microservices"""
    try:
        data = request.json
        message = data.get('message')
        send_telegram_notification(message)
        return jsonify({"success": True}), 200
    except Exception as e:
        logging.error(f"Error sending telegram notification: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/handle_detection', methods=['POST'])
def handle_detection_api():
    """Handle detection requests from processor microservices"""
    try:
        data = request.json
        app_name = data.get('app_name')
        channel_id = data.get('channel_id')
        message = data.get('message')
        is_gif = data.get('is_gif', False)
        # Note: In microservice architecture, processors handle their own media saving
        # This endpoint is for fallback/coordination purposes
        return jsonify({"success": True}), 200
    except Exception as e:
        logging.error(f"Error handling detection: {e}")
        return jsonify({"error": str(e)}), 500

def save_periodic_heatmap_snapshots():
    with app.app_context():
        for channel_id, processors in stream_processors.items():
            for p in processors:
                if isinstance(p, HeatmapProcessor):
                    if frame := p.get_snapshot_frame():
                        handle_detection('Heatmap', channel_id, [frame], 'Periodic heatmap snapshot.')

def log_queue_counts():
    if not SessionLocal: return
    with SessionLocal() as db:
        for channel_id, processors in stream_processors.items():
            for p in processors:
                if isinstance(p, QueueMonitorProcessor):
                    db.add(QueueLog(channel_id=channel_id, queue_count=p.current_queue_count))
        db.commit()

def get_app_configs():
    app_configs = defaultdict(lambda: {'channels': [], 'online_count': 0})
    if not os.path.exists(RTSP_LINKS_FILE): return {}
    
    is_docker_mode = os.environ.get('DOCKER_MODE', 'false').lower() == 'true'
    
    # Port mapping for Docker mode health checks
    processor_ports = {
        'PeopleCounter': 5010,
        'QueueMonitor': 5011,
        'Security': 5012,
        'Heatmap': 5013,
        'ShutterMonitor': 5014,
        'KitchenCompliance': 5015,
        'Shoplifting': 5016,
        'QPOS': 5016,
        'Generic': 5016
    }
    
    with open(RTSP_LINKS_FILE, 'r') as f:
        for line in [l.strip() for l in f if l.strip() and not l.startswith('#')]:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 3:
                link, name, app_names = parts[0], parts[1], parts[2:]
                channel_id = get_stable_channel_id(link)
                
                # In Docker mode, check processor health via HTTP
                if is_docker_mode:
                    channel_is_alive = False
                    
                    for app_name in app_names:
                        if app_name in APP_TASKS_CONFIG:
                            # Add channel to app config
                            if not any(d['id'] == channel_id for d in app_configs[app_name]['channels']):
                                app_configs[app_name]['channels'].append({'id': channel_id, 'name': name})
                            
                            # Check if processor is alive via health endpoint
                            port = processor_ports.get(app_name)
                            if port:
                                try:
                                    resp = requests.get(f"http://localhost:{port}/health", timeout=1)
                                    if resp.status_code == 200:
                                        channel_is_alive = True
                                        break  # One healthy processor is enough for this channel
                                except:
                                    pass
                    
                    # Increment online count if any processor is alive for this channel
                    if channel_is_alive:
                        for app_name in app_names:
                            if app_name in APP_TASKS_CONFIG:
                                # Only increment once per channel, not per app
                                current_channels = app_configs[app_name]['channels']
                                if any(ch['id'] == channel_id for ch in current_channels):
                                    app_configs[app_name]['online_count'] = sum(
                                        1 for ch in current_channels 
                                        if ch['id'] == channel_id or 
                                        any(requests.get(f"http://localhost:{processor_ports.get(app_name, 0)}/health", timeout=0.5).status_code == 200 for _ in [1])
                                    )
                
                # Traditional mode
                else:
                    is_alive = any(p.is_alive() for p in stream_processors.get(channel_id, []))
                    
                    for app_name in app_names:
                        if app_name in APP_TASKS_CONFIG:
                            if not any(d['id'] == channel_id for d in app_configs[app_name]['channels']):
                                app_configs[app_name]['channels'].append({'id': channel_id, 'name': name})
                    
                    # Recalculate online counts for traditional mode
                    for app_name, config in app_configs.items():
                        online = 0
                        for channel in config['channels']:
                            if any(p.is_alive() for p in stream_processors.get(channel['id'], [])):
                                online += 1
                        config['online_count'] = online

    # Recalculate online counts for Docker mode
    if is_docker_mode:
        for app_name, config in app_configs.items():
            online = 0
            port = processor_ports.get(app_name)
            if port:
                try:
                    resp = requests.get(f"http://localhost:{port}/health", timeout=1)
                    if resp.status_code == 200:
                        data = resp.json()
                        online = data.get('alive_count', 0)
                except:
                    pass
            config['online_count'] = online

    return dict(app_configs)

