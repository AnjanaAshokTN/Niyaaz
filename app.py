"""
Sakshi.AI - Intelligent Video Analytics Platform
Main Flask Application
"""
import os
import json
import logging
from datetime import datetime, timedelta

# Load environment variables from .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, skip .env loading
from flask import Flask, render_template, request, jsonify, Response, send_from_directory, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from flask_socketio import SocketIO, emit
from functools import wraps
import cv2
import threading
import time
from pathlib import Path

# Import custom modules
from modules.people_counter import PeopleCounter
from modules.queue_monitor import QueueMonitor
from modules.bag_detection import BagDetection
from modules.heatmap_processor import HeatmapProcessor
from modules.cash_detection import CashDetection
from modules.fall_detection import FallDetection
from modules.mopping_detection import MoppingDetection
from modules.smoking_detection import SmokingDetection
from modules.phone_usage_detection import PhoneUsageDetection
from modules.dress_code_monitoring import DressCodeMonitoring
from modules.ppe_monitoring import PPEMonitoring
from modules.crowd_detection import CrowdDetection
from modules.restricted_area_monitor import RestrictedAreaMonitor
#from modules.table_service_monitor import TableServiceMonitor
from modules.video_processor import VideoProcessor
from modules.multi_module_processor import MultiModuleVideoProcessor
from modules.shared_multi_module_processor import SharedMultiModuleVideoProcessor
from modules.rtsp_connection_pool import rtsp_pool
from modules.database import DatabaseManager

from modules.model_manager import get_model_stats, cleanup_models

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'sakshi-ai-secret-key-2025'

# Database configuration
# Support both PostgreSQL (via environment variables) and SQLite (fallback)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Check for PostgreSQL environment variables first
db_host = os.getenv('DB_HOST', 'localhost')
db_port = os.getenv('DB_PORT', '5432')
db_name = os.getenv('DB_NAME', 'sakshiai')
db_user = os.getenv('DB_USER', 'postgres')
db_password = os.getenv('DB_PASSWORD', '')

# If DB_PASSWORD is set, use PostgreSQL; otherwise try config file, then fallback to SQLite
if db_password or os.getenv('USE_POSTGRESQL', '').lower() == 'true':
    # Use PostgreSQL
    if not db_password:
        # Try loading from config file as fallback
        try:
            with open(os.path.join(BASE_DIR, 'config', 'default.json'), 'r') as f:
                config = json.load(f)
                db_config = config.get('database', {})
                db_host = db_config.get('host', db_host)
                db_port = str(db_config.get('port', db_port))
                db_name = db_config.get('name', db_name)
                db_user = db_config.get('username', db_user)
                db_password = db_config.get('password', '')
        except Exception as e:
            logger.warning(f"Could not load database config from file: {e}")
    
    if db_password:
        # Construct PostgreSQL connection string
        app.config['SQLALCHEMY_DATABASE_URI'] = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        logger.info(f"Using PostgreSQL database: {db_name}@{db_host}:{db_port}")
    else:
        # Fallback to SQLite if no password provided
        db_path = os.path.join(BASE_DIR, "data", "sakshi.db")
        app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{db_path}"
        logger.info(f"Using SQLite database: {db_path}")
else:
    # Use SQLite as default
    db_path = os.path.join(BASE_DIR, "data", "sakshi.db")
    app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{db_path}"
    logger.info(f"Using SQLite database: {db_path}")

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db = SQLAlchemy(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global variables - Updated for multi-module support
shared_video_processors = {}  # {channel_id: MultiModuleVideoProcessor}
channel_modules = {}  # {channel_id: {module_name: module_instance}}
app_configs = {
    # 1) Queue & wait time tracking
    'QueueMonitor': {
        'name': 'Queue & Wait Time',
        'description': 'Monitor queue length, wait time, and counter staffing',
        'channels': {},
        'status': 'online'
    },
    # 2) Uniform compliance
    'DressCodeMonitoring': {
        'name': 'Uniform Compliance',
        'description': 'Monitor employee uniform compliance',
        'channels': {},
        'status': 'online'
    },
    # 2b) PPE compliance (separate from uniform)
    'PPEMonitoring': {
        'name': 'PPE Compliance',
        'description': 'Monitor Personal Protective Equipment compliance (Apron, Gloves, Hairnet)',
        'channels': {},
        'status': 'online'
    },
    # 3) Cash drawer monitoring
    'CashDetection': {
        'name': 'Cash Drawer Monitoring',
        'description': 'Detect cash and open drawers to monitor transactions',
        'channels': {},
        'status': 'online'
    },
    # 4) Smoke & fire detection (reusing smoking/safety model later if needed)
    'SmokingDetection': {
        'name': 'Smoke & Fire Detection',
        'description': 'Detect smoke and fire events for safety alerts',
        'channels': {},
        'status': 'online'
    },
    # 5) Crowd detection in parking space
    'CrowdDetection': {
        'name': 'Crowd Detection',
        'description': 'Monitor crowd gathering in parking space',
        'channels': {},
        'status': 'online'
    },
    # 6) Table service monitoring
    'TableServiceMonitor': {
        'name': 'Table Service Monitoring',
        'description': 'Monitor customer wait times at tables and server attendance',
        'channels': {},
        'status': 'online'
    },
}

# Database manager
db_manager = DatabaseManager(db)

# ============= Channel Auto-Loader from Configuration =============
def load_channels_from_config(config_file='config/channels.json'):
    """
    Load and start channels from config file on application startup
    
    Args:
        config_file: Path to the channels configuration file (default: config/channels.json)
    """
    config_path = Path(config_file)
    
    if not config_path.exists():
        logger.warning(f"Channel configuration file not found: {config_path}")
        return
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        channels = config.get('channels', [])
        logger.info(f"Loading {len(channels)} channels from configuration...")
        
        for channel_config in channels:
            try:
                if not channel_config.get('enabled', False):
                    logger.info(f"Skipping disabled channel: {channel_config.get('channel_name', 'Unknown')}")
                    continue
                
                channel_id = channel_config['channel_id']
                channel_name = channel_config['channel_name']
                # Support both RTSP URLs and local video files
                rtsp_url = channel_config.get('rtsp_url', '')
                video_file = channel_config.get('video_file', '')
                video_source = video_file if video_file else rtsp_url
                modules_config = channel_config.get('modules', [])
                
                # Save channel to database for persistence (if not already saved)
                try:
                    with app.app_context():
                        existing = db_manager.get_rtsp_channel(channel_id)
                        if not existing:
                            # Use video_source (could be RTSP URL or video file path)
                            db_manager.save_rtsp_channel(channel_id, channel_name, video_source, 
                                                       description=f"Auto-loaded from channels.json")
                            logger.info(f"  💾 Saved channel '{channel_name}' to database")
                except Exception as e:
                    logger.warning(f"  ⚠ Could not save channel to database: {e}")
                
                # Determine video source with fallback logic
                # Try RTSP first, fall back to video_file if RTSP fails
                final_video_source = None
                source_type = "unknown"
                
                if rtsp_url:
                    # Try RTSP first
                    logger.info(f"Starting channel '{channel_name}' ({channel_id}) with {len(modules_config)} modules (trying RTSP first)")
                    final_video_source = rtsp_url
                    source_type = "RTSP"
                elif video_file:
                    # Use video file directly if no RTSP URL
                    logger.info(f"Starting channel '{channel_name}' ({channel_id}) with {len(modules_config)} modules (using video file)")
                    final_video_source = video_file
                    source_type = "video file"
                else:
                    logger.warning(f"Channel '{channel_name}' ({channel_id}) has no RTSP URL or video file - skipping")
                    continue
                
                # Create video processor if it doesn't exist
                if channel_id not in shared_video_processors:
                    # Use SharedMultiModuleVideoProcessor - supports both RTSP and local video files
                    processor = SharedMultiModuleVideoProcessor(
                        video_source=final_video_source,
                        channel_id=channel_id,
                        fps_limit=30
                    )
                    shared_video_processors[channel_id] = processor
                    
                    # Initialize channel modules dictionary
                    if channel_id not in channel_modules:
                        channel_modules[channel_id] = {}
                
                # Add each module to the processor
                for module_config in modules_config:
                    module_type = module_config['type']
                    module_settings = module_config.get('config', {})
                    
                    try:
                        # Create module instance based on type
                        if module_type == 'PeopleCounter':
                            module = PeopleCounter(channel_id, socketio, db_manager, app)
                            # Apply counting line configuration if provided
                            if 'counting_line' in module_settings:
                                logger.info(f"  📏 Loading counting line for {channel_id}:")
                                logger.info(f"     {json.dumps(module_settings['counting_line'], indent=6)}")
                                module.set_counting_line(module_settings['counting_line'])
                        
                        elif module_type == 'QueueMonitor':
                            module = QueueMonitor(channel_id, socketio, db_manager, app)
                            # Load configuration from database first
                            try:
                                module.load_configuration()
                            except Exception as e:
                                logger.warning(f"Could not load QueueMonitor config from DB: {e}")
                            # Apply ROI configuration from config file if provided (overrides DB)
                            if module_settings:
                                # Map the channels.json structure to QueueMonitor format
                                roi_points = {
                                    'main': module_settings.get('queue_roi', {}).get('points', []),
                                    'secondary': module_settings.get('counter_roi', {}).get('points', [])
                                }
                                if roi_points['main'] or roi_points['secondary']:
                                    logger.info(f"  📐 Loading ROI configuration from config file for {channel_id}:")
                                    logger.info(f"     Queue area: {len(roi_points['main'])} points")
                                    logger.info(f"     Counter area: {len(roi_points['secondary'])} points")
                                    module.set_roi(roi_points)
                            # Apply settings if provided
                            if 'settings' in module_settings:
                                module.settings.update(module_settings['settings'])
                        
                        elif module_type == 'BagDetection':
                            module = BagDetection(channel_id, socketio, db_manager, app)
                            # Apply settings if provided
                            if module_settings:
                                for key, value in module_settings.items():
                                    if hasattr(module, key):
                                        setattr(module, key, value)
                        
                        elif module_type == 'CashDetection':
                            module = CashDetection(channel_id, socketio, db_manager, app)
                            # Apply settings if provided
                            if module_settings:
                                for key, value in module_settings.items():
                                    if hasattr(module, key):
                                        setattr(module, key, value)
                        
                        elif module_type == 'FallDetection':
                            module = FallDetection(channel_id, socketio, db_manager, app)
                            # Apply settings if provided
                            if module_settings:
                                for key, value in module_settings.items():
                                    if hasattr(module, key):
                                        setattr(module, key, value)
                        
                        elif module_type == 'MoppingDetection':
                            module = MoppingDetection(channel_id, socketio, db_manager, app)
                            # Apply settings if provided
                            if module_settings:
                                for key, value in module_settings.items():
                                    if hasattr(module, key):
                                        setattr(module, key, value)
                        
                        elif module_type == 'SmokingDetection':
                            module = SmokingDetection(channel_id, socketio, db_manager, app)
                            # Apply settings if provided
                            if module_settings:
                                for key, value in module_settings.items():
                                    if hasattr(module, key):
                                        setattr(module, key, value)
                        
                        elif module_type == 'PhoneUsageDetection':
                            module = PhoneUsageDetection(channel_id, socketio, db_manager, app)
                            # Apply settings if provided
                            if module_settings:
                                for key, value in module_settings.items():
                                    if hasattr(module, key):
                                        setattr(module, key, value)
                        
                        elif module_type == 'RestrictedAreaMonitor':
                            model_path = module_settings.get('model_path', 'models/best.pt') if module_settings else 'models/best.pt'
                            module = RestrictedAreaMonitor(channel_id, model_path, db_manager, socketio, module_settings)
                            
                            # Load ROI points - try config first, then database
                            roi_loaded = False
                            if module_settings and 'roi_points' in module_settings:
                                module.set_roi_points(module_settings['roi_points'])
                                roi_loaded = True
                                logger.info(f"✅ Loaded ROI from config file for {channel_id}")
                            
                            # If not in config, try loading from database (with app context)
                            if not roi_loaded:
                                try:
                                    with app.app_context():
                                        saved_roi = db_manager.get_channel_config(channel_id, 'RestrictedAreaMonitor', 'roi')
                                        if saved_roi:
                                            # Handle both dict and list formats
                                            if isinstance(saved_roi, dict):
                                                if 'main' in saved_roi:
                                                    module.set_roi_points(saved_roi['main'])
                                                    logger.info(f"✅ Loaded ROI from database (dict format) for {channel_id}: {len(saved_roi['main'])} points")
                                                else:
                                                    logger.warning(f"⚠️ ROI dict has no 'main' key for {channel_id}: {saved_roi}")
                                            elif isinstance(saved_roi, list):
                                                module.set_roi_points(saved_roi)
                                                logger.info(f"✅ Loaded ROI from database (list format) for {channel_id}: {len(saved_roi)} points")
                                            else:
                                                logger.warning(f"⚠️ Unexpected ROI format for {channel_id}: {type(saved_roi)}")
                                        else:
                                            logger.info(f"ℹ️ No saved ROI found in database for {channel_id}")
                                except Exception as e:
                                    logger.error(f"❌ Could not load ROI from database for {channel_id}: {e}", exc_info=True)
                        
                        elif module_type == 'DressCodeMonitoring':
                            module = DressCodeMonitoring(channel_id, socketio, db_manager, app)
                            
                            # Load counter ROI from config file (if provided)
                            if module_settings:
                                counter_roi_points = module_settings.get('counter_roi', {})
                                if isinstance(counter_roi_points, dict) and 'points' in counter_roi_points:
                                    module.set_counter_roi(counter_roi_points['points'])
                                    logger.info(f"  📐 Loaded counter ROI from config for {channel_id}: {len(counter_roi_points['points'])} points")
                                elif isinstance(counter_roi_points, list):
                                    module.set_counter_roi(counter_roi_points)
                                    logger.info(f"  📐 Loaded counter ROI from config for {channel_id}: {len(counter_roi_points)} points")
                                
                                # Load allowed uniforms from config
                                allowed_uniforms = module_settings.get('allowed_uniforms', {})
                                if allowed_uniforms:
                                    module.set_allowed_uniforms(allowed_uniforms)
                                    logger.info(f"  📐 Loaded allowed uniforms from config for {channel_id}: {allowed_uniforms}")
                                
                                # Apply other settings
                                for key, value in module_settings.items():
                                    if key not in ['counter_roi', 'allowed_uniforms'] and hasattr(module, key):
                                        setattr(module, key, value)
                        
                        elif module_type == 'TableServiceMonitor':
                            module = TableServiceMonitor(channel_id, socketio, db_manager, app)
                            
                            # Load table ROIs from config file (if provided)
                            if module_settings:
                                table_rois = module_settings.get('table_rois', {})
                                if table_rois:
                                    for table_id, roi_data in table_rois.items():
                                        if isinstance(roi_data, dict) and 'points' in roi_data:
                                            module.set_table_roi(table_id, roi_data['points'])
                                        elif isinstance(roi_data, list):
                                            module.set_table_roi(table_id, roi_data)
                                    logger.info(f"  📐 Loaded {len(table_rois)} table ROIs from config for {channel_id}")
                                
                                # Load settings from config
                                if 'settings' in module_settings:
                                    module.settings.update(module_settings['settings'])
                                    logger.info(f"  ✅ Loaded settings from config for {channel_id}")
                        
                        elif module_type == 'PPEMonitoring':
                            module = PPEMonitoring(channel_id, socketio, db_manager, app)
                            logger.info(f" ✓ Added PPEMonitoring to channel {channel_id}")
                            # Apply settings if provided
                            if module_settings:
                                if 'required_items' in module_settings:
                                    module.set_settings({'required_items': module_settings['required_items']})
                                    logger.info(f"   Applied required_items: {module_settings['required_items']}")
                                if 'settings' in module_settings:
                                    module.set_settings(module_settings['settings'])
                                    logger.info(f"   Applied settings: {module_settings['settings']}")
                                # Also apply individual settings
                                for key, value in module_settings.items():
                                    if key not in ['required_items', 'settings'] and hasattr(module, key):
                                        setattr(module, key, value)
                        
                        elif module_type == 'CrowdDetection':
                            module = CrowdDetection(channel_id, socketio, db_manager, app)
                            # Apply ROI configuration if provided
                            if module_settings:
                                roi_points = module_settings.get('roi', {})
                                if isinstance(roi_points, dict) and 'points' in roi_points:
                                    module.set_roi({'main': roi_points['points']})
                                elif isinstance(roi_points, list):
                                    module.set_roi({'main': roi_points})
                            # Apply settings if provided
                            if 'settings' in module_settings:
                                module.set_settings(module_settings['settings'])
                        
                        elif module_type == 'Heatmap':
                            module = HeatmapProcessor(channel_id, socketio, db_manager, app)
                            # Apply settings if provided
                            if module_settings:
                                for key, value in module_settings.items():
                                    if hasattr(module, key):
                                        setattr(module, key, value)
                        
                        else:
                            logger.warning(f"Unknown module type: {module_type}")
                            continue
                        
                        # Add module to processor and track it
                        shared_video_processors[channel_id].add_module(module_type, module)
                        channel_modules[channel_id][module_type] = module
                        
                        # Also populate app_configs so modules loaded from
                        # channels.json appear correctly in the dashboard for
                        # their respective apps (not only when started via
                        # /api/start_channel).
                        if module_type in app_configs:
                            if channel_id not in app_configs[module_type]['channels']:
                                app_configs[module_type]['channels'][channel_id] = {
                                    'name': channel_name,
                                    'status': 'online',
                                    'video_source': rtsp_url,
                                    'source_type': 'rtsp',
                                    'shared': True,
                                    'active_modules': []
                                }
                            active = app_configs[module_type]['channels'][channel_id].setdefault('active_modules', [])
                            if module_type not in active:
                                active.append(module_type)
                        
                        logger.info(f"  ✓ Added {module_type} to channel {channel_id}")
                        
                    except Exception as e:
                        logger.error(f"Failed to add module {module_type} to channel {channel_id}: {e}")
                        continue
                
                # Start the video processor
                try:
                    start_result = shared_video_processors[channel_id].start()
                    logger.info(f"Channel '{channel_name}' start() returned: {start_result}")
                    if start_result:
                        logger.info(f"✓ Channel '{channel_name}' started successfully")
                    else:
                        # RTSP connection failed - try fallback to video file if available
                        if rtsp_url and video_file:
                            logger.warning(f"⚠ RTSP connection failed for '{channel_name}', trying fallback to video file: {video_file}")
                            
                            # Remove failed processor
                            if channel_id in shared_video_processors:
                                failed_processor = shared_video_processors[channel_id]
                                try:
                                    failed_processor.stop()
                                except:
                                    pass
                                del shared_video_processors[channel_id]
                            
                            # Create new processor with video file
                            try:
                                fallback_processor = SharedMultiModuleVideoProcessor(
                                    video_source=video_file,
                                    channel_id=channel_id,
                                    fps_limit=30
                                )
                                shared_video_processors[channel_id] = fallback_processor
                                
                                # Re-add all modules to the new processor
                                for module_name, module_instance in channel_modules[channel_id].items():
                                    fallback_processor.add_module(module_name, module_instance)
                                
                                # Try to start with video file
                                fallback_result = fallback_processor.start()
                                if fallback_result:
                                    logger.info(f"✅ Channel '{channel_name}' started successfully using video file fallback")
                                else:
                                    logger.warning(f"⚠ Video file fallback also failed for '{channel_name}'")
                                    logger.warning(f"⚠ Keeping channel in channel_modules for dashboard visibility (is_running=False)")
                                    if channel_id in shared_video_processors:
                                        del shared_video_processors[channel_id]
                            except Exception as fallback_error:
                                logger.error(f"❌ Video file fallback failed for '{channel_name}': {fallback_error}")
                                logger.warning(f"⚠ Keeping channel in channel_modules for dashboard visibility (is_running=False)")
                                if channel_id in shared_video_processors:
                                    del shared_video_processors[channel_id]
                        else:
                            logger.warning(f"⚠ Channel '{channel_name}' processor start() returned False - RTSP connection may have failed")
                            logger.warning(f"⚠ No video file fallback available (video_file not configured)")
                            logger.warning(f"⚠ Keeping channel in channel_modules for dashboard visibility (is_running=False)")
                            # Remove processor from shared_video_processors since it's not actually running
                            if channel_id in shared_video_processors:
                                del shared_video_processors[channel_id]
                                logger.info(f"Removed non-running processor for {channel_id} from shared_video_processors")
                except Exception as e:
                    logger.error(f"❌ Failed to start channel '{channel_name}': {e}", exc_info=True)
                    # Keep channel_modules entry even on failure so dashboard can show it
                    # Only remove processor if it exists
                    if channel_id in shared_video_processors:
                        del shared_video_processors[channel_id]
                    # Keep channel_modules so dashboard can show configured channels
                
            except KeyError as e:
                logger.error(f"Invalid channel configuration - missing required field: {e}")
                continue
            except Exception as e:
                logger.error(f"Error loading channel: {e}")
                continue
        
        # Summary of channel loading
        running_count = len(shared_video_processors)
        configured_count = len(channel_modules)
        logger.info(f"📊 Channel loading complete:")
        logger.info(f"   - {running_count} channels with running processors")
        logger.info(f"   - {configured_count} channels configured (may include non-running)")
        
        # Log which channels are running vs configured
        running_channels = set(shared_video_processors.keys())
        configured_channels = set(channel_modules.keys())
        not_running = configured_channels - running_channels
        if not_running:
            logger.warning(f"   ⚠ {len(not_running)} channels configured but not running: {sorted(not_running)}")
        if running_channels:
            logger.info(f"   ✓ {len(running_channels)} channels running: {sorted(running_channels)}")
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse channel configuration file: {e}")
    except Exception as e:
        logger.error(f"Error loading channels from config: {e}")

# ============= Authentication Decorator =============
def login_required(f):
    """Decorator to require login for routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Note: Removed admin_required decorator - all users now have access

# ============= Routes =============

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page (DEV MODE: auto-login)"""
    if request.method == 'POST':
        username = request.form.get('username') or 'devuser'

        # 🚨 Dev-only: skip DB check and just log in
        session['user_id'] = 1
        session['username'] = username
        session['role'] = 'admin'

        return redirect(url_for('dashboard'))

    # If already logged in, redirect to dashboard
    if 'user_id' in session:
        return redirect(url_for('dashboard'))

    return render_template('login.html')


@app.route('/logout')
def logout():
    """Logout user"""
    session.clear()
    return redirect(url_for('login'))

@app.route('/')
def landing():
    """Landing page"""
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    """Main dashboard with full analytics (original UI)"""
    user = db_manager.get_user_by_id(session['user_id'])
    return render_template('dashboard.html', app_configs=app_configs, user=user)


@app.route('/dashboard_clean')
@login_required
def dashboard_clean():
    """Minimal dashboard view (clean 4-module UI)"""
    user = db_manager.get_user_by_id(session['user_id'])
    return render_template('clean_dashboard.html', app_configs=app_configs, user=user)

@app.route('/static/alerts/<filename>')
@login_required
def serve_alert_gif(filename):
    """Serve alert GIF files"""
    return send_from_directory('static/alerts', filename)

@app.route('/api/get_current_user')
@login_required
def get_current_user():
    """Get current logged in user info"""
    return jsonify({
        'success': True,
        'user': {
            'id': session.get('user_id'),
            'username': session.get('username'),
            'role': session.get('role')
        }
    })

@app.route('/video_feed/<app_name>/<channel_id>')
@login_required
def video_feed(app_name, channel_id):
    """Video streaming endpoint - supports both shared and module-specific feeds"""
    def generate():
        if channel_id in shared_video_processors:
            processor = shared_video_processors[channel_id]
            while processor.is_running:
                # Get frame - either combined or module-specific
                frame = processor.get_latest_frame(module_name=app_name if app_name in processor.get_active_modules() else None)
                if frame is not None:
                    # Encode with optimized JPEG quality for faster transmission
                    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])  # 75% quality (vs 95% default)
                    if ret:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(0.066)  # ~15 FPS (reduced from 10 FPS for smoother streaming)
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/get_channels/<app_name>')
def get_channels(app_name):
    """Get available channels for an app"""
    channels = []
    
    # Get saved RTSP channels from database
    try:
        saved_channels = db_manager.get_rtsp_channels()
        for channel in saved_channels:
            channels.append({
                'id': channel['channel_id'],
                'name': channel['name'],
                'rtsp_url': channel['rtsp_url'],
                'type': 'rtsp'
            })
    except Exception as e:
        logger.error(f"Error loading RTSP channels: {e}")
    
    # Also check for video files (backward compatibility)
    videos_dir = Path('videos')
    if videos_dir.exists():
        for video_file in videos_dir.glob('*.mp4'):
            channel_id = video_file.stem
            channels.append({
                'id': channel_id,
                'name': f"Video {channel_id}",
                'path': str(video_file),
                'type': 'video'
            })
    
    return jsonify(channels)

@app.route('/api/get_active_channels')
def get_active_channels():
    """Get all currently active channels with their running modules"""
    try:
        active_channels = []
        
        logger.info(f"📊 get_active_channels: shared_video_processors has {len(shared_video_processors)} entries")
        logger.info(f"📊 get_active_channels: channel_modules has {len(channel_modules)} entries")
        
        # First, get channels from processors that are running
        for channel_id, processor in shared_video_processors.items():
            # Check if channel has modules configured
            if channel_id in channel_modules and channel_modules[channel_id]:
                # Get active modules for this channel
                active_modules = processor.get_active_modules() if hasattr(processor, 'get_active_modules') else list(channel_modules[channel_id].keys())
                is_running = processor.is_running if hasattr(processor, 'is_running') else True
                
                logger.info(f"📊 Channel {channel_id}: processor exists, modules={active_modules}, is_running={is_running}")
                
                active_channels.append({
                    'channel_id': channel_id,
                    'modules': active_modules,
                    'is_running': is_running
                })
        
        # Also include channels that are configured in channel_modules but might not have processors yet
        # (e.g., if processor failed to start but modules are still configured)
        for channel_id, modules_dict in channel_modules.items():
            if channel_id not in shared_video_processors and modules_dict:
                # Channel is configured but processor not running
                active_modules = list(modules_dict.keys())
                logger.info(f"📊 Channel {channel_id}: configured but no processor, modules={active_modules}")
                
                active_channels.append({
                    'channel_id': channel_id,
                    'modules': active_modules,
                    'is_running': False
                })
        
        # Remove duplicates (in case channel appears in both)
        seen = set()
        unique_channels = []
        for ch in active_channels:
            if ch['channel_id'] not in seen:
                seen.add(ch['channel_id'])
                unique_channels.append(ch)
        
        logger.info(f"📊 Returning {len(unique_channels)} active channels")
        
        return jsonify({
            'success': True,
            'active_channels': unique_channels,
            'count': len(unique_channels)
        })
    except Exception as e:
        logger.error(f"Error getting active channels: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/add_rtsp_channel', methods=['POST'])
@login_required
def add_rtsp_channel():
    """Add a new RTSP channel"""
    data = request.json
    channel_name = data.get('name')
    rtsp_url = data.get('rtsp_url')
    
    if not channel_name or not rtsp_url:
        return jsonify({'success': False, 'error': 'Name and RTSP URL are required'})
    
    try:
        # Generate channel ID from name
        channel_id = channel_name.lower().replace(' ', '_').replace('-', '_')
        
        # Save to database
        success = db_manager.save_rtsp_channel(channel_id, channel_name, rtsp_url)
        
        if success:
            return jsonify({
                'success': True, 
                'channel_id': channel_id,
                'message': f'RTSP channel "{channel_name}" added successfully'
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to save RTSP channel'})
    
    except Exception as e:
        logger.error(f"Error adding RTSP channel: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/remove_rtsp_channel', methods=['POST'])
@login_required
def remove_rtsp_channel():
    """Remove an RTSP channel"""
    data = request.json
    channel_id = data.get('channel_id')
    
    if not channel_id:
        return jsonify({'success': False, 'error': 'Channel ID is required'})
    
    try:
        # Stop channel if running
        if channel_id in shared_video_processors:
            processor = shared_video_processors[channel_id]
            processor.stop()
            del shared_video_processors[channel_id]
            if channel_id in channel_modules:
                del channel_modules[channel_id]
        
        # Remove from database
        success = db_manager.remove_rtsp_channel(channel_id)
        
        if success:
            return jsonify({'success': True, 'message': 'RTSP channel removed successfully'})
        else:
            return jsonify({'success': False, 'error': 'Failed to remove RTSP channel'})
    
    except Exception as e:
        logger.error(f"Error removing RTSP channel: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/start_channel', methods=['POST'])
@login_required
def start_channel():
    """Start processing a video channel - supports multiple modules on same video"""
    data = request.json
    app_name = data.get('app_name')
    channel_id = data.get('channel_id')
    
    # Determine video source (RTSP URL or video file path)
    rtsp_url = data.get('rtsp_url')
    video_path = data.get('video_path')
    
    if rtsp_url:
        video_source = rtsp_url
        source_type = 'rtsp'
    elif video_path:
        video_source = video_path
        source_type = 'video'
    else:
        # Try to get from database or fallback to video file
        try:
            rtsp_channel = db_manager.get_rtsp_channel(channel_id)
            if rtsp_channel:
                video_source = rtsp_channel['rtsp_url']
                source_type = 'rtsp'
            else:
                video_source = f'videos/{channel_id}.mp4'
                source_type = 'video'
        except:
            video_source = f'videos/{channel_id}.mp4'
            source_type = 'video'
    
    try:
        # Check if video processor already exists for this channel
        if channel_id not in shared_video_processors:
            # Create new multi-module processor
            processor = MultiModuleVideoProcessor(video_source, channel_id)
            shared_video_processors[channel_id] = processor
            channel_modules[channel_id] = {}
            
            # Start the processor
            if not processor.start():
                del shared_video_processors[channel_id]
                del channel_modules[channel_id]
                return jsonify({'success': False, 'error': f'Failed to start video processor for {source_type} source'})
        
        processor = shared_video_processors[channel_id]
        
        # Check if this module is already active for this channel
        if app_name in channel_modules[channel_id]:
            return jsonify({'success': True, 'message': f'{app_name} already active on channel {channel_id}'})
        
        # Create and add the analysis module
        if app_name == 'PeopleCounter':
            module = PeopleCounter(channel_id, socketio, db_manager, app)
        elif app_name == 'QueueMonitor':
            module = QueueMonitor(channel_id, socketio, db_manager, app)
            # Load saved ROI configuration from database
            try:
                module.load_configuration()
                logger.info(f"✅ Loaded QueueMonitor configuration from database for {channel_id}")
            except Exception as e:
                logger.warning(f"⚠️ Could not load QueueMonitor config from DB for {channel_id}: {e}")
        elif app_name == 'BagDetection':
            module = BagDetection(channel_id, socketio, db_manager, app)
        elif app_name == 'Heatmap':
            module = HeatmapProcessor(channel_id, socketio, db_manager, app)
        elif app_name == 'CashDetection':
            module = CashDetection(channel_id, socketio, db_manager, app)
        elif app_name == 'FallDetection':
            module = FallDetection(channel_id, socketio, db_manager, app)
        elif app_name == 'MoppingDetection':
            module = MoppingDetection(channel_id, socketio, db_manager, app)
        elif app_name == 'SmokingDetection':
            module = SmokingDetection(channel_id, socketio, db_manager, app)
        elif app_name == 'PhoneUsageDetection':
            module = PhoneUsageDetection(channel_id, socketio, db_manager, app)
        elif app_name == 'RestrictedAreaMonitor':
            model_path = 'models/best.pt'
            module = RestrictedAreaMonitor(channel_id, model_path, db_manager, socketio, config={})
            # Load saved ROI points from database
            try:
                saved_roi = db_manager.get_channel_config(channel_id, 'RestrictedAreaMonitor', 'roi')
                logger.info(f"📐 Loading ROI from database for {channel_id}: type={type(saved_roi)}, data={saved_roi}")
                if saved_roi:
                    # Handle both formats: list or dict with 'main' key
                    if isinstance(saved_roi, dict) and 'main' in saved_roi:
                        module.set_roi_points(saved_roi['main'])
                        logger.info(f"✅ Loaded saved ROI (dict format) for RestrictedAreaMonitor on channel {channel_id}: {len(saved_roi['main'])} points")
                    elif isinstance(saved_roi, list):
                        module.set_roi_points(saved_roi)
                        logger.info(f"✅ Loaded saved ROI (list format) for RestrictedAreaMonitor on channel {channel_id}: {len(saved_roi)} points")
                    else:
                        logger.warning(f"⚠️ Unknown ROI format for {channel_id}: {type(saved_roi)}")
                else:
                    logger.info(f"ℹ️ No saved ROI found for RestrictedAreaMonitor on channel {channel_id}")
            except Exception as e:
                logger.error(f"❌ Could not load saved ROI for RestrictedAreaMonitor: {e}", exc_info=True)
        elif app_name == 'DressCodeMonitoring':
            module = DressCodeMonitoring(channel_id, socketio, db_manager, app)
        elif app_name == 'PPEMonitoring':
            module = PPEMonitoring(channel_id, socketio, db_manager, app)
        elif app_name == 'CrowdDetection':
            module = CrowdDetection(channel_id, socketio, db_manager, app)
        else:
            return jsonify({'success': False, 'error': 'Unknown app type'})
        
        # Add module to processor
        processor.add_module(app_name, module)
        channel_modules[channel_id][app_name] = module
        
        # Update config
        if channel_id not in app_configs[app_name]['channels']:
            app_configs[app_name]['channels'][channel_id] = {
                'name': f"Channel {channel_id}",
                'status': 'online',
                'video_source': video_source,
                'source_type': source_type,
                'shared': True,
                'active_modules': []
            }
        
        # Add this module to active modules list
        if app_name not in app_configs[app_name]['channels'][channel_id].get('active_modules', []):
            app_configs[app_name]['channels'][channel_id].setdefault('active_modules', []).append(app_name)
        
        # Update other app configs to show shared status
        for other_app in app_configs:
            if other_app != app_name and channel_id in channel_modules[channel_id]:
                if channel_id not in app_configs[other_app]['channels']:
                    app_configs[other_app]['channels'][channel_id] = {
                        'name': f"Channel {channel_id}",
                        'status': 'online',
                        'video_source': video_source,
                        'source_type': source_type,
                        'shared': True,
                        'active_modules': []
                    }
                # Update active modules for other apps
                current_modules = list(channel_modules[channel_id].keys())
                app_configs[other_app]['channels'][channel_id]['active_modules'] = current_modules
        
        logger.info(f"Started {app_name} on channel {channel_id} ({source_type}: {video_source})")
        return jsonify({
            'success': True, 
            'shared': True, 
            'active_modules': list(channel_modules[channel_id].keys()),
            'source_type': source_type,
            'video_source': video_source
        })
        
    except Exception as e:
        logger.error(f"Error starting channel: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/stop_channel', methods=['POST'])
@login_required
def stop_channel():
    """Stop processing a video channel or remove a module from shared channel"""
    data = request.json
    app_name = data.get('app_name')
    channel_id = data.get('channel_id')
    
    try:
        if channel_id in shared_video_processors:
            processor = shared_video_processors[channel_id]
            
            # Remove the specific module
            if app_name in channel_modules.get(channel_id, {}):
                processor.remove_module(app_name)
                del channel_modules[channel_id][app_name]
                
                # Update app config
                if channel_id in app_configs[app_name]['channels']:
                    app_configs[app_name]['channels'][channel_id]['status'] = 'offline'
                
                # If no modules left, stop the entire processor
                if not channel_modules[channel_id]:
                    processor.stop()
                    del shared_video_processors[channel_id]
                    del channel_modules[channel_id]
                    
                    # Update all app configs
                    for app_config in app_configs.values():
                        if channel_id in app_config['channels']:
                            app_config['channels'][channel_id]['status'] = 'offline'
                else:
                    # Update active modules list for all apps
                    remaining_modules = list(channel_modules[channel_id].keys())
                    for app_config in app_configs.values():
                        if channel_id in app_config['channels']:
                            app_config['channels'][channel_id]['active_modules'] = remaining_modules
        
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error stopping channel: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/set_roi', methods=['POST'])
def set_roi():
    """Set ROI for queue monitoring on shared channel"""
    data = request.json
    app_name = data.get('app_name')
    channel_id = data.get('channel_id')
    roi_points = data.get('roi_points')
    
    try:
        logger.info(f"Setting ROI for {app_name} on channel {channel_id}")
        logger.info(f"ROI Points: {json.dumps(roi_points, indent=2)}")
        
        if channel_id in channel_modules and app_name in channel_modules[channel_id]:
            module = channel_modules[channel_id][app_name]
            if hasattr(module, 'set_roi'):
                module.set_roi(roi_points)
                logger.info(f"✓ ROI successfully set for {app_name} on channel {channel_id}")
                return jsonify({'success': True})
        
        logger.warning(f"Module {app_name} not found on channel {channel_id}")
        return jsonify({'success': False, 'error': 'Module not found on this channel'})
    except Exception as e:
        logger.error(f"Error setting ROI: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/get_roi/<app_name>/<channel_id>')
def get_roi(app_name, channel_id):
    """Get current ROI configuration for queue monitoring"""
    try:
        if channel_id in channel_modules and app_name in channel_modules[channel_id]:
            module = channel_modules[channel_id][app_name]
            if hasattr(module, 'get_roi'):
                roi_config = module.get_roi()
                return jsonify({'success': True, 'roi_config': roi_config})
        
        return jsonify({'success': False, 'error': 'Module not found on this channel'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/set_counting_line', methods=['POST'])
def set_counting_line():
    """Set counting line for people counter on shared channel"""
    data = request.json
    app_name = data.get('app_name')
    channel_id = data.get('channel_id')
    line_config = data.get('line_config')
    
    try:
        logger.info(f"Setting counting line for {app_name} on channel {channel_id}")
        logger.info(f"Counting Line Config: {json.dumps(line_config, indent=2)}")
        
        if channel_id in channel_modules and app_name in channel_modules[channel_id]:
            module = channel_modules[channel_id][app_name]
            if hasattr(module, 'set_counting_line'):
                module.set_counting_line(line_config)
                logger.info(f"✓ Counting line successfully set for {app_name} on channel {channel_id}")
                return jsonify({'success': True})
        
        logger.warning(f"Module {app_name} not found on channel {channel_id}")
        return jsonify({'success': False, 'error': 'Module not found on this channel'})
    except Exception as e:
        logger.error(f"Error setting counting line: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/get_counting_line/<app_name>/<channel_id>')
def get_counting_line(app_name, channel_id):
    """Get current counting line configuration for people counter"""
    try:
        if channel_id in channel_modules and app_name in channel_modules[channel_id]:
            module = channel_modules[channel_id][app_name]
            if hasattr(module, 'get_counting_line'):
                line_config = module.get_counting_line()
                return jsonify({'success': True, 'line_config': line_config})
        
        return jsonify({'success': False, 'error': 'Module not found on this channel'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/set_bag_settings', methods=['POST'])
def set_bag_settings():
    """Set bag detection settings (time threshold, proximity threshold, confidence)"""
    data = request.json
    app_name = 'BagDetection'
    channel_id = data.get('channel_id')
    time_threshold = data.get('time_threshold')
    proximity_threshold = data.get('proximity_threshold')
    confidence = data.get('confidence')
    alert_cooldown = data.get('alert_cooldown')
    
    try:
        if channel_id in channel_modules and app_name in channel_modules[channel_id]:
            module = channel_modules[channel_id][app_name]
            
            # Update module configuration
            if time_threshold is not None:
                module.time_threshold = float(time_threshold)
            if proximity_threshold is not None:
                module.proximity_threshold = float(proximity_threshold)
            if confidence is not None:
                module.confidence = float(confidence)
            if alert_cooldown is not None:
                module.alert_cooldown = float(alert_cooldown)
            
            return jsonify({
                'success': True,
                'message': 'Bag detection settings updated successfully',
                'settings': {
                    'time_threshold': module.time_threshold,
                    'proximity_threshold': module.proximity_threshold,
                    'confidence': module.confidence,
                    'alert_cooldown': module.alert_cooldown
                }
            })
        
        return jsonify({'success': False, 'error': 'Bag detection module not running on this channel'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/get_channel_status/<channel_id>')
def get_channel_status(channel_id):
    """Get status of all modules running on a channel"""
    try:
        if channel_id in shared_video_processors:
            processor = shared_video_processors[channel_id]
            status = processor.get_status()
            
            # Add module-specific information
            module_info = {}
            for module_name, module in channel_modules.get(channel_id, {}).items():
                if hasattr(module, 'get_current_counts'):
                    module_info[module_name] = module.get_current_counts()
                elif hasattr(module, 'get_current_status'):
                    module_info[module_name] = module.get_current_status()
            
            status['module_info'] = module_info
            return jsonify(status)
        else:
            return jsonify({'error': 'Channel not found', 'channel_id': channel_id})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/get_module_analytics/<module_name>')
@login_required
def get_module_analytics(module_name):
    """Get analytics summary for a specific module"""
    try:
        analytics = {}
        
        if module_name == 'PeopleCounter':
            # Get today's IN/OUT counts from database (not in-memory counter)
            total_in = 0
            total_out = 0
            active_channels = []
            daily_data = []
            
            # Get list of active channels
            for channel_id, processor in shared_video_processors.items():
                if processor.is_running and 'PeopleCounter' in processor.get_active_modules():
                    active_channels.append(channel_id)
            
            # Get today's counts from database
            try:
                with app.app_context():
                    today_data = db_manager.get_today_footfall_count()
                    total_in = today_data.get('total_in', 0)
                    total_out = today_data.get('total_out', 0)
            except Exception as e:
                logger.error(f"Error getting today's footfall count: {e}")
                total_in = 0
                total_out = 0
            
            # Get daily footfall data for the last 7 days
            try:
                with app.app_context():
                    # Aggregate data from all active channels
                    from datetime import datetime, timedelta
                    end_date = datetime.now().date()
                    start_date = end_date - timedelta(days=6)
                    
                    # Create a dictionary to aggregate daily counts
                    daily_aggregated = {}
                    
                    for channel_id in active_channels:
                        report = db_manager.get_footfall_report(channel_id, period='7days')
                        for day_data in report.get('data', []):
                            date = day_data['date']
                            if date not in daily_aggregated:
                                daily_aggregated[date] = {'in_count': 0, 'out_count': 0}
                            daily_aggregated[date]['in_count'] += day_data['in_count']
                            daily_aggregated[date]['out_count'] += day_data['out_count']
                    
                    # Convert to sorted list
                    daily_data = [
                        {
                            'date': date,
                            'in_count': counts['in_count'],
                            'out_count': counts['out_count'],
                            'total': counts['in_count'] + counts['out_count']
                        }
                        for date, counts in sorted(daily_aggregated.items())
                    ]
            except Exception as e:
                logger.error(f"Error getting daily footfall data: {e}")
                daily_data = []
            
            # Get peak hour data
            peak_hour_info = {'peak_hour': 'N/A', 'traffic_count': 0}
            try:
                with app.app_context():
                    peak_hour_info = db_manager.get_people_counter_peak_hour()
            except Exception as e:
                logger.error(f"Error getting peak hour data: {e}")
            
            analytics = {
                'module': 'People Counter',
                'total_in': total_in,
                'total_out': total_out,
                'net_count': total_in - total_out,
                'active_channels': len(active_channels),
                'channels': active_channels,
                'daily_data': daily_data,  # Add daily breakdown
                'peak_hour': peak_hour_info.get('peak_hour', 'N/A'),
                'peak_hour_traffic': peak_hour_info.get('traffic_count', 0)
            }
            
        elif module_name == 'QueueMonitor':
            # Get total alerts and current queue stats
            total_alerts = 0
            current_queue_total = 0
            current_counter_total = 0
            active_channels = []
            
            for channel_id, processor in shared_video_processors.items():
                if processor.is_running and 'QueueMonitor' in processor.get_active_modules():
                    active_channels.append(channel_id)
                    if channel_id in channel_modules and 'QueueMonitor' in channel_modules[channel_id]:
                        module = channel_modules[channel_id]['QueueMonitor']
                        status = module.get_status() if hasattr(module, 'get_status') else {}
                        # Get queue and counter counts from status
                        current_queue_total += status.get('queue_count', 0)
                        current_counter_total += status.get('counter_count', 0)
            
            # Get alert count from database (queue_alert type)
            try:
                with app.app_context():
                    alert_count = db_manager.get_alert_count('queue_alert', days=7)
                    total_alerts = alert_count if alert_count is not None else 0
            except Exception as e:
                logger.error(f"Error getting queue alert count: {e}")
                total_alerts = 0
            
            # Get queue violation count from database
            violation_count = 0
            try:
                with app.app_context():
                    violations = db_manager.get_queue_violations(limit=1000)
                    violation_count = len(violations) if violations else 0
            except Exception as e:
                logger.error(f"Error getting queue violations: {e}")
                violation_count = 0
            
            analytics = {
                'module': 'Queue Monitor',
                'total_alerts_7days': total_alerts,
                'total_violations': violation_count,
                'current_queue_count': current_queue_total,
                'current_counter_count': current_counter_total,
                'active_channels': len(active_channels),
                'channels': active_channels
            }
            
        elif module_name == 'BagDetection':
            # Get comprehensive bag detection analytics
            total_alerts = 0
            active_channels = []
            channel_details = []
            current_bags_tracked = 0
            current_unattended = 0
            
            # Get real-time data from active processors
            for channel_id, processor in shared_video_processors.items():
                if processor.is_running and 'BagDetection' in processor.get_active_modules():
                    active_channels.append(channel_id)
                    if channel_id in channel_modules and 'BagDetection' in channel_modules[channel_id]:
                        module = channel_modules[channel_id]['BagDetection']
                        stats = module.get_statistics() if hasattr(module, 'get_statistics') else {}
                        current_bags_tracked += stats.get('bags_tracked', 0)
                        current_unattended += stats.get('current_unattended_bags', 0)
                        
                        channel_details.append({
                            'channel_id': channel_id,
                            'bags_tracked': stats.get('bags_tracked', 0),
                            'active_alerts': stats.get('active_alerts', 0),
                            'total_alerts': stats.get('total_alerts_triggered', 0),
                            'longest_unattended': stats.get('longest_unattended_time', 0),
                            'peak_bags': stats.get('peak_bags_count', 0)
                        })
            
            # Get historical analytics from database
            db_analytics = {}
            try:
                with app.app_context():
                    db_analytics = db_manager.get_bag_detection_analytics(days=7)
                    total_alerts = db_analytics.get('total_alerts', 0)
            except Exception as e:
                logger.error(f"Error getting bag detection analytics: {e}")
            
            # Sort channels by alert count (most risky first)
            channel_details.sort(key=lambda x: x['total_alerts'], reverse=True)
            
            # Identify most risky zone
            most_risky_now = channel_details[0] if channel_details else None
            most_risky_historical = db_analytics.get('most_risky_channel')
            
            analytics = {
                'module': 'Bag Detection',
                'total_alerts_7days': total_alerts,
                'active_channels': len(active_channels),
                'channels': active_channels,
                'current_bags_tracked': current_bags_tracked,
                'current_unattended_bags': current_unattended,
                'channel_details': channel_details,
                'most_risky_now': most_risky_now,
                'historical_data': {
                    'total_alerts': total_alerts,
                    'most_risky_channel': most_risky_historical,
                    'daily_trend': db_analytics.get('daily_trend', []),
                    'channels': db_analytics.get('channels', []),
                    'period_days': 7
                }
            }
            
        elif module_name == 'Heatmap':
            active_channels = []
            total_hotspots = 0
            channel_details = []
            
            # Get real-time data from active processors
            for channel_id, processor in shared_video_processors.items():
                if processor.is_running and 'HeatmapProcessor' in processor.get_active_modules():
                    active_channels.append(channel_id)
                    if channel_id in channel_modules and 'HeatmapProcessor' in channel_modules[channel_id]:
                        module = channel_modules[channel_id]['HeatmapProcessor']
                        status = module.get_status() if hasattr(module, 'get_status') else {}
                        current_hotspots = status.get('hotspot_count', 0)
                        total_hotspots += current_hotspots
                        
                        channel_details.append({
                            'channel_id': channel_id,
                            'current_hotspots': current_hotspots,
                            'peak_hotspots': status.get('peak_hotspot_count', 0),
                            'peak_person_count': status.get('peak_person_count', 0),
                            'active_cells': status.get('active_cells', 0)
                        })
            
            # Sort channels by current hotspots (most crowded first)
            channel_details.sort(key=lambda x: x['current_hotspots'], reverse=True)
            
            # Get historical analytics from database
            db_analytics = {}
            try:
                with app.app_context():
                    db_analytics = db_manager.get_heatmap_analytics(days=7)
            except Exception as e:
                logger.error(f"Error getting heatmap analytics: {e}")
            
            # Identify most crowded zone
            most_crowded_now = channel_details[0] if channel_details else None
            most_crowded_historical = db_analytics.get('most_crowded_channel')
            
            analytics = {
                'module': 'Heatmap',
                'current_hotspots': total_hotspots,
                'active_channels': len(active_channels),
                'channels': active_channels,
                'channel_details': channel_details,
                'most_crowded_now': most_crowded_now,
                'historical_data': {
                    'total_snapshots': db_analytics.get('total_snapshots', 0),
                    'total_hotspots': db_analytics.get('total_hotspots', 0),
                    'most_crowded_channel': most_crowded_historical,
                    'period_days': 7,
                    'channels': db_analytics.get('channels', [])
                }
            }

            
        elif module_name == 'CashDetection':
            # Enhanced cash detection analytics
            active_channels = []
            channel_details = []
            current_detections_total = 0
            
            # Get real-time data from active processors
            for channel_id, processor in shared_video_processors.items():
                if processor.is_running and 'CashDetection' in processor.get_active_modules():
                    active_channels.append(channel_id)
                    if channel_id in channel_modules and 'CashDetection' in channel_modules[channel_id]:
                        module = channel_modules[channel_id]['CashDetection']
                        stats = module.get_statistics() if hasattr(module, 'get_statistics') else {}
                        current_detections_total += stats.get('current_detections', 0)
                        
                        channel_details.append({
                            'channel_id': channel_id,
                            'current_detections': stats.get('current_detections', 0),
                            'total_alerts': stats.get('total_alerts', 0),
                            'total_detections': stats.get('total_detections', 0),
                            'peak_detections': stats.get('peak_detections', 0),
                            'detection_sessions': stats.get('detection_sessions', 0),
                            'avg_confidence': stats.get('avg_confidence', 0),
                            'highest_confidence': stats.get('highest_confidence', 0)
                        })
            
            # Get historical analytics from database
            db_analytics = {}
            try:
                with app.app_context():
                    db_analytics = db_manager.get_cash_detection_analytics(days=7)
            except Exception as e:
                logger.error(f"Error getting cash detection analytics: {e}")
            
            # Sort channels by snapshot count (most active first)
            channel_details.sort(key=lambda x: x['total_detections'], reverse=True)
            
            # Identify most active zone
            most_active_now = channel_details[0] if channel_details else None
            most_active_historical = db_analytics.get('most_active_channel')
            peak_hour = db_analytics.get('peak_hour')
            
            analytics = {
                'module': 'Cash Detection',
                'total_alerts_7days': db_analytics.get('total_snapshots', 0),
                'total_detections_7days': db_analytics.get('total_detections', 0),
                'active_channels': len(active_channels),
                'channels': active_channels,
                'current_detections': current_detections_total,
                'channel_details': channel_details,
                'most_active_now': most_active_now,
                'peak_hour': peak_hour,
                'historical_data': {
                    'total_snapshots': db_analytics.get('total_snapshots', 0),
                    'total_detections': db_analytics.get('total_detections', 0),
                    'most_active_channel': most_active_historical,
                    'daily_trend': db_analytics.get('daily_trend', []),
                    'hourly_distribution': db_analytics.get('hourly_distribution', []),
                    'channels': db_analytics.get('channels', []),
                    'period_days': 7
                }
            }
            
        elif module_name == 'FallDetection':
            active_channels = []
            current_falls = 0
            
            for channel_id, processor in shared_video_processors.items():
                if processor.is_running and 'FallDetection' in processor.get_active_modules():
                    active_channels.append(channel_id)
                    # Get current fall count from module if available
                    if channel_id in channel_modules and 'FallDetection' in channel_modules[channel_id]:
                        module = channel_modules[channel_id]['FallDetection']
                        if hasattr(module, 'current_fall_count'):
                            current_falls += module.current_fall_count
            
            # Get comprehensive analytics from database
            db_analytics = {}
            try:
                with app.app_context():
                    db_analytics = db_manager.get_fall_detection_analytics(days=7)
            except Exception as e:
                logger.error(f"Error getting fall detection analytics: {e}")
            
            # Get channel details
            channel_details = []
            most_risky_now = None
            
            for ch_data in db_analytics.get('channels', []):
                channel_details.append(ch_data)
                if most_risky_now is None or ch_data['fall_count'] > most_risky_now.get('fall_count', 0):
                    most_risky_now = ch_data
            
            analytics = {
                'module': 'Fall Detection',
                'total_falls_7days': db_analytics.get('total_falls', 0),
                'today_falls': db_analytics.get('today_falls', 0),
                'active_channels': len(active_channels),
                'channels': active_channels,
                'current_falls': current_falls,
                'avg_fall_duration': db_analytics.get('avg_fall_duration', 0),
                'max_fall_duration': db_analytics.get('max_fall_duration', 0),
                'peak_hour': db_analytics.get('peak_hour', 'N/A'),
                'peak_hour_count': db_analytics.get('peak_hour_count', 0),
                'channel_details': channel_details,
                'most_risky_channel': most_risky_now,
                'response_categories': db_analytics.get('response_categories', {}),
                'historical_data': {
                    'daily_trend': db_analytics.get('daily_trend', []),
                    'most_incidents_channel': db_analytics.get('most_incidents_channel'),
                    'period_days': 7
                }
            }
        
        elif module_name == 'MoppingDetection':
            # Enhanced mopping detection analytics
            active_channels = []
            channel_details = []
            current_detections_total = 0
            
            # Get real-time data from active processors
            for channel_id, processor in shared_video_processors.items():
                if processor.is_running and 'MoppingDetection' in processor.get_active_modules():
                    active_channels.append(channel_id)
                    if channel_id in channel_modules and 'MoppingDetection' in channel_modules[channel_id]:
                        module = channel_modules[channel_id]['MoppingDetection']
                        current_detections = getattr(module, 'detection_count', 0)
                        current_detections_total += current_detections
                        
                        # Get channel-specific stats from database
                        try:
                            with app.app_context():
                                ch_stats = db_manager.get_mopping_statistics(channel_id=channel_id, days=7)
                                channel_details.append({
                                    'channel_id': channel_id,
                                    'total_alerts': ch_stats.get('total_alerts', 0),
                                    'total_detections': ch_stats.get('total_detections', 0),
                                    'current_detections': current_detections
                                })
                        except Exception as e:
                            logger.error(f"Error getting channel stats for {channel_id}: {e}")
            
            # Get comprehensive analytics from database
            db_stats = {}
            most_active_channel = None
            try:
                with app.app_context():
                    db_stats = db_manager.get_mopping_statistics(days=7)
                    
                    # Find most active channel from channel_details
                    if channel_details:
                        most_active_channel = max(channel_details, key=lambda x: x['total_alerts'])
            except Exception as e:
                logger.error(f"Error getting mopping detection statistics: {e}")
            
            # Sort channels by alert count (most active first)
            channel_details.sort(key=lambda x: x['total_alerts'], reverse=True)
            
            # Get today's alerts
            today_alerts = 0
            try:
                with app.app_context():
                    today_stats = db_manager.get_mopping_statistics(days=1)
                    today_alerts = today_stats.get('total_alerts', 0)
            except Exception as e:
                logger.error(f"Error getting today's mopping stats: {e}")
            
            analytics = {
                'module': 'Mopping Detection',
                'total_alerts_7days': db_stats.get('total_alerts', 0),
                'today_alerts': today_alerts,
                'total_detections_7days': db_stats.get('total_detections', 0),
                'active_channels': len(active_channels),
                'channels': active_channels,
                'current_detections': current_detections_total,
                'channel_details': channel_details,
                'most_active_channel': most_active_channel,
                'peak_hour': 'N/A',  # Can be enhanced later
                'historical_data': {
                    'total_snapshots': db_stats.get('total_alerts', 0),
                    'most_active_channel': most_active_channel,
                    'daily_counts': db_stats.get('daily_counts', {}),
                    'period_days': 7
                }
            }
        
        elif module_name == 'SmokingDetection':
            # Enhanced smoking detection analytics
            active_channels = []
            channel_details = []
            current_detections_total = 0
            
            # Get real-time data from active processors
            for channel_id, processor in shared_video_processors.items():
                if processor.is_running and 'SmokingDetection' in processor.get_active_modules():
                    active_channels.append(channel_id)
                    if channel_id in channel_modules and 'SmokingDetection' in channel_modules[channel_id]:
                        module = channel_modules[channel_id]['SmokingDetection']
                        current_detections = getattr(module, 'detection_count', 0)
                        current_detections_total += current_detections
                        
                        # Get channel-specific stats from database
                        try:
                            with app.app_context():
                                ch_stats = db_manager.get_smoking_statistics(channel_id=channel_id, days=7)
                                channel_details.append({
                                    'channel_id': channel_id,
                                    'total_alerts': ch_stats.get('total_alerts', 0),
                                    'total_detections': ch_stats.get('total_detections', 0),
                                    'current_detections': current_detections
                                })
                        except Exception as e:
                            logger.error(f"Error getting channel stats for {channel_id}: {e}")
            
            # Get comprehensive analytics from database
            db_stats = {}
            most_active_channel = None
            try:
                with app.app_context():
                    db_stats = db_manager.get_smoking_statistics(days=7)
                    
                    # Find most active channel from channel_details
                    if channel_details:
                        most_active_channel = max(channel_details, key=lambda x: x['total_alerts'])
            except Exception as e:
                logger.error(f"Error getting smoking detection statistics: {e}")
            
            # Sort channels by alert count (most active first)
            channel_details.sort(key=lambda x: x['total_alerts'], reverse=True)
            
            # Get today's alerts
            today_alerts = 0
            try:
                with app.app_context():
                    today_stats = db_manager.get_smoking_statistics(days=1)
                    today_alerts = today_stats.get('total_alerts', 0)
            except Exception as e:
                logger.error(f"Error getting today's smoking stats: {e}")
            
            analytics = {
                'module': 'Smoking Detection',
                'total_alerts_7days': db_stats.get('total_alerts', 0),
                'today_alerts': today_alerts,
                'total_detections_7days': db_stats.get('total_detections', 0),
                'active_channels': len(active_channels),
                'channels': active_channels,
                'current_detections': current_detections_total,
                'channel_details': channel_details,
                'most_active_channel': most_active_channel,
                'peak_hour': 'N/A',  # Can be enhanced later
                'historical_data': {
                    'total_snapshots': db_stats.get('total_alerts', 0),
                    'most_active_channel': most_active_channel,
                    'daily_counts': db_stats.get('daily_counts', {}),
                    'period_days': 7
                }
            }
        
        elif module_name == 'PhoneUsageDetection':
            # Enhanced phone usage detection analytics
            active_channels = []
            channel_details = []
            current_detections_total = 0
            
            # Get real-time data from active processors
            for channel_id, processor in shared_video_processors.items():
                if processor.is_running and 'PhoneUsageDetection' in processor.get_active_modules():
                    active_channels.append(channel_id)
                    if channel_id in channel_modules and 'PhoneUsageDetection' in channel_modules[channel_id]:
                        module = channel_modules[channel_id]['PhoneUsageDetection']
                        current_detections = getattr(module, 'detection_count', 0)
                        current_detections_total += current_detections
                        
                        # Get channel-specific stats from database
                        try:
                            with app.app_context():
                                ch_stats = db_manager.get_phone_statistics(channel_id=channel_id, days=7)
                                channel_details.append({
                                    'channel_id': channel_id,
                                    'total_alerts': ch_stats.get('total_alerts', 0),
                                    'total_detections': ch_stats.get('total_detections', 0),
                                    'current_detections': current_detections
                                })
                        except Exception as e:
                            logger.error(f"Error getting channel stats for {channel_id}: {e}")
            
            # Get comprehensive analytics from database
            db_stats = {}
            most_active_channel = None
            try:
                with app.app_context():
                    db_stats = db_manager.get_phone_statistics(days=7)
                    
                    # Find most active channel from channel_details
                    if channel_details:
                        most_active_channel = max(channel_details, key=lambda x: x['total_alerts'])
            except Exception as e:
                logger.error(f"Error getting phone usage detection statistics: {e}")
            
            # Sort channels by alert count (most active first)
            channel_details.sort(key=lambda x: x['total_alerts'], reverse=True)
            
            # Get today's alerts
            today_alerts = 0
            try:
                with app.app_context():
                    today_stats = db_manager.get_phone_statistics(days=1)
                    today_alerts = today_stats.get('total_alerts', 0)
            except Exception as e:
                logger.error(f"Error getting today's phone stats: {e}")
            
            analytics = {
                'module': 'Phone Usage Detection',
                'total_alerts_7days': db_stats.get('total_alerts', 0),
                'today_alerts': today_alerts,
                'total_detections_7days': db_stats.get('total_detections', 0),
                'active_channels': len(active_channels),
                'channels': active_channels,
                'current_detections': current_detections_total,
                'channel_details': channel_details,
                'most_active_channel': most_active_channel,
                'peak_hour': 'N/A',  # Can be enhanced later
                'historical_data': {
                    'total_snapshots': db_stats.get('total_alerts', 0),
                    'most_active_channel': most_active_channel,
                    'daily_counts': db_stats.get('daily_counts', {}),
                    'period_days': 7
                }
            }
        
        elif module_name == 'RestrictedAreaMonitor':
            # Restricted Area Monitor analytics
            active_channels = []
            channel_details = []
            current_violations_total = 0
            
            # Get real-time data from active processors
            for channel_id, processor in shared_video_processors.items():
                if processor.is_running and 'RestrictedAreaMonitor' in processor.get_active_modules():
                    active_channels.append(channel_id)
                    if channel_id in channel_modules and 'RestrictedAreaMonitor' in channel_modules[channel_id]:
                        module = channel_modules[channel_id]['RestrictedAreaMonitor']
                        current_violations = module.stats.get('total_violations', 0)
                        current_violations_total += current_violations
                        
                        # Get channel-specific stats from database
                        try:
                            with app.app_context():
                                ch_stats = db_manager.get_restricted_area_statistics(channel_id=channel_id, days=7)
                                channel_details.append({
                                    'channel_id': channel_id,
                                    'total_alerts': ch_stats.get('total_alerts', 0),
                                    'total_violations': ch_stats.get('total_violations', 0),
                                    'roi_defined': len(module.roi_points) >= 3
                                })
                        except Exception as e:
                            logger.error(f"Error getting restricted area stats for {channel_id}: {e}")
            
            # Get aggregated database statistics
            db_stats = {}
            most_active_channel = None
            try:
                with app.app_context():
                    db_stats = db_manager.get_restricted_area_statistics(days=7)
                    
                    # Find most active channel from channel_details
                    if channel_details:
                        most_active_channel = max(channel_details, key=lambda x: x['total_alerts'])
            except Exception as e:
                logger.error(f"Error getting restricted area statistics: {e}")
            
            # Sort channels by alert count (most active first)
            channel_details.sort(key=lambda x: x['total_alerts'], reverse=True)
            
            # Get today's alerts
            today_alerts = 0
            try:
                with app.app_context():
                    today_stats = db_manager.get_restricted_area_statistics(days=1)
                    today_alerts = today_stats.get('total_alerts', 0)
            except Exception as e:
                logger.error(f"Error getting today's restricted area stats: {e}")
            
            analytics = {
                'module': 'Restricted Area Monitor',
                'total_alerts_7days': db_stats.get('total_alerts', 0),
                'today_alerts': today_alerts,
                'total_violations_7days': db_stats.get('total_violations', 0),
                'active_channels': len(active_channels),
                'channels': active_channels,
                'current_violations': current_violations_total,
                'channel_details': channel_details,
                'most_active_channel': most_active_channel,
                'historical_data': {
                    'total_snapshots': db_stats.get('total_alerts', 0),
                    'most_active_channel': most_active_channel,
                    'daily_counts': db_stats.get('daily_counts', {}),
                    'period_days': 7
                }
            }
        
        elif module_name == 'PPEMonitoring':
            try:
                with app.app_context():
                    total_alerts = db_manager.get_alert_count('ppe_alert', days=365)  # All time (last year)
                    today_alerts = db_manager.get_alert_count('ppe_alert', days=1)
                    
                    # Get active channels - check both running processors and configured modules
                    active_channels = []
                    total_violations = 0
                    
                    # First check running processors
                    for channel_id, processor in shared_video_processors.items():
                        if processor.is_running and 'PPEMonitoring' in processor.get_active_modules():
                            active_channels.append(channel_id)
                            # Get stats from module if available
                            if channel_id in channel_modules and 'PPEMonitoring' in channel_modules[channel_id]:
                                module = channel_modules[channel_id]['PPEMonitoring']
                                if hasattr(module, 'total_violations'):
                                    total_violations += module.total_violations
                    
                    # Also check configured modules (even if processor not running yet)
                    for channel_id, modules_dict in channel_modules.items():
                        if 'PPEMonitoring' in modules_dict and channel_id not in active_channels:
                            # Module is configured but processor might not be running
                            processor = shared_video_processors.get(channel_id)
                            if processor and processor.is_running:
                                # Should have been caught above, but double-check
                                if channel_id not in active_channels:
                                    active_channels.append(channel_id)
                            # Even if not running, we can still show it as configured
                            # (This helps show channels that are set up but not yet started)
                    
                    analytics = {
                        'module': 'PPE Compliance',
                        'total_alerts': total_alerts,
                        'today_alerts': today_alerts,
                        'active_channels': len(active_channels),
                        'total_violations': total_violations,
                        'channels': [{'channel_id': ch_id, 'status': 'active'} for ch_id in active_channels]
                    }
                    
                    return jsonify({'success': True, 'analytics': analytics})
            except Exception as e:
                logger.error(f"Error getting PPE analytics: {e}", exc_info=True)
                return jsonify({'success': False, 'error': str(e)})
        
        elif module_name == 'DressCodeMonitoring':
            active_channels = []
            total_violations = 0
            
            for channel_id, processor in shared_video_processors.items():
                if processor.is_running and 'DressCodeMonitoring' in processor.get_active_modules():
                    active_channels.append(channel_id)
                    # Get stats from module if available
                    if channel_id in channel_modules and 'DressCodeMonitoring' in channel_modules[channel_id]:
                        module = channel_modules[channel_id]['DressCodeMonitoring']
                        if hasattr(module, 'total_violations'):
                            total_violations += module.total_violations
            
            # Get comprehensive statistics from database
            db_stats = {}
            try:
                with app.app_context():
                    db_stats = db_manager.get_dresscode_stats(days=7)
            except Exception as e:
                logger.error(f"Error getting dress code stats: {e}")
            
            analytics = {
                'module': 'Dress Code Monitoring',
                'total_violations': db_stats.get('total_violations', 0),
                'violation_types': db_stats.get('violation_types', {}),
                'uniform_colors': db_stats.get('uniform_colors', {}),
                'active_channels': len(active_channels),
                'channels': active_channels,
                'current_violations': total_violations,
                'period_days': 7
            }
            
        elif module_name == 'GroomingDetection':
            active_channels = []
            total_alerts = 0
            
            for channel_id, processor in shared_video_processors.items():
                if processor.is_running and module_name in processor.get_active_modules():
                    active_channels.append(channel_id)
        
        elif module_name == 'CrowdDetection':
            active_channels = []
            current_crowd_total = 0
            total_alerts = 0
            
            for channel_id, processor in shared_video_processors.items():
                if processor.is_running and 'CrowdDetection' in processor.get_active_modules():
                    active_channels.append(channel_id)
                    if channel_id in channel_modules and 'CrowdDetection' in channel_modules[channel_id]:
                        module = channel_modules[channel_id]['CrowdDetection']
                        status = module.get_status() if hasattr(module, 'get_status') else {}
                        current_crowd_total += status.get('crowd_count', 0)
            
            # Get alert count from database
            try:
                with app.app_context():
                    alert_count = db_manager.get_alert_count('crowd_alert', days=7)
                    total_alerts = alert_count
            except:
                total_alerts = 0
            
            analytics = {
                'module': 'Crowd Detection',
                'total_alerts_7days': total_alerts,
                'current_crowd_count': current_crowd_total,
                'crowd_threshold': 5,  # Default threshold
                'active_channels': len(active_channels),
                'channels': active_channels
            }
        
        else:
            analytics = {
                'module': module_name,
                'error': 'Module not found'
            }
        
        return jsonify({'success': True, 'analytics': analytics})
    
    except Exception as e:
        logger.error(f"Error getting module analytics: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/get_fo otfall_report/<channel_id>')
def get_footfall_report(channel_id):
    """Get footfall report for a channel"""
    period = request.args.get('period', '7days')
    try:
        report_data = db_manager.get_footfall_report(channel_id, period)
        return jsonify(report_data)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/get_queue_report/<channel_id>')
def get_queue_report(channel_id):
    """Get queue analytics report"""
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    try:
        report_data = db_manager.get_queue_report(channel_id, start_date, end_date)
        return jsonify(report_data)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/get_alert_gifs')
def get_alert_gifs():
    """Get alert GIFs with optional filtering"""
    channel_id = request.args.get('channel_id')
    alert_type = request.args.get('alert_type')
    limit = int(request.args.get('limit', 20))
    
    try:
        alert_gifs = db_manager.get_alert_gifs(channel_id, alert_type, limit)
        return jsonify({
            'success': True,
            'alert_gifs': alert_gifs,
            'count': len(alert_gifs)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/delete_alert_gif/<int:gif_id>', methods=['DELETE'])
def delete_alert_gif(gif_id):
    """Delete an alert GIF"""
    try:
        success = db_manager.delete_alert_gif(gif_id)
        if success:
            return jsonify({'success': True, 'message': 'Alert GIF deleted successfully'})
        else:
            return jsonify({'success': False, 'error': 'Alert GIF not found'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/clear_old_alerts', methods=['POST'])
def clear_old_alerts():
    """Clear old queue monitor and bag detection alert GIFs"""
    data = request.json
    days = data.get('days', 7)
    alert_type = data.get('alert_type', 'all')  # 'queue_alert', 'bag_unattended', or 'all'
    
    try:
        with app.app_context():
            deleted_count = db_manager.cleanup_old_alert_gifs(max_age_days=days, alert_type=alert_type)
        
        return jsonify({
            'success': True,
            'deleted_count': deleted_count,
            'message': f'Deleted {deleted_count} alerts older than {days} days'
        })
    except Exception as e:
        logger.error(f"Error clearing old alerts: {e}")
        return jsonify({'success': False, 'error': str(e)})

# Heatmap Routes
@app.route('/api/get_heatmap_snapshots')
def get_heatmap_snapshots():
    """Get heatmap snapshots with optional filtering"""
    channel_id = request.args.get('channel_id')
    limit = int(request.args.get('limit', 20))
    
    try:
        snapshots = db_manager.get_heatmap_snapshots(channel_id, limit)
        return jsonify({
            'success': True,
            'snapshots': snapshots,
            'count': len(snapshots)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/delete_heatmap_snapshot/<int:snapshot_id>', methods=['DELETE'])
def delete_heatmap_snapshot(snapshot_id):
    """Delete a heatmap snapshot"""
    try:
        success = db_manager.delete_heatmap_snapshot(snapshot_id)
        if success:
            return jsonify({'success': True, 'message': 'Heatmap snapshot deleted successfully'})
        else:
            return jsonify({'success': False, 'error': 'Snapshot not found'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/capture_heatmap_snapshot', methods=['POST'])
def capture_heatmap_snapshot():
    """Manually capture a heatmap snapshot"""
    data = request.json
    channel_id = data.get('channel_id')
    
    if not channel_id:
        return jsonify({'success': False, 'error': 'Channel ID is required'})
    
    try:
        # Check if Heatmap module is running for this channel
        if channel_id not in channel_modules or 'Heatmap' not in channel_modules[channel_id]:
            return jsonify({'success': False, 'error': 'Heatmap not running on this channel'})
        
        heatmap_module = channel_modules[channel_id]['Heatmap']
        
        # Get snapshot frame from the module
        snapshot_frame = heatmap_module.get_snapshot_frame()
        
        if snapshot_frame is None:
            return jsonify({'success': False, 'error': 'Failed to capture snapshot'})
        
        # Save the snapshot
        import cv2
        import os
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"heatmap_{channel_id}_{timestamp}.jpg"
        filepath = os.path.join('static/heatmaps', filename)
        
        # Ensure directory exists
        os.makedirs('static/heatmaps', exist_ok=True)
        
        # Save image
        cv2.imwrite(filepath, snapshot_frame)
        
        # Get file size
        file_size = os.path.getsize(filepath)
        
        # Save to database
        with app.app_context():
            snapshot_id = db_manager.save_heatmap_snapshot(
                channel_id=channel_id,
                filename=filename,
                filepath=filepath,
                file_size=file_size
            )
        
        return jsonify({
            'success': True,
            'message': 'Snapshot captured successfully',
            'snapshot_id': snapshot_id,
            'filename': filename
        })
        
    except Exception as e:
        logger.error(f"Error capturing heatmap snapshot: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/clear_old_heatmap_snapshots', methods=['POST'])
def clear_old_heatmap_snapshots():
    """Clear old heatmap snapshots"""
    data = request.json
    days = data.get('days', 7)
    
    try:
        with app.app_context():
            deleted_count = db_manager.clear_old_heatmap_snapshots(days)
        
        return jsonify({
            'success': True,
            'deleted_count': deleted_count,
            'message': f'Deleted {deleted_count} snapshots older than {days} days'
        })
    except Exception as e:
        logger.error(f"Error clearing old heatmap snapshots: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/set_heatmap_settings', methods=['POST'])
def set_heatmap_settings():
    """Set heatmap settings (decay rate, snapshot interval, intensity)"""
    data = request.json
    channel_id = data.get('channel_id')
    decay_rate = data.get('decay_rate')
    snapshot_interval = data.get('snapshot_interval')
    intensity = data.get('intensity')
    
    if not channel_id:
        return jsonify({'success': False, 'error': 'Channel ID is required'})
    
    try:
        if channel_id not in channel_modules or 'Heatmap' not in channel_modules[channel_id]:
            return jsonify({'success': False, 'error': 'Heatmap not running on this channel'})
        
        heatmap_module = channel_modules[channel_id]['Heatmap']
        
        # Update module settings
        if decay_rate is not None:
            heatmap_module.decay_rate = float(decay_rate)
        if snapshot_interval is not None:
            heatmap_module.snapshot_interval = int(snapshot_interval)
        if intensity is not None:
            heatmap_module.intensity = float(intensity)
        
        return jsonify({
            'success': True,
            'message': 'Heatmap settings updated successfully',
            'settings': {
                'decay_rate': heatmap_module.decay_rate,
                'snapshot_interval': heatmap_module.snapshot_interval,
                'intensity': heatmap_module.intensity
            }
        })
        
    except Exception as e:
        logger.error(f"Error setting heatmap settings: {e}")
        return jsonify({'success': False, 'error': str(e)})

# Configuration Management Routes
@app.route('/api/save_config', methods=['POST'])
def save_config():
    """Save channel configuration (ROI, counting line, etc.)"""
    try:
        data = request.json
        channel_id = data.get('channel_id')
        app_name = data.get('app_name')  # 'PeopleCounter' or 'QueueMonitor'
        config_type = data.get('config_type')  # 'roi' or 'counting_line'
        config_data = data.get('config_data')
        
        if not all([channel_id, app_name, config_type, config_data]):
            return jsonify({'success': False, 'error': 'Missing required fields'})
        
        logger.info(f"💾 Saving configuration: {channel_id} - {app_name} - {config_type}")
        logger.info(f"Configuration Data:\n{json.dumps(config_data, indent=2)}")
        
        db_manager.save_channel_config(channel_id, app_name, config_type, config_data)
        logger.info(f"✓ Configuration saved successfully to database")
        
        return jsonify({'success': True, 'message': 'Configuration saved successfully'})
    except Exception as e:
        logger.error(f"Error saving configuration: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/get_config/<channel_id>/<app_name>/<config_type>')
def get_config(channel_id, app_name, config_type):
    """Get channel configuration"""
    try:
        config_data = db_manager.get_channel_config(channel_id, app_name, config_type)
        if config_data:
            return jsonify({'success': True, 'config': config_data})
        else:
            return jsonify({'success': False, 'message': 'No configuration found'})
    except Exception as e:
        logger.error(f"Error retrieving configuration: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/get_all_configs/<channel_id>/<app_name>')
def get_all_configs(channel_id, app_name):
    """Get all configurations for a channel and app"""
    try:
        roi_config = db_manager.get_channel_config(channel_id, app_name, 'roi')
        line_config = db_manager.get_channel_config(channel_id, app_name, 'counting_line')
        
        return jsonify({
            'success': True,
            'configs': {
                'roi': roi_config,
                'counting_line': line_config
            }
        })
    except Exception as e:
        logger.error(f"Error retrieving configurations: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/update_bag_detection_config', methods=['POST'])
def update_bag_detection_config():
    """Update bag detection configuration"""
    try:
        data = request.json
        channel_id = data.get('channel_id')
        config = data.get('config', {})
        
        if not channel_id:
            return jsonify({'success': False, 'error': 'Channel ID required'})
        
        # Update the active module if it exists
        if channel_id in channel_modules and 'BagDetection' in channel_modules[channel_id]:
            module = channel_modules[channel_id]['BagDetection']
            module.update_config(config)
            return jsonify({'success': True, 'message': 'Configuration updated'})
        else:
            return jsonify({'success': False, 'error': 'Bag detection not running on this channel'})
            
    except Exception as e:
        logger.error(f"Error updating bag detection config: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/get_bag_detection_stats/<channel_id>')
def get_bag_detection_stats(channel_id):
    """Get bag detection statistics"""
    try:
        if channel_id in channel_modules and 'BagDetection' in channel_modules[channel_id]:
            module = channel_modules[channel_id]['BagDetection']
            stats = module.get_statistics()
            return jsonify({'success': True, 'stats': stats})
        else:
            return jsonify({'success': False, 'error': 'Bag detection not running'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# Cash Detection Routes
@app.route('/static/cash_snapshots/<filename>')
def serve_cash_snapshot(filename):
    """Serve cash detection snapshot files"""
    return send_from_directory('static/cash_snapshots', filename)

@app.route('/api/get_cash_snapshots')
def get_cash_snapshots():
    """Get cash detection snapshots with optional filtering"""
    channel_id = request.args.get('channel_id')
    limit = int(request.args.get('limit', 50))
    
    try:
        snapshots = db_manager.get_cash_snapshots(channel_id, limit)
        return jsonify({
            'success': True,
            'snapshots': snapshots,
            'count': len(snapshots)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/delete_cash_snapshot/<int:snapshot_id>', methods=['DELETE'])
def delete_cash_snapshot(snapshot_id):
    """Delete a cash detection snapshot"""
    try:
        success = db_manager.delete_cash_snapshot(snapshot_id)
        if success:
            return jsonify({'success': True, 'message': 'Cash snapshot deleted successfully'})
        else:
            return jsonify({'success': False, 'error': 'Snapshot not found'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/clear_old_cash_snapshots', methods=['POST'])
def clear_old_cash_snapshots():
    """Clear old cash detection snapshots"""
    data = request.json
    days = data.get('days', 7)
    
    try:
        with app.app_context():
            deleted_count = db_manager.clear_old_cash_snapshots(days)
        
        return jsonify({
            'success': True,
            'deleted_count': deleted_count,
            'message': f'Deleted {deleted_count} snapshots older than {days} days'
        })
    except Exception as e:
        logger.error(f"Error clearing old cash snapshots: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/update_cash_detection_config', methods=['POST'])
def update_cash_detection_config():
    """Update cash detection configuration"""
    try:
        data = request.json
        channel_id = data.get('channel_id')
        config = data.get('config', {})
        
        if not channel_id:
            return jsonify({'success': False, 'error': 'Channel ID required'})
        
        # Update the active module if it exists
        if channel_id in channel_modules and 'CashDetection' in channel_modules[channel_id]:
            module = channel_modules[channel_id]['CashDetection']
            module.update_config(config)
            return jsonify({'success': True, 'message': 'Configuration updated'})
        else:
            return jsonify({'success': False, 'error': 'Cash detection not running on this channel'})
            
    except Exception as e:
        logger.error(f"Error updating cash detection config: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/get_cash_detection_stats/<channel_id>')
def get_cash_detection_stats(channel_id):
    """Get cash detection statistics"""
    try:
        if channel_id in channel_modules and 'CashDetection' in channel_modules[channel_id]:
            module = channel_modules[channel_id]['CashDetection']
            stats = module.get_statistics()
            return jsonify({'success': True, 'stats': stats})
        else:
            return jsonify({'success': False, 'error': 'Cash detection not running'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# Fall Detection routes
@app.route('/static/fall_snapshots/<filename>')
def fall_snapshot(filename):
    """Serve fall detection snapshot images"""
    return send_from_directory('static/fall_snapshots', filename)

@app.route('/api/get_fall_snapshots')
def get_fall_snapshots():
    """Get fall detection snapshots"""
    try:
        channel_id = request.args.get('channel_id')
        limit = int(request.args.get('limit', 50))
        
        snapshots = db_manager.get_fall_snapshots(channel_id=channel_id, limit=limit)
        
        return jsonify({
            'success': True,
            'snapshots': snapshots
        })
    except Exception as e:
        logger.error(f"Error getting fall snapshots: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/delete_fall_snapshot/<int:snapshot_id>', methods=['DELETE'])
def delete_fall_snapshot(snapshot_id):
    """Delete a fall detection snapshot"""
    try:
        success = db_manager.delete_fall_snapshot(snapshot_id)
        if success:
            return jsonify({'success': True, 'message': 'Snapshot deleted successfully'})
        else:
            return jsonify({'success': False, 'error': 'Snapshot not found'})
    except Exception as e:
        logger.error(f"Error deleting fall snapshot: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/clear_old_fall_snapshots', methods=['POST'])
def clear_old_fall_snapshots():
    """Clear old fall detection snapshots"""
    try:
        data = request.json or {}
        days = int(data.get('days', 7))
        
        deleted_count = db_manager.clear_old_fall_snapshots(days)
        
        return jsonify({
            'success': True,
            'deleted_count': deleted_count,
            'message': f'Deleted {deleted_count} snapshots older than {days} days'
        })
    except Exception as e:
        logger.error(f"Error clearing old fall snapshots: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/update_fall_detection_config', methods=['POST'])
def update_fall_detection_config():
    """Update fall detection configuration"""
    try:
        data = request.json
        channel_id = data.get('channel_id')
        config = data.get('config', {})
        
        if not channel_id:
            return jsonify({'success': False, 'error': 'Channel ID required'})
        
        # Update the active module if it exists
        if channel_id in channel_modules and 'FallDetection' in channel_modules[channel_id]:
            module = channel_modules[channel_id]['FallDetection']
            module.update_config(config)
            return jsonify({'success': True, 'message': 'Configuration updated'})
        else:
            return jsonify({'success': False, 'error': 'Fall detection not running on this channel'})
            
    except Exception as e:
        logger.error(f"Error updating fall detection config: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/get_fall_detection_stats/<channel_id>')
def get_fall_detection_stats(channel_id):
    """Get fall detection statistics"""
    try:
        if channel_id in channel_modules and 'FallDetection' in channel_modules[channel_id]:
            module = channel_modules[channel_id]['FallDetection']
            stats = module.get_statistics()
            return jsonify({'success': True, 'stats': stats})
        else:
            return jsonify({'success': False, 'error': 'Fall detection not running'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# ============= Mopping Detection Endpoints =============
@app.route('/static/mopping_snapshots/<filename>')
def serve_mopping_snapshot(filename):
    """Serve mopping snapshot images"""
    return send_from_directory('static/mopping_snapshots', filename)

@app.route('/api/get_mopping_snapshots')
def get_mopping_snapshots():
    """Get mopping detection snapshots from database"""
    try:
        channel_id = request.args.get('channel_id')
        limit = int(request.args.get('limit', 50))
        offset = int(request.args.get('offset', 0))
        
        snapshots = db_manager.get_mopping_snapshots(channel_id=channel_id, limit=limit, offset=offset)
        return jsonify({'success': True, 'snapshots': snapshots})
    except Exception as e:
        logger.error(f"Error getting mopping snapshots: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/delete_mopping_snapshot/<int:snapshot_id>', methods=['DELETE'])
def delete_mopping_snapshot(snapshot_id):
    """Delete a mopping snapshot"""
    try:
        logger.info(f"Attempting to delete mopping snapshot ID: {snapshot_id}")
        success = db_manager.delete_mopping_snapshot(snapshot_id)
        if success:
            logger.info(f"Successfully deleted mopping snapshot ID: {snapshot_id}")
            return jsonify({'success': True, 'message': 'Snapshot deleted'})
        else:
            logger.warning(f"Mopping snapshot ID {snapshot_id} not found in database")
            return jsonify({'success': False, 'error': 'Snapshot not found'})
    except Exception as e:
        logger.error(f"Error deleting mopping snapshot {snapshot_id}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/clear_old_mopping_snapshots', methods=['POST'])
def clear_old_mopping_snapshots():
    """Clear old mopping snapshots older than specified days"""
    try:
        data = request.json or {}
        days = data.get('days', 7)
        deleted_count = db_manager.clear_old_mopping_snapshots(days)
        return jsonify({
            'success': True, 
            'message': f'Cleared {deleted_count} snapshots older than {days} days',
            'deleted_count': deleted_count
        })
    except Exception as e:
        logger.error(f"Error clearing old mopping snapshots: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/update_mopping_detection_config', methods=['POST'])
def update_mopping_detection_config():
    """Update mopping detection configuration"""
    try:
        data = request.json
        channel_id = data.get('channel_id')
        
        if channel_id in channel_modules and 'MoppingDetection' in channel_modules[channel_id]:
            module = channel_modules[channel_id]['MoppingDetection']
            module.update_config(data)
            return jsonify({'success': True, 'message': 'Configuration updated'})
        else:
            return jsonify({'success': False, 'error': 'Mopping detection not running'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/get_mopping_detection_stats/<channel_id>')
def get_mopping_detection_stats(channel_id):
    """Get mopping detection statistics"""
    try:
        if channel_id in channel_modules and 'MoppingDetection' in channel_modules[channel_id]:
            module = channel_modules[channel_id]['MoppingDetection']
            stats = module.get_statistics()
            return jsonify({'success': True, 'stats': stats})
        else:
            return jsonify({'success': False, 'error': 'Mopping detection not running'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# ============= Smoking Detection Routes =============
@app.route('/static/smoking_snapshots/<filename>')
def serve_smoking_snapshot(filename):
    """Serve smoking detection snapshot images"""
    return send_from_directory('static/smoking_snapshots', filename)

@app.route('/api/get_smoking_snapshots')
def get_smoking_snapshots():
    """Get all smoking detection snapshots"""
    try:
        channel_id = request.args.get('channel_id')
        limit = int(request.args.get('limit', 50))
        
        snapshots = db_manager.get_smoking_snapshots(channel_id=channel_id, limit=limit)
        return jsonify({'success': True, 'snapshots': snapshots})
    except Exception as e:
        logger.error(f"Error getting smoking snapshots: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/delete_smoking_snapshot/<int:snapshot_id>', methods=['DELETE'])
def delete_smoking_snapshot(snapshot_id):
    """Delete a smoking detection snapshot"""
    try:
        # Implementation would delete from DB and file system
        return jsonify({'success': True, 'message': 'Snapshot deleted'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/clear_old_smoking_snapshots', methods=['POST'])
def clear_old_smoking_snapshots():
    """Clear old smoking detection snapshots"""
    try:
        days = request.json.get('days', 7)
        # Implementation would delete old snapshots
        return jsonify({'success': True, 'message': f'Cleared snapshots older than {days} days'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/update_smoking_detection_config', methods=['POST'])
def update_smoking_detection_config():
    """Update smoking detection configuration"""
    try:
        data = request.json
        channel_id = data.get('channel_id')
        config = data.get('config', {})
        
        # Update config
        return jsonify({'success': True, 'message': 'Configuration updated'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/get_smoking_detection_stats/<channel_id>')
def get_smoking_detection_stats(channel_id):
    """Get smoking detection statistics"""
    try:
        if channel_id in channel_modules and 'SmokingDetection' in channel_modules[channel_id]:
            module = channel_modules[channel_id]['SmokingDetection']
            stats = module.get_statistics()
            return jsonify({'success': True, 'stats': stats})
        else:
            return jsonify({'success': False, 'error': 'Smoking detection not running'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ==================== Phone Usage Detection Routes ====================

@app.route('/static/phone_snapshots/<filename>')
def serve_phone_snapshot(filename):
    """Serve phone usage detection snapshot images"""
    return send_from_directory('static/phone_snapshots', filename)

@app.route('/api/get_phone_snapshots')
def get_phone_snapshots():
    """Get all phone usage detection snapshots"""
    try:
        channel_id = request.args.get('channel_id')
        limit = int(request.args.get('limit', 50))
        
        snapshots = db_manager.get_phone_snapshots(channel_id=channel_id, limit=limit)
        return jsonify({'success': True, 'snapshots': snapshots})
    except Exception as e:
        logger.error(f"Error getting phone snapshots: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/delete_phone_snapshot/<int:snapshot_id>', methods=['DELETE'])
def delete_phone_snapshot(snapshot_id):
    """Delete a phone usage detection snapshot"""
    try:
        success = db_manager.delete_phone_snapshot(snapshot_id)
        if success:
            return jsonify({'success': True, 'message': 'Snapshot deleted'})
        else:
            return jsonify({'success': False, 'error': 'Snapshot not found'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/clear_old_phone_snapshots', methods=['POST'])
def clear_old_phone_snapshots():
    """Clear old phone usage detection snapshots"""
    try:
        days = request.json.get('days', 7)
        count = db_manager.clear_old_phone_snapshots(days=days)
        return jsonify({'success': True, 'message': f'Cleared {count} snapshots older than {days} days'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/get_phone_detection_stats/<channel_id>')
def get_phone_detection_stats(channel_id):
    """Get phone usage detection statistics"""
    try:
        if channel_id in channel_modules and 'PhoneUsageDetection' in channel_modules[channel_id]:
            module = channel_modules[channel_id]['PhoneUsageDetection']
            stats = module.get_statistics()
            return jsonify({'success': True, 'stats': stats})
        else:
            return jsonify({'success': False, 'error': 'Phone detection not running'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# Restricted Area Monitor API Routes
@app.route('/api/get_restricted_area_snapshots')
def get_restricted_area_snapshots():
    """Get restricted area violation snapshots"""
    try:
        channel_id = request.args.get('channel_id')
        limit = int(request.args.get('limit', 50))
        offset = int(request.args.get('offset', 0))
        
        snapshots = db_manager.get_restricted_area_snapshots(channel_id, limit, offset)
        return jsonify({'success': True, 'snapshots': snapshots})
    except Exception as e:
        logger.error(f"Error getting restricted area snapshots: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/delete_restricted_area_snapshot/<int:snapshot_id>', methods=['DELETE'])
def delete_restricted_area_snapshot(snapshot_id):
    """Delete a restricted area snapshot"""
    try:
        logger.info(f"Attempting to delete restricted area snapshot ID: {snapshot_id}")
        success = db_manager.delete_restricted_area_snapshot(snapshot_id)
        if success:
            logger.info(f"Successfully deleted restricted area snapshot ID: {snapshot_id}")
            return jsonify({'success': True, 'message': 'Snapshot deleted'})
        else:
            logger.warning(f"Restricted area snapshot ID {snapshot_id} not found in database")
            return jsonify({'success': False, 'error': 'Snapshot not found'})
    except Exception as e:
        logger.error(f"Error deleting restricted area snapshot {snapshot_id}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/clear_old_restricted_area_snapshots', methods=['POST'])
def clear_old_restricted_area_snapshots():
    """Clear old restricted area snapshots"""
    try:
        data = request.json or {}
        days = data.get('days', 7)
        deleted_count = db_manager.clear_old_restricted_area_snapshots(days)
        return jsonify({
            'success': True,
            'message': f'Cleared {deleted_count} snapshots older than {days} days',
            'deleted_count': deleted_count
        })
    except Exception as e:
        logger.error(f"Error clearing old restricted area snapshots: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/static/restricted_area_snapshots/<filename>')
def serve_restricted_area_snapshot(filename):
    """Serve restricted area snapshot file"""
    return send_from_directory('static/restricted_area_snapshots', filename)

@app.route('/api/get_restricted_area_stats/<channel_id>')
def get_restricted_area_stats(channel_id):
    """Get restricted area monitoring statistics for a channel"""
    try:
        if channel_id in channel_modules and 'RestrictedAreaMonitor' in channel_modules[channel_id]:
            module = channel_modules[channel_id]['RestrictedAreaMonitor']
            stats = module.get_statistics()
            return jsonify({'success': True, 'stats': stats})
        else:
            return jsonify({'success': False, 'error': 'Restricted area monitor not running'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/set_restricted_area_roi/<channel_id>', methods=['POST'])
def set_restricted_area_roi(channel_id):
    """Set ROI points for restricted area monitor"""
    try:
        data = request.json
        roi_points = data.get('roi_points', [])
        
        if channel_id in channel_modules and 'RestrictedAreaMonitor' in channel_modules[channel_id]:
            module = channel_modules[channel_id]['RestrictedAreaMonitor']
            module.set_roi_points(roi_points)
            
            # Save to database using the generic config method
            db_manager.save_channel_config(channel_id, 'RestrictedAreaMonitor', 'roi', roi_points)
            logger.info(f"✓ ROI points saved to database for RestrictedAreaMonitor on channel {channel_id}")
            
            return jsonify({'success': True, 'message': 'ROI points saved'})
        else:
            return jsonify({'success': False, 'error': 'Restricted area monitor not running'})
    except Exception as e:
        logger.error(f"Error setting restricted area ROI: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/get_restricted_area_roi/<channel_id>')
def get_restricted_area_roi(channel_id):
    """Get ROI points for restricted area monitor"""
    try:
        if channel_id in channel_modules and 'RestrictedAreaMonitor' in channel_modules[channel_id]:
            module = channel_modules[channel_id]['RestrictedAreaMonitor']
            roi_points = module.get_roi_points()
            return jsonify({'success': True, 'roi_points': roi_points})
        else:
            return jsonify({'success': False, 'error': 'Restricted area monitor not running'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/model_stats')
@login_required
def get_model_stats_api():
    """Get global model manager statistics"""
    try:
        stats = get_model_stats()
        return jsonify({'success': True, 'stats': stats})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/cleanup_models', methods=['POST'])
@login_required
def cleanup_models_api():
    """Clean up unused models"""
    try:
        data = request.json
        max_age = data.get('max_age_seconds', 3600)  # Default 1 hour
        
        cleaned_count = cleanup_models(max_age)
        return jsonify({
            'success': True, 
            'message': f'Cleaned up {cleaned_count} unused models',
            'cleaned_count': cleaned_count
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/get_grooming_detection_stats/<channel_id>')
def get_grooming_detection_stats(channel_id):
    """Get grooming detection statistics"""
    try:
        if channel_id in channel_modules and 'GroomingDetection' in channel_modules[channel_id]:
            module = channel_modules[channel_id]['GroomingDetection']
            stats = module.get_statistics()
            return jsonify({'success': True, 'stats': stats})
        else:
            return jsonify({'success': False, 'error': 'Grooming detection not running'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# Socket.IO events
# Track which clients are subscribed to which video streams
active_stream_subscriptions = {}  # {session_id: {channel_id: True}}
stream_broadcast_threads = {}  # {channel_id: thread}
stream_broadcast_stop_flags = {}  # {channel_id: threading.Event}

@socketio.on('connect')
def handle_connect():
    logger.info('Client connected')
    emit('status', {'message': 'Connected to Sakshi.AI'})
    # Initialize subscription tracking for this client
    active_stream_subscriptions[request.sid] = {}

@socketio.on('disconnect')
def handle_disconnect():
    logger.info('Client disconnected')
    # Clean up subscriptions for this client
    if request.sid in active_stream_subscriptions:
        del active_stream_subscriptions[request.sid]

@socketio.on('subscribe_stream')
def handle_subscribe_stream(data):
    """Client subscribes to a video stream"""
    try:
        app_name = data.get('app_name')
        channel_id = data.get('channel_id')
        
        if not app_name or not channel_id:
            emit('stream_error', {'error': 'Missing app_name or channel_id'})
            return
        
        # Track this subscription
        if request.sid not in active_stream_subscriptions:
            active_stream_subscriptions[request.sid] = {}
        active_stream_subscriptions[request.sid][f"{app_name}:{channel_id}"] = True
        
        logger.info(f"Client {request.sid} subscribed to {app_name}/{channel_id}")
        
        # Check if channel processor exists and is running
        if channel_id not in shared_video_processors:
            # Channel is configured but processor not running - try to restart it
            logger.info(f"Channel {channel_id} not in shared_video_processors, attempting to restart...")
            
            # Get channel configuration from channel_modules, database, or config file
            rtsp_url = None
            
            # First, try to get from database
            try:
                with app.app_context():
                    rtsp_channel = db_manager.get_rtsp_channel(channel_id)
                    if rtsp_channel:
                        rtsp_url = rtsp_channel.get('rtsp_url')
                        logger.info(f"Found RTSP URL for {channel_id} in database")
            except Exception as e:
                logger.debug(f"Could not get RTSP URL from database: {e}")
            
            # If not in database, try app_configs
            if not rtsp_url and channel_id in channel_modules:
                for module_type in channel_modules[channel_id].keys():
                    if module_type in app_configs and channel_id in app_configs[module_type]['channels']:
                        rtsp_url = app_configs[module_type]['channels'][channel_id].get('video_source')
                        if rtsp_url:
                            logger.info(f"Found RTSP URL for {channel_id} in app_configs")
                            break
            
            # If still not found, try channels.json
            if not rtsp_url:
                try:
                    config_path = Path('config/channels.json')
                    if config_path.exists():
                        with open(config_path, 'r') as f:
                            config = json.load(f)
                            for ch in config.get('channels', []):
                                if ch.get('channel_id') == channel_id:
                                    rtsp_url = ch.get('rtsp_url')
                                    if rtsp_url:
                                        logger.info(f"Found RTSP URL for {channel_id} in channels.json")
                                    break
                except Exception as e:
                    logger.error(f"Error reading channels.json: {e}")
            
            if rtsp_url:
                # Attempt to create and start processor
                try:
                    logger.info(f"Attempting to start processor for {channel_id} with RTSP: {rtsp_url}")
                    processor = SharedMultiModuleVideoProcessor(
                        video_source=rtsp_url,
                        channel_id=channel_id,
                        fps_limit=30
                    )
                    shared_video_processors[channel_id] = processor
                    
                    # Add existing modules back to processor
                    if channel_id in channel_modules:
                        for module_type, module in channel_modules[channel_id].items():
                            processor.add_module(module_type, module)
                    
                    # Start the processor
                    if processor.start():
                        logger.info(f"✓ Successfully restarted channel {channel_id}")
                    else:
                        logger.warning(f"⚠ Failed to start channel {channel_id} - RTSP connection may be unavailable")
                        emit('stream_error', {
                            'app_name': app_name,
                            'channel_id': channel_id,
                            'error': f'Channel {channel_id} is configured but RTSP connection failed. Please check camera connectivity.'
                        })
                except Exception as e:
                    logger.error(f"Error restarting channel {channel_id}: {e}")
                    emit('stream_error', {
                        'app_name': app_name,
                        'channel_id': channel_id,
                        'error': f'Failed to restart channel {channel_id}: {str(e)}'
                    })
            else:
                logger.warning(f"Could not find RTSP URL for channel {channel_id}")
                emit('stream_error', {
                    'app_name': app_name,
                    'channel_id': channel_id,
                    'error': f'Channel {channel_id} is configured but RTSP URL not found. Cannot start video stream.'
                })
        elif not shared_video_processors[channel_id].is_running:
            # Processor exists but not running
            logger.warning(f"Channel {channel_id} processor exists but is not running")
            emit('stream_error', {
                'app_name': app_name,
                'channel_id': channel_id,
                'error': f'Channel {channel_id} is not running. RTSP connection may have failed.'
            })
        
        # Start broadcast thread for this channel if not already running
        stream_key = f"{app_name}:{channel_id}"
        if stream_key not in stream_broadcast_threads or not stream_broadcast_threads[stream_key].is_alive():
            stop_flag = threading.Event()
            stream_broadcast_stop_flags[stream_key] = stop_flag
            
            thread = threading.Thread(
                target=broadcast_video_frames,
                args=(app_name, channel_id, stop_flag),
                daemon=True
            )
            thread.start()
            stream_broadcast_threads[stream_key] = thread
            logger.info(f"Started broadcast thread for {stream_key}")
        
        emit('stream_subscribed', {'app_name': app_name, 'channel_id': channel_id})
        
    except Exception as e:
        logger.error(f"Error subscribing to stream: {e}")
        emit('stream_error', {'error': str(e)})

@socketio.on('unsubscribe_stream')
def handle_unsubscribe_stream(data):
    """Client unsubscribes from a video stream"""
    try:
        app_name = data.get('app_name')
        channel_id = data.get('channel_id')
        
        if not app_name or not channel_id:
            return
        
        # Remove subscription
        stream_key = f"{app_name}:{channel_id}"
        if request.sid in active_stream_subscriptions:
            active_stream_subscriptions[request.sid].pop(stream_key, None)
        
        logger.info(f"Client {request.sid} unsubscribed from {app_name}/{channel_id}")
        
        # Check if anyone is still subscribed to this stream
        still_subscribed = False
        for client_subs in active_stream_subscriptions.values():
            if stream_key in client_subs:
                still_subscribed = True
                break
        
        # If no one is subscribed, stop the broadcast thread
        if not still_subscribed and stream_key in stream_broadcast_stop_flags:
            stream_broadcast_stop_flags[stream_key].set()
            logger.info(f"Stopping broadcast thread for {stream_key} (no subscribers)")
        
    except Exception as e:
        logger.error(f"Error unsubscribing from stream: {e}")

def broadcast_video_frames(app_name, channel_id, stop_flag):
    """Broadcast video frames to subscribed clients via Socket.IO"""
    import base64
    stream_key = f"{app_name}:{channel_id}"
    frame_interval = 1.0 / 15  # 15 FPS for web streaming
    last_frame_time = 0
    no_frame_start_time = None
    error_emitted = False
    
    logger.info(f"Broadcasting frames for {stream_key}")
    
    while not stop_flag.is_set():
        try:
            current_time = time.time()
            
            # Rate limiting
            if current_time - last_frame_time < frame_interval:
                time.sleep(0.01)
                continue
            
            # Get frame from video processor
            if channel_id not in shared_video_processors:
                if no_frame_start_time is None:
                    no_frame_start_time = current_time
                elif current_time - no_frame_start_time > 3.0 and not error_emitted:
                    # No processor after 3 seconds - emit error
                    socketio.emit('stream_error', {
                        'app_name': app_name,
                        'channel_id': channel_id,
                        'error': f'Channel {channel_id} is not running. RTSP connection may have failed. Please check camera connectivity.'
                    }, room=None)
                    error_emitted = True
                time.sleep(0.5)
                continue
            
            processor = shared_video_processors[channel_id]
            if not processor.is_running:
                if no_frame_start_time is None:
                    no_frame_start_time = current_time
                elif current_time - no_frame_start_time > 3.0 and not error_emitted:
                    # Processor not running after 3 seconds - emit error
                    socketio.emit('stream_error', {
                        'app_name': app_name,
                        'channel_id': channel_id,
                        'error': f'Channel {channel_id} processor is not running. RTSP connection may have failed.'
                    }, room=None)
                    error_emitted = True
                time.sleep(0.5)
                continue
            
            # Reset error tracking if we have a processor and it's running
            if no_frame_start_time is not None:
                no_frame_start_time = None
                error_emitted = False
            
            # Get latest frame (either module-specific or combined)
            # Try module-specific first, then fall back to combined frame
            # Use timeout to avoid blocking too long
            frame = None
            try:
                if app_name in processor.get_active_modules():
                    # Get module-specific frame
                    frame = processor.get_latest_frame(module_name=app_name)
                    # If module-specific frame is None or invalid, DON'T use combined frame
                    # Combined frame might have other modules' annotations (e.g., BagDetection)
                    # Return None instead so we can skip this frame
                    if frame is None or (hasattr(frame, 'size') and frame.size == 0):
                        logger.debug(f"Module-specific frame for {app_name}/{channel_id} not available, skipping")
                        time.sleep(0.05)
                        continue
                else:
                    # Module not active - return None to skip
                    logger.debug(f"{app_name} not in active modules for {channel_id}, skipping")
                    time.sleep(0.05)
                    continue
            except Exception as e:
                logger.warning(f"Error getting frame for {stream_key}: {e}")
                time.sleep(0.1)
                continue
            
            if frame is None:
                consecutive_none_frames += 1
                if consecutive_none_frames > max_consecutive_none:
                    if no_frame_start_time is None:
                        no_frame_start_time = current_time
                    elif current_time - no_frame_start_time > 3.0 and not error_emitted:
                        # No frames received after 3 seconds - emit error
                        socketio.emit('stream_error', {
                            'app_name': app_name,
                            'channel_id': channel_id,
                            'error': f'No video frames received from channel {channel_id}. Camera may be offline or RTSP stream unavailable.'
                        }, room=None)
                        error_emitted = True
                time.sleep(0.05)  # Shorter sleep for faster recovery
                continue
            
            # Reset counters on successful frame
            consecutive_none_frames = 0
            
            # Reset error tracking if we got a frame
            if no_frame_start_time is not None:
                no_frame_start_time = None
                error_emitted = False
            
            # Check frame freshness - skip if frame is too old (stale)
            # This prevents sending frozen/stuck frames
            frame_timestamp = getattr(frame, 'timestamp', None)
            if frame_timestamp and (current_time - frame_timestamp) > 2.0:
                logger.debug(f"Skipping stale frame for {stream_key} (age: {current_time - frame_timestamp:.2f}s)")
                time.sleep(0.05)
                continue
            
            # Resize frame if too large to reduce encoding time
            frame_height, frame_width = frame.shape[:2]
            if frame_width > 1280 or frame_height > 720:
                scale = min(1280 / frame_width, 720 / frame_height)
                new_width = int(frame_width * scale)
                new_height = int(frame_height * scale)
                frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            
            # Encode frame as JPEG (lower quality for faster encoding)
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not ret:
                time.sleep(0.1)
                continue
            
            # Convert to base64 (optimized)
            try:
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
            except Exception as e:
                logger.warning(f"Error encoding frame to base64 for {stream_key}: {e}")
                time.sleep(0.1)
                continue
            
            # Get FPS data
            fps_data = processor.get_live_fps() if hasattr(processor, 'get_live_fps') else {'live_feed_fps': 0, 'processing_fps': 0}
            
            # Broadcast to all subscribed clients
            socketio.emit('video_frame', {
                'app_name': app_name,
                'channel_id': channel_id,
                'frame': frame_base64,
                'timestamp': current_time,
                'fps': fps_data['live_feed_fps'],
                'processing_fps': fps_data['processing_fps']
            }, room=None)  # Broadcast to all connected clients
            
            last_frame_time = current_time
            
        except Exception as e:
            logger.error(f"Error broadcasting frame for {stream_key}: {e}")
            time.sleep(0.5)
    
    # Clean up
    if stream_key in stream_broadcast_threads:
        del stream_broadcast_threads[stream_key]
    if stream_key in stream_broadcast_stop_flags:
        del stream_broadcast_stop_flags[stream_key]
    
    logger.info(f"Stopped broadcasting frames for {stream_key}")

def create_database_tables():
    """Create database tables"""
    with app.app_context():
        db.create_all()
        logger.info("Database tables created")


# ============= Dress Code Monitoring Routes =============

@app.route('/static/dresscode_snapshots/<filename>')
def serve_dresscode_snapshot(filename):
    """Serve dress code violation snapshot"""
    return send_from_directory('static/dresscode_snapshots', filename)

@app.route('/api/get_dresscode_alerts')
def get_dresscode_alerts():
    """Get dress code violation alerts"""
    channel_id = request.args.get('channel_id')
    limit = int(request.args.get('limit', 50))
    
    try:
        with app.app_context():
            alerts = db_manager.get_dresscode_alerts(channel_id=channel_id, limit=limit)
        
        return jsonify({
            'success': True,
            'alerts': alerts,
            'count': len(alerts)
        })
    except Exception as e:
        logger.error(f"Error getting dress code alerts: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/set_counter_roi', methods=['POST'])
@login_required
def set_counter_roi():
    """Set counter ROI for dress code monitoring (uses best.pt for uniform detection)"""
    data = request.json
    channel_id = data.get('channel_id')
    roi_points = data.get('roi_points', [])
    
    if not channel_id:
        return jsonify({'success': False, 'error': 'channel_id is required'})
    
    try:
        # Get the DressCodeMonitoring module for this channel
        if channel_id in shared_video_processors:
            processor = shared_video_processors[channel_id]
            if 'DressCodeMonitoring' in processor.modules:
                module = processor.modules['DressCodeMonitoring']
                module.set_counter_roi(roi_points)
                return jsonify({'success': True, 'message': 'Counter ROI set successfully (best.pt for uniforms)'})
        
        return jsonify({'success': False, 'error': f'DressCodeMonitoring not found for channel {channel_id}'})
    except Exception as e:
        logger.error(f"Error setting counter ROI: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/set_queue_roi', methods=['POST'])
@login_required
def set_queue_roi():
    """Set queue ROI for dress code monitoring (uses YOLOv11 for person detection)"""
    data = request.json
    channel_id = data.get('channel_id')
    roi_points = data.get('roi_points', [])
    
    if not channel_id:
        return jsonify({'success': False, 'error': 'channel_id is required'})
    
    try:
        # Get the DressCodeMonitoring module for this channel
        if channel_id in shared_video_processors:
            processor = shared_video_processors[channel_id]
            if 'DressCodeMonitoring' in processor.modules:
                module = processor.modules['DressCodeMonitoring']
                module.set_queue_roi(roi_points)
                return jsonify({'success': True, 'message': 'Queue ROI set successfully (YOLOv11 for persons)'})
        
        return jsonify({'success': False, 'error': f'DressCodeMonitoring not found for channel {channel_id}'})
    except Exception as e:
        logger.error(f"Error setting queue ROI: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/set_table_roi', methods=['POST'])
@login_required
def set_table_roi():
    """Set table ROI for table service monitoring"""
    data = request.json
    channel_id = data.get('channel_id')
    table_id = data.get('table_id')
    roi_points = data.get('roi_points', [])
    
    if not channel_id or not table_id:
        return jsonify({'success': False, 'error': 'channel_id and table_id are required'})
    
    try:
        # Get the TableServiceMonitor module for this channel
        if channel_id in shared_video_processors:
            processor = shared_video_processors[channel_id]
            if 'TableServiceMonitor' in processor.modules:
                module = processor.modules['TableServiceMonitor']
                module.set_table_roi(table_id, roi_points)
                return jsonify({'success': True, 'message': f'Table ROI set successfully for table {table_id}'})
        
        return jsonify({'success': False, 'error': f'TableServiceMonitor not found for channel {channel_id}'})
    except Exception as e:
        logger.error(f"Error setting table ROI: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/get_table_service_violations')
@login_required
def get_table_service_violations():
    """Get table service violations"""
    channel_id = request.args.get('channel_id')
    limit = int(request.args.get('limit', 50))
    
    try:
        with app.app_context():
            violations = db_manager.get_table_service_violations(channel_id=channel_id, limit=limit)
        
        return jsonify({
            'success': True,
            'violations': violations,
            'count': len(violations)
        })
    except Exception as e:
        logger.error(f"Error getting table service violations: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/delete_table_service_violation', methods=['POST'])
@login_required
def delete_table_service_violation():
    """Delete a table service violation"""
    data = request.json
    violation_id = data.get('violation_id')
    
    if not violation_id:
        return jsonify({'success': False, 'error': 'violation_id is required'})
    
    try:
        with app.app_context():
            success = db_manager.delete_table_service_violation(violation_id)
            if success:
                return jsonify({'success': True, 'message': 'Violation deleted successfully'})
            else:
                return jsonify({'success': False, 'error': 'Violation not found'})
    except Exception as e:
        logger.error(f"Error deleting table service violation: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/clear_old_table_service_violations', methods=['POST'])
@login_required
def clear_old_table_service_violations():
    """Clear old table service violations"""
    data = request.json
    days = data.get('days', 7)
    
    try:
        with app.app_context():
            deleted_count = db_manager.clear_old_table_service_violations(days=days)
            return jsonify({
                'success': True,
                'deleted_count': deleted_count,
                'message': f'Deleted {deleted_count} violations older than {days} days'
            })
    except Exception as e:
        logger.error(f"Error clearing old table service violations: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/set_allowed_uniforms', methods=['POST'])
@login_required
def set_allowed_uniforms():
    """Set allowed uniform colors for counter area"""
    data = request.json
    channel_id = data.get('channel_id')
    allowed_uniforms = data.get('allowed_uniforms', {})
    
    if not channel_id:
        return jsonify({'success': False, 'error': 'channel_id is required'})
    
    try:
        # Get the DressCodeMonitoring module for this channel
        if channel_id in shared_video_processors:
            processor = shared_video_processors[channel_id]
            if 'DressCodeMonitoring' in processor.modules:
                module = processor.modules['DressCodeMonitoring']
                module.set_allowed_uniforms(allowed_uniforms)
                return jsonify({'success': True, 'message': 'Allowed uniforms set successfully'})
        
        return jsonify({'success': False, 'error': f'DressCodeMonitoring not found for channel {channel_id}'})
    except Exception as e:
        logger.error(f"Error setting allowed uniforms: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/get_counter_roi/<channel_id>')
@login_required
def get_counter_roi(channel_id):
    """Get counter and queue ROI configuration"""
    try:
        if channel_id in shared_video_processors:
            processor = shared_video_processors[channel_id]
            if 'DressCodeMonitoring' in processor.modules:
                module = processor.modules['DressCodeMonitoring']
                return jsonify({
                    'success': True,
                    'counter_roi': module.counter_roi,
                    'queue_roi': module.queue_roi,
                    'allowed_uniforms': module.allowed_uniforms_in_counter
                })
        
        return jsonify({'success': False, 'error': f'DressCodeMonitoring not found for channel {channel_id}'})
    except Exception as e:
        logger.error(f"Error getting ROI configuration: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/delete_dresscode_alert/<int:alert_id>', methods=['DELETE'])
@login_required
def delete_dresscode_alert(alert_id):
    """Delete a dress code alert"""
    try:
        with app.app_context():
            success = db_manager.delete_dresscode_alert(alert_id)
        
        if success:
            return jsonify({'success': True, 'message': 'Alert deleted successfully'})
        else:
            return jsonify({'success': False, 'error': 'Alert not found'})
    except Exception as e:
        logger.error(f"Error deleting dress code alert: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/get_dresscode_stats')
def get_dresscode_stats():
    """Get dress code violation statistics"""
    channel_id = request.args.get('channel_id')
    days = int(request.args.get('days', 7))
    
    try:
        with app.app_context():
            stats = db_manager.get_dresscode_stats(channel_id=channel_id, days=days)
        
        return jsonify({
            'success': True,
            'stats': stats
        })
    except Exception as e:
        logger.error(f"Error getting dress code stats: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/get_ppe_alerts')
def get_ppe_alerts():
    """Get PPE violation alerts"""
    channel_id = request.args.get('channel_id')
    limit = int(request.args.get('limit', 50))
    
    try:
        with app.app_context():
            alerts = db_manager.get_ppe_alerts(channel_id=channel_id, limit=limit)
        
        return jsonify({
            'success': True,
            'alerts': alerts,
            'count': len(alerts)
        })
    except Exception as e:
        logger.error(f"Error getting PPE alerts: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/delete_ppe_alert/<int:alert_id>', methods=['DELETE'])
@login_required
def delete_ppe_alert(alert_id):
    """Delete a PPE violation alert"""
    try:
        with app.app_context():
            success = db_manager.delete_ppe_alert(alert_id)
        
        if success:
            return jsonify({'success': True, 'message': 'Alert deleted successfully'})
        else:
            return jsonify({'success': False, 'error': 'Alert not found'})
    except Exception as e:
        logger.error(f"Error deleting PPE alert: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/get_ppe_stats')
def get_ppe_stats():
    """Get PPE violation statistics"""
    channel_id = request.args.get('channel_id')
    days = int(request.args.get('days', 7))
    
    try:
        with app.app_context():
            stats = db_manager.get_ppe_stats(channel_id=channel_id, days=days)
        
        return jsonify({
            'success': True,
            'stats': stats
        })
    except Exception as e:
        logger.error(f"Error getting PPE stats: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/get_queue_violations')
def get_queue_violations():
    """Get queue violation alerts"""
    channel_id = request.args.get('channel_id')
    limit = int(request.args.get('limit', 50))
    
    try:
        with app.app_context():
            violations = db_manager.get_queue_violations(channel_id=channel_id, limit=limit)
        
        logger.info(f"Retrieved {len(violations)} queue violations (channel_id={channel_id}, limit={limit})")
        
        # Ensure each violation has a timestamp field (for compatibility)
        for violation in violations:
            if 'timestamp' not in violation and 'created_at' in violation:
                violation['timestamp'] = violation['created_at']
        
        return jsonify({
            'success': True,
            'violations': violations,
            'count': len(violations)
        })
    except Exception as e:
        logger.error(f"Error getting queue violations: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/delete_queue_violation/<int:violation_id>', methods=['DELETE'])
@login_required
def delete_queue_violation(violation_id):
    """Delete a queue violation"""
    try:
        with app.app_context():
            success = db_manager.delete_queue_violation(violation_id)
        
        if success:
            return jsonify({'success': True, 'message': 'Violation deleted successfully'})
        else:
            return jsonify({'success': False, 'error': 'Violation not found'})
    except Exception as e:
        logger.error(f"Error deleting queue violation: {e}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/update_dresscode_config', methods=['POST'])
@login_required
def update_dresscode_config():
    """Update dress code monitoring configuration"""
    data = request.json
    channel_id = data.get('channel_id')
    config = data.get('config', {})
    
    try:
        # Get the module
        if channel_id in channel_modules and 'DressCodeMonitoring' in channel_modules[channel_id]:
            module = channel_modules[channel_id]['DressCodeMonitoring']
            
            # Update configuration
            if 'alert_cooldown' in config:
                module.alert_cooldown = float(config['alert_cooldown'])
            if 'violation_duration_threshold' in config:
                module.violation_duration_threshold = float(config['violation_duration_threshold'])
            if 'conf_threshold' in config:
                module.conf_threshold = float(config['conf_threshold'])
            
            return jsonify({
                'success': True,
                'message': 'Configuration updated successfully'
            })
        else:
            return jsonify({'success': False, 'error': 'Module not found'})
            
    except Exception as e:
        logger.error(f"Error updating dress code config: {e}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/system_resources')
@login_required
def get_system_resources():
    """Get current system resource usage"""
    try:
        import psutil
        resources = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'available_memory_gb': psutil.virtual_memory().available / (1024**3)
        }
        return jsonify({
            'success': True,
            'resources': resources,
            'active_channels': len(shared_video_processors),
            'max_channels': 50
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    # Create database tables
    create_database_tables()
    
    # Start the application FIRST (non-blocking)
    logger.info("Starting Sakshi.AI Video Analytics Platform")
    logger.info("Dashboard will be available at http://localhost:5000")
    
    # Load and start channels from configuration in BACKGROUND (non-blocking)
    def load_channels_in_background():
        """Load channels in background thread so server starts immediately"""
        time.sleep(1)  # Give server a moment to start
        logger.info("Loading channels from configuration in background...")
        load_channels_from_config()
        logger.info("✅ Channel loading completed")
    
    # Start channel loading in background thread
    channel_loader_thread = threading.Thread(target=load_channels_in_background, daemon=True)
    channel_loader_thread.start()
    
    # Start Flask server (this will block, but server is now running)
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)