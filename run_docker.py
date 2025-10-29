#!/usr/bin/env python3
"""
Sakshi AI Docker Entry Point - Main App Only (No Processors)
For Docker deployment where processors run in separate containers
"""

import eventlet
eventlet.monkey_patch()

import logging
import signal
import sys
from apscheduler.schedulers.background import BackgroundScheduler

# Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
for logger_name in ['ultralytics', 'apscheduler', 'sqlalchemy.engine', 'werkzeug', 'socketio', 'engineio']:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

# Import Flask app but not processor functions
from main_app import (
    app, socketio, initialize_database, graceful_shutdown, IST
)

def run_app():
    """Run Flask app only (processors run in separate containers)"""
    print("🚀 Starting Sakshi AI Main App (Docker Mode)...")
    print("========================================")
    print("Note: Processors run in separate containers")
    
    # Initialize Database Connection
    if not initialize_database():
        logging.error("❌ Halting application start due to database connection failure.")
        sys.exit(1)
    
    # Setup Background Scheduler for periodic tasks (disabled - processors handle their own)
    # scheduler = BackgroundScheduler(timezone=str(IST))
    # scheduler.start()
    
    logging.info("✅ Starting Flask-SocketIO server on http://0.0.0.0:5001")
    try:
        socketio.run(app, host='0.0.0.0', port=5001, debug=False)
    except Exception as e:
        logging.error(f"❌ Server failed to run: {e}")
    finally:
        graceful_shutdown()

if __name__ == '__main__':
    signal.signal(signal.SIGINT, graceful_shutdown)
    signal.signal(signal.SIGTERM, graceful_shutdown)
    run_app()

