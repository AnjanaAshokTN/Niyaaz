import eventlet
# Monkey patching is essential for eventlet to work with standard libraries
eventlet.monkey_patch()

import logging
import signal
import sys
from apscheduler.schedulers.background import BackgroundScheduler

# --- Logging Configuration ---
# Configure logging before importing application parts to ensure consistency
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Quieten down libraries that are too verbose
for logger_name in ['ultralytics', 'apscheduler', 'sqlalchemy.engine', 'werkzeug', 'socketio', 'engineio']:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

# --- Application Import ---
# This block attempts to import the core components of the app.
# An error here points to a fundamental issue like a circular import.
try:
    from main_app import (
        app, socketio, initialize_database, start_streams, graceful_shutdown,
        save_periodic_heatmap_snapshots, log_queue_counts, IST
    )
except ImportError as e:
    logging.error(f"❌ Failed to import application components: {e}")
    logging.error("This might be a circular import or a Python path issue. Please check processor files for errors.")
    sys.exit(1)


def run_app():
    """Initializes and runs the entire Sakshi.AI application in the correct order."""
    print("🚀 Starting Sakshi AI...")
    print("========================================")
    
    # 1. Initialize Database Connection
    if not initialize_database():
        logging.error("❌ Halting application start due to database connection failure.")
        sys.exit(1)
    
    # 2. Setup Background Scheduler for periodic tasks
    scheduler = BackgroundScheduler(timezone=str(IST))
    scheduler.add_job(save_periodic_heatmap_snapshots, 'cron', hour='*/4', misfire_grace_time=60)
    scheduler.add_job(log_queue_counts, 'interval', minutes=5, misfire_grace_time=30)
    scheduler.start()
    
    # 3. Start all video processing threads
    start_streams()
    
    # 4. Run the Web Server using SocketIO and Eventlet
    logging.info("✅ Starting Flask-SocketIO server on http://localhost:5001")
    try:
        socketio.run(app, host='0.0.0.0', port=5001, debug=False)
    except Exception as e:
        logging.error(f"❌ Server failed to run: {e}")
    finally:
        # This will be called after the server stops (e.g., via Ctrl+C)
        if scheduler.running:
            scheduler.shutdown()
        graceful_shutdown()

if __name__ == '__main__':
    # Register signal handlers to trigger our graceful shutdown function
    signal.signal(signal.SIGINT, graceful_shutdown)
    signal.signal(signal.SIGTERM, graceful_shutdown)
    
    run_app()
