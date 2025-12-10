# Essential Files for Running the Application

## Core Application Files (Minimum Required)

### 1. Main Application
- **`app.py`** - Main Flask application entry point

### 2. Core Modules (Required)
- **`modules/__init__.py`** - Python package marker
- **`modules/database.py`** - Database management
- **`modules/yolo_detector.py`** - YOLO detection service
- **`modules/rtsp_connection_pool.py`** - RTSP stream management
- **`modules/shared_multi_module_processor.py`** - Video processing engine
- **`modules/gif_recorder.py`** - Alert GIF recording

### 3. Use Case Modules (Your 5 Active Modules)
- **`modules/queue_monitor.py`** - Queue & Wait Time tracking
- **`modules/dress_code_monitoring.py`** - Uniform & PPE Compliance
- **`modules/cash_detection.py`** - Cash Drawer Monitoring
- **`modules/smoking_detection.py`** - Smoke & Fire Detection
- **`modules/crowd_detection.py`** - Crowd Detection

### 4. Configuration Files
- **`config/channels.json`** - Camera and module configuration
- **`config/default.json`** - Default settings (if used)

### 5. Templates
- **`templates/dashboard.html`** - Main dashboard UI
- **`templates/login.html`** - Login page

### 6. Models
- **`models/best.pt`** - Your custom YOLO model
- **`models/yolo11n.pt`** - YOLOv11 model (for QueueMonitor, if used)

### 7. Database
- **`data/sakshi.db`** - SQLite database (created automatically if missing)

### 8. Static Directories (Created automatically)
- **`static/alerts/`** - Alert GIFs storage
- **`static/debug/`** - Debug snapshots
- **`static/dresscode_snapshots/`** - Dress code violation images
- **`static/cash_snapshots/`** - Cash detection images
- **`static/smoking_snapshots/`** - Smoke/fire detection images

### 9. Dependencies
- **`requirements.txt`** - Python package dependencies

---

## Total Essential Files: ~20 files

### Absolutely Critical (13 files):
1. `app.py`
2. `modules/__init__.py`
3. `modules/database.py`
4. `modules/yolo_detector.py`
5. `modules/rtsp_connection_pool.py`
6. `modules/shared_multi_module_processor.py`
7. `modules/gif_recorder.py`
8. `modules/queue_monitor.py`
9. `modules/dress_code_monitoring.py`
10. `modules/cash_detection.py`
11. `modules/smoking_detection.py`
12. `modules/crowd_detection.py`
13. `config/channels.json`

### Also Needed:
14. `templates/dashboard.html`
15. `templates/login.html`
16. `models/best.pt`
17. `requirements.txt`

---

## Optional/Supporting Files (Can be removed if not needed):
- `modules/model_manager.py` - Model loading optimization
- `modules/gpu_monitor.py` - GPU monitoring
- `modules/tensorrt_engine.py` - TensorRT support
- `init_users.py` - User initialization script
- `explore_database.py` - Database exploration tool
- All documentation files (*.md)
- All test/debug scripts

---

## To Run the Application:

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Initialize database (if first run):**
   ```bash
   python init_users.py
   ```

3. **Run the application:**
   ```bash
   python app.py
   ```

---

**Note:** The application will automatically create:
- Database file (`data/sakshi.db`) if it doesn't exist
- Static directories if they don't exist
- Required database tables on first run


