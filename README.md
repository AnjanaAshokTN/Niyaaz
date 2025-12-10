# Sakshi.AI - Intelligent Video Analytics Platform

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-2.3.3-green.svg)](https://flask.palletsprojects.com/)
[![YOLOv8/YOLOv11](https://img.shields.io/badge/YOLO-Ultralytics-orange.svg)](https://github.com/ultralytics/ultralytics)

A comprehensive video analytics platform that transforms any CCTV camera into an intelligent monitoring system. Sakshi.AI provides real-time monitoring, compliance tracking, safety detection, and advanced analytics with a modern web dashboard.

## 🎯 Features

### Core Monitoring Modules

#### 1. **Queue & Wait Time Monitoring**
- Real-time queue length tracking
- Wait time analysis (alerts when queue wait > 3 seconds)
- Counter staffing detection
- ROI-based queue and counter area configuration
- Violation alerts for:
  - Queue overflow (> 3 people)
  - Long wait times (≥ 3 seconds)
  - No staff at counter while queue exists
- Snapshot capture of violations

#### 2. **Uniform Compliance Monitoring**
- Real-time uniform detection (Black, Blue, Cream uniforms)
- Dress code violation tracking
- Snapshot capture of violations
- Historical compliance reports
- Employee identification support

#### 3. **PPE Compliance Monitoring**
- Personal Protective Equipment detection:
  - Apron detection
  - Gloves detection
  - Hairnet detection
- Real-time violation alerts
- Snapshot capture of violations
- Compliance statistics and reporting

#### 4. **Cash Drawer Monitoring**
- Cash drawer open detection
- Transaction monitoring
- Alert generation for cash handling events
- Snapshot capture of events

#### 5. **Smoke & Fire Detection**
- Real-time smoke detection
- Fire detection
- Safety alert system
- Snapshot capture of incidents

#### 6. **Crowd Detection**
- Crowd gathering detection in designated areas
- Density analysis
- Alert generation for crowd events

#### 7. **People Counter**
- Real-time footfall tracking
- Direction detection (IN/OUT)
- Configurable counting line
- Historical analytics (daily, weekly, monthly)
- Peak hours analysis

### Additional Modules
- **Bag Detection**: Unattended baggage detection
- **Fall Detection**: Person fall detection and emergency alerts
- **Phone Usage Detection**: Mobile phone usage monitoring
- **Mopping Detection**: Cleaning activity monitoring
- **Restricted Area Monitor**: Unauthorized access detection
- **Grooming Detection**: Behavioral pattern analysis

### Technical Features
- **Multi-Module Support**: Run multiple analysis modules on the same video feed
- **YOLOv8/YOLOv11 Detection**: State-of-the-art AI for accurate object detection
- **Real-time Streaming**: Live video feeds with AI annotations
- **WebSocket Communication**: Instant updates without page refresh
- **RTSP Support**: Direct camera stream integration
- **GPU Acceleration**: CUDA support for improved performance
- **Database Analytics**: Comprehensive data storage and reporting
- **Responsive Dashboard**: Modern web interface for all devices
- **Violation Tracking**: Automatic snapshot capture and alert management

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- Git
- 4GB+ RAM recommended
- CUDA-compatible GPU (optional, for better performance)
- RTSP camera or video files for testing

### Quick Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Niyaz
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/Mac:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure channels**
   - Edit `config/channels.json` to add your camera RTSP URLs
   - Or place video files in the `videos/` directory

5. **Add YOLO models**
   - Place your custom model (`best.pt`) in the `models/` directory
   - YOLOv11 model (`yolo11n.pt`) for person detection should also be in `models/`

6. **Run the application**
   ```bash
   python app.py
   ```

7. **Access the dashboard**
   - Open your browser and go to: `http://localhost:5000`
   - Login with your credentials
   - Access the analytics dashboard

## 📁 Project Structure

```
Niyaz/
├── app.py                          # Main Flask application
├── requirements.txt                 # Python dependencies
├── README.md                       # This file
├── config/                         # Configuration files
│   ├── channels.json              # Camera and module configuration
│   └── default.json               # Default settings
├── data/                          # Database storage
│   └── sakshi.db                  # SQLite database
├── models/                        # YOLO model files
│   ├── best.pt                    # Custom YOLO model
│   └── yolo11n.pt                 # YOLOv11 person detection model
├── videos/                        # Video files for testing
├── static/                        # Static files
│   ├── alerts/                    # Alert GIFs
│   ├── dresscode_snapshots/       # Dress code violation images
│   ├── ppe_snapshots/             # PPE violation images
│   ├── cash_snapshots/            # Cash detection images
│   ├── queue_violations/          # Queue violation images
│   └── smoking_snapshots/         # Smoke/fire detection images
├── templates/                     # HTML templates
│   ├── landing.html               # Landing page
│   ├── login.html                 # Login page
│   └── dashboard.html             # Analytics dashboard
└── modules/                       # Core AI modules
    ├── database.py                # Database models and management
    ├── yolo_detector.py           # YOLO detection service
    ├── queue_monitor.py           # Queue monitoring logic
    ├── dress_code_monitoring.py   # Uniform compliance
    ├── ppe_monitoring.py          # PPE compliance
    ├── cash_detection.py          # Cash drawer monitoring
    ├── smoking_detection.py       # Smoke & fire detection
    ├── crowd_detection.py         # Crowd detection
    ├── people_counter.py          # People counting logic
    ├── shared_multi_module_processor.py  # Multi-module video processor
    └── rtsp_connection_pool.py    # RTSP stream management
```

## 🚀 Usage

### Setting Up Monitoring Modules

1. **Start the application** and navigate to the dashboard
2. **Select a channel** from the available cameras
3. **Configure ROIs** (Region of Interest) for modules that require it:
   - **Queue Monitor**: Draw queue area (yellow) and counter area (cyan)
   - **People Counter**: Configure counting line
4. **Start monitoring** by clicking the "Start" button for each module
5. **View real-time analytics** in the dashboard

### Module-Specific Configuration

#### Queue Monitor
- Configure queue ROI (main area)
- Configure counter ROI (secondary area)
- Set thresholds:
  - Queue alert threshold (default: 3 people)
  - Wait time threshold (default: 3 seconds)
  - Counter threshold (default: 1 person)

#### PPE Monitoring
- Configure required PPE items (Apron, Gloves, Hairnet)
- Set violation duration threshold
- Set alert cooldown period

#### Dress Code Monitoring
- Configure required uniform types
- Set violation thresholds
- Enable employee ID tracking (optional)

#### Cash Detection
- Set confidence threshold (default: 0.8)
- Configure alert cooldown

### Viewing Analytics

- **Real-time counts** are displayed in the dashboard
- **Violation snapshots** are saved automatically
- **Historical reports** available in the Analytics & Reports tabs
- **Alerts** appear automatically when thresholds are exceeded

## ⚙️ Configuration

### Channel Configuration (`config/channels.json`)

```json
{
  "camera_1": {
    "rtsp_url": "rtsp://user:pass@ip:port/stream",
    "modules": ["QueueMonitor", "DressCodeMonitoring", "CashDetection"]
  }
}
```

### Detection Settings
- **Confidence threshold**: Adjust per module (default: 0.5-0.8)
- **Tracking distance**: Modify in person tracker settings
- **Dwell time**: Configure minimum wait time for queue counting
- **Alert cooldown**: Time between alerts (default: 60 seconds)

### Database
- Default: SQLite database in `data/sakshi.db`
- For production, configure PostgreSQL in `app.py`
- Database is automatically initialized on first run

## 🔧 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Landing page |
| `/dashboard` | GET | Analytics dashboard |
| `/api/login` | POST | User authentication |
| `/api/get_active_channels` | GET | List active channels |
| `/api/start_channel` | POST | Start monitoring a channel |
| `/api/stop_channel` | POST | Stop monitoring a channel |
| `/api/set_roi` | POST | Configure ROI areas |
| `/api/get_ppe_alerts` | GET | Get PPE violation alerts |
| `/api/delete_ppe_alert/<id>` | DELETE | Delete PPE alert |
| `/api/get_queue_violations` | GET | Get queue violations |
| `/api/delete_queue_violation/<id>` | DELETE | Delete queue violation |
| `/api/get_module_analytics/<module>` | GET | Get module analytics |
| `/api/get_dresscode_alerts` | GET | Get dress code alerts |

## 📊 Database Schema

### Key Tables
- **ppe_alerts**: PPE violation records with snapshots
- **queue_violations**: Queue violation records with snapshots
- **dresscode_alerts**: Dress code violation records
- **cash_snapshots**: Cash detection events
- **smoking_snapshots**: Smoke/fire detection events
- **daily_footfall**: Daily in/out counts per channel
- **hourly_footfall**: Hourly breakdown of footfall data
- **queue_analytics**: Queue monitoring data
- **channel_config**: ROI and line configurations
- **detection_events**: Audit trail of all detections

## 🚨 Troubleshooting

### Common Issues

1. **"Cannot open video source" error**
   - Ensure RTSP URLs are correct in `config/channels.json`
   - Check camera credentials and network connectivity
   - Verify video files are in the `videos/` directory

2. **High CPU usage**
   - Reduce video resolution in channel configuration
   - Lower the processing FPS limit
   - Consider using GPU acceleration
   - Disable unused modules

3. **Socket.IO connection issues**
   - Check firewall settings
   - Ensure port 5000 is not blocked
   - Try running on different port in `app.py`

4. **YOLO model errors**
   - Ensure model files (`best.pt`, `yolo11n.pt`) are in `models/` directory
   - Check model file permissions
   - Verify model compatibility with Ultralytics version

5. **Database errors**
   - Ensure `data/` directory exists and is writable
   - Check database file permissions
   - For PostgreSQL, verify connection settings

### Performance Optimization

1. **For better performance:**
   - Use GPU acceleration (install CUDA + PyTorch GPU version)
   - Reduce video resolution before processing
   - Adjust confidence thresholds to reduce false positives
   - Use TensorRT for model optimization (if available)

2. **For production deployment:**
   - Use PostgreSQL or MySQL instead of SQLite
   - Set up Redis for session management
   - Use nginx for serving static files
   - Consider Docker deployment
   - Enable GPU acceleration
   - Set up proper logging and monitoring

## 🔐 Security Considerations

- Change default `SECRET_KEY` in `app.py` for production
- Use strong passwords for database connections
- Secure RTSP credentials
- Enable HTTPS for production deployments
- Implement proper authentication and authorization
- Regularly update dependencies for security patches

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🏢 About

**Sakshi.AI** is developed by **ThinkNeural.AI** - a cutting-edge AI research and development company focused on computer vision and intelligent systems.

- Website: [ThinkNeural.AI](https://thinkneural.ai)
- Email: contact@thinkneural.ai

---

**Sakshi** (Sanskrit: साक्षी) means "witness" - representing our mission to create an intelligent digital witness for your physical spaces.

## 🔮 Roadmap

- [ ] Real-time heatmap analysis
- [ ] Facial recognition integration
- [ ] Mobile app for alerts
- [ ] Cloud deployment options
- [ ] Advanced behavioral analytics
- [ ] Integration with existing security systems
- [ ] Multi-language support
- [ ] Advanced reporting and analytics

---

For support and questions, please open an issue or contact our team.
