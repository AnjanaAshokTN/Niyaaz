# Sakshi AI - Intelligent Video Analytics Platform

Sakshi AI is a comprehensive video analytics platform that provides real-time monitoring and analysis capabilities for retail, security, and business intelligence applications. The platform uses advanced computer vision and machine learning to deliver actionable insights from video streams.

## 🚀 Features

### Core Analytics Modules

1. **People Counter** - Bidirectional footfall tracking with hourly and daily analytics
2. **Shoplifting Detection** - AI-powered suspicious behavior detection
3. **Queue Monitor** - Real-time queue length monitoring and analytics
4. **Heatmap Generation** - Customer engagement zone analysis
5. **Shutter Monitor** - Shop open/close time tracking with video evidence
6. **Security Monitor** - Security personnel interaction monitoring
7. **Kitchen Compliance** - Kitchen safety and compliance monitoring
8. **QPOS Monitoring** - Point-of-sale screen monitoring
9. **Generic Object Detection** - Multi-class object detection and tracking

### Key Capabilities

- **Real-time Processing**: Live video stream analysis with minimal latency
- **Multi-camera Support**: Simultaneous processing of multiple video sources
- **Web Dashboard**: Interactive web interface for monitoring and configuration
- **Telegram Notifications**: Real-time alerts via Telegram bot
- **Data Analytics**: Comprehensive reporting and analytics
- **ROI Configuration**: Region of Interest setup for targeted monitoring
- **Video Evidence**: Automatic capture and storage of detection events

## 📁 Project Structure

```
Sakshi-15-October/
├── main_app.py                 # Main Flask application
├── sakshi.db                  # SQLite database
├── models/                    # AI model files (.pt)
│   ├── best_shoplift.pt
│   ├── best_qpos.pt
│   ├── best_generic.pt
│   ├── shutter_model.pt
│   ├── yolov8n.pt
│   └── ...
├── processors/                # Video processing modules
│   ├── detection_processor.py
│   ├── people_counter_processor.py
│   ├── queue_monitor_processor.py
│   ├── heatmap_processor.py
│   ├── shutter_monitor_processor006.py
│   ├── security_monitor_1.py
│   └── kitchen_compliance_monitor.py
├── config/                    # Configuration files
│   ├── requirements.txt
│   └── rtsp_links.txt
├── static/                    # Static web assets
│   ├── assets/
│   └── detections/           # Detection media storage
├── templates/                # HTML templates
│   ├── dashboard.html
│   └── landing.html
├── test-videos/              # Test video files
└── docs/                     # Documentation
```

## 🛠️ Installation

### 🐳 Option 1: Docker (Recommended)

**Prerequisites:**
- Docker Engine 20.10+
- Docker Compose 2.0+
- 16GB+ RAM
- 50GB+ free disk space
- Optional: NVIDIA GPU with Docker support

**Quick Setup:**
```bash
cd Sakshi-21-OCT
./docker-start.sh
```

**📖 See [DOCKER_README.md](DOCKER_README.md) for detailed instructions**

### 🐍 Option 2: Traditional Python Installation

**Prerequisites:**
- Python 3.8 or higher
- CUDA-compatible GPU (recommended for optimal performance)
- 8GB+ RAM
- 10GB+ free disk space

**Setup Instructions:**

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Sakshi-15-October
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r config/requirements.txt
   ```

4. **Setup PostgreSQL database**
   ```bash
   python3 setup_postgresql.py
   ```

5. **Configure video sources**
   Edit `config/rtsp_links.txt` to add your video sources:
   ```
   # Format: VIDEO_PATH, Channel_Name, App1, App2, App3, ...
   rtsp://your-camera-ip/stream, Main Entrance, PeopleCounter
   /path/to/video.mp4, Checkout Area, QueueMonitor, Shoplifting
   ```

5. **Configure Telegram notifications (optional)**
   Edit `main_app.py` and update:
   ```python
   TELEGRAM_BOT_TOKEN = "your_bot_token"
   TELEGRAM_CHAT_ID = "your_chat_id"
   ```

## 🐳 Docker Deployment (Recommended)

**NEW**: Sakshi AI now supports Docker deployment with microservices architecture!

Each processor runs in its own container for better:
- ✅ **Scalability** - Scale individual processors independently
- ✅ **Reliability** - One processor failure doesn't affect others
- ✅ **Easy deployment** - One command to start everything
- ✅ **Isolation** - Better resource management and security

### Quick Start with Docker
```bash
# Start all services
./docker-start.sh

# View status
./docker-status.sh

# View logs
./docker-logs.sh

# Stop services
./docker-stop.sh
```

**📖 See [DOCKER_README.md](DOCKER_README.md) for complete Docker documentation**

## 🚀 Traditional Usage (Without Docker)

### Starting the Application

```bash
python run.py
```

The application will start on `http://localhost:5001`

### Web Interface

- **Landing Page**: `http://localhost:5001/`
- **Dashboard**: `http://localhost:5001/dashboard`

### API Endpoints

- `GET /video_feed/<app_name>/<channel_id>` - Live video feed
- `GET /history/<app_name>` - Detection history
- `POST /api/set_roi` - Configure ROI regions
- `GET /report/<channel_id>/<date>` - Analytics reports
- `GET /queue_report/<channel_id>` - Queue analytics
- `GET /shutter_report/<channel_id>` - Shutter reports
- `GET /reports/security/<channel_id>` - Security reports

## 📊 Analytics & Reporting

### People Counter Analytics
- Real-time footfall tracking
- Hourly and daily reports
- Peak hour analysis
- Traffic pattern insights

### Queue Monitoring
- Live queue length monitoring
- Queue analytics and trends
- Alert system for long queues
- Historical queue data

### Security Monitoring
- Security personnel tracking
- Violation detection and logging
- Real-time alerts
- Incident reporting

### Shutter Monitoring
- Open/close time tracking
- Video evidence capture
- Daily operation reports
- Compliance monitoring

## 🔧 Configuration

### Model Configuration
Models are configured in `main_app.py` under `APP_TASKS_CONFIG`:

```python
APP_TASKS_CONFIG = {
    'Shoplifting': {
        'model_path': 'models/best_shoplift.pt',
        'target_class_id': 1,
        'confidence': 0.8,
        'is_gif': True
    },
    # ... other configurations
}
```

### ROI Configuration
Use the web interface to set Region of Interest (ROI) for specific monitoring areas.

### Database Configuration
The application is configured to use PostgreSQL for production:

```python
DATABASE_URL = "postgresql://postgres:Tneural01@localhost:5432/sakshi"
```

**PostgreSQL Setup:**
1. Install PostgreSQL dependencies: `pip install psycopg2-binary`
2. Setup database: `python3 setup_postgresql.py`
3. Migrate from SQLite (if needed): `python3 migrate_to_postgresql.py`

## 🎯 Supported Video Sources

- **RTSP Streams**: IP cameras and network video sources
- **Video Files**: MP4, AVI, MOV, MKV, and other formats
- **USB Cameras**: Direct camera input
- **Network Streams**: HTTP/HTTPS video streams

## 📈 Performance Optimization

### GPU Acceleration
Ensure CUDA is properly installed for GPU acceleration:
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

### Memory Management
- Adjust batch sizes based on available memory
- Use appropriate model sizes for your hardware
- Monitor system resources during operation

## 🔒 Security Considerations

- Change default database credentials
- Use HTTPS in production
- Secure API endpoints
- Regular security updates
- Data encryption for sensitive information

## 🐛 Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Ensure model files are in the `models/` directory
   - Check file permissions
   - Verify model compatibility

2. **Video Stream Issues**
   - Check network connectivity for RTSP streams
   - Verify video file paths
   - Ensure proper codec support

3. **Database Errors**
   - Check database file permissions
   - Verify SQLite installation
   - Clear database if corrupted

4. **Performance Issues**
   - Monitor GPU/CPU usage
   - Adjust processing parameters
   - Consider hardware upgrades

### Logs and Debugging

Enable detailed logging by modifying the logging level in `main_app.py`:
```python
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Check the documentation
- Review the troubleshooting section

## 🔄 Updates and Maintenance

- Regular model updates
- Security patches
- Feature enhancements
- Performance optimizations

---

**Sakshi AI** - Intelligent Video Analytics for Modern Businesses
