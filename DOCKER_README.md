# Sakshi AI - Docker Deployment Guide

This guide explains how to deploy Sakshi AI using Docker containers with a microservices architecture.

## 🏗️ Architecture Overview

Sakshi AI uses a **microservices architecture** where each processor runs in its own container:

```
┌─────────────────────────────────────────────────────────────┐
│                     Docker Network                          │
│                                                               │
│  ┌──────────────┐      ┌──────────────────────────┐        │
│  │  PostgreSQL  │◄─────┤   Main Flask App         │        │
│  │   Database   │      │   (Web UI + API Gateway) │        │
│  └──────────────┘      └──────────────────────────┘        │
│         ▲                         ▲                          │
│         │                         │                          │
│         ├─────────┬───────┬───────┴──────┬────────┬────────┤
│         │         │       │              │        │        │
│  ┌──────┴────┐ ┌─┴──────┴┐  ┌──────────┴──┐  ┌─┴────────┴┐│
│  │ Detection │ │ People   │  │  Heatmap    │  │  Kitchen  ││
│  │ Processor │ │ Counter  │  │  Processor  │  │Compliance ││
│  └───────────┘ └──────────┘  └─────────────┘  └───────────┘│
│  ┌───────────┐ ┌──────────┐  ┌─────────────┐               │
│  │  Queue    │ │ Security │  │  Shutter    │               │
│  │  Monitor  │ │ Monitor  │  │  Monitor    │               │
│  └───────────┘ └──────────┘  └─────────────┘               │
└─────────────────────────────────────────────────────────────┘
```

### Services Breakdown

1. **postgres** - PostgreSQL database (shared by all services)
2. **main-app** - Flask web application (Dashboard + API Gateway)
3. **detection-processor** - Handles Shoplifting, QPOS, Generic detection
4. **people-counter-processor** - Bidirectional footfall tracking
5. **heatmap-processor** - Customer engagement heatmaps
6. **kitchen-compliance-processor** - Kitchen safety monitoring
7. **queue-monitor-processor** - Queue length analytics
8. **security-monitor-processor** - Security personnel monitoring
9. **shutter-monitor-processor** - Shop open/close time tracking

## 📋 Prerequisites

### Required
- Docker Engine 20.10+
- Docker Compose 2.0+
- 16GB+ RAM
- 50GB+ free disk space

### Optional (for GPU acceleration)
- NVIDIA GPU with CUDA support
- NVIDIA Docker runtime (`nvidia-docker2`)

### Install Docker

**Ubuntu/Debian:**
```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
```

**For NVIDIA GPU support:**
```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

## 🚀 Quick Start

### 1. Clone and Navigate to Project
```bash
cd /home/abhijith/Sakshi-21-OCT
```

### 2. Prepare Model Files
Ensure all required model files are in the `models/` directory:
- `yolov8n.pt`
- `best_shoplift.pt`
- `best_qpos.pt`
- `best_generic.pt`
- `shutter_model.pt`

### 3. Configure Video Sources
Edit `config/rtsp_links.txt` to add your camera streams:
```
# Format: RTSP_URL, Channel_Name, App1, App2, ...
rtsp://admin:password@192.168.1.100:554/stream, Main Entrance, PeopleCounter
rtsp://admin:password@192.168.1.101:554/stream, Store Floor, Heatmap, Shoplifting
```

### 4. Configure Environment (Optional)
Create a `.env` file (use `.env.example` as template):
```bash
cp .env.example .env
nano .env
```

Update Telegram settings if needed:
```
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

### 5. Build and Start Services

**With GPU:**
```bash
docker-compose up -d
```

**Without GPU (CPU only):**
```bash
# Create override file for CPU
cp docker-compose.override.yml.example docker-compose.override.yml
docker-compose up -d
```

### 6. Check Status
```bash
docker-compose ps
```

All services should show as "Up". Example output:
```
NAME                         STATUS          PORTS
sakshi-postgres              Up (healthy)    5432/tcp
sakshi-main-app              Up              0.0.0.0:5001->5001/tcp
sakshi-detection-processor   Up
sakshi-people-counter        Up
sakshi-heatmap              Up
...
```

### 7. Access Dashboard
Open browser and navigate to:
```
http://localhost:5001
```

## 🔧 Management Commands

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f main-app
docker-compose logs -f detection-processor
docker-compose logs -f people-counter-processor
```

### Stop Services
```bash
docker-compose stop
```

### Restart Services
```bash
# Restart all
docker-compose restart

# Restart specific service
docker-compose restart detection-processor
```

### Stop and Remove Containers
```bash
docker-compose down
```

### Stop and Remove Everything (including volumes)
```bash
docker-compose down -v
```

### Rebuild After Code Changes
```bash
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### Scale Specific Processors
```bash
# Run multiple instances of a processor (if load balancing is needed)
docker-compose up -d --scale detection-processor=2
```

## 📊 Monitoring

### Check Resource Usage
```bash
docker stats
```

### Check Service Health
```bash
# Main app health
curl http://localhost:5001/health

# Individual container health
docker inspect --format='{{.State.Health.Status}}' sakshi-main-app
```

### Access PostgreSQL Database
```bash
docker exec -it sakshi-postgres psql -U postgres -d sakshi
```

Common queries:
```sql
-- Check detections
SELECT * FROM detections ORDER BY timestamp DESC LIMIT 10;

-- Check footfall data
SELECT * FROM daily_footfall ORDER BY report_date DESC;

-- Check queue logs
SELECT * FROM queue_logs ORDER BY timestamp DESC LIMIT 20;
```

## 🔄 Adding New Processors

The architecture is designed to be **easily extensible**. To add a new processor:

### 1. Create Processor Code
Add your processor to `processors/your_new_processor.py`

### 2. Create Service Wrapper
Create `services/your_new_processor_service.py`:
```python
#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processors.your_new_processor import YourNewProcessor
# ... rest of service code
```

### 3. Create Dockerfile
Create `docker/Dockerfile.your_new_processor`:
```dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 \
    libpq-dev gcc g++ && rm -rf /var/lib/apt/lists/*

COPY config/requirements.txt /app/config/requirements.txt
RUN pip install --no-cache-dir -r config/requirements.txt

# Copy processor code
COPY processors/your_new_processor.py /app/processors/
COPY services/your_new_processor_service.py /app/services/

RUN mkdir -p /app/static/detections /app/models

ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "services/your_new_processor_service.py"]
```

### 4. Add to Docker Compose
Edit `docker-compose.yml` and add:
```yaml
  your-new-processor:
    build:
      context: .
      dockerfile: docker/Dockerfile.your_new_processor
    container_name: sakshi-your-new-processor
    environment:
      - DATABASE_URL=postgresql+psycopg2://postgres:Tneural01@postgres:5432/sakshi
      - MAIN_APP_URL=http://main-app:5001
      - PROCESSOR_TYPE=your_new_processor
    volumes:
      - ./config:/app/config
      - ./models:/app/models
      - ./static:/app/static
    depends_on:
      - postgres
      - main-app
    networks:
      - sakshi-network
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### 5. Update Configuration
Add your processor to `config/rtsp_links.txt`:
```
rtsp://your-camera-url, Camera Name, YourNewProcessor
```

### 6. Deploy
```bash
docker-compose up -d --build
```

That's it! Your new processor is now part of the system! 🎉

## 🛠️ Troubleshooting

### Service Won't Start
```bash
# Check logs
docker-compose logs service-name

# Check if port is already in use
sudo netstat -tlnp | grep 5001

# Remove and rebuild
docker-compose down
docker-compose up -d --build
```

### Database Connection Issues
```bash
# Check if PostgreSQL is healthy
docker-compose ps postgres

# Restart database
docker-compose restart postgres

# Check database logs
docker-compose logs postgres
```

### GPU Not Detected
```bash
# Check NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# If not working, check docker daemon.json
cat /etc/docker/daemon.json
# Should contain: {"default-runtime": "nvidia"}
```

### Out of Memory
```bash
# Check memory usage
docker stats

# Reduce number of processors or adjust limits in docker-compose.yml
```

### Model Files Not Found
```bash
# Ensure models are in the correct location
ls -lh models/

# Check volume mounts
docker inspect sakshi-detection-processor | grep -A 10 Mounts
```

## 📝 Configuration Files

### docker-compose.yml
Main orchestration file defining all services

### docker-compose.override.yml
Local overrides (e.g., for CPU-only deployment)

### .env
Environment variables (not committed to git)

### config/rtsp_links.txt
Camera stream and processor assignments

### config/requirements.txt
Python dependencies

## 🔒 Security Best Practices

1. **Change default database password**
   ```bash
   # Edit docker-compose.yml or use .env file
   POSTGRES_PASSWORD=your_secure_password
   ```

2. **Use secrets for Telegram tokens**
   ```bash
   # Store in .env file (not in docker-compose.yml)
   TELEGRAM_BOT_TOKEN=your_token
   ```

3. **Limit network exposure**
   ```yaml
   # In docker-compose.yml, only expose what's needed
   ports:
     - "127.0.0.1:5001:5001"  # Only localhost access
   ```

4. **Regular updates**
   ```bash
   docker-compose pull
   docker-compose up -d
   ```

## 📈 Performance Tuning

### For Multiple Cameras
Adjust resource limits in `docker-compose.yml`:
```yaml
deploy:
  resources:
    limits:
      cpus: '4'
      memory: 8G
```

### For High-Resolution Streams
Increase shared memory:
```yaml
shm_size: '2gb'
```

### For Better GPU Utilization
Assign specific GPUs to services:
```yaml
environment:
  - CUDA_VISIBLE_DEVICES=0,1
```

## 🆘 Support

For issues or questions:
1. Check logs: `docker-compose logs -f`
2. Verify configuration files
3. Review this documentation
4. Check GitHub issues

---

**Sakshi AI - Intelligent Video Analytics Platform**

