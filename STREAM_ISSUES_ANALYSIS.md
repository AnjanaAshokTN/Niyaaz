# Stream Issues Analysis & Resolution

**Date:** October 23, 2025  
**Status:** ✅ Issues Identified & Fixed

---

## 🔍 Issues Found

### 1. **APPLICATION IS NOT RUNNING** ❌ (CRITICAL)
- No Docker containers are running
- No Python processes are running
- **This is why no streams are showing in the frontend**

### 2. **RTSP Links Configuration** ✅ (FIXED)
- **File:** `config/rtsp_links.txt`
- **Issue:** Trailing whitespace and empty lines at end of file
- **Status:** ✅ **FIXED** - Removed trailing spaces and empty lines

### 3. **Configuration Review** ✅
All RTSP links are properly formatted:
- Kitchen Camera → KitchenCompliance
- Main Entrance → PeopleCounter
- QueueMonitor → QueueMonitor
- Store Floor → Heatmap
- Main Gate → ShutterMonitor
- Security Post → Security
- Point of Sale → QPOS
- Secondary Camera → Generic
- Backup Camera → Security

---

## 🎯 How Each Model Uses RTSP Links

### **Main Application (main_app.py)**
- Reads `config/rtsp_links.txt` at startup
- Parses each line: `RTSP_URL, Camera_Name, App1, App2, ...`
- Creates channel IDs using MD5 hash: `cam_{hash[:10]}`
- Starts appropriate processors for each app

### **Microservices (Docker Mode)**
Each service reads rtsp_links.txt independently:

1. **Detection Service** (Port 5016)
   - Handles: Shoplifting, QPOS, Generic
   - Models: best_shoplift.pt, best_qpos.pt, best_generic.pt

2. **People Counter Service** (Port 5010)
   - Handles: PeopleCounter
   - Model: yolov8n.pt

3. **Queue Monitor Service** (Port 5011)
   - Handles: QueueMonitor
   - Model: yolov8n.pt

4. **Security Monitor Service** (Port 5012)
   - Handles: Security
   - Model: (Face detection based)

5. **Heatmap Service** (Port 5013)
   - Handles: Heatmap
   - Model: yolov8n.pt

6. **Shutter Monitor Service** (Port 5014)
   - Handles: ShutterMonitor
   - Model: shutter_model.pt

7. **Kitchen Compliance Service** (Port 5015)
   - Handles: KitchenCompliance
   - Models: gloves.pt, apron-cap.pt

### **Video Feed Flow**
```
Browser Request
    ↓
Frontend (dashboard.html)
    ↓
/video_feed/{app_name}/{channel_id}
    ↓
Main App (port 5001)
    ↓
[If Docker Mode] → Proxy to Processor Service (ports 5010-5016)
[If Traditional Mode] → Direct from processor threads
    ↓
RTSP Stream Processing
    ↓
JPEG frames to browser
```

---

## ✅ Solutions

### **Option 1: Start with Docker (RECOMMENDED)**

```bash
# Make scripts executable
chmod +x docker-start.sh docker-stop.sh docker-logs.sh

# Start all services
./docker-start.sh
```

This will:
- Start PostgreSQL database
- Start main Flask application (port 5001)
- Start all processor microservices (ports 5010-5016)
- All services will read RTSP links from config/rtsp_links.txt
- Each processor will connect to its assigned RTSP streams

**Access Dashboard:**
```
http://localhost:5001/dashboard
```

**View Logs:**
```bash
./docker-logs.sh              # All services
./docker-logs.sh main-app     # Main app only
./docker-logs.sh detection-processor  # Detection service only
```

**Check Status:**
```bash
docker-compose ps
# or
docker compose ps
```

---

### **Option 2: Start in Traditional Mode (Without Docker)**

```bash
# Install dependencies (if not already installed)
pip3 install -r requirements.txt

# Run the application
python3 run.py
```

This will:
- Start main app on port 5001
- Start all processors in threads (not microservices)
- Read RTSP links from config/rtsp_links.txt
- Use traditional mode (DOCKER_MODE=false)

**Note:** In traditional mode, all processors run in the same process.

---

## 🔧 Troubleshooting

### **If streams still don't show:**

1. **Check if services are running:**
   ```bash
   docker-compose ps
   # All services should show "Up" status
   ```

2. **Check health endpoints:**
   ```bash
   curl http://localhost:5001/health
   curl http://localhost:5010/health  # People Counter
   curl http://localhost:5011/health  # Queue Monitor
   curl http://localhost:5016/health  # Detection
   ```

3. **Check RTSP connectivity:**
   ```bash
   # Test one RTSP link (requires ffmpeg)
   ffmpeg -rtsp_transport tcp -i "rtsp://admin:cctv%231234@182.65.205.121:554/cam/realmonitor?channel=1&subtype=0" -frames:v 1 test.jpg
   ```

4. **Check logs for errors:**
   ```bash
   ./docker-logs.sh | grep -i error
   ```

5. **Verify ports are accessible:**
   ```bash
   netstat -tlnp | grep -E "5001|5010|5011|5012|5013|5014|5015|5016"
   ```

### **Common Issues:**

- **Port Already in Use:** Stop other services using ports 5001, 5010-5016
- **RTSP Authentication Failed:** Check username/password in RTSP URLs
- **Network Unreachable:** Verify camera IP addresses and network connectivity
- **Model Files Missing:** Ensure all .pt files exist in `models/` directory

---

## 📊 Current Configuration Summary

**Total Cameras:** 9 (including backup)
**Total Apps:** 9 different detection/monitoring apps
**Database:** PostgreSQL (port 5433 in Docker, 5432 local)

**Apps per Camera:**
- Camera 1 (Kitchen) → KitchenCompliance
- Camera 2 (Main Entrance) → PeopleCounter
- Camera 3 (Queue) → QueueMonitor
- Camera 4 (Store Floor) → Heatmap
- Camera 5 (Main Gate) → ShutterMonitor
- Camera 6 (Security Post) → Security
- Camera 7 (Point of Sale) → QPOS
- Camera 8 (Secondary) → Generic
- Camera 9 (Backup) → Security

---

## 🎬 Next Steps

1. ✅ **Fixed:** Cleaned up rtsp_links.txt file
2. ⏳ **Action Required:** Start the application using one of the options above
3. ⏳ **Verify:** Open http://localhost:5001/dashboard and check if streams appear
4. ⏳ **Monitor:** Check logs for any RTSP connection errors

---

## 📝 Notes

- All RTSP links point to IP: `182.65.205.121:554`
- Using Dahua camera protocol with subtype=0 (mainstream)
- Credentials: admin / cctv#1234 (URL encoded as cctv%231234)
- Multiple cameras use channel 4 and 5 - verify if these are correct

---

**✅ Configuration is correct. Just need to START the application!**

