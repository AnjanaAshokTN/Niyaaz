# 🚨 QUICK FIX GUIDE - Streams Not Showing

## THE PROBLEM
**Your application is NOT running!** That's why no streams are showing in the frontend.

---

## ✅ SOLUTION (Choose One)

### **Option A: Start with Docker (Easy & Recommended)**

```bash
# Just run this:
./docker-start.sh

# Wait 30 seconds, then open browser:
# http://localhost:5001/dashboard
```

### **Option B: Start without Docker**

```bash
# Run this:
python3 run.py

# Wait 30 seconds, then open browser:
# http://localhost:5001/dashboard
```

### **Option C: Use the Check Script (Safest)**

```bash
# This will check everything first:
./check-and-start.sh

# Follow the prompts
```

---

## 🔍 HOW TO VERIFY IT'S WORKING

### 1. Check if Main App is Running
```bash
curl http://localhost:5001/health
# Should return: {"status":"healthy","timestamp":"..."}
```

### 2. Open Dashboard
```
http://localhost:5001/dashboard
```

### 3. You Should See:
- ✅ Live video streams in the dashboard
- ✅ Each camera showing its feed
- ✅ Detection boxes on the video
- ✅ Status showing "X / Y Online"

---

## 🐛 IF STILL NOT WORKING

### Check Logs (Docker Mode)
```bash
./docker-logs.sh
```

### Check Logs (Traditional Mode)
Look at the terminal where you ran `python3 run.py`

### Common Error Messages & Fixes:

**"Failed to connect to RTSP"**
- Check if camera IPs are accessible
- Verify camera credentials
- Test with VLC: `vlc rtsp://admin:cctv#1234@182.65.205.121:554/cam/realmonitor?channel=1&subtype=0`

**"Port 5001 already in use"**
```bash
# Find and kill the process
sudo lsof -ti:5001 | xargs kill -9
```

**"Model file not found"**
- Ensure all .pt files are in `models/` directory:
  - best_shoplift.pt
  - best_qpos.pt
  - best_generic.pt
  - yolov8n.pt
  - shutter_model.pt
  - gloves.pt
  - apron-cap.pt
  - security.pt

**"Database connection failed"**
```bash
# Start PostgreSQL (Docker)
docker-compose up -d postgres

# Or install locally
sudo apt-get install postgresql
```

---

## 📸 RTSP Links Verified

All 9 camera streams are configured:

| Camera | Channel | App | Status |
|--------|---------|-----|--------|
| Kitchen Camera | 10 | KitchenCompliance | ✅ |
| Main Entrance | 1 | PeopleCounter | ✅ |
| QueueMonitor | 5 | QueueMonitor | ✅ |
| Store Floor | 7 | Heatmap | ✅ |
| Main Gate | 4 | ShutterMonitor | ✅ |
| Security Post | 4 | Security | ✅ |
| Point of Sale | 4 | QPOS | ✅ |
| Secondary Camera | 5 | Generic | ✅ |
| Backup Camera | 4 | Security | ✅ |

---

## ⚠️ IMPORTANT NOTES

1. **Multiple cameras use same channel numbers (4, 5)**
   - Channel 4 used by: Main Gate, Security Post, Point of Sale, Backup
   - Channel 5 used by: QueueMonitor, Secondary Camera
   - **This might be intentional OR a configuration error**
   - Verify with your camera setup documentation

2. **All cameras use same IP: 182.65.205.121**
   - This is typical for multi-channel DVR/NVR systems
   - Different channels = different camera feeds

3. **Credentials:** admin / cctv#1234
   - Make sure these are correct for your system

---

## 🎯 EXPECTED BEHAVIOR AFTER STARTING

1. **Console Output:**
   ```
   🚀 Starting Sakshi AI...
   ✅ PostgreSQL database connection successful.
   ✅ Loaded model 'yolov8n.pt' on 'cuda'
   ✅ Started processor 'PeopleCounter_cam_...'
   ✅ Starting Flask-SocketIO server on http://localhost:5001
   ```

2. **Browser (http://localhost:5001/dashboard):**
   - Shows 9 different app sections
   - Each with live video feed
   - Real-time detection overlays
   - Online status indicators

3. **Database:**
   - Detections saved automatically
   - Reports generated on demand
   - Historical data accessible

---

## 🆘 STILL STUCK?

1. Check `/home/abhijith/Sakshi-21-OCT/STREAM_ISSUES_ANALYSIS.md` for detailed analysis
2. Run `./check-and-start.sh` to diagnose all issues
3. Check logs for specific error messages

---

**TL;DR: Just run `./docker-start.sh` or `python3 run.py` - That's it!** 🚀

