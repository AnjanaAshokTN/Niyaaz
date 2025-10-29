# 🐳 Docker Service Control Guide

Complete guide for starting/stopping individual Docker containers in Sakshi AI.

---

## 📋 Available Services

| Service Name | Container Name | Port | App Handled |
|--------------|----------------|------|-------------|
| `postgres` | sakshi-postgres | 5433 | Database (Required) |
| `main-app` | sakshi-main-app | 5001 | Frontend & API (Required) |
| `people-counter-processor` | sakshi-people-counter | 5010 | PeopleCounter |
| `kitchen-compliance-processor` | sakshi-kitchen-compliance | 5015 | KitchenCompliance |
| `queue-monitor-processor` | sakshi-queue-monitor | 5011 | QueueMonitor |
| `heatmap-processor` | sakshi-heatmap | 5013 | Heatmap |
| `security-monitor-processor` | sakshi-security-monitor | 5012 | Security |
| `shutter-monitor-processor` | sakshi-shutter-monitor | 5014 | ShutterMonitor |
| `detection-processor` | sakshi-detection-processor | 5016 | Shoplifting, QPOS, Generic |

---

## 🚀 Quick Start Options

### **Option 1: Start ONLY Selected Services (RECOMMENDED for your case)**

```bash
./start-selected-services.sh
```

This starts:
- ✅ PostgreSQL
- ✅ Main App
- ✅ People Counter
- ✅ Kitchen Compliance
- ✅ Queue Monitor

**All others stay OFF** - they won't appear in the dashboard!

---

### **Option 2: Manual Selection**

Start only the services you want:

```bash
# For PeopleCounter, Kitchen, Queue ONLY:
docker-compose up -d postgres main-app people-counter-processor kitchen-compliance-processor queue-monitor-processor
```

Add any service name to include it, remove to exclude it.

---

### **Option 3: Start All Services**

```bash
./docker-start.sh
# or
docker-compose up -d
```

This starts **everything** (all 9 apps).

---

## 🎛️ Individual Container Control

### **Start a Single Service**

```bash
# Start just one processor
docker-compose up -d people-counter-processor

# Start multiple specific services
docker-compose up -d kitchen-compliance-processor queue-monitor-processor
```

### **Stop a Single Service**

```bash
# Stop one service
docker-compose stop people-counter-processor

# Stop multiple
docker-compose stop heatmap-processor security-monitor-processor
```

### **Remove a Service (Stop + Delete Container)**

```bash
docker-compose rm -sf heatmap-processor
```

### **Restart a Service**

```bash
docker-compose restart people-counter-processor
```

---

## 📊 Monitoring Services

### **Check Which Services Are Running**

```bash
docker-compose ps
```

Output shows:
- `Up` = Running ✅
- `Exit` = Stopped ❌

### **Check Service Health**

```bash
# Main app
curl http://localhost:5001/health

# People Counter
curl http://localhost:5010/health

# Kitchen Compliance
curl http://localhost:5015/health

# Queue Monitor
curl http://localhost:5011/health
```

### **View Logs**

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f people-counter-processor

# Last 100 lines
docker-compose logs --tail=100 kitchen-compliance-processor

# Multiple services
docker-compose logs -f people-counter-processor queue-monitor-processor
```

---

## 🛑 Stopping Services

### **Stop All Services**

```bash
docker-compose down
```

### **Stop But Keep Data**

```bash
docker-compose stop
```

### **Stop and Remove Everything (Including Database)**

```bash
docker-compose down -v
```
⚠️ **Warning:** This deletes all database data!

---

## 🎯 Your Current Setup (3 Apps Only)

Based on your `rtsp_links.txt`, you want:

### **Services to START:**
```bash
docker-compose up -d \
    postgres \
    main-app \
    people-counter-processor \
    kitchen-compliance-processor \
    queue-monitor-processor
```

### **Services to KEEP OFF:**
- detection-processor (Shoplifting, QPOS, Generic)
- heatmap-processor (Heatmap)
- security-monitor-processor (Security)
- shutter-monitor-processor (ShutterMonitor)

---

## 🔄 Switching Configurations

### **Scenario 1: Add Heatmap Later**

1. Uncomment Heatmap line in `rtsp_links.txt`
2. Start the processor:
   ```bash
   docker-compose up -d heatmap-processor
   ```
3. Refresh dashboard - Heatmap will appear!

### **Scenario 2: Remove Queue Monitor**

1. Stop the service:
   ```bash
   docker-compose stop queue-monitor-processor
   ```
2. Comment Queue line in `rtsp_links.txt`
3. Refresh dashboard - Queue will disappear!

### **Scenario 3: Restart Everything Fresh**

```bash
# Stop all
docker-compose down

# Edit rtsp_links.txt as needed

# Start selected services
./start-selected-services.sh
```

---

## 🏥 Health Check Commands

Create a quick health check script:

```bash
#!/bin/bash
echo "Service Health Status:"
echo "======================"

services=(
    "5001:Main App"
    "5010:People Counter"
    "5011:Queue Monitor"
    "5015:Kitchen Compliance"
)

for service in "${services[@]}"; do
    port="${service%%:*}"
    name="${service##*:}"
    if curl -s http://localhost:$port/health > /dev/null 2>&1; then
        echo "✅ $name (port $port)"
    else
        echo "❌ $name (port $port)"
    fi
done
```

---

## 🎨 Frontend Display

**Only running processors appear in the dashboard!**

If you start only:
- people-counter-processor
- kitchen-compliance-processor  
- queue-monitor-processor

Then the dashboard at http://localhost:5001/dashboard will show **ONLY**:
- PeopleCounter section
- KitchenCompliance section
- QueueMonitor section

Other apps (Heatmap, Security, etc.) **won't appear at all**.

---

## 📝 Common Commands Cheat Sheet

```bash
# Start your 3 apps only
./start-selected-services.sh

# Check what's running
docker-compose ps

# View logs (all)
docker-compose logs -f

# View logs (one service)
docker-compose logs -f people-counter-processor

# Stop all
docker-compose down

# Stop one service
docker-compose stop queue-monitor-processor

# Start one service
docker-compose up -d queue-monitor-processor

# Restart one service
docker-compose restart people-counter-processor

# Health check
curl http://localhost:5001/health
curl http://localhost:5010/health
curl http://localhost:5011/health
curl http://localhost:5015/health
```

---

## 🚨 Troubleshooting

**Service won't start:**
```bash
# Check logs
docker-compose logs service-name

# Check if port is in use
netstat -tlnp | grep PORT_NUMBER

# Rebuild and restart
docker-compose up -d --build service-name
```

**Frontend shows wrong apps:**
- Main app caches the service list
- Restart main app after changing processors:
  ```bash
  docker-compose restart main-app
  ```

**Database connection issues:**
- Make sure postgres is running first:
  ```bash
  docker-compose up -d postgres
  sleep 10
  docker-compose up -d main-app people-counter-processor
  ```

---

## ✅ Recommended Startup Sequence

For your 3-app setup:

```bash
# 1. Start database first
docker-compose up -d postgres

# 2. Wait for database to be ready
sleep 10

# 3. Start main app
docker-compose up -d main-app

# 4. Wait for main app
sleep 5

# 5. Start processors
docker-compose up -d \
    people-counter-processor \
    kitchen-compliance-processor \
    queue-monitor-processor

# 6. Check status
docker-compose ps

# 7. Open dashboard
# http://localhost:5001/dashboard
```

**OR just use:** `./start-selected-services.sh` (does all of this!) 🚀

---

**Your services are now fully configurable! Start only what you need.** 🎯

