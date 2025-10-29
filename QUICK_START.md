# 🚀 Sakshi AI Docker - Quick Start Guide

## Prerequisites Checklist
- [ ] Docker installed and running
- [ ] NVIDIA GPU drivers (optional, for GPU support)
- [ ] Model files in `models/` directory
- [ ] Camera streams configured in `config/rtsp_links.txt`

## 3-Step Deployment

### Step 1: Validate Setup
```bash
./docker-validate.sh
```

### Step 2: Start Services
```bash
./docker-start.sh
```

### Step 3: Access Dashboard
Open browser: **http://localhost:5001**

## Essential Commands

| Task | Command |
|------|---------|
| Start all services | `./docker-start.sh` |
| Stop all services | `./docker-stop.sh` |
| View all logs | `./docker-logs.sh` |
| View service logs | `./docker-logs.sh service-name` |
| Check status | `./docker-status.sh` |
| Validate setup | `./docker-validate.sh` |
| Restart service | `docker compose restart service-name` |
| Rebuild service | `docker compose up -d --build service-name` |
| Remove everything | `docker compose down -v` |

## Service Names
- `postgres` - Database
- `main-app` - Web UI
- `detection-processor` - Shoplifting/QPOS/Generic
- `people-counter-processor` - Footfall
- `heatmap-processor` - Heatmap
- `kitchen-compliance-processor` - Kitchen
- `queue-monitor-processor` - Queue
- `security-monitor-processor` - Security
- `shutter-monitor-processor` - Shutter

## Common Tasks

### View Logs
```bash
# All services
./docker-logs.sh

# Specific service
./docker-logs.sh detection-processor
./docker-logs.sh main-app
```

### Check Status
```bash
./docker-status.sh
```

### Restart a Service
```bash
docker compose restart detection-processor
```

### Update Code and Rebuild
```bash
# Stop services
./docker-stop.sh

# Rebuild
docker compose build --no-cache

# Start
./docker-start.sh
```

### Database Access
```bash
docker exec -it sakshi-postgres psql -U postgres -d sakshi
```

### View Resource Usage
```bash
docker stats
```

## Troubleshooting

### Services Won't Start
```bash
# Check logs
./docker-logs.sh

# Check Docker
docker ps
docker compose ps
```

### Port Already in Use
Edit `docker-compose.yml`:
```yaml
ports:
  - "5002:5001"  # Change 5001 to any free port
```

### Out of Memory
```bash
# Check usage
docker stats

# Reduce services or adjust limits in docker-compose.yml
```

### GPU Issues
```bash
# Test GPU
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Use CPU only
cp docker-compose.override.yml.example docker-compose.override.yml
```

## Adding New Camera

1. Edit `config/rtsp_links.txt`:
```
rtsp://camera-url, Camera Name, PeopleCounter, Shoplifting
```

2. Restart services:
```bash
./docker-stop.sh
./docker-start.sh
```

## Health Check
```bash
# Main app
curl http://localhost:5001/health

# Database
docker exec sakshi-postgres pg_isready -U postgres
```

## Configuration Files

| File | Purpose |
|------|---------|
| `docker-compose.yml` | Service definitions |
| `config/rtsp_links.txt` | Camera streams |
| `.env` | Environment variables (optional) |
| `config/requirements.txt` | Python dependencies |

## Endpoints

| URL | Description |
|-----|-------------|
| http://localhost:5001 | Landing page |
| http://localhost:5001/dashboard | Main dashboard |
| http://localhost:5001/health | Health check |

## Default Credentials

**PostgreSQL Database:**
- Host: `localhost:5432`
- Database: `sakshi`
- User: `postgres`
- Password: `Tneural01`

⚠️ **Change in production!**

## Getting Help

1. Check validation: `./docker-validate.sh`
2. View logs: `./docker-logs.sh`
3. Read full guide: `DOCKER_README.md`
4. Check summary: `DOCKER_SETUP_SUMMARY.md`

## Emergency Stop

```bash
# Stop all services
docker compose down

# Stop and remove everything (including data)
docker compose down -v
```

---

**For detailed information, see:**
- `DOCKER_README.md` - Complete Docker guide
- `DOCKER_SETUP_SUMMARY.md` - Setup summary
- `README.md` - Main documentation

