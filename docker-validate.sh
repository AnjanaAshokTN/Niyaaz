#!/bin/bash
# Sakshi AI - Docker Setup Validation Script

set -e

echo "🔍 Sakshi AI Docker Setup Validation"
echo "========================================"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

ERRORS=0
WARNINGS=0

check_command() {
    if command -v $1 &> /dev/null; then
        echo -e "${GREEN}✅${NC} $1 is installed"
        return 0
    else
        echo -e "${RED}❌${NC} $1 is not installed"
        ERRORS=$((ERRORS + 1))
        return 1
    fi
}

check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✅${NC} $1 exists"
        return 0
    else
        echo -e "${RED}❌${NC} $1 not found"
        ERRORS=$((ERRORS + 1))
        return 1
    fi
}

check_dir() {
    if [ -d "$1" ]; then
        echo -e "${GREEN}✅${NC} $1 exists"
        return 0
    else
        echo -e "${YELLOW}⚠️${NC}  $1 not found"
        WARNINGS=$((WARNINGS + 1))
        return 1
    fi
}

echo "1️⃣  Checking System Requirements"
echo "-----------------------------------"
check_command docker
check_command docker-compose

if docker info &> /dev/null; then
    echo -e "${GREEN}✅${NC} Docker daemon is running"
else
    echo -e "${RED}❌${NC} Docker daemon is not running"
    ERRORS=$((ERRORS + 1))
fi

# Check Docker version
DOCKER_VERSION=$(docker version --format '{{.Server.Version}}' 2>/dev/null || echo "0.0.0")
echo "   Docker version: $DOCKER_VERSION"

# Check Docker Compose version
if command -v docker-compose &> /dev/null; then
    COMPOSE_VERSION=$(docker-compose version --short 2>/dev/null || echo "0.0.0")
    echo "   Docker Compose version: $COMPOSE_VERSION"
elif docker compose version &> /dev/null; then
    COMPOSE_VERSION=$(docker compose version --short 2>/dev/null || echo "0.0.0")
    echo "   Docker Compose version: $COMPOSE_VERSION (v2)"
else
    COMPOSE_VERSION="0.0.0"
    echo "   Docker Compose version: Not found"
fi

echo ""
echo "2️⃣  Checking NVIDIA GPU Support (Optional)"
echo "-------------------------------------------"
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✅${NC} nvidia-smi found"
    if docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
        echo -e "${GREEN}✅${NC} NVIDIA Docker runtime is working"
    else
        echo -e "${YELLOW}⚠️${NC}  NVIDIA Docker runtime not working (will use CPU)"
        WARNINGS=$((WARNINGS + 1))
    fi
else
    echo -e "${YELLOW}⚠️${NC}  No NVIDIA GPU detected (will use CPU)"
    WARNINGS=$((WARNINGS + 1))
fi

echo ""
echo "3️⃣  Checking Required Files"
echo "----------------------------"
check_file "docker-compose.yml"
check_file "config/requirements.txt"
check_file "config/rtsp_links.txt"
check_file "main_app.py"
check_file "run.py"

echo ""
echo "4️⃣  Checking Docker Files"
echo "-------------------------"
check_file "docker/Dockerfile.main"
check_file "docker/Dockerfile.detection"
check_file "docker/Dockerfile.people_counter"
check_file "docker/Dockerfile.heatmap"
check_file "docker/Dockerfile.kitchen_compliance"
check_file "docker/Dockerfile.queue_monitor"
check_file "docker/Dockerfile.security_monitor"
check_file "docker/Dockerfile.shutter_monitor"
check_file ".dockerignore"

echo ""
echo "5️⃣  Checking Service Files"
echo "---------------------------"
check_file "services/detection_service.py"
check_file "services/people_counter_service.py"
check_file "services/heatmap_service.py"
check_file "services/kitchen_compliance_service.py"
check_file "services/queue_monitor_service.py"
check_file "services/security_monitor_service.py"
check_file "services/shutter_monitor_service.py"

echo ""
echo "6️⃣  Checking Processor Files"
echo "-----------------------------"
check_file "processors/detection_processor.py"
check_file "processors/people_counter_processor.py"
check_file "processors/heatmap_processor.py"
check_file "processors/kitchen_compliance_monitor.py"
check_file "processors/queue_monitor_processor.py"
check_file "processors/security_monitor_1.py"
check_file "processors/shutter_monitor_processor006.py"

echo ""
echo "7️⃣  Checking Model Files"
echo "------------------------"
check_dir "models"
if [ -d "models" ]; then
    MODEL_COUNT=$(ls models/*.pt 2>/dev/null | wc -l)
    if [ $MODEL_COUNT -eq 0 ]; then
        echo -e "${RED}❌${NC} No model files found in models/"
        ERRORS=$((ERRORS + 1))
    else
        echo -e "${GREEN}✅${NC} Found $MODEL_COUNT model file(s)"
        ls models/*.pt 2>/dev/null | while read model; do
            echo "   - $(basename $model)"
        done
    fi
fi

echo ""
echo "8️⃣  Checking Directories"
echo "------------------------"
check_dir "static"
check_dir "static/detections"
check_dir "templates"

# Create missing directories
if [ ! -d "static/detections" ]; then
    echo "   Creating static/detections..."
    mkdir -p static/detections
fi
if [ ! -d "static/detections/shutter_videos" ]; then
    echo "   Creating static/detections/shutter_videos..."
    mkdir -p static/detections/shutter_videos
fi

echo ""
echo "9️⃣  Checking Configuration"
echo "--------------------------"

# Check rtsp_links.txt
if [ -f "config/rtsp_links.txt" ]; then
    LINE_COUNT=$(grep -v '^#' config/rtsp_links.txt | grep -v '^$' | wc -l)
    if [ $LINE_COUNT -eq 0 ]; then
        echo -e "${YELLOW}⚠️${NC}  No camera streams configured in config/rtsp_links.txt"
        WARNINGS=$((WARNINGS + 1))
    else
        echo -e "${GREEN}✅${NC} Found $LINE_COUNT camera stream(s) configured"
    fi
fi

# Check for .env file
if [ -f ".env" ]; then
    echo -e "${GREEN}✅${NC} .env file exists"
else
    echo -e "${YELLOW}⚠️${NC}  .env file not found (using defaults)"
    WARNINGS=$((WARNINGS + 1))
fi

echo ""
echo "🔟 Checking Utility Scripts"
echo "---------------------------"
check_file "docker-start.sh"
check_file "docker-stop.sh"
check_file "docker-logs.sh"
check_file "docker-status.sh"

# Check if scripts are executable
for script in docker-start.sh docker-stop.sh docker-logs.sh docker-status.sh; do
    if [ -x "$script" ]; then
        echo -e "${GREEN}✅${NC} $script is executable"
    else
        echo -e "${YELLOW}⚠️${NC}  $script is not executable (running chmod +x)"
        chmod +x "$script"
        WARNINGS=$((WARNINGS + 1))
    fi
done

echo ""
echo "1️⃣1️⃣  Docker Compose Validation"
echo "--------------------------------"
if docker compose config &> /dev/null 2>&1; then
    echo -e "${GREEN}✅${NC} docker-compose.yml is valid"
elif docker-compose config &> /dev/null 2>&1; then
    echo -e "${GREEN}✅${NC} docker-compose.yml is valid"
else
    echo -e "${RED}❌${NC} docker-compose.yml has errors"
    ERRORS=$((ERRORS + 1))
fi

echo ""
echo "========================================"
echo "📊 Validation Summary"
echo "========================================"
echo -e "Errors:   ${RED}$ERRORS${NC}"
echo -e "Warnings: ${YELLOW}$WARNINGS${NC}"
echo ""

if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✅ Setup is ready for Docker deployment!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Review config/rtsp_links.txt and add your camera streams"
    echo "  2. Ensure model files are in models/ directory"
    echo "  3. Run: ./docker-start.sh"
    echo ""
    exit 0
else
    echo -e "${RED}❌ Please fix the errors above before deploying${NC}"
    echo ""
    exit 1
fi

