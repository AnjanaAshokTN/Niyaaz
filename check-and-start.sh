#!/bin/bash
# Sakshi AI - System Check and Start Script

set -e

echo "🔍 Sakshi AI - System Check & Start"
echo "===================================="
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check 1: RTSP Links Configuration
echo "📋 Checking RTSP configuration..."
if [ -f "config/rtsp_links.txt" ]; then
    lines=$(grep -v "^#" config/rtsp_links.txt | grep -v "^$" | wc -l)
    echo -e "${GREEN}✅ RTSP config found with $lines camera streams${NC}"
else
    echo -e "${RED}❌ config/rtsp_links.txt not found${NC}"
    exit 1
fi

# Check 2: Model files
echo ""
echo "🤖 Checking model files..."
missing_models=0
for model in best_shoplift.pt best_qpos.pt best_generic.pt yolov8n.pt shutter_model.pt gloves.pt apron-cap.pt security.pt; do
    if [ -f "models/$model" ]; then
        echo -e "${GREEN}✅ $model${NC}"
    else
        echo -e "${RED}❌ $model MISSING${NC}"
        missing_models=$((missing_models + 1))
    fi
done

if [ $missing_models -gt 0 ]; then
    echo -e "${YELLOW}⚠️  Warning: $missing_models model file(s) missing${NC}"
    echo "   Some detection apps may not work properly"
fi

# Check 3: Python environment
echo ""
echo "🐍 Checking Python environment..."
if command -v python3 &> /dev/null; then
    python_version=$(python3 --version 2>&1)
    echo -e "${GREEN}✅ $python_version${NC}"
else
    echo -e "${RED}❌ Python3 not found${NC}"
    exit 1
fi

# Check 4: Required Python packages
echo ""
echo "📦 Checking Python packages..."
required_packages=("flask" "cv2" "torch" "ultralytics" "sqlalchemy")
missing_packages=0

for package in "${required_packages[@]}"; do
    if python3 -c "import $package" 2>/dev/null; then
        echo -e "${GREEN}✅ $package${NC}"
    else
        echo -e "${RED}❌ $package not installed${NC}"
        missing_packages=$((missing_packages + 1))
    fi
done

if [ $missing_packages -gt 0 ]; then
    echo ""
    echo -e "${YELLOW}⚠️  Some packages are missing. Installing...${NC}"
    pip3 install -r requirements.txt
fi

# Check 5: Docker availability
echo ""
echo "🐳 Checking Docker..."
if docker info > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Docker is running${NC}"
    DOCKER_AVAILABLE=true
else
    echo -e "${YELLOW}⚠️  Docker is not running${NC}"
    DOCKER_AVAILABLE=false
fi

# Check 6: PostgreSQL
echo ""
echo "🗄️  Checking PostgreSQL..."
if command -v psql &> /dev/null; then
    if pg_isready -h localhost -p 5432 > /dev/null 2>&1; then
        echo -e "${GREEN}✅ PostgreSQL is running (port 5432)${NC}"
    elif pg_isready -h localhost -p 5433 > /dev/null 2>&1; then
        echo -e "${GREEN}✅ PostgreSQL is running (port 5433)${NC}"
    else
        echo -e "${YELLOW}⚠️  PostgreSQL not responding${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  PostgreSQL client not installed${NC}"
fi

# Check 7: Current running services
echo ""
echo "🏃 Checking currently running services..."
if netstat -tlnp 2>/dev/null | grep -q ":5001"; then
    echo -e "${GREEN}✅ Main app is already running on port 5001${NC}"
    ALREADY_RUNNING=true
else
    echo -e "${YELLOW}⚠️  Main app is NOT running${NC}"
    ALREADY_RUNNING=false
fi

# Summary and start options
echo ""
echo "========================================"
echo "📊 SYSTEM STATUS SUMMARY"
echo "========================================"

if [ "$ALREADY_RUNNING" = true ]; then
    echo -e "${GREEN}✅ Application is already running!${NC}"
    echo ""
    echo "🌐 Dashboard: http://localhost:5001/dashboard"
    echo ""
    echo "To restart:"
    if [ "$DOCKER_AVAILABLE" = true ]; then
        echo "  1. Stop: ./docker-stop.sh"
        echo "  2. Start: ./docker-start.sh"
    else
        echo "  1. Stop: pkill -f 'python3 run.py'"
        echo "  2. Start: python3 run.py"
    fi
    exit 0
fi

echo ""
echo "Select how to start Sakshi AI:"
echo ""
if [ "$DOCKER_AVAILABLE" = true ]; then
    echo "  1) Docker Mode (RECOMMENDED)"
    echo "     - All services in containers"
    echo "     - Better isolation and resource management"
    echo "     - Run: ./docker-start.sh"
    echo ""
fi
echo "  2) Traditional Mode"
echo "     - All services in one process"
echo "     - Simpler deployment"
echo "     - Run: python3 run.py"
echo ""

read -p "Choose option (1 or 2): " choice

case $choice in
    1)
        if [ "$DOCKER_AVAILABLE" = true ]; then
            echo ""
            echo "🚀 Starting in Docker mode..."
            chmod +x docker-start.sh
            ./docker-start.sh
        else
            echo -e "${RED}❌ Docker is not available${NC}"
            exit 1
        fi
        ;;
    2)
        echo ""
        echo "🚀 Starting in Traditional mode..."
        echo "   Main app will start on http://localhost:5001"
        echo ""
        echo "   Press Ctrl+C to stop"
        echo ""
        python3 run.py
        ;;
    *)
        echo -e "${RED}❌ Invalid option${NC}"
        exit 1
        ;;
esac

