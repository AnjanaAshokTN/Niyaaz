#!/bin/bash
# Sakshi AI - Docker Start Script

set -e

echo "🚀 Starting Sakshi AI Docker Services..."
echo "========================================"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Check if docker compose is available (try v2 first, then v1)
if docker compose version &> /dev/null 2>&1; then
    DOCKER_COMPOSE="docker compose"
    echo "✅ Using Docker Compose v2"
elif command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE="docker-compose"
    echo "✅ Using Docker Compose v1"
else
    echo "❌ Docker Compose not found. Please enable WSL integration in Docker Desktop."
    exit 1
fi

# Check if models directory exists
if [ ! -d "models" ]; then
    echo "⚠️  Warning: models/ directory not found"
fi

# Check if rtsp_links.txt exists
if [ ! -f "config/rtsp_links.txt" ]; then
    echo "⚠️  Warning: config/rtsp_links.txt not found"
    echo "   Please configure your camera streams before starting."
    exit 1
fi

# Create necessary directories
mkdir -p static/detections
mkdir -p static/detections/shutter_videos

echo ""
echo "📦 Building and starting containers..."
$DOCKER_COMPOSE up -d --build

echo ""
echo "⏳ Waiting for services to be healthy..."
sleep 10

# Check service status
echo ""
echo "📊 Service Status:"
$DOCKER_COMPOSE ps

echo ""
echo "✅ Sakshi AI is starting up!"
echo ""
echo "📍 Access the dashboard at: http://localhost:5001"
echo ""
echo "🔍 To view logs:"
echo "   ./docker-logs.sh"
echo ""
echo "🛑 To stop services:"
echo "   ./docker-stop.sh"
echo ""

# Check if main app is responding
echo "🏥 Checking main app health..."
sleep 5
if curl -s http://localhost:5001/health > /dev/null; then
    echo "✅ Main app is healthy!"
else
    echo "⚠️  Main app is not responding yet. Please check logs:"
    echo "   ./docker-logs.sh main-app"
fi

