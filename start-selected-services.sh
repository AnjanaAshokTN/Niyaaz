#!/bin/bash
# Start only selected Sakshi AI services
# For: PeopleCounter, KitchenCompliance, QueueMonitor

set -e

echo "🚀 Starting Selected Sakshi AI Services"
echo "========================================"
echo "Services to start:"
echo "  ✅ PostgreSQL Database"
echo "  ✅ Main App (Frontend & API)"
echo "  ✅ People Counter Processor"
echo "  ✅ Kitchen Compliance Processor"
echo "  ✅ Queue Monitor Processor"
echo ""
echo "Services NOT starting:"
echo "  ❌ Detection Processor (Shoplifting, QPOS, Generic)"
echo "  ❌ Heatmap Processor"
echo "  ❌ Security Monitor Processor"
echo "  ❌ Shutter Monitor Processor"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Determine docker-compose command
if docker compose version &> /dev/null 2>&1; then
    DOCKER_COMPOSE="docker compose"
elif command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE="docker-compose"
else
    echo "❌ Docker Compose not found."
    exit 1
fi

# Create necessary directories
mkdir -p static/detections
mkdir -p static/detections/shutter_videos

echo "📦 Building and starting selected containers..."
echo ""

# Start only the required services
$DOCKER_COMPOSE up -d --build \
    postgres \
    main-app \
    people-counter-processor \
    kitchen-compliance-processor \
    queue-monitor-processor

echo ""
echo "⏳ Waiting for services to initialize..."
sleep 10

# Check service status
echo ""
echo "📊 Running Containers:"
$DOCKER_COMPOSE ps

echo ""
echo "✅ Selected services started!"
echo ""
echo "📍 Access the dashboard at: http://localhost:5001/dashboard"
echo ""
echo "🔍 To view logs:"
echo "   docker-compose logs -f main-app"
echo "   docker-compose logs -f people-counter-processor"
echo "   docker-compose logs -f kitchen-compliance-processor"
echo "   docker-compose logs -f queue-monitor-processor"
echo ""
echo "🛑 To stop services:"
echo "   docker-compose down"
echo ""

# Health check
echo "🏥 Checking service health..."
sleep 5

if curl -s http://localhost:5001/health > /dev/null; then
    echo "✅ Main app is healthy!"
else
    echo "⚠️  Main app not responding yet. Check logs:"
    echo "   docker-compose logs main-app"
fi

if curl -s http://localhost:5010/health > /dev/null; then
    echo "✅ People Counter processor is healthy!"
else
    echo "⚠️  People Counter processor not responding"
fi

if curl -s http://localhost:5015/health > /dev/null; then
    echo "✅ Kitchen Compliance processor is healthy!"
else
    echo "⚠️  Kitchen Compliance processor not responding"
fi

if curl -s http://localhost:5011/health > /dev/null; then
    echo "✅ Queue Monitor processor is healthy!"
else
    echo "⚠️  Queue Monitor processor not responding"
fi

echo ""
echo "🎯 Only PeopleCounter, KitchenCompliance, and QueueMonitor will appear in the dashboard!"
echo ""

