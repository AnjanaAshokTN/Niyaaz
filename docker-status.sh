#!/bin/bash
# Sakshi AI - Docker Status Script

# Detect docker compose command (try v2 first, then v1)
if docker compose version &> /dev/null 2>&1; then
    DOCKER_COMPOSE="docker compose"
elif command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE="docker-compose"
else
    echo "❌ Docker Compose not found."
    exit 1
fi

echo "📊 Sakshi AI Service Status"
echo "========================================"
echo ""

echo "🐳 Container Status:"
$DOCKER_COMPOSE ps

echo ""
echo "💾 Resource Usage:"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"

echo ""
echo "🏥 Health Checks:"

# Check main app
if curl -s http://localhost:5001/health > /dev/null 2>&1; then
    echo "✅ Main App: Healthy"
else
    echo "❌ Main App: Unhealthy"
fi

# Check PostgreSQL
if docker exec sakshi-postgres pg_isready -U postgres > /dev/null 2>&1; then
    echo "✅ PostgreSQL: Healthy"
else
    echo "❌ PostgreSQL: Unhealthy"
fi

echo ""
echo "📊 Database Records:"
docker exec sakshi-postgres psql -U postgres -d sakshi -t -c "SELECT COUNT(*) FROM detections;" 2>/dev/null | xargs echo "Detections:" || echo "Detections: N/A"
docker exec sakshi-postgres psql -U postgres -d sakshi -t -c "SELECT COUNT(*) FROM daily_footfall;" 2>/dev/null | xargs echo "Daily Footfall Records:" || echo "Daily Footfall: N/A"
docker exec sakshi-postgres psql -U postgres -d sakshi -t -c "SELECT COUNT(*) FROM queue_logs;" 2>/dev/null | xargs echo "Queue Logs:" || echo "Queue Logs: N/A"

echo ""
echo "🌐 Access Points:"
echo "   Dashboard: http://localhost:5001"
echo "   Database: localhost:5432 (postgres/Tneural01)"

