#!/bin/bash
# Sakshi AI - Docker Stop Script

set -e

echo "🛑 Stopping Sakshi AI Docker Services..."
echo "========================================"

# Detect docker compose command (try v2 first, then v1)
if docker compose version &> /dev/null 2>&1; then
    DOCKER_COMPOSE="docker compose"
elif command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE="docker-compose"
else
    echo "❌ Docker Compose not found."
    exit 1
fi

$DOCKER_COMPOSE stop

echo ""
echo "✅ All services stopped."
echo ""
echo "💡 To start again: ./docker-start.sh"
echo "💡 To remove containers: $DOCKER_COMPOSE down"
echo "💡 To remove everything (including volumes): $DOCKER_COMPOSE down -v"

