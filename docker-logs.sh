#!/bin/bash
# Sakshi AI - Docker Logs Viewer Script

# Detect docker compose command (try v2 first, then v1)
if docker compose version &> /dev/null 2>&1; then
    DOCKER_COMPOSE="docker compose"
elif command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE="docker-compose"
else
    echo "❌ Docker Compose not found."
    exit 1
fi

if [ -z "$1" ]; then
    echo "📋 Viewing logs for all services..."
    echo "Press Ctrl+C to exit"
    echo ""
    $DOCKER_COMPOSE logs -f
else
    echo "📋 Viewing logs for: $1"
    echo "Press Ctrl+C to exit"
    echo ""
    $DOCKER_COMPOSE logs -f "$1"
fi

