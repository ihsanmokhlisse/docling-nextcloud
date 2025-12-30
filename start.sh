#!/bin/bash
# ==============================================================================
# Docling ExApp Startup Script
# ==============================================================================

set -e

echo "Starting Docling ExApp for Nextcloud..."
echo "Python version: $(python --version)"
echo "Working directory: $(pwd)"

# Set default values if not provided
export APP_HOST="${APP_HOST:-0.0.0.0}"
export APP_PORT="${APP_PORT:-9000}"
export APP_ID="${APP_ID:-docling}"

# Log level
export LOG_LEVEL="${LOG_LEVEL:-info}"

# Download Docling models on first run (if needed)
if [ ! -d "/app/cache/models" ]; then
    echo "Initializing Docling models (this may take a moment on first run)..."
fi

# Start the application
echo "Starting FastAPI application on ${APP_HOST}:${APP_PORT}..."
exec python -m uvicorn main:APP \
    --host "${APP_HOST}" \
    --port "${APP_PORT}" \
    --log-level "${LOG_LEVEL}" \
    --no-access-log

