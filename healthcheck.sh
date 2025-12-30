#!/bin/bash
# ==============================================================================
# Health check script for Docling ExApp
# ==============================================================================

set -e

# Check if the application is responding
APP_PORT="${APP_PORT:-9000}"

response=$(curl -sf "http://localhost:${APP_PORT}/heartbeat" 2>/dev/null) || exit 1

# Verify response contains expected status
if echo "$response" | grep -q '"status":"ok"'; then
    exit 0
else
    exit 1
fi

