#!/bin/bash
# shutdown.sh
# ===========
# Gracefully shutdown all services
#
# Usage: ./shutdown.sh

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
BACKEND_PORT=8000
FRONTEND_PORT=5173
PID_DIR="logs"
BACKEND_PID_FILE="$PID_DIR/backend.pid"
FRONTEND_PID_FILE="$PID_DIR/frontend.pid"

echo -e "${BLUE}================================================================================================${NC}"
echo -e "${BLUE}                     F&O NEURAL NETWORK PREDICTOR - SHUTDOWN${NC}"
echo -e "${BLUE}================================================================================================${NC}"
echo ""

stopped=0

# Stop backend
if [ -f "$BACKEND_PID_FILE" ]; then
    BACKEND_PID=$(cat "$BACKEND_PID_FILE")
    if ps -p $BACKEND_PID > /dev/null 2>&1; then
        echo -e "${YELLOW}Stopping backend (PID: $BACKEND_PID)...${NC}"
        kill $BACKEND_PID 2>/dev/null || true
        sleep 1
        if ! ps -p $BACKEND_PID > /dev/null 2>&1; then
            echo -e "  ${GREEN}✓ Backend stopped${NC}"
            stopped=$((stopped + 1))
        else
            echo -e "  ${YELLOW}Force killing backend...${NC}"
            kill -9 $BACKEND_PID 2>/dev/null || true
        fi
        rm "$BACKEND_PID_FILE"
    else
        echo -e "${YELLOW}Backend PID file exists but process not running${NC}"
        rm "$BACKEND_PID_FILE"
    fi
fi

# Stop frontend
if [ -f "$FRONTEND_PID_FILE" ]; then
    FRONTEND_PID=$(cat "$FRONTEND_PID_FILE")
    if ps -p $FRONTEND_PID > /dev/null 2>&1; then
        echo -e "${YELLOW}Stopping frontend (PID: $FRONTEND_PID)...${NC}"
        kill $FRONTEND_PID 2>/dev/null || true
        sleep 1
        if ! ps -p $FRONTEND_PID > /dev/null 2>&1; then
            echo -e "  ${GREEN}✓ Frontend stopped${NC}"
            stopped=$((stopped + 1))
        else
            echo -e "  ${YELLOW}Force killing frontend...${NC}"
            kill -9 $FRONTEND_PID 2>/dev/null || true
        fi
        rm "$FRONTEND_PID_FILE"
    else
        echo -e "${YELLOW}Frontend PID file exists but process not running${NC}"
        rm "$FRONTEND_PID_FILE"
    fi
fi

# Kill any remaining processes on ports
if lsof -Pi :$BACKEND_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${YELLOW}Killing remaining processes on port $BACKEND_PORT...${NC}"
    lsof -ti:$BACKEND_PORT | xargs kill -9 2>/dev/null || true
    stopped=$((stopped + 1))
fi

if lsof -Pi :$FRONTEND_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${YELLOW}Killing remaining processes on port $FRONTEND_PORT...${NC}"
    lsof -ti:$FRONTEND_PORT | xargs kill -9 2>/dev/null || true
    stopped=$((stopped + 1))
fi

echo ""
if [ $stopped -gt 0 ]; then
    echo -e "${GREEN}✓ All services stopped${NC}"
else
    echo -e "${YELLOW}No services were running${NC}"
fi

echo -e "${BLUE}================================================================================================${NC}"
