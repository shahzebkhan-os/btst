#!/bin/bash
# startup.sh
# ==========
# Comprehensive startup script for F&O Neural Network Predictor
# Starts both backend (FastAPI) and frontend (React/Vite) with health checks
#
# Usage:
#   ./startup.sh                 # Start both backend and frontend
#   ./startup.sh --backend-only  # Start only backend
#   ./startup.sh --frontend-only # Start only frontend
#   ./startup.sh --test          # Run in test mode with quick checks

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BACKEND_PORT=8000
FRONTEND_PORT=5173
BACKEND_LOG="logs/backend.log"
FRONTEND_LOG="logs/frontend.log"
PID_DIR="logs"
BACKEND_PID_FILE="$PID_DIR/backend.pid"
FRONTEND_PID_FILE="$PID_DIR/frontend.pid"

# Flags
START_BACKEND=1
START_FRONTEND=1
TEST_MODE=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --backend-only)
            START_FRONTEND=0
            shift
            ;;
        --frontend-only)
            START_BACKEND=0
            shift
            ;;
        --test)
            TEST_MODE=1
            shift
            ;;
        --help)
            echo "Usage: ./startup.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --backend-only   Start only the backend API"
            echo "  --frontend-only  Start only the frontend"
            echo "  --test           Run in test mode with quick checks"
            echo "  --help           Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Create necessary directories
mkdir -p logs models output data

echo -e "${BLUE}================================================================================================${NC}"
echo -e "${BLUE}                     F&O NEURAL NETWORK PREDICTOR - STARTUP${NC}"
echo -e "${BLUE}================================================================================================${NC}"
echo ""
echo -e "${GREEN}Starting services...${NC}"
echo -e "  Backend Port: ${BACKEND_PORT}"
echo -e "  Frontend Port: ${FRONTEND_PORT}"
echo -e "  Test Mode: $([ $TEST_MODE -eq 1 ] && echo 'Yes' || echo 'No')"
echo ""

# Function to check if port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Function to wait for service to be ready
wait_for_service() {
    local url=$1
    local name=$2
    local max_attempts=30
    local attempt=1

    echo -n "Waiting for $name to be ready..."

    while [ $attempt -le $max_attempts ]; do
        if curl -s "$url" > /dev/null 2>&1; then
            echo -e " ${GREEN}✓${NC}"
            return 0
        fi
        echo -n "."
        sleep 1
        attempt=$((attempt + 1))
    done

    echo -e " ${RED}✗ (timeout)${NC}"
    return 1
}

# Function to stop existing services
stop_services() {
    echo -e "\n${YELLOW}Stopping existing services...${NC}"

    # Stop backend
    if [ -f "$BACKEND_PID_FILE" ]; then
        BACKEND_PID=$(cat "$BACKEND_PID_FILE")
        if ps -p $BACKEND_PID > /dev/null 2>&1; then
            echo "  Stopping backend (PID: $BACKEND_PID)"
            kill $BACKEND_PID 2>/dev/null || true
            rm "$BACKEND_PID_FILE"
        fi
    fi

    # Stop frontend
    if [ -f "$FRONTEND_PID_FILE" ]; then
        FRONTEND_PID=$(cat "$FRONTEND_PID_FILE")
        if ps -p $FRONTEND_PID > /dev/null 2>&1; then
            echo "  Stopping frontend (PID: $FRONTEND_PID)"
            kill $FRONTEND_PID 2>/dev/null || true
            rm "$FRONTEND_PID_FILE"
        fi
    fi

    # Force kill any processes on the ports
    if check_port $BACKEND_PORT; then
        echo "  Killing process on port $BACKEND_PORT"
        lsof -ti:$BACKEND_PORT | xargs kill -9 2>/dev/null || true
    fi

    if check_port $FRONTEND_PORT; then
        echo "  Killing process on port $FRONTEND_PORT"
        lsof -ti:$FRONTEND_PORT | xargs kill -9 2>/dev/null || true
    fi

    sleep 2
}

# Check Python version
check_python() {
    echo -e "\n${BLUE}Checking Python environment...${NC}"

    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}✗ Python3 not found${NC}"
        exit 1
    fi

    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    echo -e "  Python version: $PYTHON_VERSION ${GREEN}✓${NC}"

    # Check critical packages
    echo -n "  Checking critical packages..."
    if python3 -c "import fastapi, uvicorn, pandas, numpy" 2>/dev/null; then
        echo -e " ${GREEN}✓${NC}"
    else
        echo -e " ${RED}✗${NC}"
        echo -e "${YELLOW}Run: pip install -r requirements.txt${NC}"
        exit 1
    fi
}

# Check Node.js
check_node() {
    echo -e "\n${BLUE}Checking Node.js environment...${NC}"

    if ! command -v node &> /dev/null; then
        echo -e "${RED}✗ Node.js not found${NC}"
        echo -e "${YELLOW}Install Node.js from: https://nodejs.org/${NC}"
        return 1
    fi

    NODE_VERSION=$(node --version)
    echo -e "  Node version: $NODE_VERSION ${GREEN}✓${NC}"

    if [ ! -d "dashboard/frontend/node_modules" ]; then
        echo -e "${YELLOW}  Node modules not found. Installing...${NC}"
        cd dashboard/frontend
        npm install
        cd ../..
    else
        echo -e "  Node modules ${GREEN}✓${NC}"
    fi

    return 0
}

# Check data availability
check_data() {
    echo -e "\n${BLUE}Checking data availability...${NC}"

    local issues=0

    if [ ! -f "data/vix/india_vix.csv" ]; then
        echo -e "  ${YELLOW}⚠ VIX data missing${NC}"
        issues=$((issues + 1))
    else
        echo -e "  VIX data ${GREEN}✓${NC}"
    fi

    if [ ! -f "data/historical_data/reconstructed.csv" ]; then
        echo -e "  ${YELLOW}⚠ Historical data missing${NC}"
        issues=$((issues + 1))
    else
        echo -e "  Historical data ${GREEN}✓${NC}"
    fi

    if [ ! -f "data/extended/market_data_extended.parquet" ]; then
        echo -e "  ${YELLOW}⚠ Extended market data missing${NC}"
        issues=$((issues + 1))
    else
        echo -e "  Extended market data ${GREEN}✓${NC}"
    fi

    if [ $issues -gt 0 ]; then
        echo -e "\n${YELLOW}Some data files are missing. Run:${NC}"
        echo -e "  ${YELLOW}python data_downloader.py --all${NC}"
        if [ $TEST_MODE -eq 0 ]; then
            read -p "Continue anyway? (y/N) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
        fi
    fi
}

# Start backend
start_backend() {
    echo -e "\n${BLUE}Starting Backend (FastAPI)...${NC}"

    # Check if backend is already running
    if check_port $BACKEND_PORT; then
        echo -e "${YELLOW}Backend port $BACKEND_PORT is already in use${NC}"
        stop_services
    fi

    # Start backend
    nohup python3 -m uvicorn dashboard.backend.api:app \
        --host 0.0.0.0 \
        --port $BACKEND_PORT \
        --reload \
        > "$BACKEND_LOG" 2>&1 &

    BACKEND_PID=$!
    echo $BACKEND_PID > "$BACKEND_PID_FILE"

    echo -e "  Backend started (PID: $BACKEND_PID)"
    echo -e "  Logs: $BACKEND_LOG"

    # Wait for backend to be ready
    if wait_for_service "http://localhost:$BACKEND_PORT/" "Backend"; then
        echo -e "  ${GREEN}Backend is ready!${NC}"
        echo -e "  API: http://localhost:$BACKEND_PORT"
        echo -e "  Docs: http://localhost:$BACKEND_PORT/docs"
        return 0
    else
        echo -e "  ${RED}Backend failed to start. Check logs: $BACKEND_LOG${NC}"
        return 1
    fi
}

# Start frontend
start_frontend() {
    echo -e "\n${BLUE}Starting Frontend (React/Vite)...${NC}"

    # Check if frontend is already running
    if check_port $FRONTEND_PORT; then
        echo -e "${YELLOW}Frontend port $FRONTEND_PORT is already in use${NC}"
        stop_services
    fi

    # Check Node.js
    if ! check_node; then
        echo -e "${RED}Cannot start frontend without Node.js${NC}"
        return 1
    fi

    # Start frontend
    cd dashboard/frontend
    nohup npm run dev -- --host 0.0.0.0 --port $FRONTEND_PORT \
        > "../../$FRONTEND_LOG" 2>&1 &

    FRONTEND_PID=$!
    echo $FRONTEND_PID > "../../$FRONTEND_PID_FILE"
    cd ../..

    echo -e "  Frontend started (PID: $FRONTEND_PID)"
    echo -e "  Logs: $FRONTEND_LOG"

    # Wait for frontend to be ready
    if wait_for_service "http://localhost:$FRONTEND_PORT/" "Frontend"; then
        echo -e "  ${GREEN}Frontend is ready!${NC}"
        echo -e "  Dashboard: http://localhost:$FRONTEND_PORT"
        return 0
    else
        echo -e "  ${RED}Frontend failed to start. Check logs: $FRONTEND_LOG${NC}"
        return 1
    fi
}

# Main execution
main() {
    # Run checks
    check_python
    check_data

    # Stop existing services
    stop_services

    # Start services
    local backend_ok=0
    local frontend_ok=0

    if [ $START_BACKEND -eq 1 ]; then
        if start_backend; then
            backend_ok=1
        fi
    fi

    if [ $START_FRONTEND -eq 1 ]; then
        if start_frontend; then
            frontend_ok=1
        fi
    fi

    # Print summary
    echo ""
    echo -e "${BLUE}================================================================================================${NC}"
    echo -e "${BLUE}                                    STARTUP COMPLETE${NC}"
    echo -e "${BLUE}================================================================================================${NC}"
    echo ""

    if [ $START_BACKEND -eq 1 ]; then
        if [ $backend_ok -eq 1 ]; then
            echo -e "${GREEN}✓ Backend:${NC}  http://localhost:$BACKEND_PORT"
            echo -e "            API Docs: http://localhost:$BACKEND_PORT/docs"
        else
            echo -e "${RED}✗ Backend failed to start${NC}"
        fi
    fi

    if [ $START_FRONTEND -eq 1 ]; then
        if [ $frontend_ok -eq 1 ]; then
            echo -e "${GREEN}✓ Frontend:${NC} http://localhost:$FRONTEND_PORT"
        else
            echo -e "${RED}✗ Frontend failed to start${NC}"
        fi
    fi

    echo ""
    echo -e "${BLUE}Logs:${NC}"
    [ $START_BACKEND -eq 1 ] && echo -e "  Backend:  tail -f $BACKEND_LOG"
    [ $START_FRONTEND -eq 1 ] && echo -e "  Frontend: tail -f $FRONTEND_LOG"
    echo ""
    echo -e "${BLUE}To stop:${NC}  ./shutdown.sh"
    echo -e "${BLUE}================================================================================================${NC}"

    # Exit with error if any service failed
    if [ $START_BACKEND -eq 1 ] && [ $backend_ok -eq 0 ]; then
        exit 1
    fi
    if [ $START_FRONTEND -eq 1 ] && [ $frontend_ok -eq 0 ]; then
        exit 1
    fi
}

# Run main
main
