#!/bin/bash

# Start script for F&O Neural Network Dashboard
# Starts both FastAPI backend and Vite frontend in parallel

set -e

echo "🚀 Starting F&O Neural Network Dashboard..."

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$SCRIPT_DIR"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Check if Node is available
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is required but not installed."
    exit 1
fi

# Function to kill processes on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down servers..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit 0
}

trap cleanup INT TERM

# Start FastAPI backend
echo -e "${BLUE}Starting FastAPI backend...${NC}"
cd "$REPO_ROOT"
python3 -m uvicorn dashboard.backend.api:app --reload --port 8000 &
BACKEND_PID=$!
echo -e "${GREEN}✓ Backend started (PID: $BACKEND_PID)${NC}"

# Wait a moment for backend to start
sleep 2

# Start Vite frontend
echo -e "${BLUE}Starting Vite frontend...${NC}"
cd "$SCRIPT_DIR/dashboard/frontend"

# Install dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
    echo "📦 Installing frontend dependencies..."
    npm install
fi

npm run dev &
FRONTEND_PID=$!
echo -e "${GREEN}✓ Frontend started (PID: $FRONTEND_PID)${NC}"

# Wait a bit and open browser
sleep 3
echo ""
echo -e "${GREEN}✓ Dashboard is ready!${NC}"
echo ""
echo "  Backend API:  http://localhost:8000"
echo "  API Docs:     http://localhost:8000/docs"
echo "  Frontend:     http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop both servers"

# Open browser (optional - comment out if not desired)
if command -v open &> /dev/null; then
    open http://localhost:5173
elif command -v xdg-open &> /dev/null; then
    xdg-open http://localhost:5173
fi

# Wait for background processes
wait $BACKEND_PID $FRONTEND_PID
