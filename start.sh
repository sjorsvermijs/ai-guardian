#!/bin/bash
# AI Guardian - Start script (Mac/Linux)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "🛡️  AI Guardian - Starting..."

# Kill any existing processes on our ports and wait for release
echo "Cleaning up old processes..."
found=false
for port in 8000 5173 5174 5175; do
    pids=$(lsof -ti:$port 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo "  Killing process(es) on port $port (PIDs: $pids)"
        echo "$pids" | xargs kill -9 2>/dev/null || true
        found=true
    fi
done
if [ "$found" = true ]; then
    echo "  Waiting for ports to free up..."
    sleep 2
fi

# Activate venv if present
if [ -f "$SCRIPT_DIR/venv/bin/activate" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
    echo "  Activated venv"
fi

# Use nvm if available to ensure correct Node version
if [ -s "$HOME/.nvm/nvm.sh" ]; then
    source "$HOME/.nvm/nvm.sh"
    nvm use 20 --silent 2>/dev/null || echo "  Warning: Node 20 not available via nvm, using system Node"
fi

# Trap Ctrl+C to kill both
cleanup() {
    echo ""
    echo "Shutting down..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
    wait $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
    echo "Done."
    exit 0
}
trap cleanup INT TERM

# Start backend
echo "Starting backend (loading models, this takes ~30s)..."
cd "$SCRIPT_DIR"
python -m src.backend.api &
BACKEND_PID=$!

# Wait for backend to be ready
echo -n "  Waiting for backend"
for i in $(seq 1 60); do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo " ready!"
        break
    fi
    if ! kill -0 $BACKEND_PID 2>/dev/null; then
        echo " FAILED (backend process died)"
        exit 1
    fi
    echo -n "."
    sleep 1
done

# Check if backend actually started
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo " TIMEOUT (backend not ready after 60s)"
    kill $BACKEND_PID 2>/dev/null || true
    exit 1
fi

# Start frontend only after backend is ready
echo "Starting frontend..."
cd "$SCRIPT_DIR/src/frontend"
npm run dev &
FRONTEND_PID=$!
cd "$SCRIPT_DIR"

echo ""
echo "✅ AI Guardian running!"
echo "   Backend:  http://localhost:8000"
echo "   Frontend: http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop both services."

# Wait for either to exit
wait
