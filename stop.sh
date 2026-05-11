#!/usr/bin/env bash
# Kill the ECHO SCOPE kiosk: Python backend + Chromium kiosk window.
#
# Use when start.sh got disconnected (ssh dropped, closed terminal)
# and Ctrl+C isn't available to fire the script's own trap. Otherwise
# Ctrl+C in the start.sh terminal does the same thing.
#
# Usage:
#   ./stop.sh              # graceful TERM, then KILL after 3 s
#   FORCE=1 ./stop.sh      # immediate KILL
#   ./stop.sh --ollama     # also stop the local hailo-ollama daemon

set -u

KIOSK_PORT="${KIOSK_PORT:-8090}"
FORCE="${FORCE:-0}"
STOP_OLLAMA=0
for arg in "$@"; do
    case "$arg" in
        --ollama) STOP_OLLAMA=1 ;;
        *) ;;
    esac
done

signal_then_kill() {
    local pattern="$1" label="$2"
    local pids
    pids=$(pgrep -f "$pattern" || true)
    if [[ -z "$pids" ]]; then
        echo "   $label : not running"
        return
    fi
    if [[ "$FORCE" == 1 ]]; then
        echo "   $label : KILLing pids $pids"
        kill -9 $pids 2>/dev/null || true
        return
    fi
    echo "   $label : TERM pids $pids"
    kill $pids 2>/dev/null || true
    for _ in 1 2 3 4 5 6; do
        sleep 0.5
        if ! pgrep -f "$pattern" >/dev/null; then
            return
        fi
    done
    echo "   $label : still alive, KILLing"
    kill -9 $(pgrep -f "$pattern") 2>/dev/null || true
}

echo "Stopping ECHO SCOPE..."

# Backend Python process.
signal_then_kill 'python.* main\.py' 'backend  '

# Chromium kiosk window.
signal_then_kill 'chromium.*--kiosk' 'chromium '
signal_then_kill 'chromium-browser.*--kiosk' 'chromium '

# Anything still listening on the kiosk port.
if command -v fuser >/dev/null 2>&1; then
    if fuser "${KIOSK_PORT}/tcp" >/dev/null 2>&1; then
        echo "   port $KIOSK_PORT: still bound -- releasing"
        fuser -k "${KIOSK_PORT}/tcp" 2>/dev/null || true
    else
        echo "   port $KIOSK_PORT: free"
    fi
fi

# Optionally tear down hailo-ollama too.
if [[ "$STOP_OLLAMA" == 1 ]]; then
    signal_then_kill 'hailo-ollama' 'ollama   '
fi

echo "done."
