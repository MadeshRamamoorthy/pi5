#!/usr/bin/env bash
# Start the web admin UI for the face DB.
#
# Override host/port:
#   ADMIN_HOST=127.0.0.1 ADMIN_PORT=9000 ./start_admin.sh

set -euo pipefail

: "${PI5_DIR:=/home/echo/Documents/code/pi5}"
: "${ADMIN_HOST:=0.0.0.0}"
: "${ADMIN_PORT:=8081}"

cd "$PI5_DIR"

if [[ ! -f .venv/bin/activate ]]; then
    echo "Error: .venv not found at $PI5_DIR/.venv" >&2
    exit 1
fi

# shellcheck source=/dev/null
source .venv/bin/activate

if ! python -c "import flask" 2>/dev/null; then
    echo "Flask not installed in venv; running 'pip install flask'..." >&2
    pip install flask
fi

export ADMIN_HOST ADMIN_PORT

echo "---------------------------------------------------------------"
echo " pi5 face DB admin UI"
echo "   project : $PI5_DIR"
echo "   url     : http://$ADMIN_HOST:$ADMIN_PORT"
echo "---------------------------------------------------------------"

exec python admin_web.py
