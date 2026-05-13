#!/usr/bin/env bash
# Standalone launcher for the ECHO SCOPE admin server.
#
# Runs the Flask admin app on :8081 with no dependencies on the kiosk
# (no camera worker, no Hailo NPU at boot, no state bus). Same .env,
# same faces.db, same admin credentials as start.sh. Either side can
# run without the other.
#
# Override host/port:
#   ADMIN_HOST=127.0.0.1 ADMIN_PORT=9000 ./start_admin.sh

set -euo pipefail

: "${PI5_DIR:=/home/echo/Documents/code/pi5}"

cd "$PI5_DIR"

# ---- Local config (.env) ------------------------------------------------
if [[ -f .env ]]; then
    set -a                     # auto-export everything sourced below
    # shellcheck disable=SC1091
    source .env
    set +a
    ECHO_ENV_LOADED=1
else
    ECHO_ENV_LOADED=0
fi

: "${ADMIN_HOST:=0.0.0.0}"
: "${ADMIN_PORT:=8081}"

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

# ---- Admin credentials --------------------------------------------------
# Required to reach /admin. If unset we generate one per launch so the
# page is always behind auth. Set KIOSK_ADMIN_PASS in .env for stability.
: "${KIOSK_ADMIN_USER:=admin}"
if [[ -z "${KIOSK_ADMIN_PASS:-}" ]]; then
    KIOSK_ADMIN_PASS="$(python -c 'import secrets; print(secrets.token_urlsafe(6))')"
    ADMIN_PASS_GENERATED=1
else
    ADMIN_PASS_GENERATED=0
fi
export ADMIN_HOST ADMIN_PORT KIOSK_ADMIN_USER KIOSK_ADMIN_PASS

echo "---------------------------------------------------------------"
echo " ECHO SCOPE admin"
echo "   project    : $PI5_DIR"
if [[ "$ECHO_ENV_LOADED" == 1 ]]; then
    echo "   config     : .env loaded"
else
    echo "   config     : (no .env)"
fi
echo "   web URL    : http://$ADMIN_HOST:$ADMIN_PORT/"
if [[ "$ADMIN_PASS_GENERATED" == 1 ]]; then
    echo "   admin user : $KIOSK_ADMIN_USER"
    echo "   admin pass : $KIOSK_ADMIN_PASS    (generated -- set KIOSK_ADMIN_PASS to pin)"
else
    echo "   admin auth : $KIOSK_ADMIN_USER (KIOSK_ADMIN_PASS from env)"
fi
echo "---------------------------------------------------------------"

exec python admin_web.py
