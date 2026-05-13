"""Standalone admin server entrypoint.

Run via `./start_admin.sh` or directly:

    source .venv/bin/activate
    python admin_web.py

Serves /admin and the admin API on port 8081. Reads + writes the same
faces.db as the kiosk, so either side can run independently and they
share content (projects, sessions, employees) automatically. See
docs/admin.md.

Env vars (also picked up from .env via the launcher):
    ADMIN_HOST            bind address (default 0.0.0.0)
    ADMIN_PORT            port (default 8081)
    KIOSK_ADMIN_USER      HTTP basic auth user (default "admin")
    KIOSK_ADMIN_PASS      HTTP basic auth password (REQUIRED)
"""

from __future__ import annotations

import os

from database import FaceDB
from web import create_admin_app


def main() -> None:
    host = os.environ.get("ADMIN_HOST", "0.0.0.0")
    port = int(os.environ.get("ADMIN_PORT", "8081"))

    db = FaceDB()
    app = create_admin_app(db)

    user = os.environ.get("KIOSK_ADMIN_USER", "admin")
    pw_set = bool(os.environ.get("KIOSK_ADMIN_PASS"))
    print(
        "---------------------------------------------------------------\n"
        " ECHO SCOPE admin\n"
        f"   web URL    : http://{host}:{port}/\n"
        f"   admin user : {user}\n"
        f"   admin auth : {'env-provided' if pw_set else 'NOT SET (requests will 401)'}\n"
        "---------------------------------------------------------------"
    )
    app.run(host=host, port=port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
