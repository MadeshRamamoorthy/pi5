"""HTTP basic auth helpers shared between the kiosk app and the
standalone admin app. Both look at the same env vars
(`KIOSK_ADMIN_USER` / `KIOSK_ADMIN_PASS`) so a single .env line covers
both processes.
"""

from __future__ import annotations

import os

from flask import Response, request


def check_admin_auth() -> bool:
    user = os.environ.get("KIOSK_ADMIN_USER") or "admin"
    pw = os.environ.get("KIOSK_ADMIN_PASS")
    if not pw:
        # No password configured -- refuse rather than silently allow.
        # start.sh / start_admin.sh generate a random password at launch
        # when the user didn't provide one, so this path only hits when
        # somebody bypassed the launcher.
        return False
    auth = request.authorization
    return bool(auth and auth.username == user and auth.password == pw)


def request_admin_auth():
    return Response(
        "Admin authentication required.\n"
        "If you didn't set KIOSK_ADMIN_PASS yourself, look at the\n"
        "'admin pass:' line in the launcher banner.",
        401,
        {"WWW-Authenticate": 'Basic realm="ECHO SCOPE admin"'},
    )
