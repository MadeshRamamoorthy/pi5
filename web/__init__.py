"""Flask app factories.

- create_app:       kiosk SPA + state + camera + worker callbacks.
                    Served by main.py on port 8090.
- create_admin_app: standalone admin (CRUD + photo register).
                    Served by admin_web.py on port 8081.
"""

from .admin_app import create_admin_app
from .app import create_app

__all__ = ["create_app", "create_admin_app"]
