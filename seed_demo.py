"""Seed the kiosk DB with a few example projects + an upcoming session.

Run once on a fresh DB to populate the IDLE dashboard with something
to look at:

    python seed_demo.py

Idempotent: skips rows whose title already exists.
"""

from __future__ import annotations

import datetime as dt

from database import FaceDB
from set_projects import PROJECTS


# Mirror the authoritative project list (set_projects.py) so a fresh
# seed and an explicit reset show the same thing. seed_demo is additive
# (skips existing titles); use set_projects.py to REPLACE the list.
DEFAULT_PROJECTS = [(title, desc) for title, desc, _ordering in PROJECTS]


def upcoming_session():
    """A session ~3 days out at 15:30-16:30 local."""
    soon = dt.datetime.now().replace(microsecond=0)
    target = soon + dt.timedelta(days=3)
    target = target.replace(hour=15, minute=30, second=0)
    end = target.replace(hour=16, minute=30)
    return (
        "Build Your Own Document Chatbot",
        target.strftime("%Y-%m-%d %H:%M:%S"),
        end.strftime("%Y-%m-%d %H:%M:%S"),
        "Hands-on session: RAG over your own PDFs.",
    )


def main():
    db = FaceDB()
    try:
        existing = {row[1] for row in db.list_projects()}
        added_p = 0
        for title, desc in DEFAULT_PROJECTS:
            if title in existing:
                continue
            db.add_project(title, desc, 0)
            added_p += 1
        print(f"projects: {added_p} new (existing {len(existing)})")

        existing_sessions = {row[1] for row in db.list_sessions(upcoming_only=False)}
        title, starts_at, ends_at, notes = upcoming_session()
        if title not in existing_sessions:
            db.add_session(title, starts_at, ends_at, notes)
            print(f"session : added '{title}' @ {starts_at}")
        else:
            print(f"session : '{title}' already present")
    finally:
        db.close()


if __name__ == "__main__":
    main()
