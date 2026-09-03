"""Set the kiosk's upcoming sessions (the idle dashboard "Upcoming
Session" card).

REPLACES the sessions table with exactly the list below. Run on the Pi:

    python set_sessions.py

The dashboard shows the *next* session whose start time is still in the
future, so once a session's date passes it automatically rolls to the
following one. This is separate from config.AI_LAB_SESSIONS_PLANNED
(which feeds the chat "what's coming up?" answer) -- keep both in sync.

Each entry is (title, starts_at, ends_at, notes). Times are full
"YYYY-MM-DD HH:MM:SS" local. NOTE: the times below are placeholders
(5:00-6:00 PM) -- edit them to the real schedule, or set them in the
admin panel's Sessions tab.
"""

from __future__ import annotations

from database import FaceDB


SESSIONS = [
    ("Claude Certified Associate - Foundation",
     "2026-09-12 14:54:00", "2026-09-12 14:54:00",
     "Prep session for CCAO-F certificate"),
]


def main():
    db = FaceDB()
    try:
        existing = db.list_sessions(upcoming_only=False)
        for row in existing:
            db.delete_session(row[0])
        for title, starts_at, ends_at, notes in SESSIONS:
            db.add_session(title, starts_at, ends_at or None, notes)
        print(f"sessions: removed {len(existing)}, inserted {len(SESSIONS)}")
        for title, starts_at, *_rest in SESSIONS:
            print(f"  {title} @ {starts_at}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
