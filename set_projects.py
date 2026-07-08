"""Set the kiosk's project list to the authoritative demo list.

Unlike seed_demo.py (which only ADDS missing rows), this REPLACES the
projects table wholesale: it deletes whatever is there and inserts
exactly the list below, in order. Run it on the Pi after editing the
list, then the idle dashboard + ECHO's "what's on display today?"
answer reflect it immediately (next idle-data refresh, ~5 s).

    python set_projects.py

Edit PROJECTS to change what's shown. Each entry is (title, description);
the ordering is taken from the list index, so just re-arrange the lines
to reorder the dashboard card.
"""

from __future__ import annotations

from database import FaceDB


# The single source of truth for what's "on display today". seed_demo.py
# imports this so a fresh install and an explicit reset stay in sync.
# Order in the list == order on the dashboard.
PROJECTS = [
    ("Railcar Passport AIoT Model", ""),
    ("AI Contract Lifecycle Management", ""),
    ("Cobalt Migration Accelerator (Transform Hub)", ""),
    ("DB Migration", ""),
    ("FinOps and Compliance", ""),
    ("AWS AI Ops", ""),
    ("AI-Powered Smart Plant Maintenance Management (Connected Ops)", ""),
    ("PowerBI Usage Analytics", ""),
    ("Databricks Platform Strategy & Enterprise Data Hub", ""),
]


def main():
    db = FaceDB()
    try:
        existing = db.list_projects()
        for row in existing:
            db.delete_project(row[0])
        for ordering, (title, desc) in enumerate(PROJECTS, start=1):
            db.add_project(title, desc, ordering)
        print(f"projects: removed {len(existing)}, inserted {len(PROJECTS)}")
        for i, (title, _desc) in enumerate(PROJECTS, start=1):
            print(f"  {i}. {title}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
