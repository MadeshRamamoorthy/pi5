"""Set the kiosk's project list to the authoritative demo list.

Unlike seed_demo.py (which only ADDS missing rows), this REPLACES the
projects table wholesale: it deletes whatever is there and inserts
exactly the list below, in order. Run it on the Pi after editing the
list, then the idle dashboard + ECHO's "what's on display today?"
answer reflect it immediately (next idle-data refresh, ~5 s).

    python set_projects.py

Edit PROJECTS to change what's shown. Each entry is
(title, description, ordering); lower ordering shows first.
"""

from __future__ import annotations

from database import FaceDB


# The single source of truth for what's "on display today". seed_demo.py
# imports this so a fresh install and an explicit reset stay in sync.
PROJECTS = [
    ("Digital Quality Railcar Passport",
     "IoT-driven potash quality control and railcar traceability.",
     1),
    ("Business Incubator",
     "AI-powered tool for airline smart troubleshooting & automated "
     "operation support; TrustLayer AI, an enterprise AI reliability "
     "platform; and Gov Cycle, reclaiming lost assets.",
     2),
    ("AWS COE", "", 3),
    ("Resources COE", "", 4),
    ("Locally Developed Apps", "", 5),
]


def main():
    db = FaceDB()
    try:
        existing = db.list_projects()
        for row in existing:
            db.delete_project(row[0])
        for title, desc, ordering in PROJECTS:
            db.add_project(title, desc, ordering)
        print(f"projects: removed {len(existing)}, inserted {len(PROJECTS)}")
        for title, _desc, ordering in PROJECTS:
            print(f"  {ordering}. {title}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
