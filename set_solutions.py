"""Set the kiosk's solutions catalog (the things ECHO can match a
visitor's question against).

REPLACES the solutions table with exactly the list below. Run on the Pi:

    python set_solutions.py

Mirrors the same demos that are on display today (see set_projects.py),
so questions like "what solutions have you built?" / "what tools do
you have?" / "do you have something for X?" all draw from the same
canonical list.

Each entry is (name, description, domain, link); ordering is taken from
the list index. Descriptions/domains/links are left blank for now --
fill them in per demo when you're ready and re-run this script.
"""

from __future__ import annotations

from database import FaceDB


SOLUTIONS = [
    ("Railcar Passport AIoT Model", "", "", ""),
    ("AI Contract Lifecycle Management", "", "", ""),
    ("Cobalt Migration Accelerator (Transform Hub)", "", "", ""),
    ("DB Migration", "", "", ""),
    ("FinOps and Compliance", "", "", ""),
    ("AWS AI Ops", "", "", ""),
    ("AI-Powered Smart Plant Maintenance Management (Connected Ops)", "", "", ""),
    ("PowerBI Usage Analytics", "", "", ""),
    ("Databricks Platform Strategy & Enterprise Data Hub", "", "", ""),
]


def main():
    db = FaceDB()
    try:
        existing = db.list_solutions()
        for row in existing:
            db.delete_solution(row[0])
        for i, (name, desc, domain, link) in enumerate(SOLUTIONS, start=1):
            db.add_solution(name, desc, domain, link, ordering=i)
        print(f"solutions: removed {len(existing)}, inserted {len(SOLUTIONS)}")
        for i, (name, *_rest) in enumerate(SOLUTIONS, start=1):
            print(f"  {i:>2}. {name}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
