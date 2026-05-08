"""Admin CLI for the face DB.

Usage:
    python admin.py list                 # all employees + sample counts
    python admin.py show E001            # one employee, with embedding metadata
    python admin.py delete E001          # remove employee + all their embeddings
    python admin.py export employees.csv # dump employees to CSV (no embeddings)
    python admin.py path                 # print DB file location
"""

from __future__ import annotations

import argparse
import csv
import sys

from config import DB_PATH
from database import FaceDB


def cmd_list(_args):
    db = FaceDB()
    rows = db.list_employees()
    if not rows:
        print("(no employees registered)")
        return
    print(f"{'EMP_ID':<12} {'NAME':<24} {'SAMPLES':>8}  CREATED_AT")
    print("-" * 70)
    for emp_id, name, count, created in rows:
        print(f"{emp_id:<12} {name:<24} {count:>8}  {created}")
    print(f"\nTotal: {len(rows)} employees, DB at {DB_PATH}")
    db.close()


def cmd_show(args):
    db = FaceDB()
    if not db.employee_exists(args.emp_id):
        print(f"No such employee: {args.emp_id}")
        sys.exit(1)
    name = db.get_name(args.emp_id)
    rows = db.conn.execute(
        "SELECT id, length(embedding), created_at "
        "FROM face_embeddings WHERE emp_id = ? ORDER BY id",
        (args.emp_id,),
    ).fetchall()
    print(f"Employee: {args.emp_id}  ({name})")
    print(f"Embeddings: {len(rows)}")
    for row_id, blob_len, created in rows:
        dim = blob_len // 4  # float32
        print(f"  #{row_id:<4} dim={dim:<4} created={created}")
    db.close()


def cmd_delete(args):
    db = FaceDB()
    if not db.employee_exists(args.emp_id):
        print(f"No such employee: {args.emp_id}")
        sys.exit(1)
    name = db.get_name(args.emp_id)
    if not args.yes:
        ans = input(f"Delete {args.emp_id} ({name}) and all embeddings? [y/N] ")
        if ans.strip().lower() != "y":
            print("Cancelled.")
            return
    db.delete_employee(args.emp_id)
    print(f"Deleted {args.emp_id}.")
    db.close()


def cmd_export(args):
    db = FaceDB()
    rows = db.list_employees()
    with open(args.csv_path, "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["emp_id", "name", "sample_count", "created_at"])
        for emp_id, name, count, created in rows:
            writer.writerow([emp_id, name, count, created])
    print(f"Wrote {len(rows)} rows to {args.csv_path}")
    db.close()


def cmd_path(_args):
    print(DB_PATH)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list").set_defaults(func=cmd_list)

    s = sub.add_parser("show")
    s.add_argument("emp_id")
    s.set_defaults(func=cmd_show)

    s = sub.add_parser("delete")
    s.add_argument("emp_id")
    s.add_argument("--yes", "-y", action="store_true", help="Skip confirmation.")
    s.set_defaults(func=cmd_delete)

    s = sub.add_parser("export")
    s.add_argument("csv_path")
    s.set_defaults(func=cmd_export)

    sub.add_parser("path").set_defaults(func=cmd_path)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
