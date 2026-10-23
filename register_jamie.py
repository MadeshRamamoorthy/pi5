"""One-shot registration for Jamie (client visit).

Usage on the Pi (from the project root, venv active):

    python register_jamie.py path/to/jamie.jpg
    # or with multiple angles:
    python register_jamie.py jamie_front.jpg jamie_left.jpg

Registers a new employee with emp_id=EMP_ID / name=NAME below, using the
photo(s) you point at, then pins CUSTOM_WELCOME as the exact greeting the
kiosk speaks whenever Jamie is recognised (overrides any LLM-generated
welcome). Safe to re-run: appends embeddings if the emp_id already exists.

Edit EMP_ID / NAME / CUSTOM_WELCOME below to change any of those.
"""

from __future__ import annotations

import sys
from pathlib import Path

from database import FaceDB
from hailo_infer import HailoFacePipeline
from photo_register import register_from_photos
import config


EMP_ID = "jamie"
NAME = "Jamie"
CUSTOM_WELCOME = (
    "Hi Jamie, welcome to Echo AI Lab! Great to have you with us today. "
    "We're proud of our Nutrien-Infosys partnership -- together, feeding "
    "the future. This is where ideas turn into real AI solutions. Let's "
    "build the future of AI together."
)


def main(paths: list[str]) -> int:
    if not paths:
        print("usage: python register_jamie.py <photo1> [<photo2> ...]",
              file=sys.stderr)
        return 2
    blobs = []
    for p in paths:
        f = Path(p)
        if not f.is_file():
            print(f"error: {p} not found", file=sys.stderr)
            return 1
        blobs.append(f.read_bytes())

    db = FaceDB()
    pipe = HailoFacePipeline(config.DETECTOR_HEF, config.EMBEDDER_HEF)
    try:
        result = register_from_photos(db, pipe, EMP_ID, NAME, blobs)
        print(f"register_from_photos: {result}")
        if not result.get("ok"):
            return 1
        db.set_employee_profile(EMP_ID, custom_welcome=CUSTOM_WELCOME)
        print(f"custom_welcome set for {EMP_ID}.")
        # Quick read-back so you can eyeball the row.
        profile = db.get_employee_profile(EMP_ID)
        print(f"profile: {profile}")
    finally:
        try:
            pipe.close()
        except Exception:
            pass
        db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
