"""Photo-only face registration -- detect, embed, persist.

Shared between the standalone admin app (web/admin_app.py) and the
deprecated kiosk-side admin path. Pure function: callers own the
HailoFacePipeline lifecycle and the database connection.
"""

from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

import config


def _largest_detection(dets):
    if not dets:
        return None
    return max(
        dets,
        key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]),
    )


def register_from_photos(
    db,
    pipeline,
    emp_id: str,
    name: str,
    photos: Iterable[bytes],
) -> dict:
    """Process one or more photo bytes blobs into face embeddings.

    Returns:
        {"ok": True,  "samples": N}                if N >= 1 face embeddings saved.
        {"ok": False, "error": "<reason>"}         if zero usable photos.

    Args:
        db:        FaceDB instance. Writer's responsibility.
        pipeline:  HailoFacePipeline instance with .detect / .embed.
                   Caller owns the lifecycle (close after this returns).
        emp_id:    SQLite primary key. New row created if missing, else
                   embeddings are appended to the existing row.
        name:      Human-readable name. Ignored when emp_id already exists.
        photos:    Iterable of raw image bytes (JPEG / PNG / etc.).
    """
    # Local import so the admin process doesn't fail to load when
    # hailo_infer's heavy deps aren't installed.
    from hailo_infer import align_face

    photos = list(photos or [])
    if not photos:
        return {"ok": False, "error": "no photos provided"}

    embeddings = []
    skipped = 0
    for blob in photos:
        arr = np.frombuffer(blob, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            skipped += 1
            continue
        # cv2 decodes as BGR; Hailo's SCRFD was trained on RGB.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        dets = pipeline.detect(
            img,
            config.DETECTOR_SCORE_THRESHOLD,
            config.DETECTOR_NMS_IOU,
        )
        det = _largest_detection(dets)
        if det is None:
            skipped += 1
            continue
        aligned = align_face(img, det.landmarks)
        embeddings.append(pipeline.embed(aligned))

    if not embeddings:
        return {
            "ok": False,
            "error": (
                f"no clear face found in any of {len(photos)} photo(s); "
                "try frontal, well-lit images"
            ),
        }

    if db.employee_exists(emp_id):
        for e in embeddings:
            db.add_embedding(emp_id, e)
    else:
        db.add_employee(emp_id, name, embeddings)

    return {"ok": True, "samples": len(embeddings), "skipped": skipped}
