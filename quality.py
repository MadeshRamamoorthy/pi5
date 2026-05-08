"""Face-quality and pose-change helpers.

These keep the recognition / enrolment pipeline robust against:
- partial occlusions (a hand near the face, side-on faces)
- low-confidence detections
- people walking past the edge of the frame
- enrolling 5 nearly-identical frames because the user didn't actually move

All checks operate on a `Detection` (from hailo_infer) and the original frame
shape — no model state needed.
"""

from __future__ import annotations

import numpy as np

import config

# Landmark order from SCRFD (ArcFace standard 5-pt layout):
LE, RE, NOSE, LM, RM = 0, 1, 2, 3, 4


def is_quality_face(det, frame_shape) -> tuple[bool, str]:
    """Return (ok, reason). reason is empty when ok is True."""
    if det.score < config.QUALITY_SCORE_THRESHOLD:
        return False, f"score {det.score:.2f} < {config.QUALITY_SCORE_THRESHOLD}"

    x1, y1, x2, y2 = det.bbox
    w, h = x2 - x1, y2 - y1
    if w < config.QUALITY_MIN_FACE_PIXELS or h < config.QUALITY_MIN_FACE_PIXELS:
        return False, f"face {int(w)}x{int(h)}px is too small"

    fh, fw = frame_shape[:2]
    m = config.QUALITY_FRAME_EDGE_MARGIN
    if x1 < m or y1 < m or x2 > fw - m or y2 > fh - m:
        return False, "face touches frame edge"

    lms = det.landmarks
    le, re, nose, lm, rm = lms[LE], lms[RE], lms[NOSE], lms[LM], lms[RM]

    # Eyes above nose, mouth below nose — basic anatomy sanity.
    if not (le[1] < nose[1] and re[1] < nose[1]
            and lm[1] > nose[1] and rm[1] > nose[1]):
        return False, "landmark geometry invalid"

    eye_dx = re[0] - le[0]
    eye_dy = re[1] - le[1]
    eye_dist = float(np.hypot(eye_dx, eye_dy))
    if eye_dist < config.QUALITY_MIN_EYE_DISTANCE:
        return False, f"eye distance {eye_dist:.0f}px too small"

    # Roll: |dy / dx| between the two eyes. Normal frontal faces stay below ~0.3.
    if abs(eye_dx) > 1e-3:
        tilt = abs(eye_dy / eye_dx)
        if tilt > config.QUALITY_MAX_EYE_TILT:
            return False, f"head tilted too far (tilt={tilt:.2f})"

    return True, ""


def landmark_anchor(det) -> tuple[np.ndarray, float]:
    """Return (anchor_xy, eye_dist) for the face. Anchor = midpoint of the eyes,
    expressed in original-image pixels. eye_dist is used to normalise shifts."""
    lms = det.landmarks
    eye_mid = (lms[LE] + lms[RE]) / 2.0
    eye_dist = float(np.hypot(*(lms[RE] - lms[LE])))
    return eye_mid, max(eye_dist, 1.0)


def shift_matches_direction(
    current_anchor: np.ndarray,
    baseline_anchor: np.ndarray,
    eye_dist: float,
    direction: str,
) -> bool:
    """Has the face shifted in the requested direction since the baseline?

    `direction` is "right" / "left" / "up" / "down" (or None to skip the check).
    Threshold is config.POSE_MIN_SHIFT, in inter-eye-distance units.
    """
    if direction is None:
        return True
    dx = (current_anchor[0] - baseline_anchor[0]) / eye_dist
    dy = (current_anchor[1] - baseline_anchor[1]) / eye_dist
    t = config.POSE_MIN_SHIFT
    if direction == "right":
        return dx >= t
    if direction == "left":
        return dx <= -t
    if direction == "down":
        return dy >= t
    if direction == "up":
        return dy <= -t
    return True


def landmarks_drift(a: np.ndarray, b: np.ndarray) -> float:
    """Max per-point pixel distance between two 5x2 landmark sets."""
    return float(np.linalg.norm(a - b, axis=1).max())
