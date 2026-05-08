"""Passive liveness check.

Designed to defeat the most common spoofing case for an unattended kiosk:
a printed photo or a phone screen held up to the camera. Two signals are
combined over a short sliding window:

1. Relative landmark motion: per-frame face landmarks, after subtracting
   the centroid, must show non-trivial standard deviation. This rejects
   "photo waved in front of camera" because all landmarks move *together*
   on a 2D photo; subtracting the centroid leaves ~zero relative motion.
   A live face has small but measurable micro-jitter (breathing, mouth
   twitches, eye saccades).

2. Face-region pixel jitter: the mean absolute frame-to-frame difference
   inside the face bbox. Real faces produce more pixel jitter than a
   static photo (camera noise alone is ~1-2 grey levels).

Both must clear their thresholds within the sliding window before the
face is treated as "live". Limitation: a high-quality video replay on a
sufficiently large screen can defeat this. For that you need active
challenges (already done at enrolment) or a dedicated anti-spoofing
model (out of scope).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

import config


@dataclass
class _Sample:
    landmarks: np.ndarray  # (5, 2)
    gray_face: np.ndarray  # LIVENESS_CROP_PX x LIVENESS_CROP_PX uint8


class LivenessChecker:
    """Maintains a sliding window per face track. Currently only tracks the
    largest face (which is what main.py asks about anyway)."""

    def __init__(self) -> None:
        self._window: deque[_Sample] = deque(maxlen=config.LIVENESS_WINDOW_FRAMES)
        self.last_relative_motion = 0.0
        self.last_pixel_jitter = 0.0
        self.last_reason = "warming up"

    def reset(self) -> None:
        self._window.clear()
        self.last_relative_motion = 0.0
        self.last_pixel_jitter = 0.0
        self.last_reason = "warming up"

    def update(self, frame: np.ndarray, det) -> bool:
        """Push a new (frame, detection) sample and return current liveness."""
        if det is None:
            self.reset()
            return False

        x1, y1, x2, y2 = (int(round(v)) for v in det.bbox)
        x1, y1 = max(x1, 0), max(y1, 0)
        x2 = min(x2, frame.shape[1])
        y2 = min(y2, frame.shape[0])
        if x2 - x1 < 8 or y2 - y1 < 8:
            self.reset()
            self.last_reason = "face too small"
            return False

        gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (config.LIVENESS_CROP_PX, config.LIVENESS_CROP_PX))
        self._window.append(_Sample(det.landmarks.copy(), gray))

        if len(self._window) < self._window.maxlen // 2:
            self.last_reason = "warming up"
            return False

        # 1) Relative landmark motion: subtract per-frame centroid first.
        lms = np.stack([s.landmarks for s in self._window])           # (T, 5, 2)
        centred = lms - lms.mean(axis=1, keepdims=True)
        rel_motion = float(centred.std(axis=0).mean())

        # 2) Pixel jitter inside the face crop.
        grays = np.stack([s.gray_face for s in self._window]).astype(np.int16)
        diffs = np.abs(np.diff(grays, axis=0))
        pixel_jitter = float(diffs.mean())

        self.last_relative_motion = rel_motion
        self.last_pixel_jitter = pixel_jitter

        ok_motion = rel_motion >= config.LIVENESS_REL_MOTION_MIN
        ok_pixel = pixel_jitter >= config.LIVENESS_PIXEL_JITTER_MIN

        if ok_motion and ok_pixel:
            self.last_reason = ""
            return True
        if not ok_motion and not ok_pixel:
            self.last_reason = "static (photo?)"
        elif not ok_motion:
            self.last_reason = "too rigid (photo?)"
        else:
            self.last_reason = "no facial micro-motion"
        return False
