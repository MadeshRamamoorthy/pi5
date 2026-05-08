"""Passive liveness check.

Defeats the most common kiosk-spoofing attempts:

- Printed photo (held still or waved)
- Phone / monitor screen showing a still image, including glare-on-screen

Combines four signals:

A. SINGLE-FRAME (cheap rejects, run first)
   1. Specular highlight ratio. Phone/monitor screens reflecting room
      light produce large near-saturated regions. Real skin is rarely
      >5% saturated even with glasses or forehead sheen.
   2. Texture variance (Laplacian). Real skin has fine pore/wrinkle
      texture; a screen-displayed face is smoothed by camera capture +
      display + recapture and ends up flatter.

B. TEMPORAL (over a sliding window)
   3. Relative landmark motion: per-frame landmarks minus their centroid
      then std over the window. A waved photo's landmarks all move
      together -> centroid-subtracted std is ~0. A live face shows
      micro-jitter (breathing, eye saccades, mouth twitches).
   4. Face-region pixel jitter: mean abs frame-to-frame diff inside the
      face bbox. Real faces produce more pixel jitter than camera read
      noise alone.

Single-frame failures CLEAR the temporal window so a screen attack can
never accumulate enough frames to look "live".

Limitation: a high-quality video replay on a screen large/clean enough
to dodge the specular and texture gates can still pass. For that, plug
in a dedicated anti-spoofing model (Silent-Face-Anti-Spoofing or similar)
or add an active challenge.
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
    def __init__(self) -> None:
        self._window: deque[_Sample] = deque(maxlen=config.LIVENESS_WINDOW_FRAMES)
        self.last_relative_motion = 0.0
        self.last_pixel_jitter = 0.0
        self.last_specular_ratio = 0.0
        self.last_texture_var = 0.0
        self.last_reason = "warming up"

    def reset(self) -> None:
        self._window.clear()
        self.last_relative_motion = 0.0
        self.last_pixel_jitter = 0.0
        self.last_specular_ratio = 0.0
        self.last_texture_var = 0.0
        self.last_reason = "warming up"

    def update(self, frame: np.ndarray, det) -> bool:
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

        crop_bgr = frame[y1:y2, x1:x2]
        color = cv2.resize(
            crop_bgr, (config.LIVENESS_CROP_PX, config.LIVENESS_CROP_PX)
        )
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

        # ---- A. Single-frame screen-attack gates ----------------------
        hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
        specular_ratio = float((hsv[:, :, 2] > 240).mean())
        self.last_specular_ratio = specular_ratio
        if specular_ratio > config.LIVENESS_MAX_SPECULAR_RATIO:
            self._window.clear()
            self.last_reason = f"glare/screen ({specular_ratio:.0%} bright)"
            return False

        texture_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        self.last_texture_var = texture_var
        if texture_var < config.LIVENESS_MIN_TEXTURE_VAR:
            self._window.clear()
            self.last_reason = f"too smooth (tex={texture_var:.0f})"
            return False

        # ---- B. Temporal sliding-window checks ------------------------
        self._window.append(_Sample(det.landmarks.copy(), gray))

        if len(self._window) < self._window.maxlen // 2:
            self.last_reason = "warming up"
            return False

        lms = np.stack([s.landmarks for s in self._window])           # (T, 5, 2)
        centred = lms - lms.mean(axis=1, keepdims=True)
        rel_motion = float(centred.std(axis=0).mean())

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
