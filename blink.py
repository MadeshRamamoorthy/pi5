"""Active liveness: a short blink challenge.

After passive liveness gates (specular, texture, micro-motion) say a
face looks live, the blink challenge adds an explicit signal: ask the
user to blink and look for a clear dip in eye-region pixel std (open
eye = high std from iris/sclera contrast, closed eye = uniform skin).

Without dedicated eye-corner landmarks we can't compute Eye Aspect
Ratio, so the heuristic is the *range* of std across a short observation
window: a real blink produces a clear dip, while a static photo / video
of someone NOT blinking produces nearly constant std. Combined with the
spoken instruction, this defeats both photos and held-up screens
showing someone who happens to not be blinking.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

import config


class BlinkChecker:
    def __init__(self, pipe, grab_frame, draw_hud):
        self.pipe = pipe
        self.grab_frame = grab_frame
        self.draw_hud = draw_hud

    def run(self, cam, greeter, show_preview: bool, target_emp_id: str | None = None) -> bool:
        """Returns True iff a blink-pattern was observed before the timeout."""
        greeter.say(config.LIVENESS_BLINK_PROMPT)
        history: list[float] = []  # combined eye-region std per frame
        timestamps: list[float] = []
        deadline = time.time() + config.LIVENESS_BLINK_TIMEOUT_SEC
        passed = False

        while not passed and time.time() < deadline:
            frame = self.grab_frame(cam)
            dets = self.pipe.detect(
                frame,
                config.DETECTOR_SCORE_THRESHOLD,
                config.DETECTOR_NMS_IOU,
            )
            det = max(
                dets,
                key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]),
                default=None,
            )

            value = None
            if det is not None:
                value = self._eye_signal(frame, det.landmarks)
                history.append(value)
                timestamps.append(time.time())
                # keep last 2.5 s
                while timestamps and timestamps[0] < time.time() - 2.5:
                    timestamps.pop(0)
                    history.pop(0)
                if len(history) >= 8:
                    rng = float(max(history) - min(history))
                    if rng >= config.LIVENESS_BLINK_DELTA_MIN:
                        passed = True

            if show_preview:
                if det is not None:
                    x1, y1, x2, y2 = (int(v) for v in det.bbox)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
                rng_now = (
                    f"range={(max(history)-min(history)):.1f}"
                    if len(history) >= 2 else "..."
                )
                self.draw_hud(
                    frame, [], lambda d: "",
                    banner=config.LIVENESS_BLINK_PROMPT,
                    status=f"watching for blink  {rng_now}/"
                           f"{config.LIVENESS_BLINK_DELTA_MIN}",
                )
                cv2.imshow("Face Recognition", frame)
                cv2.waitKey(20)

        return passed

    def _eye_signal(self, frame: np.ndarray, landmarks: np.ndarray) -> float:
        """Combined std of small patches around each eye landmark."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        r = config.LIVENESS_BLINK_PATCH_PX

        def patch_std(pt):
            x, y = int(pt[0]), int(pt[1])
            x1, y1 = max(0, x - r), max(0, y - r)
            x2, y2 = min(w, x + r), min(h, y + r)
            if x2 - x1 < 4 or y2 - y1 < 4:
                return 0.0
            return float(gray[y1:y2, x1:x2].std())

        return (patch_std(landmarks[0]) + patch_std(landmarks[1])) / 2.0
