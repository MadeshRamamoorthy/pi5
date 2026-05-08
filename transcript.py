"""Right-side transcript panel.

Thread-safe event log + renderer. The wake-word listener thread pushes
Vosk partials/finals; the main thread pushes TTS, recognition, and
state-machine events. The renderer composes a fixed-width image that
main.py hstacks next to the camera frame.

Event kinds (used to colour-code the panel):

  partial   Vosk partial transcription (replaceable -- the latest
            partial overwrites any previous partial in the log).
  heard     Vosk final transcription.
  wake      Wake-word matched.
  tts       TTS speaking something.
  match     Face recognised.
  unknown   Face seen but not matched.
  liveness  Liveness gate decision.
  state     IDLE/ACTIVE state transition.
  system    Generic info.
"""

from __future__ import annotations

import threading
import time
from collections import deque

import cv2
import numpy as np

import config


_KIND_COLOUR = {
    "partial":  (140, 140, 140),
    "heard":    (220, 220, 220),
    "wake":     (0, 255, 255),
    "tts":      (80, 200, 255),
    "match":    (80, 220, 80),
    "unknown":  (140, 140, 200),
    "liveness": (180, 180, 80),
    "state":    (200, 160, 0),
    "system":   (160, 160, 160),
}

_KIND_PREFIX = {
    "partial":  "...",
    "heard":    "MIC",
    "wake":     "WAKE",
    "tts":      "SAY",
    "match":    "OK ",
    "unknown":  "?  ",
    "liveness": "LIV",
    "state":    "STA",
    "system":   "INF",
}


class Transcript:
    def __init__(self, max_events: int = 24):
        self._events: deque[tuple[float, str, str]] = deque(maxlen=max_events)
        self._lock = threading.Lock()

    def add(self, kind: str, text: str) -> None:
        text = text.strip()
        if not text:
            return
        with self._lock:
            if (
                kind == "partial"
                and self._events
                and self._events[-1][1] == "partial"
            ):
                # Replace the rolling partial in place.
                self._events[-1] = (time.time(), kind, text)
                return
            if (
                self._events
                and self._events[-1][1] == kind
                and self._events[-1][2] == text
            ):
                # Don't append exact repeats.
                return
            self._events.append((time.time(), kind, text))

    def render(self, height: int, width: int) -> np.ndarray:
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:] = (24, 24, 24)

        # Header
        cv2.putText(
            canvas, "Transcript",
            (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 1
        )
        cv2.line(canvas, (8, 36), (width - 8, 36), (60, 60, 60), 1)

        with self._lock:
            events = list(self._events)

        # Approx chars-per-line at scale 0.45 -> ~7px per char.
        chars_per_line = max(20, (width - 24) // 8)
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Render newest at the bottom (chronological top-down).
        y = 60
        for ts, kind, text in events:
            colour = _KIND_COLOUR.get(kind, (200, 200, 200))
            tag = _KIND_PREFIX.get(kind, "   ")
            t_str = time.strftime("%H:%M:%S", time.localtime(ts))
            head = f"{t_str}  {tag}"

            cv2.putText(canvas, head, (12, y), font, 0.42, (110, 110, 110), 1)
            y += 16

            for line in _wrap(text, chars_per_line):
                cv2.putText(canvas, line, (24, y), font, 0.5, colour, 1)
                y += 18
                if y > height - 12:
                    return canvas
            y += 4

        return canvas


def _wrap(text: str, max_chars: int) -> list[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    out: list[str] = []
    while text:
        if len(text) <= max_chars:
            out.append(text)
            break
        cut = text.rfind(" ", 0, max_chars)
        if cut <= 0:
            cut = max_chars
        out.append(text[:cut])
        text = text[cut:].lstrip()
    return out


def compose_with_camera(frame: np.ndarray, panel: np.ndarray) -> np.ndarray:
    """Resize panel to the camera height (if needed) and hstack."""
    if panel.shape[0] != frame.shape[0]:
        panel = cv2.resize(panel, (panel.shape[1], frame.shape[0]))
    return np.hstack([frame, panel])
