"""Right-side conversation panel.

This is the user-facing log -- not a debug stream. Three event roles:

  user    Something the person said (e.g. the wake word).
  system  Something the assistant said out loud (TTS).
  event   A short status line (recognised X, new face, going to sleep).

Vosk partials, internal state transitions, streak counters, and the
like are NOT shown here -- they go to stdout for debugging instead.
The right panel reads like a chat log.

Thread-safety: the wake-word listener thread and the main thread can
both push events; a single lock serialises access to the ring buffer.
"""

from __future__ import annotations

import threading
import time
from collections import deque

import cv2
import numpy as np


_ROLE_STYLE = {
    # role     -> (prefix,   colour BGR)
    "user":     ("You ",     (240, 230, 140)),   # pale cyan
    "system":   ("Echo",     (180, 220, 255)),   # warm amber
    "event":    ("·   ",     (140, 220, 140)),   # muted green
}


class Transcript:
    def __init__(self, max_events: int = 16):
        self._events: deque[tuple[float, str, str]] = deque(maxlen=max_events)
        self._lock = threading.Lock()

    # ---- public API ----------------------------------------------------

    def user(self, text: str) -> None:
        self._add("user", text)

    def system(self, text: str) -> None:
        self._add("system", text)

    def event(self, text: str) -> None:
        self._add("event", text)

    # ---- rendering -----------------------------------------------------

    def render(self, height: int, width: int) -> np.ndarray:
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:] = (24, 24, 24)

        cv2.putText(
            canvas, "Conversation",
            (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 1
        )
        cv2.line(canvas, (10, 42), (width - 10, 42), (60, 60, 60), 1)

        with self._lock:
            events = list(self._events)

        # ~7 px per char at scale 0.5
        chars_per_line = max(20, (width - 28) // 8)
        font = cv2.FONT_HERSHEY_SIMPLEX

        y = 72
        for ts, role, text in events:
            prefix, colour = _ROLE_STYLE.get(role, ("    ", (200, 200, 200)))
            t_str = time.strftime("%H:%M", time.localtime(ts))

            cv2.putText(canvas, f"{t_str}  {prefix}",
                        (14, y), font, 0.45, (110, 110, 110), 1)
            y += 18

            for line in _wrap(text, chars_per_line):
                cv2.putText(canvas, line, (28, y), font, 0.55, colour, 1)
                y += 22
                if y > height - 18:
                    return canvas
            y += 6

        if not events:
            cv2.putText(canvas,
                        "(say 'hello echo' to begin)",
                        (14, 80), font, 0.5, (110, 110, 110), 1)
        return canvas

    # ---- internals -----------------------------------------------------

    def _add(self, role: str, text: str) -> None:
        text = text.strip()
        if not text:
            return
        with self._lock:
            if (
                self._events
                and self._events[-1][1] == role
                and self._events[-1][2] == text
            ):
                return  # de-dup exact repeats
            self._events.append((time.time(), role, text))


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
    if panel.shape[0] != frame.shape[0]:
        panel = cv2.resize(panel, (panel.shape[1], frame.shape[0]))
    return np.hstack([frame, panel])
