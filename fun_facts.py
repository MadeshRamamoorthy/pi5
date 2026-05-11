"""Background thread that rotates the ACTIVE-screen "AI Fun Fact" /
"AI Tip" card every config.FUN_FACT_ROTATE_SEC seconds.

Idle when the kiosk isn't in ACTIVE state -- no reason to burn cycles
picking a fact nobody is reading.
"""

from __future__ import annotations

import threading
import time

import config
import messages


class FunFactRotator:
    def __init__(self, state):
        self._state = state
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        # Push one immediately so the first ACTIVE frame has fresh content.
        self._state.update(fun_fact=messages.get_rotating_content())
        while not self._stop.is_set():
            interval = max(5, int(config.FUN_FACT_ROTATE_SEC))
            self._stop.wait(interval)
            if self._stop.is_set():
                return
            snap = self._state.snapshot()
            if snap.get("state") != "ACTIVE":
                # Save CPU when nobody's watching; the SPA shows the
                # last value when ACTIVE re-engages, then we resume
                # rotating from here.
                continue
            self._state.update(fun_fact=messages.get_rotating_content())
