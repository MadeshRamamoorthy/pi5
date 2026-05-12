"""Non-blocking TTS wrapper.

The main camera loop used to call backend.speak() inline, which blocks
for the entire duration of `aplay` (~0.5-1.5 s for a typical greeting).
That froze cv2.imshow during the audio. AsyncTTS pushes utterances onto
a queue; a daemon thread drains it serially. The camera thread keeps
rendering at full FPS.

For places that need ordering (e.g. the pose-capture loop expects the
"Look straight at the camera" prompt to be spoken BEFORE we start
watching for stability), call `wait_idle()` before reading frames.
"""

from __future__ import annotations

import queue
import threading
import time

from tts import _Backend


_SHUTDOWN = object()


class AsyncTTS:
    def __init__(self, backend: _Backend):
        self._backend = backend
        self._q: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    @property
    def name(self) -> str:
        return getattr(self._backend, "name", "?")

    def speak(self, text: str, on_start=None) -> None:
        """Queue text for spoken output. If `on_start` is provided, it
        is invoked from the TTS worker thread right BEFORE the backend
        starts playing this utterance. Use that hook to push UI state
        (toasts, transcript banners) so the picture and audio land
        together rather than the UI racing ahead of the queue."""
        text = (text or "").strip()
        if not text:
            return
        self._q.put((text, on_start))

    def wait_idle(self, timeout: float | None = None) -> bool:
        """Block until the queue is fully drained. Returns True if drained
        within timeout, False otherwise."""
        deadline = (time.time() + timeout) if timeout is not None else None
        while True:
            if self._q.unfinished_tasks == 0:
                return True
            if deadline is not None and time.time() >= deadline:
                return False
            time.sleep(0.05)

    def flush(self) -> None:
        """Drop pending utterances. The currently-playing one (if any) is
        not interrupted -- it has already been pulled off the queue."""
        try:
            while True:
                self._q.get_nowait()
                self._q.task_done()
        except queue.Empty:
            pass

    def interrupt(self) -> None:
        """Stop the currently-playing audio AND drop the pending queue.

        Used by the chat-mic flow so the kiosk shuts up the moment the
        user taps "tap to speak" -- it shouldn't talk over them. Safe
        to call when nothing is playing."""
        try:
            self._backend.stop()
        except Exception as exc:  # noqa: BLE001
            print(f"[async-tts] backend.stop failed: {exc!r}")
        self.flush()

    def stop(self) -> None:
        self._q.put(_SHUTDOWN)
        self._thread.join(timeout=2.0)

    # internal -----------------------------------------------------------

    def _run(self) -> None:
        while True:
            item = self._q.get()
            try:
                if item is _SHUTDOWN:
                    return
                text, on_start = item
                if on_start is not None:
                    try:
                        on_start()
                    except Exception as exc:  # noqa: BLE001
                        print(f"[async-tts] on_start callback failed: {exc!r}")
                try:
                    self._backend.speak(text)
                except Exception as exc:  # noqa: BLE001
                    print(f"[async-tts] speak failed: {exc!r}")
            finally:
                self._q.task_done()
