"""Chat-voice capture: record PCM from the mic, hand to ChatASR.

Replaces the previous "swap Vosk recognizer to no-grammar" approach.
The wake-word listener is paused (which releases the ALSA device), this
module opens its own sounddevice InputStream, buffers PCM until either:

  - the user explicitly stops listening, OR
  - `CHAT_VOICE_SILENCE_SEC` of below-threshold audio elapses (VAD), OR
  - `CHAT_VOICE_MAX_SEC` total wall-clock elapses.

The captured PCM is then handed to a ChatASR worker thread; final
transcripts land on `out_queue` for the main loop to drain. The
wake-word listener resumes immediately after transcription.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Callable

import numpy as np

try:
    import sounddevice as sd

    from audio_utils import pick_input_device
except Exception as exc:  # noqa: BLE001
    sd = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

import config
from asr import ChatASR


class ChatVoiceCaptureError(RuntimeError):
    pass


class ChatVoiceCapture:
    """One instance lives for the whole app lifetime. Each user-initiated
    chat utterance spawns a short-lived capture run via `start()`."""

    def __init__(
        self,
        asr: ChatASR,
        out_queue: "queue.Queue[str]",
        wake_listener=None,
        device: int | None = None,
        on_listening_changed: Callable[[bool], None] | None = None,
    ):
        if _IMPORT_ERROR is not None:
            raise ChatVoiceCaptureError(
                f"sounddevice unavailable: {_IMPORT_ERROR!r}"
            )
        self._asr = asr
        self._out_q = out_queue
        self._listener = wake_listener
        self._on_changed = on_listening_changed
        try:
            self._device, self._native_rate = pick_input_device(device)
        except Exception as exc:  # noqa: BLE001
            raise ChatVoiceCaptureError(f"input device probe failed: {exc}")
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._listening = False

    @property
    def listening(self) -> bool:
        return self._listening

    def start(self) -> None:
        """Begin recording. No-op if already running."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Ask the capture thread to finalise the recording."""
        self._stop.set()

    # internal -----------------------------------------------------------

    def _set_listening(self, on: bool) -> None:
        self._listening = on
        if self._on_changed:
            try:
                self._on_changed(on)
            except Exception:
                pass

    def _run(self) -> None:
        if self._listener is not None:
            self._listener.pause()
            # Tiny grace period so the ALSA device fully closes before
            # we try to reopen it.
            time.sleep(0.15)
        try:
            self._set_listening(True)
            pcm = self._record()
        finally:
            self._set_listening(False)
            if self._listener is not None:
                self._listener.resume()

        if not pcm:
            return
        try:
            text = self._asr.transcribe(pcm, self._native_rate)
        except Exception as exc:  # noqa: BLE001
            print(f"[chat-voice] transcribe failed: {exc!r}")
            return
        if text:
            self._out_q.put(text)

    def _record(self) -> bytes:
        chunks: list[bytes] = []
        # ~100 ms blocks make VAD responsive without burning CPU.
        block = max(1, int(self._native_rate * 0.1))
        silence_run = 0.0
        had_voice = False
        start = time.time()
        q: "queue.Queue[bytes]" = queue.Queue()

        def cb(indata, frames, time_info, status):  # noqa: ARG001
            q.put(bytes(indata))

        try:
            stream = sd.RawInputStream(
                samplerate=self._native_rate,
                blocksize=block,
                device=self._device,
                dtype="int16",
                channels=1,
                callback=cb,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[chat-voice] mic open failed: {exc!r}")
            return b""

        with stream:
            while True:
                if self._stop.is_set():
                    break
                if time.time() - start > config.CHAT_VOICE_MAX_SEC:
                    break
                try:
                    data = q.get(timeout=0.2)
                except queue.Empty:
                    continue
                chunks.append(data)
                arr = np.frombuffer(data, dtype=np.int16)
                # RMS as int16 amplitude; tweak via config.
                if arr.size == 0:
                    continue
                rms = float(np.sqrt(np.mean(arr.astype(np.float32) ** 2)))
                if rms >= config.CHAT_VOICE_SILENCE_RMS:
                    had_voice = True
                    silence_run = 0.0
                else:
                    silence_run += block / self._native_rate
                if had_voice and silence_run >= config.CHAT_VOICE_SILENCE_SEC:
                    break

        return b"".join(chunks) if had_voice else b""
