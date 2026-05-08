"""Offline wake-word listener using Vosk.

Captures audio at the mic's native samplerate (USB conferencing devices
like the Anker A3301 don't support 16 kHz directly), resamples to the
rate Vosk wants, and sets an Event whenever the configured WAKE_WORD
phrase is heard.

The main loop polls `is_activated()` and clears it via `deactivate()`
when it wants to drop back to idle.
"""

from __future__ import annotations

import json
import queue
import threading
from pathlib import Path

try:
    import numpy as np
    import sounddevice as sd
    import vosk

    from audio_utils import pick_input_device, resample_int16
except Exception as exc:  # noqa: BLE001
    sd = None
    vosk = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class WakeWordError(RuntimeError):
    pass


class WakeWordListener:
    def __init__(
        self,
        model_dir: Path,
        keyword: str,
        samplerate: int = 16000,
        blocksize: int = 8000,  # kept for API compatibility; computed from native rate now
        device: int | None = None,
    ):
        if _IMPORT_ERROR is not None:
            raise WakeWordError(
                "Wake-word deps missing. Install vosk + sounddevice "
                "(see README §2.8). Original error: " f"{_IMPORT_ERROR!r}"
            )
        if not Path(model_dir).is_dir():
            raise WakeWordError(
                f"Vosk model not found at {model_dir}. "
                "Download via README §2.8."
            )

        self.keyword = keyword.strip().lower()
        self.target_rate = samplerate

        try:
            self.device, self.native_rate = pick_input_device(device)
        except Exception as exc:  # noqa: BLE001
            raise WakeWordError(f"could not query input device: {exc}")

        grammar = json.dumps([self.keyword, "[unk]"])
        self._model = vosk.Model(str(model_dir))
        self._recognizer = vosk.KaldiRecognizer(self._model, self.target_rate, grammar)

        self._activated = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._queue: queue.Queue[bytes] = queue.Queue()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._thread = None

    def is_activated(self) -> bool:
        return self._activated.is_set()

    def deactivate(self) -> None:
        self._activated.clear()
        try:
            self._recognizer.Reset()
        except Exception:
            pass

    # internal -----------------------------------------------------------

    def _audio_cb(self, indata, frames, time_info, status):  # noqa: ARG002
        if status:
            pass
        if self.native_rate != self.target_rate:
            mono = np.frombuffer(bytes(indata), dtype=np.int16)
            mono = resample_int16(mono, self.native_rate, self.target_rate)
            self._queue.put(mono.tobytes())
        else:
            self._queue.put(bytes(indata))

    def _run(self) -> None:
        block = max(1, int(self.native_rate / 2))  # ~500 ms blocks
        try:
            stream = sd.RawInputStream(
                samplerate=self.native_rate,
                blocksize=block,
                device=self.device,
                dtype="int16",
                channels=1,
                callback=self._audio_cb,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[wake-word] mic open failed at {self.native_rate} Hz on "
                  f"device={self.device}: {exc}")
            return

        with stream:
            while not self._stop.is_set():
                try:
                    data = self._queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                self._consume(data)

    def _consume(self, data: bytes) -> None:
        if self._recognizer.AcceptWaveform(data):
            text = json.loads(self._recognizer.Result()).get("text", "")
        else:
            text = json.loads(self._recognizer.PartialResult()).get("partial", "")
        if self.keyword in text.lower():
            self._activated.set()
