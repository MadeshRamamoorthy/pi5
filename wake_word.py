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
        on_partial=None,
        on_final=None,
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
        self._on_partial = on_partial
        self._on_final = on_final
        self._chat_on_final = None     # set by set_freeform()
        self._chat_on_partial = None
        self._mode = "wake"            # "wake" | "freeform"

        try:
            self.device, self.native_rate = pick_input_device(device)
        except Exception as exc:  # noqa: BLE001
            raise WakeWordError(f"could not query input device: {exc}")

        self._grammar = json.dumps([self.keyword, "[unk]"])
        self._model = vosk.Model(str(model_dir))
        self._recognizer = vosk.KaldiRecognizer(self._model, self.target_rate, self._grammar)
        self._lock = threading.Lock()

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

    @property
    def mode(self) -> str:
        return self._mode

    def deactivate(self) -> None:
        self._activated.clear()
        try:
            self._recognizer.Reset()
        except Exception:
            pass

    def set_freeform(self, on_final, on_partial=None) -> None:
        """Switch the recognizer out of wake-word grammar mode and into
        unrestricted dictation. Final transcripts go to `on_final(text)`.
        Use for the CHAT tab's voice input. Call set_wake() to revert."""
        with self._lock:
            self._chat_on_final = on_final
            self._chat_on_partial = on_partial
            # Build a no-grammar recognizer so any speech can transcribe.
            self._recognizer = vosk.KaldiRecognizer(self._model, self.target_rate)
            self._mode = "freeform"

    def set_wake(self) -> None:
        with self._lock:
            self._chat_on_final = None
            self._chat_on_partial = None
            self._recognizer = vosk.KaldiRecognizer(
                self._model, self.target_rate, self._grammar
            )
            self._mode = "wake"

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
        with self._lock:
            recog = self._recognizer
            mode = self._mode
            chat_on_final = self._chat_on_final
            chat_on_partial = self._chat_on_partial
        if recog.AcceptWaveform(data):
            text = json.loads(recog.Result()).get("text", "").strip().lower()
            if mode == "freeform":
                if text and chat_on_final:
                    try:
                        chat_on_final(text)
                    except Exception:
                        pass
                return
            if text and self._on_final:
                try:
                    self._on_final(text)
                except Exception:
                    pass
            # Only fire on FINAL transcripts. Partial results from Vosk's
            # grammar-restricted recognizer ("hello echo" vs "[unk]") tend
            # to lock onto the wake phrase before the audio has settled,
            # producing false positives on ambient speech / clatter. The
            # final transcript is much more reliable.
            #
            # Match the *whole* utterance (or the keyword followed by
            # filler tokens like "the"). A bare substring check would
            # accept any sentence happening to contain "hello echo".
            if text == self.keyword or text.startswith(self.keyword + " "):
                self._activated.set()
        else:
            text = json.loads(recog.PartialResult()).get("partial", "")
            if mode == "freeform":
                if text and chat_on_partial:
                    try:
                        chat_on_partial(text)
                    except Exception:
                        pass
                return
            if text and self._on_partial:
                try:
                    self._on_partial(text)
                except Exception:
                    pass
