"""TTS backends.

Two backends:

1. PiperPyBackend  -- preferred. Uses the piper1-gpl Python package
                      (`pip install piper-tts`). Loads the ONNX voice
                      once at startup so subsequent utterances are fast.
2. PyttsxBackend   -- espeak-ng fallback. Always available.

Both honour config.AUDIO_OUTPUT_DEVICE -- when set, audio is played
through `aplay -D <device>` so it goes to the Anker (or whatever sink
you've pinned) instead of the system default.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import wave
from pathlib import Path

import pyttsx3

import config


def _prepend_silence(wav_path: str, ms: int) -> None:
    """Prepend `ms` of silence to an existing WAV (in place).

    USB speakerphones often clip the first ~300 ms of audio while their
    DAC/amp wakes up. Buffering the WAV with silence at the start gives
    the device time to come up before the first phoneme is spoken,
    instead of using a blocking sleep before aplay.
    """
    if ms <= 0:
        return
    with wave.open(wav_path, "rb") as f:
        params = f.getparams()
        audio = f.readframes(f.getnframes())
    nsil = int(params.framerate * ms / 1000)
    silence = b"\x00" * (nsil * params.sampwidth * params.nchannels)
    with wave.open(wav_path, "wb") as f:
        f.setparams(params)
        f.writeframes(silence + audio)


class _Backend:
    name = "?"

    def speak(self, text: str) -> None:
        raise NotImplementedError


# ---------------- pyttsx3 / espeak-ng ----------------


class PyttsxBackend(_Backend):
    name = "pyttsx3"

    def __init__(self, output_device: str | None = None):
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 170)
        self.device = output_device

    def speak(self, text: str) -> None:
        if not self.device and config.TTS_PREBUFFER_MS <= 0:
            # Cheapest path: let pyttsx3 drive the system default sink.
            self.engine.say(text)
            self.engine.runAndWait()
            return
        fd, path = tempfile.mkstemp(suffix=".wav", prefix="tts_")
        os.close(fd)
        try:
            self.engine.save_to_file(text, path)
            self.engine.runAndWait()
            _prepend_silence(path, config.TTS_PREBUFFER_MS)
            cmd = ["aplay", "-q"]
            if self.device:
                cmd += ["-D", self.device]
            cmd.append(path)
            subprocess.run(cmd, check=False)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass


# ---------------- Piper (Python: piper1-gpl) ----------------


class PiperPyBackend(_Backend):
    name = "piper-py"

    def __init__(self, model_path: Path, output_device: str | None = None):
        from piper import PiperVoice  # imported lazily so we can fall back
        self._voice = PiperVoice.load(str(model_path))
        self.model_name = model_path.name
        self.device = output_device

    def speak(self, text: str) -> None:
        fd, wav = tempfile.mkstemp(suffix=".wav", prefix="tts_")
        os.close(fd)
        try:
            with wave.open(wav, "wb") as wf:
                # piper1-gpl exposes synthesize_wav(text, wave.Wave_write).
                # Older piper-tts wrote raw PCM via .synthesize(); handle both.
                if hasattr(self._voice, "synthesize_wav"):
                    self._voice.synthesize_wav(text, wf)
                else:
                    self._voice.synthesize(text, wf)
            _prepend_silence(wav, config.TTS_PREBUFFER_MS)
            cmd = ["aplay", "-q"]
            if self.device:
                cmd += ["-D", self.device]
            cmd.append(wav)
            subprocess.run(cmd, check=False)
        finally:
            try:
                os.unlink(wav)
            except OSError:
                pass


# ---------------- factory ----------------


def make_backend() -> _Backend:
    if config.TTS_BACKEND.lower() == "piper":
        model = config.PIPER_MODEL_PATH
        if not model.exists():
            print(f"[TTS] Piper model missing at {model}; falling back to pyttsx3.")
        else:
            try:
                backend = PiperPyBackend(model, config.AUDIO_OUTPUT_DEVICE)
            except ImportError as exc:
                print(f"[TTS] piper Python pkg not installed ({exc.name}); "
                      "falling back to pyttsx3. See README §2.9.")
            except Exception as exc:  # noqa: BLE001
                print(f"[TTS] piper load failed: {exc!r}; falling back to pyttsx3.")
            else:
                print(f"[TTS] using Piper (Python): {backend.model_name}")
                return backend
    print("[TTS] using pyttsx3 / espeak-ng")
    return PyttsxBackend(config.AUDIO_OUTPUT_DEVICE)
