"""TTS backends.

Default is Piper (neural, on-device, much more natural than espeak-ng).
Falls back to pyttsx3 / espeak-ng if Piper isn't installed yet, so the
app keeps working even before you've done the optional Piper setup.

Three backends, in preference order:

1. PiperPyBackend  -- uses the piper1-gpl Python package (pip install
                      piper-tts). Loads the ONNX voice once at startup
                      so subsequent utterances are fast.
2. PiperCliBackend -- uses the legacy piper binary (the C++ build from
                      rhasspy/piper releases). Reloads model per call.
3. PyttsxBackend   -- espeak-ng. Always available.

All backends honour config.AUDIO_OUTPUT_DEVICE -- when set, audio is
played via `aplay -D <device>` so it goes to the Anker (or whatever
sink you've pinned) instead of the system default.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

import pyttsx3

import config


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
        if not self.device:
            self.engine.say(text)
            self.engine.runAndWait()
            return
        fd, path = tempfile.mkstemp(suffix=".wav", prefix="tts_")
        os.close(fd)
        try:
            self.engine.save_to_file(text, path)
            self.engine.runAndWait()
            subprocess.run(
                ["aplay", "-q", "-D", self.device, path], check=False
            )
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
                # piper1-gpl exposes synthesize_wav(text, wave.Wave_write)
                # Older piper-tts (pre-rewrite) used .synthesize() that
                # writes raw PCM; we handle both.
                if hasattr(self._voice, "synthesize_wav"):
                    self._voice.synthesize_wav(text, wf)
                else:
                    self._voice.synthesize(text, wf)
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


# ---------------- Piper (CLI binary) ----------------


class PiperCliBackend(_Backend):
    name = "piper-cli"

    def __init__(
        self,
        piper_bin: Path,
        model_path: Path,
        output_device: str | None = None,
    ):
        self.piper = str(piper_bin)
        self.model = str(model_path)
        self.model_name = model_path.name
        self.device = output_device

    def speak(self, text: str) -> None:
        fd, wav = tempfile.mkstemp(suffix=".wav", prefix="tts_")
        os.close(fd)
        try:
            subprocess.run(
                [self.piper, "--model", self.model, "--output_file", wav],
                input=text.encode(),
                check=True,
                stderr=subprocess.DEVNULL,
            )
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


def _resolve_piper_bin() -> Path | None:
    if config.PIPER_BIN.exists():
        return config.PIPER_BIN
    sys_piper = shutil.which("piper")
    return Path(sys_piper) if sys_piper else None


def _try_piper_python(model: Path, device: str | None) -> _Backend | None:
    try:
        backend = PiperPyBackend(model, device)
    except ImportError as exc:
        print(f"[TTS] piper Python pkg not installed ({exc.name}); "
              "trying CLI fallback.")
        return None
    except Exception as exc:  # noqa: BLE001
        print(f"[TTS] piper Python load failed: {exc!r}; trying CLI fallback.")
        return None
    print(f"[TTS] using Piper (Python): {backend.model_name}")
    return backend


def _try_piper_cli(model: Path, device: str | None) -> _Backend | None:
    piper_bin = _resolve_piper_bin()
    if not piper_bin:
        return None
    backend = PiperCliBackend(piper_bin, model, device)
    print(f"[TTS] using Piper (CLI): {backend.model_name}  ({piper_bin})")
    return backend


def make_backend() -> _Backend:
    if config.TTS_BACKEND.lower() == "piper":
        model = config.PIPER_MODEL_PATH
        if not model.exists():
            print(f"[TTS] Piper model missing at {model}; falling back to pyttsx3.")
        else:
            backend = _try_piper_python(model, config.AUDIO_OUTPUT_DEVICE)
            if backend is None:
                backend = _try_piper_cli(model, config.AUDIO_OUTPUT_DEVICE)
            if backend is not None:
                return backend
            print("[TTS] No usable Piper backend; falling back to pyttsx3. "
                  "See README §2.9.")
    print("[TTS] using pyttsx3 / espeak-ng")
    return PyttsxBackend(config.AUDIO_OUTPUT_DEVICE)
