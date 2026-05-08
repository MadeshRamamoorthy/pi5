"""TTS backends.

Default is Piper (neural, on-device, much more natural than espeak-ng).
Falls back to pyttsx3 / espeak-ng if Piper isn't installed yet, so the
app keeps working even before you've done the optional Piper setup.

Both backends honour config.AUDIO_OUTPUT_DEVICE -- if set, audio is
played via `aplay -D <device>` so it ends up on the Anker (or whatever
device you've pinned) instead of the system default sink.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
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


# ---------------- Piper ----------------


class PiperBackend(_Backend):
    name = "piper"

    def __init__(
        self,
        piper_bin: Path,
        model_path: Path,
        output_device: str | None = None,
    ):
        self.piper = str(piper_bin)
        self.model = str(model_path)
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
    """Project-local binary first, then anything on PATH."""
    if config.PIPER_BIN.exists():
        return config.PIPER_BIN
    sys_piper = shutil.which("piper")
    return Path(sys_piper) if sys_piper else None


def make_backend() -> _Backend:
    if config.TTS_BACKEND.lower() == "piper":
        piper_bin = _resolve_piper_bin()
        model = config.PIPER_MODEL_PATH
        if piper_bin and model.exists():
            print(f"[TTS] using Piper voice: {model.name}  ({piper_bin})")
            return PiperBackend(piper_bin, model, config.AUDIO_OUTPUT_DEVICE)
        missing = []
        if not piper_bin:
            missing.append(
                f"binary (looked at {config.PIPER_BIN} and PATH)"
            )
        if not model.exists():
            missing.append(f"model at {model}")
        print(
            f"[TTS] Piper requested but missing: {', '.join(missing)}; "
            "falling back to pyttsx3. See README §2.9."
        )
    print("[TTS] using pyttsx3 / espeak-ng")
    return PyttsxBackend(config.AUDIO_OUTPUT_DEVICE)
