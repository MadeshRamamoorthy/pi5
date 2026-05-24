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
import time
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
    """Backends must support speak(text) and stop(). stop() should
    interrupt any audio currently playing -- the chat-mic flow taps it
    when the user starts speaking so the kiosk doesn't talk over them."""
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
        self._proc: subprocess.Popen | None = None

    def speak(self, text: str) -> None:
        if not self.device and config.TTS_PREBUFFER_MS <= 0:
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
            self._proc = subprocess.Popen(cmd)
            try:
                self._proc.wait()
            finally:
                self._proc = None
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    def stop(self) -> None:
        proc = self._proc
        if proc and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass
        try:
            self.engine.stop()
        except Exception:
            pass


# ---------------- Piper (Python: piper1-gpl) ----------------


class PiperPyBackend(_Backend):
    name = "piper-py"

    def __init__(self, model_path: Path, output_device: str | None = None):
        from piper import PiperVoice  # imported lazily so we can fall back
        self._voice = PiperVoice.load(str(model_path))
        self.model_name = model_path.name
        self.device = output_device
        self._proc: subprocess.Popen | None = None
        # Probe whether piper1-gpl's generator API is available + how
        # AudioChunk exposes its PCM. Once at init -- cheap (single
        # synth of " ") and avoids per-utterance hasattr dispatch.
        self._streaming_ok = False
        self._chunk_to_bytes = None
        self._chunk_sample_rate = None
        self._chunk_sample_width = 2
        self._chunk_channels = 1
        self._probe_streaming()

    def _probe_streaming(self) -> None:
        """Try calling voice.synthesize(<probe>) as a generator. If it yields
        an AudioChunk-like object with discoverable PCM bytes, cache the
        accessor and we're set. Anything that fails leaves streaming off
        and we fall back to the file path.

        Probe with real words, NOT a single space: a space synthesizes to an
        empty first chunk, so the "len(bytes) > 0" accessor check below would
        wrongly fail and silently disable streaming."""
        try:
            gen = self._voice.synthesize("Hello there.")
            if not hasattr(gen, "__next__"):
                return
            first = next(iter(gen))
        except (TypeError, StopIteration, Exception):
            return
        # Find a way to get raw int16 PCM bytes out of `first`.
        accessor = None
        for name, fn in (
            ("audio_int16_bytes", lambda c: c.audio_int16_bytes),
            ("audio_int16_array.tobytes()",
             lambda c: c.audio_int16_array.tobytes()),
            ("audio_bytes", lambda c: c.audio_bytes),
            ("bytes(chunk)", lambda c: bytes(c)),
        ):
            try:
                buf = fn(first)
            except Exception:
                continue
            if isinstance(buf, (bytes, bytearray, memoryview)) and len(buf) > 0:
                accessor = fn
                break
        if accessor is None:
            return
        self._chunk_to_bytes = accessor
        self._chunk_sample_rate = getattr(first, "sample_rate", None) or \
            getattr(getattr(self._voice, "config", None), "sample_rate", None)
        self._chunk_sample_width = getattr(first, "sample_width", 2)
        self._chunk_channels = getattr(first, "sample_channels", 1)
        if self._chunk_sample_rate:
            self._streaming_ok = True

    def speak(self, text: str) -> None:
        if self._streaming_ok and getattr(config, "TTS_STREAMING", True):
            self._speak_stream(text)
        else:
            self._speak_file(text)

    def _speak_stream(self, text: str) -> None:
        """Pipe Piper PCM straight to aplay's stdin. First chunk lands
        in ~200-300 ms vs ~1.5-2 s for the file path on long replies."""
        t0 = time.time()
        chunks = iter(self._voice.synthesize(text))
        try:
            first = next(chunks)
        except StopIteration:
            return
        print(f"[tts] first chunk in {(time.time() - t0) * 1000:.0f}ms "
              f"({len(text)} chars)", flush=True)
        sr = getattr(first, "sample_rate", None) or self._chunk_sample_rate
        sw = getattr(first, "sample_width", self._chunk_sample_width)
        ch = getattr(first, "sample_channels", self._chunk_channels)
        fmt = {1: "U8", 2: "S16_LE", 4: "S32_LE"}.get(sw, "S16_LE")
        cmd = ["aplay", "-q", "-r", str(sr), "-c", str(ch), "-f", fmt]
        if self.device:
            cmd += ["-D", self.device]
        # bufsize=0: each write flushes to aplay immediately. Default 8 KiB
        # buffer at 22 kHz adds ~180 ms before aplay sees anything.
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, bufsize=0)
        try:
            if config.TTS_PREBUFFER_MS > 0:
                n_samples = int(sr * config.TTS_PREBUFFER_MS / 1000)
                self._proc.stdin.write(b"\x00" * (n_samples * sw * ch))
            self._proc.stdin.write(self._chunk_to_bytes(first))
            for chunk in chunks:
                self._proc.stdin.write(self._chunk_to_bytes(chunk))
        except (BrokenPipeError, OSError):
            # stop() was called -- aplay terminated, pipe is dead. Quiet exit.
            pass
        finally:
            try:
                if self._proc.stdin:
                    self._proc.stdin.close()
            except (BrokenPipeError, OSError):
                pass
            try:
                self._proc.wait()
            finally:
                self._proc = None

    def _speak_file(self, text: str) -> None:
        """Legacy path: synthesize whole utterance to a temp WAV, then
        aplay it. Slower to start but bulletproof on older piper or odd
        voices."""
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
            self._proc = subprocess.Popen(cmd)
            try:
                self._proc.wait()
            finally:
                self._proc = None
        finally:
            try:
                os.unlink(wav)
            except OSError:
                pass

    def stop(self) -> None:
        proc = self._proc
        if proc and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
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
