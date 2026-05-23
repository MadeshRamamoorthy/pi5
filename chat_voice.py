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
        on_transcribing_changed: Callable[[bool], None] | None = None,
    ):
        if _IMPORT_ERROR is not None:
            raise ChatVoiceCaptureError(
                f"sounddevice unavailable: {_IMPORT_ERROR!r}"
            )
        self._asr = asr
        self._out_q = out_queue
        self._listener = wake_listener
        self._on_changed = on_listening_changed
        self._on_transcribing = on_transcribing_changed
        try:
            self._device, self._native_rate = pick_input_device(device)
        except Exception as exc:  # noqa: BLE001
            raise ChatVoiceCaptureError(f"input device probe failed: {exc}")
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._listening = False
        # Set by _record when the user tapped the mic but never spoke within
        # CHAT_VOICE_NO_SPEECH_SEC (vs tapping stop). _run uses it to nudge.
        self._no_speech_timeout = False

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

    def _set_transcribing(self, on: bool) -> None:
        if self._on_transcribing:
            try:
                self._on_transcribing(on)
            except Exception:
                pass

    def _run(self) -> None:
        t0 = time.time()
        if self._listener is not None:
            self._listener.pause()
            # Tiny grace period so the ALSA device fully closes before
            # we try to reopen it.
            time.sleep(0.15)
        try:
            self._set_listening(True)
            t_rec_start = time.time()
            pcm = self._record()
            t_rec_end = time.time()
        finally:
            self._set_listening(False)
            if self._listener is not None:
                self._listener.resume()

        rec_dur = t_rec_end - t_rec_start
        audio_kb = len(pcm) / 1024 if pcm else 0
        if not pcm:
            print(f"[chat-voice] no audio captured "
                   f"(record window {rec_dur:.2f}s -- VAD only saw silence)",
                   flush=True)
            if self._no_speech_timeout:
                # Tapped the mic but never spoke -> signal the main loop to
                # nudge the user ("please speak when you're ready").
                self._out_q.put({"event": "no_speech"})
            return
        print(f"[chat-voice] recorded {audio_kb:.0f} KB / "
               f"{rec_dur:.2f}s of audio, transcribing...", flush=True)
        # Transcribing flag stays true for the entire ASR call -- on
        # CPU faster-whisper this is 3-4 s of dead time the user would
        # otherwise see as a frozen UI. With it set the SPA can keep
        # the listen overlay up with a "Processing..." label.
        self._set_transcribing(True)
        t_asr_start = time.time()
        try:
            text = self._asr.transcribe(pcm, self._native_rate)
        except Exception as exc:  # noqa: BLE001
            t_asr_end = time.time()
            print(f"[chat-voice] transcribe FAILED after "
                   f"{(t_asr_end - t_asr_start) * 1000:.0f}ms: {exc!r}",
                   flush=True)
            return
        finally:
            self._set_transcribing(False)
        t_asr_end = time.time()
        asr_ms = (t_asr_end - t_asr_start) * 1000
        total_ms = (t_asr_end - t0) * 1000
        backend = getattr(self._asr, "label", "?")
        print(f"[chat-voice] {backend} transcribed in {asr_ms:.0f}ms "
               f"(total since mic-on: {total_ms:.0f}ms): {text!r}",
               flush=True)
        if text:
            self._out_q.put(text)

    def _record(self) -> bytes:
        chunks: list[bytes] = []
        # ~100 ms blocks make VAD responsive without burning CPU.
        block = max(1, int(self._native_rate * 0.1))
        silence_run = 0.0
        had_voice = False
        start = time.time()
        speech_start = None
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
                now = time.time()
                if had_voice:
                    # Once talking, cap a single utterance from speech start.
                    if speech_start and now - speech_start > config.CHAT_VOICE_MAX_SEC:
                        break
                else:
                    # Still waiting for the user to begin -- give them up to
                    # CHAT_VOICE_NO_SPEECH_SEC before we close the mic.
                    if now - start > config.CHAT_VOICE_NO_SPEECH_SEC:
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
                if getattr(config, "CHAT_VOICE_DEBUG_RMS", False):
                    print(f"[chat-voice] rms={rms:.0f} "
                           f"(silence threshold {config.CHAT_VOICE_SILENCE_RMS}, "
                           f"silence_run={silence_run:.2f}s)", flush=True)
                if rms >= config.CHAT_VOICE_SILENCE_RMS:
                    if not had_voice:
                        had_voice = True
                        speech_start = time.time()
                    silence_run = 0.0
                else:
                    silence_run += block / self._native_rate
                if had_voice and silence_run >= config.CHAT_VOICE_SILENCE_SEC:
                    break

        if not had_voice:
            # Distinguish a no-speech *timeout* (nudge the user) from the
            # user tapping stop (just close quietly).
            self._no_speech_timeout = not self._stop.is_set()
            return b""
        self._no_speech_timeout = False
        pcm = b"".join(chunks)
        # Peak-normalise to ~-3 dBFS. USB mics on the Pi often record
        # quietly (peak ~-25 to -35 dBFS) and Whisper accuracy drops
        # sharply on quiet input. This gives the model the strongest
        # possible signal without clipping. No-op when audio is already
        # loud or silent.
        arr = np.frombuffer(pcm, dtype=np.int16)
        if arr.size:
            peak = int(np.abs(arr).max())
            if 50 < peak < 23000:  # not silent, not already loud/clipping
                gain = 23000 / peak
                arr = np.clip(arr.astype(np.float32) * gain,
                              -32767, 32767).astype(np.int16)
                pcm = arr.tobytes()
        return pcm
