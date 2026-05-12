"""Chat-voice ASR backends.

Three implementations behind a tiny interface:

  ChatASR.transcribe(pcm: bytes, samplerate: int) -> str

`pcm` is mono int16 PCM at `samplerate` Hz. Returns the recognised
text (empty string if nothing was heard).

The wake-word listener stays on the small Vosk model; this module is
loaded only when the user actually opens chat voice mode, so the Pi
doesn't pay the memory bill at idle.
"""

from __future__ import annotations

import io
import os
import tempfile
import wave

import numpy as np

import config


class ChatASR:
    """Abstract base."""

    label: str = "?"

    def transcribe(self, pcm: bytes, samplerate: int) -> str:
        raise NotImplementedError

    def warmup(self) -> None:
        """Optional: pre-load model files so the first real transcript
        doesn't pay the load latency."""


# ---------------- faster-whisper (default) ----------------


class FasterWhisperASR(ChatASR):
    label = "faster-whisper"

    def __init__(self):
        self._model = None
        self._model_name = config.WHISPER_MODEL

    def _ensure_model(self):
        if self._model is not None:
            return
        from faster_whisper import WhisperModel
        cache = str(config.WHISPER_CACHE_DIR)
        os.makedirs(cache, exist_ok=True)
        print(f"[asr] loading faster-whisper {self._model_name} "
              f"({config.WHISPER_COMPUTE_TYPE}) ...")
        self._model = WhisperModel(
            self._model_name,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE_TYPE,
            download_root=cache,
        )

    def warmup(self) -> None:
        self._ensure_model()
        # Run the model once on a half-second of silence so internal
        # buffers / GEMM kernels are warm.
        silence = np.zeros(int(16000 * 0.5), dtype=np.float32)
        list(self._model.transcribe(silence, language="en", beam_size=1)[0])

    def transcribe(self, pcm: bytes, samplerate: int) -> str:
        self._ensure_model()
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        if samplerate != 16000:
            # Cheap linear resample is good enough for ASR.
            from audio_utils import resample_int16
            resampled = resample_int16(
                (audio * 32768).astype(np.int16), samplerate, 16000,
            )
            audio = resampled.astype(np.float32) / 32768.0
        segments, _info = self._model.transcribe(
            audio, language="en", beam_size=1, vad_filter=False,
        )
        return " ".join(seg.text for seg in segments).strip()


# ---------------- OpenAI Whisper API ----------------


class OpenAIWhisperASR(ChatASR):
    label = "openai-whisper"

    def __init__(self):
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return
        from openai import OpenAI
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set")
        self._client = OpenAI(api_key=key)

    def transcribe(self, pcm: bytes, samplerate: int) -> str:
        self._ensure_client()
        # Wrap the PCM in a WAV header so the API recognises the format.
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(samplerate)
            wf.writeframes(pcm)
        buf.seek(0)
        buf.name = "speech.wav"   # SDK uses .name to set MIME type
        resp = self._client.audio.transcriptions.create(
            model="whisper-1", file=buf,
        )
        return (resp.text or "").strip()


# ---------------- Vosk fallback (legacy, low quality) ----------------


class VoskFreeformASR(ChatASR):
    """Reuses the wake-word listener's Vosk model in a no-grammar
    recognizer. Quality is worse than Whisper but no extra model
    download required. Kept as the fully-offline fallback."""

    label = "vosk-freeform"

    def __init__(self, model_dir):
        import vosk
        self._vosk = vosk
        self._model = vosk.Model(str(model_dir))

    def transcribe(self, pcm: bytes, samplerate: int) -> str:
        recog = self._vosk.KaldiRecognizer(self._model, samplerate)
        recog.AcceptWaveform(pcm)
        import json
        return (json.loads(recog.FinalResult()).get("text") or "").strip()


# ---------------- factory ----------------


def make_chat_asr() -> ChatASR:
    backend = config.CHAT_ASR_BACKEND.lower()
    if backend == "auto":
        # OpenAI Whisper is ~2-3x faster end-to-end than CPU-bound
        # faster-whisper on the Pi 5 (1-2 s round-trip vs 3-4 s
        # decode). Default to it whenever a key is available; fall
        # back to local faster-whisper for offline operation.
        backend = "openai" if os.environ.get("OPENAI_API_KEY") else "faster-whisper"
        print(f"[asr] auto-selected backend: {backend}")
    if backend in ("faster-whisper", "fasterwhisper", "whisper"):
        return FasterWhisperASR()
    if backend in ("openai", "openai-whisper"):
        return OpenAIWhisperASR()
    if backend == "vosk":
        return VoskFreeformASR(config.VOSK_MODEL_DIR)
    raise ValueError(f"unknown CHAT_ASR_BACKEND: {backend!r}")
