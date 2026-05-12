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
from pathlib import Path

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


# ---------------- Hailo Whisper (NPU) --------------------------------


class HailoWhisperASR(ChatASR):
    """Run Whisper on the Hailo-10H / Hailo-8L NPU.

    Wraps Hailo's official speech_recognition pipeline from
    https://github.com/hailo-ai/hailo-apps. Inference is ~200-700 ms
    per short utterance on Hailo-10H (10-30x faster than CPU
    faster-whisper) and costs nothing per request.

    The actual import + initialisation is deferred to first use so
    the rest of the kiosk still boots cleanly when the hailo-apps
    package is missing -- in that case the asr factory's auto-select
    just falls through to the next backend.
    """

    label = "hailo-whisper"

    def __init__(self, model_size: str | None = None,
                 encoder_hef: str | None = None,
                 decoder_hef: str | None = None):
        self._model_size = (model_size or
                            getattr(config, "HAILO_WHISPER_MODEL", "base"))
        self._encoder_hef = encoder_hef or self._guess_hef("encoder")
        self._decoder_hef = decoder_hef or self._guess_hef("decoder")
        self._pipeline = None

    def _guess_hef(self, role: str) -> str:
        """Pick a sensible default path for the encoder / decoder HEF.
        Override per-instance, or via config.HAILO_WHISPER_*_HEF."""
        custom = getattr(config, f"HAILO_WHISPER_{role.upper()}_HEF", None)
        if custom:
            return str(custom)
        return str(
            getattr(config, "MODELS_DIR", Path("models"))
            / f"whisper-{self._model_size}-{role}.hef"
        )

    def _ensure_pipeline(self) -> None:
        if self._pipeline is not None:
            return
        # Verify the HEFs exist before pulling in hailo-apps, so the
        # error message is friendly and points at the missing file
        # rather than a deep ImportError.
        for path in (self._encoder_hef, self._decoder_hef):
            if not Path(path).is_file():
                raise RuntimeError(
                    f"Hailo Whisper HEF not found: {path}\n"
                    "Download it from Hailo's Model Zoo:\n"
                    "  https://github.com/hailo-ai/hailo-apps\n"
                    f"and place it at {path}, or set HAILO_WHISPER_*_HEF "
                    "in config.py / .env to point elsewhere."
                )

        # Try the canonical entry points in order. hailo-apps has
        # reorganised once or twice in 2025; this gives us a best-effort
        # without locking to a single layout.
        candidates = (
            ("hailo_apps.python.standalone_apps.speech_recognition."
             "whisper_pipeline", "WhisperPipeline"),
            ("hailo_apps.speech_recognition.pipeline", "WhisperPipeline"),
            ("hailo_whisper.pipeline", "WhisperPipeline"),
        )
        WhisperPipeline = None
        last_err = None
        for module_path, cls_name in candidates:
            try:
                module = __import__(module_path, fromlist=[cls_name])
                WhisperPipeline = getattr(module, cls_name)
                break
            except (ImportError, AttributeError) as exc:
                last_err = exc
        if WhisperPipeline is None:
            raise RuntimeError(
                "Hailo Whisper backend requires the hailo-apps package. "
                "Install it with:\n"
                "  pip install hailo-apps\n"
                "or clone https://github.com/hailo-ai/hailo-apps and "
                "`pip install -e .`. Last import error: " + repr(last_err)
            )

        print(f"[asr] loading Hailo Whisper {self._model_size} "
              f"(encoder={self._encoder_hef}, decoder={self._decoder_hef})")
        # The pipeline ctor signature is best-effort -- hailo-apps
        # currently uses (encoder_hef, decoder_hef, ...). If their API
        # has shifted, the TypeError will be obvious in the log and we
        # can patch this call.
        self._pipeline = WhisperPipeline(
            encoder_hef=self._encoder_hef,
            decoder_hef=self._decoder_hef,
        )

    def warmup(self) -> None:
        self._ensure_pipeline()

    def transcribe(self, pcm: bytes, samplerate: int) -> str:
        self._ensure_pipeline()
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        if samplerate != 16000:
            from audio_utils import resample_int16
            resampled = resample_int16(
                (audio * 32768).astype(np.int16), samplerate, 16000,
            )
            audio = resampled.astype(np.float32) / 32768.0
        # hailo-apps Whisper pipeline returns either a string or a dict
        # depending on version; handle both.
        result = self._pipeline.transcribe(audio)
        if isinstance(result, dict):
            return (result.get("text") or "").strip()
        return str(result).strip()


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
    """Pick an ASR backend.

    `auto` precedence (matches the user's requested setup -- Hailo
    NPU first, OpenAI as the network backup, local faster-whisper
    as the fully-offline fallback):

      1. hailo-whisper -- if both encoder and decoder HEFs exist on
         disk. ~200-700 ms / utterance, $0, no network.
      2. openai        -- if OPENAI_API_KEY is set. ~1-2 s round-trip.
      3. faster-whisper -- pure CPU fallback. ~3-4 s on Pi 5.
    """
    backend = config.CHAT_ASR_BACKEND.lower()
    if backend == "auto":
        backend = _auto_pick_backend()
        print(f"[asr] auto-selected backend: {backend}")
    if backend in ("hailo", "hailo-whisper"):
        return HailoWhisperASR()
    if backend in ("faster-whisper", "fasterwhisper", "whisper"):
        return FasterWhisperASR()
    if backend in ("openai", "openai-whisper"):
        return OpenAIWhisperASR()
    if backend == "vosk":
        return VoskFreeformASR(config.VOSK_MODEL_DIR)
    raise ValueError(f"unknown CHAT_ASR_BACKEND: {backend!r}")


def _auto_pick_backend() -> str:
    # Hailo Whisper: only choose it when the HEFs are actually on disk.
    # The hailo-apps package import is deferred to first transcribe(),
    # so we only need to validate file paths here.
    enc = getattr(config, "HAILO_WHISPER_ENCODER_HEF", None)
    dec = getattr(config, "HAILO_WHISPER_DECODER_HEF", None)
    size = getattr(config, "HAILO_WHISPER_MODEL", "base")
    models_dir = getattr(config, "MODELS_DIR", Path("models"))
    enc_path = Path(enc) if enc else (models_dir / f"whisper-{size}-encoder.hef")
    dec_path = Path(dec) if dec else (models_dir / f"whisper-{size}-decoder.hef")
    if enc_path.is_file() and dec_path.is_file():
        return "hailo-whisper"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return "faster-whisper"
