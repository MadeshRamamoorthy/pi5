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
import threading
import time
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
#
# Two paths:
#   1. HailoWhisperNativeASR -- uses HailoRT 5.2+ built-in genai.Speech2Text
#      API. Takes a single combined HEF, handles encoder+decoder
#      internally. No .npy / add_embed / network-group fiddling.
#      Recommended.
#   2. HailoWhisperASR -- legacy path via hailo-apps's whisper_pipeline.py.
#      Needs separate encoder + decoder HEFs + .npy assets.
#      Kept for older HailoRT installs (< 5.2.0).


class HailoWhisperNativeASR(ChatASR):
    """Whisper on Hailo NPU via HailoRT 5.2+'s native genai.Speech2Text.

    Per the official example in hailo_platform.genai, all you need is a
    single combined HEF that contains encoder + decoder network groups.
    HailoRT loads, schedules, and runs the autoregressive decode loop
    internally -- no application-level pipeline code, no .npy files, no
    add_embed flag.

    Requires HailoRT >= 5.2.0. Falls through to a clear error otherwise
    so the factory can pick the legacy backend.
    """

    label = "hailo-whisper-native"

    def __init__(self, hef_path: str | None = None,
                 language: str = "en", timeout_ms: int = 15_000):
        self._hef_path = str(hef_path or
                             getattr(config, "HAILO_WHISPER_HEF", "")
                             or "")
        self._language = language
        self._timeout_ms = timeout_ms
        self._vdevice = None
        self._speech2text = None
        self._task = None
        self._lock = threading.Lock()

    def _ensure_pipeline(self) -> None:
        if self._speech2text is not None:
            return
        if not self._hef_path or not Path(self._hef_path).is_file():
            raise RuntimeError(
                f"Hailo Whisper HEF not found: {self._hef_path}\n"
                "Set HAILO_WHISPER_HEF to a combined Whisper HEF (e.g. "
                "Whisper-Small.hef from Hailo's Model Zoo)."
            )
        try:
            from hailo_platform import VDevice, HailoSchedulingAlgorithm
            from hailo_platform.genai import Speech2Text, Speech2TextTask
        except ImportError as exc:
            raise RuntimeError(
                "hailo_platform.genai.Speech2Text is missing. This API "
                "requires HailoRT >= 5.2.0. Check your version with "
                "`hailortcli -v` and upgrade if older."
            ) from exc

        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        params.group_id = "SHARED"
        print(f"[asr] loading Hailo Whisper (native API) "
              f"from {self._hef_path}")
        self._vdevice = VDevice(params)
        self._speech2text = Speech2Text(self._vdevice, self._hef_path)
        self._task = Speech2TextTask

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
        audio = audio.astype("<f4")
        with self._lock:
            segments = self._speech2text.generate_all_segments(
                audio_data=audio,
                task=self._task.TRANSCRIBE,
                language=self._language,
                timeout_ms=self._timeout_ms,
            )
        if not segments:
            return ""
        return "".join(seg.text for seg in segments).strip()

    def stop(self) -> None:
        if self._speech2text is not None:
            try:
                self._speech2text.release()
            except Exception:
                pass
            self._speech2text = None
        if self._vdevice is not None:
            try:
                self._vdevice.release()
            except Exception:
                pass
            self._vdevice = None


class HailoWhisperASR(ChatASR):
    """Run Whisper on the Hailo-10H / Hailo-8 / Hailo-8L NPU.

    Wraps Hailo's official speech_recognition pipeline from
    https://github.com/hailo-ai/hailo-apps, matching the usage pattern
    in their standalone_apps/speech_recognition reference app:

        pipeline = WhisperPipeline(encoder_path, decoder_path,
                                   variant=..., npy_dir=..., add_embed=...)
        mels = preprocess_audio(audio, chunk_length=pipeline.get_chunk_length())
        for mel in mels:
            pipeline.send_data(mel)
            time.sleep(0.1)
            text = pipeline.get_transcription()
            results.append(text)
        return clean_transcription(" ".join(results))

    Inference is ~200-700 ms per short utterance on Hailo-10H
    (10-30x faster than CPU faster-whisper) and costs nothing per
    request. Imports are deferred to first use so the rest of the
    kiosk still boots cleanly when the hailo-apps package is missing
    -- the asr factory's auto-select then falls through to the next
    backend.

    add_embed: hailo-apps sets True for Hailo-8 / Hailo-8L (embedding
    runs on the host CPU) and False for Hailo-10H (embedding runs on
    the chip). Default False matches the Pi 5 + AI HAT 2+ setup; flip
    it via config.HAILO_WHISPER_ADD_EMBED for older hardware.
    """

    label = "hailo-whisper"

    def __init__(self, model_variant: str | None = None,
                 encoder_path: str | None = None,
                 decoder_path: str | None = None,
                 npy_dir: str | None = None,
                 add_embed: bool | None = None):
        self._variant = (model_variant or
                         getattr(config, "HAILO_WHISPER_MODEL", "base"))
        self._encoder_path = str(encoder_path or
                                 getattr(config, "HAILO_WHISPER_ENCODER_HEF",
                                          self._guess_hef("encoder")))
        self._decoder_path = str(decoder_path or
                                 getattr(config, "HAILO_WHISPER_DECODER_HEF",
                                          self._guess_hef("decoder")))
        self._npy_dir = str(npy_dir or
                            getattr(config, "HAILO_WHISPER_NPY_DIR",
                                    self._guess_npy_dir()))
        self._add_embed = (add_embed if add_embed is not None
                           else bool(getattr(config,
                                              "HAILO_WHISPER_ADD_EMBED",
                                              False)))
        self._pipeline = None
        self._preprocess_audio = None
        self._clean_transcription = None
        self._chunk_length = None
        # Serialise transcribe() calls -- the pipeline keeps internal
        # state across send_data/get_transcription pairs and can't be
        # shared across overlapping requests.
        self._lock = threading.Lock()

    def _guess_hef(self, role: str) -> str:
        return str(
            getattr(config, "MODELS_DIR", Path("models"))
            / f"whisper-{self._variant}-{role}.hef"
        )

    def _guess_npy_dir(self) -> str:
        return str(
            getattr(config, "MODELS_DIR", Path("models"))
            / f"whisper-{self._variant}-assets"
        )

    def _ensure_pipeline(self) -> None:
        if self._pipeline is not None:
            return
        # File-existence checks up front so the error message points
        # at exactly what's missing, not at a deep ImportError or a
        # HailoRT crash later.
        for path in (self._encoder_path, self._decoder_path):
            if not Path(path).is_file():
                raise RuntimeError(
                    f"Hailo Whisper HEF not found: {path}\n"
                    "Install hailo-apps (not on PyPI -- clone the repo):\n"
                    "  git clone https://github.com/hailo-ai/hailo-apps.git\n"
                    "  cd hailo-apps && pip install -e '.[speech-rec]'\n"
                    "Hailo's CLI auto-downloads the HEFs on first run; "
                    "symlink them into models/ or override "
                    "HAILO_WHISPER_ENCODER_HEF / "
                    "HAILO_WHISPER_DECODER_HEF in config.py."
                )
        if not Path(self._npy_dir).is_dir():
            raise RuntimeError(
                f"Hailo Whisper assets directory not found: {self._npy_dir}\n"
                "This holds the decoder tokenization .npy files that "
                "hailo-apps ships alongside the HEFs. Clone hailo-apps "
                "from https://github.com/hailo-ai/hailo-apps and "
                "`pip install -e '.[speech-rec]'`, or set "
                "HAILO_WHISPER_NPY_DIR in config.py to where the assets "
                "already live."
            )

        try:
            from hailo_apps.python.standalone_apps.speech_recognition.whisper_pipeline import WhisperPipeline
            from hailo_apps.python.standalone_apps.speech_recognition.audio_utils import preprocess_audio
            from hailo_apps.python.standalone_apps.speech_recognition.postprocessing import clean_transcription
        except ImportError as exc:
            raise RuntimeError(
                "Hailo Whisper backend requires hailo-apps. It isn't on "
                "PyPI -- clone the repo and install with the speech-rec "
                "extra:\n"
                "  git clone https://github.com/hailo-ai/hailo-apps.git\n"
                "  cd hailo-apps && pip install -e '.[speech-rec]'\n"
                f"Import error: {exc!r}"
            ) from exc

        print(f"[asr] loading Hailo Whisper {self._variant} "
              f"(encoder={self._encoder_path}, decoder={self._decoder_path}, "
              f"npy_dir={self._npy_dir}, add_embed={self._add_embed})")
        self._pipeline = WhisperPipeline(
            self._encoder_path,
            self._decoder_path,
            variant=self._variant,
            npy_dir=self._npy_dir,
            add_embed=self._add_embed,
        )
        self._preprocess_audio = preprocess_audio
        self._clean_transcription = clean_transcription
        self._chunk_length = self._pipeline.get_chunk_length()

    def warmup(self) -> None:
        self._ensure_pipeline()

    def transcribe(self, pcm: bytes, samplerate: int) -> str:
        self._ensure_pipeline()
        # int16 PCM at any sample rate -> float32 mono at 16 kHz, the
        # rate the Whisper preprocessor expects.
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        if samplerate != 16000:
            from audio_utils import resample_int16
            resampled = resample_int16(
                (audio * 32768).astype(np.int16), samplerate, 16000,
            )
            audio = resampled.astype(np.float32) / 32768.0

        with self._lock:
            mels = self._preprocess_audio(audio, chunk_length=self._chunk_length)
            results = []
            for mel in mels:
                self._pipeline.send_data(mel)
                # Tiny gap before pulling the result -- mirrors Hailo's
                # own reference loop. get_transcription() blocks until
                # the worker thread is ready anyway, but the sleep lets
                # short utterances pipeline cleanly.
                time.sleep(0.1)
                text = self._pipeline.get_transcription()
                if text:
                    results.append(text)

        if not results:
            return ""
        return self._clean_transcription(" ".join(results)).strip()

    def stop(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None


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

    `auto` precedence:

      1. hailo-whisper-native -- HailoRT 5.2+ built-in Speech2Text API.
         One combined HEF, no .npy / add_embed / patches. Picked when
         HAILO_WHISPER_HEF is set and the file exists.
      2. hailo-whisper        -- legacy hailo-apps WhisperPipeline path.
         Needs separate encoder + decoder HEFs + .npy assets dir.
         Picked when those three things all exist.
      3. openai               -- if OPENAI_API_KEY is set.
      4. faster-whisper       -- pure CPU fallback.
    """
    backend = config.CHAT_ASR_BACKEND.lower()
    if backend == "auto":
        backend = _auto_pick_backend()
        print(f"[asr] auto-selected backend: {backend}")
    if backend in ("hailo-native", "hailo-whisper-native", "speech2text"):
        return HailoWhisperNativeASR()
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
    # 1. Hailo native Speech2Text (HailoRT 5.2+). Picked when a single
    #    combined HEF is configured AND on disk.
    native_hef = getattr(config, "HAILO_WHISPER_HEF", None)
    if native_hef and Path(native_hef).is_file():
        return "hailo-whisper-native"

    # 2. Legacy hailo-apps pipeline. Needs separate encoder + decoder
    #    HEFs + .npy assets directory all on disk.
    size = getattr(config, "HAILO_WHISPER_MODEL", "base")
    models_dir = getattr(config, "MODELS_DIR", Path("models"))
    enc = getattr(config, "HAILO_WHISPER_ENCODER_HEF",
                  models_dir / f"whisper-{size}-encoder.hef")
    dec = getattr(config, "HAILO_WHISPER_DECODER_HEF",
                  models_dir / f"whisper-{size}-decoder.hef")
    npy = getattr(config, "HAILO_WHISPER_NPY_DIR",
                  models_dir / f"whisper-{size}-assets")
    # If encoder and decoder are the same file path it's a combined HEF
    # -- the legacy hailo-apps path needs them split, so don't pick it.
    same_file = Path(enc).resolve() == Path(dec).resolve() if (
        Path(enc).exists() and Path(dec).exists()) else False
    if (not same_file and Path(enc).is_file() and Path(dec).is_file()
            and Path(npy).is_dir()):
        return "hailo-whisper"

    # 3 / 4. Network or local CPU fallback.
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return "faster-whisper"
