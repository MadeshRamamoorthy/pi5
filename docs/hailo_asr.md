# Speech recognition on the Hailo NPU

**Status: production**. Hailo ships pre-compiled Whisper encoder +
decoder HEFs and a Python pipeline that runs the whole STT path on
the NPU.

| Variant | Approx latency on Hailo-10H | Notes |
|---------|-----------------------------|-------|
| Whisper-Tiny  | ~150-300 ms | Fastest, lowest accuracy |
| **Whisper-Base** | **~250-500 ms** | Recommended default |
| Whisper-Small | ~400-800 ms | Best accuracy, more memory |

vs. our other backends:

| Backend | Latency on a 5 s utterance | Cost |
|---------|---------------------------|------|
| **Hailo Whisper-Base** | **~250-500 ms** | $0 |
| OpenAI Whisper API     | ~1-2 s        | ~$0.0002/min |
| faster-whisper tiny.en (Pi 5 CPU) | ~3-4 s | $0 |

## How the kiosk picks an STT backend

`config.CHAT_ASR_BACKEND = "auto"` (the default) goes through this
precedence:

1. **Hailo Whisper** — chosen when both encoder + decoder HEFs are
   present on disk. The actual `hailo-apps` import is deferred to
   first use, so a missing package only fails on the first chat
   question, not at boot.
2. **OpenAI Whisper** — chosen when `OPENAI_API_KEY` is set in `.env`
   or the environment.
3. **faster-whisper** — pure CPU fallback for fully offline boxes.

Force a specific backend by setting `CHAT_ASR_BACKEND = "hailo"`
(or `"openai"`, `"faster-whisper"`, `"vosk"`).

## Setup on the Pi

```bash
# 1. Install hailo-apps (provides the Whisper pipeline class).
pip install hailo-apps
# Or clone for the latest tip:
#   git clone https://github.com/hailo-ai/hailo-apps
#   pip install -e ./hailo-apps

# 2. Download the Whisper-Base HEFs into models/.
#    Hailo's Model Zoo URL changes occasionally; see
#    https://github.com/hailo-ai/hailo-apps for the current path
#    (look in the speech_recognition app's download script).
cd models/
wget <whisper-base-encoder.hef URL>
wget <whisper-base-decoder.hef URL>
# Rename to match what config.py expects, OR override the paths
# via HAILO_WHISPER_ENCODER_HEF / HAILO_WHISPER_DECODER_HEF.

# 3. Verify selection.
./start.sh
# In the log, look for:
#   [asr] auto-selected backend: hailo-whisper
#   [asr] loading Hailo Whisper base (encoder=..., decoder=...)
```

## Chip-contention reality check

The Hailo-10H currently hosts:

- SCRFD (face detect) + ArcFace (embed) — runs on every camera frame
  while ACTIVE
- Hailo-Whisper encoder + decoder — runs during chat exchanges
- Optionally Hailo-Ollama (`qwen3:1.7b`) — only loaded if
  `PRELOAD_OLLAMA=1` is set or `OPENAI_API_KEY` is unset

In the common case (OpenAI for chat, Hailo for face + Whisper):

- Face recognition and Whisper rarely overlap. The camera worker
  *skips* recognition during a chat exchange (the `chat_busy` check in
  `main.py`), so the NPU is free for Whisper while the user is
  talking. Whisper completes in <1 s and face recognition resumes
  immediately.

If you also load Hailo-Ollama, the NPU schedules all three workloads
serially — workable but adds latency. The kiosk doesn't pre-load
Ollama when `OPENAI_API_KEY` is set, so usually you don't pay this
cost.

## Verifying the latency

```bash
tail -F /tmp/echo-backend.log | grep -E 'asr|chat-voice'
```

Expected sequence on a chat-voice tap:

```
[chat-voice] using hailo-whisper
... user speaks ...
[asr] loading Hailo Whisper base (encoder=..., decoder=...)
... ~500 ms ...
[chat-voice] transcribed: "what's the weather"
```

If you see `[asr] loading...` *every* utterance, that means the
pipeline isn't being cached -- it should only print on the first
chat after each boot.

## What to do if it doesn't work

- Missing HEFs: the wrapper prints the expected file paths and a
  download hint. Drop the files at those paths and restart.
- `hailo-apps` import fails: try `pip install hailo-apps` or clone
  the repo and `pip install -e .`. The wrapper tries three known
  module paths (`hailo_apps.python.standalone_apps.speech_recognition.
  whisper_pipeline`, `hailo_apps.speech_recognition.pipeline`,
  `hailo_whisper.pipeline`) — if Hailo reorganises again, the last
  ImportError prints clearly in the log and we patch the candidates
  list in `asr.py`.
- HailoRT errors at decode: usually means another model is holding
  the NPU. Check `pgrep -fl hailo-ollama` and stop it if you don't
  need offline chat.

## Links

- [hailo-ai/hailo-apps](https://github.com/hailo-ai/hailo-apps) —
  primary repo, has the `speech_recognition` standalone app.
- [hailocs/hailo-whisper](https://github.com/hailocs/hailo-whisper) —
  original conversion tooling.
- [Pi 5 + Whisper-Small HEF community
  thread](https://community.hailo.ai/t/pi-5-whisper-small-hef/18946) —
  troubleshooting and benchmarks from real Pi 5 users.
