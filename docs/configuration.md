# Configuration

Two layers:

1. **`.env`** — user-editable, git-ignored, sourced by `start.sh`
   before the Python process boots.
2. **`config.py`** — project defaults checked into the repo.

`.env` overrides `config.py` (it just sets environment variables that
`config.py` reads via `os.environ.get(...)`).

---

## `.env`

```bash
cp .env.example .env
nano .env
```

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | _empty_ | Primary chat backend. Also used by OpenAI Whisper if Hailo Whisper isn't available. |
| `KIOSK_ADMIN_USER` | `admin` | HTTP basic auth username for `/admin` |
| `KIOSK_ADMIN_PASS` | _random_ | Pin to a value here; otherwise `start.sh` generates one at every launch and prints it in the banner |
| `KIOSK_PORT` | `8090` | Flask port (avoid 8080 — that's Open WebUI by default) |
| `PRELOAD_OLLAMA` | `0` | Set to `1` to always pre-load Hailo-Ollama (faster fallback at the cost of 2.5 GB RAM) |
| `OLLAMA_URL` | `http://localhost:8000` | Override if your local Ollama daemon listens elsewhere |
| `SD_DEVICE` | _auto_ | sounddevice input index (`python -c "import sounddevice; print(sounddevice.query_devices())"`) |
| `AUDIO_OUTPUT_DEVICE` | _system default_ | ALSA device name for the speaker |
| `HAILO_WHISPER_MODEL` | `base` | `tiny` / `base` / `tiny.en` |
| `HAILO_WHISPER_ENCODER_HEF` | _under `models/`_ | Override if you keep them elsewhere |
| `HAILO_WHISPER_DECODER_HEF` | _under `models/`_ | Override if you keep them elsewhere |
| `HAILO_WHISPER_NPY_DIR` | _under `models/`_ | Decoder tokenization assets directory |
| `HAILO_WHISPER_ADD_EMBED` | `false` | `true` for Hailo-8 / Hailo-8L (older hardware) |

`.env` is git-ignored. `start.sh` sources it via `set -a` so every
`key=value` becomes an exported environment variable for the Python
process and all its threads.

---

## `config.py`

The big knobs are grouped near the top. Edit in-place or set the
matching env var (which takes precedence).

### Branding

| Knob | Default | Effect |
|---|---|---|
| `BRAND_NAME` | `"ECHO SCOPE"` | Header text + greeting copy |
| `CONTACT_EMAIL` | `"Calgary_AIClub@infosys.com"` | Footer link target |
| `WAKE_WORD` | `"hello echo scope"` | Primary wake phrase |
| `WAKE_WORD_ALIASES` | `["hello echo", "echo scope", "hey echo"]` | Also accepted by the Vosk grammar |

### Recognition thresholds

| Knob | Default | Effect |
|---|---|---|
| `COSINE_MATCH_THRESHOLD` | `0.38` | Lower = match more loosely (false positives risk) |
| `REREGISTER_MATCH_THRESHOLD` | `0.45` | Re-registration "is this still the same person?" gate |
| `UNKNOWN_FRAMES_BEFORE_REGISTER` | `15` | Frames of unknown before auto-register pops |
| `REGISTER_DECLINE_COOLDOWN_SEC` | `120` | "No thanks" cooldown |

### Silent learning

| Knob | Default | Effect |
|---|---|---|
| `SILENT_LEARN_ENABLED` | `True` | Master switch |
| `SILENT_LEARN_MIN_SCORE` | `0.70` | Min match score to learn from |
| `SILENT_LEARN_MIN_MARGIN` | `0.15` | Best score must beat runner-up by this much |
| `SILENT_LEARN_MAX_SIMILARITY` | `0.92` | Skip if new sample looks like an existing one |
| `SILENT_LEARN_MIN_INTERVAL_SEC` | `60` | Max one new sample per person per minute |
| `SILENT_LEARN_MAX_SAMPLES_PER_PERSON` | `30` | Cap; oldest non-enrolment samples drop first |

### Session lifetime

| Knob | Default | Effect |
|---|---|---|
| `IDLE_AFTER_LAST_INTERACTION_SEC` | `30` | Dashboard timeout after last engagement |
| `CHAT_KEEPALIVE_SEC` | `180` | Chat history keeps the kiosk ACTIVE for this long after the last message |
| `ACTIVE_SESSION_MAX_SEC` | `600` | Hard ceiling on a single ACTIVE session |

### Speech

| Knob | Default | Effect |
|---|---|---|
| `CHAT_ASR_BACKEND` | `"auto"` | Force `"hailo"` / `"openai"` / `"faster-whisper"` / `"vosk"` |
| `WHISPER_MODEL` | `"tiny.en"` | faster-whisper variant |
| `WHISPER_COMPUTE_TYPE` | `"int8"` | Quantisation for memory |
| `CHAT_VOICE_SILENCE_SEC` | `0.9` | VAD silence threshold (lower = snappier cutoff) |
| `CHAT_VOICE_MAX_SEC` | `12` | Hard cap per utterance |
| `CHAT_VOICE_SILENCE_RMS` | `350` | int16 RMS below which audio counts as silence |
| `WAKE_WORD_SAMPLERATE` | `16000` | Vosk decoding rate |
| `CHAT_MAX_QUESTIONS_PER_SESSION` | `5` | Per-emp_id chat budget |
| `CHAT_BACKEND_ORDER` | `("openai", "ollama")` | Failover order |

### Liveness

| Knob | Default | Effect |
|---|---|---|
| `LIVENESS_ENABLED` | `False` | Anti-spoofing master switch. On for spoof-prone deployments |
| `LIVENESS_WINDOW_FRAMES` | `24` | Sliding-window length for jitter / motion checks |
| `LIVENESS_REQUIRE_BLINK` | `False` | Active blink challenge |

### Web UI

| Knob | Default | Effect |
|---|---|---|
| `KIOSK_HOST` | `"127.0.0.1"` | Bind address (set to `"0.0.0.0"` to allow LAN access) |
| `KIOSK_PORT` | `8090` | Flask port |
| `FUN_FACT_ROTATE_SEC` | `15` | Active-screen fact card rotation |
| `MJPEG_QUALITY` | `80` | JPEG quality for camera stream |
| `MJPEG_MAX_FPS` | `20` | Display FPS cap |

See [Tuning](tuning.md) for an opinionated guide to adjusting these.
