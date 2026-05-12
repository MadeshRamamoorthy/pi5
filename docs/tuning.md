# Tuning

Every knob below lives in `config.py`. Override per-deployment via
the matching env var (in `.env`) when one exists.

This page is the quick-reference; for detail on each subsystem see
the linked feature pages.

## Recognition

| Knob | Default | Effect | See |
|---|---|---|---|
| `COSINE_MATCH_THRESHOLD` | `0.38` | Lower = match looser (more false positives) | [face-recognition](face-recognition.md) |
| `UNKNOWN_FRAMES_BEFORE_REGISTER` | `15` | Frames of unknown before auto-register pops | [face-recognition](face-recognition.md) |
| `REGISTER_DECLINE_COOLDOWN_SEC` | `120` | "No thanks" cooldown | [face-recognition](face-recognition.md) |
| `GREET_COOLDOWN_SEC` | `30` | Min seconds between greetings (once-per-session is the real guard) | [state-machine](state-machine.md) |

## Silent learning

| Knob | Default | Effect |
|---|---|---|
| `SILENT_LEARN_ENABLED` | `True` | Master switch |
| `SILENT_LEARN_MIN_SCORE` | `0.70` | Min match score to learn from |
| `SILENT_LEARN_MIN_MARGIN` | `0.15` | Best score must beat runner-up by this much |
| `SILENT_LEARN_MAX_SIMILARITY` | `0.92` | Skip if new sample looks like an existing one |
| `SILENT_LEARN_MIN_INTERVAL_SEC` | `60` | Max one new sample per person per minute |
| `SILENT_LEARN_MAX_SAMPLES_PER_PERSON` | `30` | Cap; oldest non-enrolment samples drop first |

## Session lifetime

| Knob | Default | Effect |
|---|---|---|
| `IDLE_AFTER_LAST_INTERACTION_SEC` | `30` | Auto-IDLE timeout |
| `CHAT_KEEPALIVE_SEC` | `180` | Chat history keeps the kiosk ACTIVE |
| `ACTIVE_SESSION_MAX_SEC` | `600` | Hard ceiling on a single ACTIVE session |

## Voice

| Knob | Default | Effect |
|---|---|---|
| `CHAT_VOICE_SILENCE_SEC` | `0.9` | VAD silence threshold (lower = snappier cutoff) |
| `CHAT_VOICE_MAX_SEC` | `12` | Hard cap per utterance |
| `CHAT_VOICE_SILENCE_RMS` | `350` | int16 RMS below which audio counts as silence |
| `WAKE_WORD_SAMPLERATE` | `16000` | Vosk decoding rate |

## Speech backends

| Knob | Default | Effect | See |
|---|---|---|---|
| `CHAT_ASR_BACKEND` | `"auto"` | `"hailo"` / `"openai"` / `"faster-whisper"` / `"vosk"` | [speech](speech.md) |
| `WHISPER_MODEL` | `"tiny.en"` | faster-whisper variant | [speech](speech.md) |
| `HAILO_WHISPER_MODEL` | `"base"` | `"tiny"` / `"base"` / `"tiny.en"` | [hailo-whisper](hailo-whisper.md) |
| `HAILO_WHISPER_ADD_EMBED` | `False` | `True` for Hailo-8 / Hailo-8L | [hailo-whisper](hailo-whisper.md) |

## Chat budget

| Knob | Default | Effect |
|---|---|---|
| `CHAT_MAX_QUESTIONS_PER_SESSION` | `5` | Per-emp_id question budget |
| `CHAT_BACKEND_ORDER` | `("openai", "ollama")` | Failover order |

## Liveness

| Knob | Default | Effect | See |
|---|---|---|---|
| `LIVENESS_ENABLED` | `False` | Anti-spoofing master switch | [face-recognition](face-recognition.md) |
| `LIVENESS_WINDOW_FRAMES` | `24` | Sliding-window length |
| `LIVENESS_REQUIRE_BLINK` | `False` | Active blink challenge |
| `LIVENESS_MIN_TEXTURE_VAR` | `60.0` | Min texture variance for "real" |
| `LIVENESS_MAX_SPECULAR_RATIO` | `0.10` | Max specular pixel fraction |
| `LIVENESS_PIXEL_JITTER_MIN` | `4.0` | Min frame-to-frame mean abs diff |
| `LIVENESS_REL_MOTION_MIN` | `0.45` | Min landmark motion std |

## Web UI

| Knob | Default | Effect |
|---|---|---|
| `KIOSK_HOST` | `"127.0.0.1"` | Bind address (set to `"0.0.0.0"` to allow LAN access) |
| `KIOSK_PORT` | `8090` | Flask port |
| `FUN_FACT_ROTATE_SEC` | `15` | Active-screen fact card rotation |
| `MJPEG_QUALITY` | `80` | JPEG quality for camera stream (0-100) |
| `MJPEG_MAX_FPS` | `20` | Display FPS cap |

## Pose capture (registration)

| Knob | Default | Effect |
|---|---|---|
| `POSE_HOLD_SEC` | `1.0` | How long the user must hold each pose |
| `POSE_CAPTURE_TIMEOUT_SEC` | `8.0` | Max wait per pose before moving on |
| `POSE_STABLE_SEC` | `0.5` | Min stability before capture |
| `POSE_STABLE_PIXEL_TOL` | `8` | Landmark drift tolerance during stability |

## Quick recipes

### "Recognition is too loose"

```python
COSINE_MATCH_THRESHOLD = 0.42      # was 0.38
SILENT_LEARN_MIN_SCORE = 0.75      # was 0.70
SILENT_LEARN_MIN_MARGIN = 0.20     # was 0.15
```

### "Kiosk drops out mid-chat"

```python
CHAT_KEEPALIVE_SEC = 300           # was 180
```

### "Wake word too unreliable"

```python
# Lower threshold of frames before deciding it's wake; add more aliases.
WAKE_WORD_ALIASES = ["hello echo", "echo scope", "hey echo",
                     "ok echo", "hi echo"]
```

### "Chat-voice cuts me off mid-sentence"

```python
CHAT_VOICE_SILENCE_SEC = 1.4       # was 0.9; wait longer for pauses
```

### "Public kiosk, anti-spoofing important"

```python
LIVENESS_ENABLED = True
LIVENESS_REQUIRE_BLINK = True
```
