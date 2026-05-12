# ECHO SCOPE

**A face-recognising kiosk for the Raspberry Pi 5 + Hailo-10H** — wake
it with your voice, chat with an LLM, and the kiosk remembers
returning visitors.

The browser SPA presents a glassmorphic 1280×800 idle dashboard
(weather, projects, upcoming sessions, hi-5 metrics) that switches to
a live camera + transcript view the moment it hears the wake phrase
"hello echo scope". Face recognition, anti-spoof liveness, voice
transcription, and LLM chat all run locally on the Hailo NPU when
possible, with OpenAI as a network-side option.

| | |
|---|---|
| **Hardware** | Raspberry Pi 5 (8 GB) · Pi AI Camera (IMX500) · Hailo-10H M.2 AI HAT 2+ |
| **OS** | Raspberry Pi OS Trixie · Python 3.13 · HailoRT 5.x |
| **NPU workloads** | SCRFD (detect) · ArcFace (embed) · Whisper-Base (STT) · optional `qwen3:1.7b` LLM |
| **Backends** | Chat: OpenAI / Hailo-Ollama / Anthropic-ready · STT: Hailo / OpenAI / faster-whisper / Vosk · TTS: Piper / pyttsx3 |
| **UI** | Chromium kiosk pointed at a local Flask app on `:8090` |

---

## Contents

1. [Quick start](#quick-start)
2. [Architecture](#architecture)
3. [Hardware checklist](#hardware-checklist)
4. [Install](#install)
5. [Configuration](#configuration)
6. [Running the kiosk](#running-the-kiosk)
7. [Speech: STT, TTS, wake word, chat](#speech)
8. [Face recognition](#face-recognition)
9. [Admin UI](#admin-ui)
10. [State machine and timeouts](#state-machine)
11. [Tuning](#tuning)
12. [Troubleshooting](#troubleshooting)
13. [Project layout](#project-layout)
14. [Privacy and limitations](#privacy-and-limitations)

---

## Quick start

For a Pi 5 + AI HAT 2+ that already has HailoRT 5.x installed, models in
`models/`, and a working `python3.13 -m venv .venv`:

```bash
git clone <your-fork-url> pi5
cd pi5
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
nano .env                                  # paste OPENAI_API_KEY=sk-...

./start.sh
```

Chromium opens in kiosk mode at `http://127.0.0.1:8090`. Say
**"hello echo scope"** to wake.

Stop with **Ctrl+C** in the launching terminal, or from another
terminal:

```bash
./stop.sh
```

If you don't yet have HailoRT / models / venv, jump to [Install](#install).

---

## Architecture

```
                          IDLE                                    ACTIVE
                                                                   (camera + AI overlays)

  ┌─────────────────────────────────────┐         ┌─────────────────────────────────────┐
  │ 👋 I'M ECHO SCOPE        Mon 3:42 PM│         │ ECHO SCOPE     🌡 21°C  🙋 247  ✕  │
  │                                     │         │                                     │
  │   Say "Hello ECHO SCOPE!" to talk   │         │  ┌────────────┐                     │
  │                                     │         │  │ live MJPEG │                     │
  │   ☀ 21°C       👋  247 hi-5s today  │         │  │ + bounding │                     │
  │   💧 45%       This week 1,432      │ ──────▶ │  │ boxes      │                     │
  │                Best day Thu (312!)  │  wake   │  │            │                     │
  │                                     │  word   │  └────────────┘                     │
  │   💡 CALGARY AI FILES ON DISPLAY    │         │                                     │
  │      ● Seeder Ideas   ● AWS COE     │         │  💡 AI FUN FACT                     │
  │      ● Digital Quality Railcar      │         │  ChatGPT reached 100M users in...   │
  │                                     │         │                                     │
  │   📅 UPCOMING SESSION               │         │           🎤 Tap to speak           │
  │      Build Your Own Doc Chatbot     │         │                                     │
  │      Fri May 29 · 3:30–4:30 PM      │         │                                     │
  └─────────────────────────────────────┘         └─────────────────────────────────────┘
                  ▲                                                   │
                  └──────── idle 30s OR ✕ tapped ─────────────────────┘
```

When the user taps the mic on the active screen, the camera + chat
panel split 50/50:

```
┌─────────────────────┬───────────────────────┐
│                     │ 💬 CHAT (5 left)      │
│  Camera 598×520     │                       │
│                     │ [user] hi             │
│                     │ [assistant] Hey!      │
│                     │ ┌─────────────┬─────┐ │
│                     │ │ type or 🎤  │  ✈  │ │
└─────────────────────┴───────────────────────┘
              🎤 Tap to speak
```

### Module map

```
   Pi AI Camera (IMX500) ──┐
                           │ picamera2 RGB888
                           ▼
   ┌──────────────────────────────────────────────────────┐
   │  CameraWorker thread (main.py)                       │
   │                                                      │
   │  pipe.detect (SCRFD)    →  Hailo NPU                 │
   │  pipe.embed  (ArcFace)  →  Hailo NPU                 │
   │  silent learner (margin-gated)                       │
   │  registration / pose capture                         │
   │  state machine IDLE ↔ ACTIVE                         │
   │                                                      │
   │  draws boxes on frame → frame_streamer.FrameStreamer │
   └──────────┬─────────────────────────────┬─────────────┘
              │                             │
   wake_word  │  state.py StateBus          │  /camera.mjpg
   (Vosk)     │  (diff broadcast via SSE)   │
              │                             │
              │                             ▼
              │                  ┌─────────────────────────┐
              │                  │ web/app.py Flask app    │
              │                  │   /            SPA      │
              │                  │   /events      SSE      │
              │                  │   /camera.mjpg          │
              │                  │   /api/*                │
              │                  │   /admin                │
              │                  └────────┬────────────────┘
              │                           ▼
              │                  Chromium --kiosk --app=http://127.0.0.1:8090
              │
   chat_voice ▼
   ChatVoiceCapture
   ┌──────────────────────────────┐
   │ Whisper backend (auto)       │
   │   1. HailoWhisperASR  (NPU)  │
   │   2. OpenAIWhisperASR (API)  │
   │   3. FasterWhisperASR (CPU)  │
   └─────────┬────────────────────┘
             │ text
             ▼
   chat.ChatClient → OpenAI / Hailo-Ollama → reply
                          │
                          ▼
   async_tts.AsyncTTS  →  Piper / pyttsx3  →  speaker
```

---

## Hardware checklist

| Item | Notes |
|---|---|
| Raspberry Pi 5 (8 GB) | 4 GB works but tight; the chat LLM benefits from headroom |
| AI HAT 2+ (Hailo-10H 26 TOPS) | Use the **Gen 3** ribbon. Older 8L works but you'd lose Whisper-Tiny.en + need `HAILO_WHISPER_ADD_EMBED=true` |
| Pi AI Camera (Sony IMX500) | The standard Pi v3 camera also works (RGB pipeline is the same) |
| USB mic + speaker | Tested on Anker PowerConf A3301 (USB conferencing device) |
| Display | Designed for **1280×800**. Smaller panels are scaled-to-fit by the SPA JS |
| Touchscreen (optional) | DOM pointer events work; mouse is equivalent |

---

## Install

### 1. OS prep + PCIe Gen 3

```bash
sudo apt update && sudo apt full-upgrade -y
sudo apt install -y python3.13-venv python3-pip git \
                    libportaudio2 portaudio19-dev \
                    espeak-ng alsa-utils \
                    ffmpeg                          # required by hailo-apps speech-rec
sudo raspi-config nonint do_pcie_gen 3              # enable Gen 3 for Hailo-10H
sudo reboot
```

### 2. Hailo driver + runtime

apt's `hailo-all` ships HailoRT 4.23 which doesn't know about
Hailo-10H. Get HailoRT 5.x as a `.deb` from
<https://hailo.ai/developer-zone/software-downloads/>:

- `hailort_5.x.y_arm64.deb`
- `hailort-5.x.y-cp313-cp313-linux_aarch64.whl` (matches Trixie's
  Python 3.13; pick `cp311` for Bookworm)

Do **not** install Hailo's `hailort-pcie-driver` `.deb` — apt
already supplied the matching kernel module.

```bash
sudo cp ~/Downloads/hailort_5.*_arm64.deb /tmp/
sudo apt install /tmp/hailort_5.*_arm64.deb

hailortcli --version                                # must show 5.x
hailortcli fw-control identify                      # Architecture: HAILO10H
```

If `apt update` flags a stale `hailo.list` source:

```bash
sudo rm -f /etc/apt/sources.list.d/hailo.list /etc/apt/keyrings/hailo.gpg
```

### 3. Clone + venv

```bash
mkdir -p ~/Documents/code && cd ~/Documents/code
git clone <your-fork-url> pi5
cd pi5

python3 -m venv --system-site-packages .venv      # for apt-installed picamera2/libcamera
source .venv/bin/activate
pip install --upgrade pip
pip install ~/Downloads/hailort-5.*-cp313-cp313-linux_aarch64.whl
pip install -r requirements.txt
```

### 4. Models

Three required, two optional. All live in `models/`.

| File | Purpose | Where to get it |
|---|---|---|
| `scrfd_10g.hef` | Face detector (Hailo) | <https://github.com/hailo-ai/hailo_model_zoo/releases> · filter HAILO10H |
| `arcface_mobilefacenet.hef` | Face embedder (Hailo) | same source |
| `vosk-model-small-en-us-0.15/` | Wake-word recogniser | `wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip` and unzip into `models/` |
| `whisper-base-encoder.hef` + `whisper-base-decoder.hef` + `whisper-base-assets/` | Hailo Whisper STT (recommended) | `pip install 'hailo-apps[speech-rec]'` auto-downloads on first use — symlink or copy into `models/`. See [docs/hailo_asr.md](docs/hailo_asr.md) |
| `en_US-hfc_female-medium.onnx` + `.json` | Piper TTS voice | `python -m piper.download_voices en_US-hfc_female-medium --data-dir models/` |

Verify the Hailo HEFs target the right chip:

```bash
hailortcli parse-hef models/scrfd_10g.hef             | head -1
hailortcli parse-hef models/arcface_mobilefacenet.hef | head -1
# Both must say: HEF Compatible for: HAILO10H
```

Smoke-test (run2 is the H10 path):

```bash
hailortcli run2 -t 5 set-net models/scrfd_10g.hef
# expect fps: 240+
```

### 5. Audio

```bash
arecord -l                                          # find a capture device
arecord -d 3 -f cd /tmp/test.wav && aplay /tmp/test.wav    # mic loopback
```

If you have a USB speakerphone you want to pin (instead of HDMI),
either set it as the system default in `wpctl`/`pactl` or set
`AUDIO_OUTPUT_DEVICE=plughw:CARD=PowerConf,DEV=0` in `.env`.

### 6. Browser

```bash
sudo apt install -y chromium fonts-noto-color-emoji
```

(start.sh tries `chromium-browser`, `chromium`, `google-chrome`,
`firefox` in that order — first one found wins.)

---

## Configuration

### .env (user-editable)

```bash
cp .env.example .env
nano .env
```

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | _empty_ | Primary chat backend. Also used by OpenAI Whisper if Hailo Whisper isn't available. |
| `KIOSK_ADMIN_USER` | `admin` | HTTP basic auth username for `/admin` |
| `KIOSK_ADMIN_PASS` | _random_ | Pin to a value here; otherwise start.sh generates one at every launch and prints it in the banner |
| `KIOSK_PORT` | `8090` | Flask port (avoid 8080 — that's Open WebUI by default) |
| `PRELOAD_OLLAMA` | `0` | Set to `1` to always pre-load Hailo-Ollama (faster fallback at the cost of 2.5 GB RAM) |
| `SD_DEVICE` | _auto_ | sounddevice input index (`python -c "import sounddevice; print(sounddevice.query_devices())"`) |
| `AUDIO_OUTPUT_DEVICE` | _system default_ | ALSA device name for the speaker |
| `HAILO_WHISPER_MODEL` | `base` | `tiny` / `base` / `tiny.en` |
| `HAILO_WHISPER_ENCODER_HEF`, `_DECODER_HEF`, `_NPY_DIR` | _under `models/`_ | Override if you keep them elsewhere |
| `HAILO_WHISPER_ADD_EMBED` | `false` | `true` for Hailo-8 / Hailo-8L (older hardware) |

`.env` is git-ignored. start.sh sources it via `set -a` so every
key=value becomes an exported environment variable for the Python
process and its threads.

### config.py (project defaults)

The big knobs are grouped near the top of `config.py`. Common
overrides — drop into `config.py` directly or set the matching env
var:

| Knob | Default | What it controls |
|---|---|---|
| `BRAND_NAME` | `"ECHO SCOPE"` | Header text + greeting copy |
| `WAKE_WORD` | `"hello echo scope"` | Primary wake phrase |
| `WAKE_WORD_ALIASES` | `["hello echo", "echo scope", "hey echo"]` | Also accepted by the Vosk grammar |
| `COSINE_MATCH_THRESHOLD` | `0.38` | Lower = match more loosely (false positives risk) |
| `LIVENESS_ENABLED` | `False` | Anti-spoofing pipeline. Off in trusted environments |
| `IDLE_AFTER_LAST_INTERACTION_SEC` | `30` | Dashboard timeout after the last engagement |
| `CHAT_KEEPALIVE_SEC` | `180` | While a chat exchange exists, kiosk stays ACTIVE for this long after the last message |
| `CHAT_MAX_QUESTIONS_PER_SESSION` | `5` | Per-emp_id chat budget |
| `CHAT_ASR_BACKEND` | `"auto"` | Force `"hailo"` / `"openai"` / `"faster-whisper"` / `"vosk"` if needed |

---

## Running the kiosk

### start.sh

```bash
./start.sh
```

Launches:

1. Sources `.env` if present
2. Activates the venv
3. Skips Hailo-Ollama when `OPENAI_API_KEY` is set (unless
   `PRELOAD_OLLAMA=1`)
4. Generates a random admin password if `KIOSK_ADMIN_PASS` is unset
5. Warms up Whisper in the background
6. Starts `main.py` (backend) in the background, logs to
   `/tmp/echo-backend.log`
7. Polls `/api/state` until the Flask app is up
8. Execs Chromium in `--kiosk --app=http://127.0.0.1:8090`

The startup banner shows everything important:

```
---------------------------------------------------------------
 ECHO SCOPE kiosk
   project    : /home/echo/Documents/code/pi5
   config     : .env loaded
   openai     : key set (***k3xL)
   mic        : sounddevice index 1
   speaker    : plughw:CARD=PowerConf,DEV=0
   web URL    : http://127.0.0.1:8090
   admin user : admin
   admin pass : kP3qN-tw    (generated -- set KIOSK_ADMIN_PASS to pin)
---------------------------------------------------------------
   backend    : up (pid 12345)
   browser    : chromium (kiosk mode)
```

### stop.sh

```bash
./stop.sh                # graceful TERM, then KILL after ~3 s
FORCE=1 ./stop.sh        # immediate KILL
./stop.sh --ollama       # also stop the hailo-ollama daemon
```

Releases the kiosk port via `fuser` if anything's still bound.

### Live logs

```bash
tail -F /tmp/echo-backend.log                          # everything
tail -F /tmp/echo-backend.log | grep -E 'wake-word|state|GREET|asr'
```

The backend emits a partial + final transcript for every Vosk
hypothesis so you can see exactly what it heard:

```
[wake-word] partial: 'hello'
[wake-word] partial: 'hello echo'
[wake-word] FINAL  : 'hello echo'
[state] IDLE -> ACTIVE (session 2c4c54cddc77)
[GREET] Welcome, Parakh!
```

### Auto-start on boot

Add a user systemd unit at `~/.config/systemd/user/echo-scope.service`:

```ini
[Unit]
Description=ECHO SCOPE kiosk
After=graphical-session.target

[Service]
Type=simple
WorkingDirectory=/home/echo/Documents/code/pi5
ExecStart=/home/echo/Documents/code/pi5/start.sh
Restart=on-failure
Environment=DISPLAY=:0

[Install]
WantedBy=default.target
```

```bash
systemctl --user daemon-reload
systemctl --user enable --now echo-scope
loginctl enable-linger echo
```

---

## Speech

### Wake word (Vosk)

A small Vosk model with a tight grammar listens continuously while
the kiosk is IDLE. The grammar accepts the primary phrase plus
`WAKE_WORD_ALIASES`:

```
"hello echo scope"  /  "hello echo"  /  "echo scope"  /  "hey echo"
```

The recogniser is recreated on every stream session, which avoids a
known Vosk `FinalizeDecoding` bug after pause/resume cycles.

While ACTIVE, the wake-word listener is **paused** entirely
(`listener.pause()`) — saves CPU, releases the mic for chat-voice,
and prevents a stray "hello" from confusing the session.

### Chat-voice (Whisper)

Tap the listening pill → the wake-word listener releases the mic →
`ChatVoiceCapture` opens its own input stream → VAD finalises after
`CHAT_VOICE_SILENCE_SEC` (0.9 s) of silence or `CHAT_VOICE_MAX_SEC`
(12 s) max → ASR backend transcribes.

Backend auto-precedence (`CHAT_ASR_BACKEND = "auto"`):

| # | Backend | Latency (5 s clip) | Cost | Local? |
|---|---|---|---|---|
| 1 | **Hailo Whisper-Base** | ~250–500 ms | $0 | ✅ |
| 2 | OpenAI Whisper API | ~1–2 s | $0.0002/min | ❌ |
| 3 | faster-whisper tiny.en (CPU) | ~3–4 s | $0 | ✅ |

For Hailo Whisper setup, see [docs/hailo_asr.md](docs/hailo_asr.md).

The kiosk shows a full-frame mic overlay for **every busy phase** so
the UI never looks frozen during processing:

| State | Label |
|---|---|
| `listening` | "Listening…" |
| `transcribing` | "Got it — transcribing…" |
| `chat_pending` | "Looking that up…" |

### Chat (LLM)

`config.CHAT_BACKEND_ORDER = ("openai", "ollama")` tries OpenAI first,
falls back to Hailo-Ollama (`qwen3:1.7b`) if the network call fails.

Per-session budget is enforced — once a user hits
`CHAT_MAX_QUESTIONS_PER_SESSION`, the next attempt receives a polite
"that's enough for now" reply.

### TTS

`async_tts.AsyncTTS` wraps the synchronous backend with a queue so
the camera loop never blocks waiting for `aplay`. The `on_start`
callback fires from the worker thread immediately before audio plays,
so toasts and spoken lines land together rather than the UI racing
ahead of the queue.

| Backend | Quality | Latency |
|---|---|---|
| Piper (`en_US-hfc_female-medium`) | Natural | ~200–400 ms |
| `pyttsx3` + espeak-ng | Robotic | ~50 ms |

Piper auto-loads when its ONNX voice is present; the kiosk falls back
to pyttsx3 if not.

---

## Face recognition

### Pipeline

```
SCRFD (face detect, NPU) →
   biggest_quality_face filter (size + landmarks + frontality) →
      LivenessChecker (optional) →
         align_face + ArcFace embed (NPU) →
            cosine match against gallery →
               score >= 0.38 (loose) → recognised
                                      → margin >= 0.15 → silent learner adds embedding
               else → bounding box "Unknown"
```

Bounding boxes are drawn server-side onto each MJPEG frame:

- 🟢 **green**: recognised — shows `Name 0.71`
- 🟡 **yellow**: unknown / low-confidence match
- 🟠 **orange**: low quality (too small / too off-axis)
- 🔵 **amber**: liveness still gathering frames

### Registration flow

1. Unknown face stays in frame for `UNKNOWN_FRAMES_BEFORE_REGISTER`
   (~15 frames, ~1 s) and `--auto-register` is enabled.
2. Registration form appears with three actions:
   - **Register**: form closes, pose capture begins, the camera
     screen shows a `Pose 1/5` glass card while it walks through the
     pose prompts.
   - **No thanks**: kiosk speaks a random `decline_registration` line
     from `messages.py` ("No worries! You can still ask me
     anything.") and drops back to the IDLE dashboard. 120 s cooldown
     before the prompt can auto-pop again.
   - **✕ close**: same as No thanks.
3. Greeting fires once per ACTIVE session per emp_id — even if you
   leave and re-enter frame.

### Photo upload (admin)

`/admin` → "Register from photo" tab. Upload 1–N images of the same
person; each one becomes an embedding. Useful for back-office
onboarding without the live capture flow. Admin uploads run silently
— no TTS announcement on the kiosk speaker.

### Silent learning safeguards

A successful greeting opportunistically adds the face's embedding
to that person's gallery, but only if:

- match score ≥ `SILENT_LEARN_MIN_SCORE` (0.70)
- runner-up score is ≥ `SILENT_LEARN_MIN_MARGIN` (0.15) below the
  best — closes the two-similar-people drift trap
- new sample isn't a near-duplicate of an existing one
  (cos ≤ `SILENT_LEARN_MAX_SIMILARITY` 0.92)
- at most one new sample per minute per person
- cap at `SILENT_LEARN_MAX_SAMPLES_PER_PERSON` (30) total

Set `SILENT_LEARN_ENABLED = False` to disable entirely.

### Liveness (optional)

Disabled by default (`LIVENESS_ENABLED = False`). When on, the
checker runs a sliding-window analysis of texture variance,
specular highlights, pixel jitter, and relative motion — plus an
optional active blink challenge — before greeting anyone. Effective
against printed photos and most phone-screen replays; not bulletproof.
See `liveness.py` for the thresholds.

---

## Admin UI

`http://<pi>:8090/admin` — HTTP basic auth, credentials from the
startup banner (or whatever you pinned in `.env`).

Four tabs:

| Tab | What's there |
|---|---|
| **Projects** | The list shown on the idle dashboard. Add / delete |
| **Sessions** | Upcoming session card (next row from this table) |
| **Employees** | Registered faces — name, sample count, rename, delete |
| **Register from photo** | Multipart upload — 1+ images per person, silent registration |

A metrics strip above the tabs summarises total / today / this week /
best-day interaction counts.

JSON API surface:

```
GET    /api/state                       full kiosk state snapshot
GET    /events                          SSE diff stream
GET    /camera.mjpg                     MJPEG camera feed
POST   /api/wake                        force IDLE -> ACTIVE
POST   /api/idle                        force ACTIVE -> IDLE
POST   /api/listen/{start,stop}         chat-voice toggle
POST   /api/chat                        {question: "..."}
POST   /api/register                    {emp_id, name}             (kiosk live capture)
POST   /api/register/photo              multipart                  (admin photo upload)
POST   /api/register/skip               decline + cooldown
GET    /api/projects                    list
POST   /api/projects                    {title, description, ordering}
POST   /api/projects/<id>               update
DELETE /api/projects/<id>               delete
GET    /api/sessions                    list (also POST/DELETE)
GET    /api/employees                   list (also POST/DELETE)
GET    /api/metrics                     {total, today, week, best_day, best_count}
```

Endpoints under `/admin` and the mutating `/api/{employees, sessions,
projects, register/photo}` require basic auth.

---

## State machine

Two top-level states (IDLE / ACTIVE) plus three sub-states the SPA
toggles on while ACTIVE (`listening`, `transcribing`, `chat_pending`).

| Transition | Trigger |
|---|---|
| IDLE → ACTIVE | wake word OR tap-anywhere on the dashboard |
| ACTIVE → IDLE | ✕ button OR `idle_for ≥ IDLE_AFTER_LAST_INTERACTION_SEC` (30 s) OR `active_for ≥ ACTIVE_SESSION_MAX_SEC` (600 s) |
| Engaged (resets idle timer) | `register_open` · `listening` · `chat_pending` · chat history non-empty within `CHAT_KEEPALIVE_SEC` (180 s) |

During chat (`chat_pending` or `listening`), face detection is
**skipped** on the camera worker — no NPU cycles wasted on a face
that won't be acted on, and the unknown-face streak resets so the
register overlay can't pop mid-conversation.

`go_idle()` clears `chat_history`, `toast`, `register_open`,
`listening`, `chat_pending`, and resumes the wake-word listener. The
SPA hides the active screen and shows the dashboard.

---

## Tuning

Every knob below lives in `config.py`. Tweak as needed.

### Recognition

| Knob | Default | Effect |
|---|---|---|
| `COSINE_MATCH_THRESHOLD` | 0.38 | Lower = match looser (more false positives) |
| `UNKNOWN_FRAMES_BEFORE_REGISTER` | 15 | Frames of unknown before auto-register pops |
| `REGISTER_DECLINE_COOLDOWN_SEC` | 120 | "No thanks" cooldown |
| `GREET_COOLDOWN_SEC` | 30 | Min seconds between greetings (within reason — once-per-session is the actual guard) |

### Session lifetime

| Knob | Default | Effect |
|---|---|---|
| `IDLE_AFTER_LAST_INTERACTION_SEC` | 30 | Auto-IDLE timeout |
| `CHAT_KEEPALIVE_SEC` | 180 | Chat history keeps the kiosk ACTIVE |
| `ACTIVE_SESSION_MAX_SEC` | 600 | Hard ceiling on a single ACTIVE session |

### Voice

| Knob | Default | Effect |
|---|---|---|
| `CHAT_VOICE_SILENCE_SEC` | 0.9 | VAD silence threshold (lower = snappier cutoff) |
| `CHAT_VOICE_MAX_SEC` | 12 | Hard cap per utterance |
| `CHAT_VOICE_SILENCE_RMS` | 350 | int16 RMS below which audio counts as silence |
| `WAKE_WORD_SAMPLERATE` | 16000 | Vosk decoding rate |

### Chat budget

| Knob | Default | Effect |
|---|---|---|
| `CHAT_MAX_QUESTIONS_PER_SESSION` | 5 | Per-emp_id question budget |
| `CHAT_BACKEND_ORDER` | `("openai", "ollama")` | Failover order |

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `Port 8090 is in use by another program` | Something else (Open WebUI?) on 8090. Override `KIOSK_PORT` in `.env` |
| `[camera-worker] init failed: ...` | Hailo HEF wrong arch or models missing. `hailortcli parse-hef models/scrfd_10g.hef \| head -1` should say HAILO10H |
| `Address already in use` | Previous backend didn't shut down. `./stop.sh` or `sudo fuser -k 8090/tcp` |
| `Package 'chromium-browser' has no installation candidate` | Pi OS package is `chromium` — `sudo apt install -y chromium`. start.sh tries both |
| `[wake-word] disabled: ...` | Vosk or sounddevice failed to import, or `models/vosk-model-small-en-us-0.15/` is missing |
| `sqlite3.ProgrammingError: SQLite objects created in a thread...` | Re-pull — fixed in commit df21737 (`check_same_thread=False`) |
| Recognising wrong people | Silent learning was over-eager; cleanup: `sqlite3 faces.db "DELETE FROM face_embeddings WHERE emp_id='X';"` then re-register. Confirm `SILENT_LEARN_MIN_MARGIN ≥ 0.15` |
| Chat says "Chat unavailable — no backend reachable" | `OPENAI_API_KEY` unset AND Hailo-Ollama not running. Either paste a key in `.env` or `PRELOAD_OLLAMA=1 ./start.sh` |
| Gibberish Whisper output | If using Hailo on Hailo-8/8L, set `HAILO_WHISPER_ADD_EMBED=true` |
| Kiosk drops out mid-chat | Bump `CHAT_KEEPALIVE_SEC`. Watch `[state] ACTIVE -> IDLE` log line for the actual reason |
| First chat takes 3-4 s, subsequent ones are 1 s | Whisper model load on first request. start.sh warms it in the background after launch — make sure `[asr] loading...` appears within ~5 s of boot |
| First boot took the system into swap | Likely on the old branch with `vosk-model-en-us-0.22`. Either remove that directory or pull the latest (pinned to the small model) |

---

## Project layout

```
pi5/
├── main.py                    Camera worker + orchestrator
├── state.py                   StateBus + KioskState dataclass
├── messages.py                Greeting / error / fact copy (from ai_lab.txt)
├── fun_facts.py               Background rotator for AI Fun Fact / AI Tip card
├── frame_streamer.py          MJPEG ring buffer
├── chat.py                    LLM client + budget tracking
├── chat_voice.py              ChatVoiceCapture (records → transcribes)
├── asr.py                     ChatASR backends (Hailo / OpenAI / faster-whisper / Vosk)
├── async_tts.py               Threaded TTS queue with on_start callbacks
├── tts.py                     Piper / pyttsx3 backends
├── wake_word.py               Vosk wake-word listener (single-concern, paused while ACTIVE)
├── hailo_infer.py             HailoFacePipeline (SCRFD + ArcFace + align)
├── liveness.py                Anti-spoofing checks (optional)
├── blink.py                   Active blink challenge (optional)
├── quality.py                 Face size / landmark / direction helpers
├── weather.py                 IP-geolocation + open-meteo poller
├── database.py                FaceDB (employees, embeddings, interactions, projects, sessions)
├── admin.py                   Legacy admin CLI (still works)
├── admin_web.py               Legacy admin Flask app on :8081 (still works alongside the new /admin)
├── enroll.py                  Pre-enrol from photos via CLI
├── seed_demo.py               Populate demo projects + a session
├── audio_utils.py             Linear resampling + device pickers
├── config.py                  All tunable knobs
├── web/
│   ├── app.py                 Flask app factory
│   ├── templates/
│   │   ├── index.html         SPA shell (idle dashboard + active camera)
│   │   └── admin.html         Tabbed admin
│   └── static/
│       ├── echo.css           Glassmorphism + 1280x800 layout + scan line
│       └── echo.js            State subscriber + DOM mirror + overlay state machine
├── docs/
│   └── hailo_asr.md           Hailo Whisper setup + chip-contention notes
├── models/                    HEFs + Vosk + Piper voices + Whisper assets
├── start.sh                   Launcher: sources .env, starts backend, opens Chromium
├── stop.sh                    Graceful shutdown
├── requirements.txt
├── .env.example               Copy to .env and fill in
└── README.md                  This file
```

---

## Privacy and limitations

**Privacy.** Face embeddings (512-d float32 vectors) and names are
stored in a local SQLite file (`faces.db`). No raw photos are kept —
images live in RAM during capture, then only the embedding is
written. Interactions store the `emp_id` and timestamp.

When `CHAT_ASR_BACKEND = "openai"` or chat falls through to OpenAI,
the audio / chat text is sent to OpenAI's API. Anthropic isn't used
unless you wire it explicitly in `chat.py`. Switch to fully-local by
configuring Hailo Whisper + Hailo-Ollama and leaving `OPENAI_API_KEY`
unset.

**Liveness.** When `LIVENESS_ENABLED = True`, the kiosk runs the
texture + motion + jitter + specular pipeline plus an optional blink
challenge. Effective against printed photos and most phone-screen
replays; **not bulletproof**. A high-quality video on a large
monitor with natural ambient motion will bypass the passive checks.
The blink challenge raises the bar. For high-stakes deployments,
combine with depth (3D structured-light or stereo) or a dedicated
liveness model.

**Recognition drift.** ArcFace is a 512-d embedding cosine-matched
against a per-person gallery. Silent learning expands the gallery
opportunistically but is gated by score + runner-up margin. False
positives still happen — especially for genuinely similar people.
Clean drifted galleries with `sqlite3 faces.db` or the Employees
admin tab.

**Chat budget.** Hard-capped per ACTIVE session at
`CHAT_MAX_QUESTIONS_PER_SESSION`. A persistent visitor can't run up a
bill by camping in front of the kiosk.

---

## Contributing / development notes

- **Branches**: feature work happens on
  `claude/<topic>` branches. Merge to the default branch via PR.
- **Hooks**: there are none — `git commit` runs straight through.
- **Tests**: there's no automated test suite yet. The validation
  surface is the kiosk itself + watching the backend log.
- **Hot-reload**: Flask runs without the reloader (`use_reloader=False`)
  because the camera worker can't be re-spawned cleanly. For SPA
  iteration, Ctrl+F5 in Chromium hard-reloads the CSS / JS.
- **Style**: 4-space indent, type hints where useful, no docstrings
  on obvious functions.

See `docs/hailo_asr.md` for Hailo Whisper integration specifics.
