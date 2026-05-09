# Echo AI Kiosk — Pi 5 + Hailo-10H

A face-recognising kiosk for the Pi 5 (8 GB) using the **Pi AI Camera**
(IMX500) and a **Hailo-10H M.2** accelerator. The screen shows a blue
"Welcome to Echo AI" banner with live weather (top-right) and a lifetime
interaction counter (bottom-right) until it hears the wake phrase
**"hello echo"** — then it splits into camera (left) + tabbed terminal
(right) for projects / on-screen registration / OpenAI-or-Ollama chat.

Each distinct face recognised before the session goes idle counts as one
interaction (persisted in SQLite, surfaced on the welcome screen and the
admin web UI). TTS runs in a worker thread so playback never freezes the
camera. New visitors register themselves on the right panel — no host
stdin required.

The system idles until it hears the wake phrase, then runs a layered
liveness stack (specular highlights + texture variance + temporal
micro-motion + active blink challenge) before recognising or greeting
anyone — so a printed photo, a phone screen with a still image, or a
held-up monitor won't trigger a greeting.

This repo was developed and tested on a Pi 5 running Raspberry Pi OS
(Trixie / Python 3.13), HailoRT 5.3.0, and the H10 PCIe driver 5.x. Working
directory in the steps below is `/home/echo/Documents/code/pi5` — adjust to
your username/path.

## How it works

```
   IDLE  (welcome screen)                  ACTIVE  (camera + right panel)
  ┌────────────────────────────┐         ┌──────────────┬──────────────┐
  │                            │         │              │ [P]rojects   │
  │       Welcome to           │         │  camera with │ [R]egister   │
  │        Echo AI             │ ──────▶ │  detections  │ [C]hat       │
  │                            │  hello  │              │              │
  │     21°C  partly cloudy    │  echo   │              │ list / form  │
  │                            │         │              │ / chat lines │
  │                  42        │         │              │              │
  │            interactions    │         │              │              │
  └────────────────────────────┘         └──────────────┴──────────────┘
                ▲                                       │
                └────────── 30s no new event ───────────┘
```

The recognition pipeline that runs in ACTIVE state:

```
 Pi AI Camera (IMX500)
        │  picamera2  RGB888 (libcamera quirk: bytes are BGR-ordered)
        ▼
 BGR frame ─▶ Hailo-10H ─▶ SCRFD face detect (640x640)
                              │ bbox + 5 landmarks per face
                              ▼
                  Quality gate  (score, size, frame edge,
                                 eye distance, head roll)
                              │
                              ▼
                  Liveness window (24 frames): relative landmark
                  motion + face-region pixel jitter
                              │ live faces only
                              ▼
                  Align to 112x112 (ArcFace 5-pt similarity)
                              │
                              ▼
                  Hailo-10H ─▶ ArcFace embedding (512-D, L2)
                              │
                              ▼
              Cosine match vs SQLite (faces.db)
                  │                       │
              match                       below threshold
                  │                       │
                  ▼                       ▼
       greet (async TTS,         after N consecutive
       no camera lag),           good-quality unknowns
       counter +=1               → REGISTER tab opens,
                                 user types emp_id + name on screen
```

TTS playback runs on a worker thread (`async_tts.py`) so the camera
never freezes during a greeting. Greetings, pose prompts, and chat
replies all queue and play serially.

The DB has four tables; `emp_id` is the primary key for employees,
`session_id` (a UUID assigned at each wake-word transition) keys
counter rows.

```sql
employees(emp_id PRIMARY KEY, name, created_at)
face_embeddings(id PK, emp_id FK, embedding BLOB, created_at)
projects(id PK, title, description, ordering, created_at)
interactions(id PK, emp_id, session_id, ts, UNIQUE(emp_id, session_id))
```

Multiple embeddings per person are stored — one per pose — and `emp_id`
is deleted-cascade so removing an employee also drops their face data.
Counter rows are deduplicated at the DB level: each `(emp_id,
session_id)` can only insert once, so re-entries inside one ACTIVE
session don't double-count.

## Project layout

| File              | Purpose |
|-------------------|---------|
| `config.py`       | Paths, thresholds, prompts, timing knobs |
| `database.py`     | SQLite schema + CRUD |
| `hailo_infer.py`  | HailoRT 5.x InferModel pipeline (SCRFD decode + NMS, ArcFace embed, alignment) |
| `quality.py`      | Face quality gate + pose-change detection |
| `liveness.py`     | Passive liveness check (relative landmark motion + pixel jitter) |
| `blink.py`        | Active liveness: blink challenge gating per-session |
| `tts.py`          | TTS backend abstraction (Piper / pyttsx3) with WAV prebuffer |
| `async_tts.py`    | Worker-thread queue around the TTS backend so greetings never freeze the camera |
| `wake_word.py`    | Vosk-based "hello echo" listener (background thread, mic) |
| `weather.py`      | IP-geolocated Open-Meteo poller for the idle widget |
| `chat.py`         | OpenAI → Ollama fallback, per-session 5-question budget |
| `views.py`        | Render functions for the welcome screen and tabbed right panel |
| `main.py`         | Live loop, IDLE/ACTIVE state machine, multi-face greet, tab handlers, registration / chat dispatch |
| `enroll.py`       | Pre-enrol an employee from N camera frames (no live loop) |
| `admin.py`        | CLI: list / show / delete / export DB entries |
| `admin_web.py`    | Flask web UI for the same operations (`./start_admin.sh`) |
| `requirements.txt`| Python deps (HailoRT, picamera2, and the Vosk model are NOT pip-installed) |

---

## 1. Hardware checklist

- Raspberry Pi 5 (8 GB) on Raspberry Pi OS 64-bit (Bookworm or Trixie)
- Active cooler (the Pi 5 + Hailo will run hot under load)
- Pi AI Camera (IMX500) on the CSI ribbon
- Hailo-10H M.2 module seated in the Pi AI HAT+ / M.2 HAT, PCIe enabled
- Speaker / 3.5 mm jack / HDMI audio out for TTS greetings
- USB / I2S microphone for the wake-word listener
- Official 27 W USB-C PSU recommended

---

## 2. Step-by-step install / activation

### 2.1 OS update + enable PCIe Gen 3

```bash
sudo apt update && sudo apt full-upgrade -y
sudo rpi-eeprom-update -a
```

In `/boot/firmware/config.txt`, under `[all]` add:

```ini
dtparam=pciex1_gen=3
camera_auto_detect=1
```

(`dtparam=pciex1` may already be set by the AI HAT+ overlay — it isn't
required if `lspci | grep -i hailo` shows the device after reboot.)

```bash
sudo reboot
lspci | grep -i hailo     # must list "Hailo Technologies Ltd. Hailo-10H AI Processor"
```

### 2.2 Install the Hailo-10H PCIe driver

The PCIe driver + firmware blobs come from Raspberry Pi's apt feed. The
**userspace** runtime does NOT — see §2.4.

```bash
sudo apt install -y hailo-all
sudo reboot
```

After reboot, confirm the driver loaded and a `/dev/h1x-0` (or `/dev/hailo0`
on older driver versions) appeared:

```bash
lsmod | grep hailo                # hailo1x_pci listed
ls /dev/h1x-0 /dev/hailo0 2>/dev/null
sudo dmesg | grep -i hailo | tail # "SOC Firmware Batch loaded successfully"
```

### 2.3 Install Pi AI Camera support

`picamera2` and `libcamera` are best installed via apt — the pip wheels
need `libcap-dev` headers and several other native libs to compile.

```bash
sudo apt install -y python3-picamera2 python3-libcamera imx500-all
rpicam-hello -t 5000               # quick preview to confirm the camera is alive
```

### 2.4 Install HailoRT 5.x from the Hailo Developer Zone

The Raspberry Pi apt feed currently ships HailoRT **4.23**, which predates
Hailo-10H support — you'll see `Hailo1X Devices are only supported in
versions 5.0.0 and above` if you try to use it. Get HailoRT 5.x as a `.deb`
from Hailo:

1. Sign in (free account) at https://hailo.ai/developer-zone/software-downloads/.
2. Filter platform **aarch64** / Raspberry Pi 5; pick the latest **HailoRT 5.x**.
3. Download:
   - `hailort_5.x.y_arm64.deb`
   - `hailort-5.x.y-cp313-cp313-linux_aarch64.whl` *(matches Trixie's
     Python 3.13; if you're on Bookworm/3.11 grab the `cp311` wheel instead)*

   Do **not** install Hailo's `hailort-pcie-driver` `.deb` — apt's
   `hailo-all` already installed the matching driver and double-installing
   conflicts.

4. Install (copy out of `~/Downloads` first to avoid the unrelated `_apt`
   permission warning):

   ```bash
   sudo cp ~/Downloads/hailort_5.*_arm64.deb /tmp/
   sudo apt install /tmp/hailort_5.*_arm64.deb

   hailortcli --version            # must show 5.x
   hailortcli scan                  # must list pci/0001:01:00.0
   hailortcli fw-control identify   # Architecture: HAILO10H, FW 5.x
   ```

If `apt update` complains about a stale `hailo.list` source, remove it:

```bash
sudo rm -f /etc/apt/sources.list.d/hailo.list /etc/apt/keyrings/hailo.gpg
```

### 2.5 Clone the project and create the venv

```bash
mkdir -p /home/echo/Documents/code
cd /home/echo/Documents/code
git clone <your-fork-url> pi5
cd /home/echo/Documents/code/pi5

# --system-site-packages so apt-installed picamera2 + libcamera are visible
python3 -m venv --system-site-packages .venv
source .venv/bin/activate

pip install --upgrade pip
pip install ~/Downloads/hailort-5.*-cp313-cp313-linux_aarch64.whl
pip install -r requirements.txt

python -c "import hailo_platform, picamera2, cv2, numpy, pyttsx3; \
           print('ok', hailo_platform.__version__)"
```

### 2.6 Audio out for TTS

```bash
sudo apt install -y espeak-ng alsa-utils
aplay -l                                          # find output devices
aplay /usr/share/sounds/alsa/Front_Center.wav    # default-device test
```

You have two ways to send TTS to a specific output (e.g. an Anker A3301
USB speakerphone instead of HDMI):

**Option A — make the device the system default (everything follows).**

PipeWire (Trixie default):

```bash
sudo apt install -y wireplumber
wpctl status                                      # find the sink ID for the Anker
wpctl set-default <SINK_ID>
```

PulseAudio compat layer:

```bash
sudo apt install -y pulseaudio-utils
pactl list short sinks
pactl set-default-sink <sink-name>
```

`raspi-config` → System → Audio also works for HDMI / headphone jack.

**Option B — pin only this app's TTS, leave system audio alone.**

Find an ALSA name for the Anker (`grep` doesn't match — the card label
is `PowerConf`, not `Anker`):

```bash
aplay -L | grep -B1 -iE 'powerconf|a3301|usb audio'
# example match:
#   plughw:CARD=PowerConf,DEV=0
#       PowerConf, USB Audio
```

Then in `config.py`:

```python
AUDIO_OUTPUT_DEVICE = "plughw:CARD=PowerConf,DEV=0"   # or "plughw:2,0"
```

The Greeter synthesises TTS to a temp WAV and plays it via `aplay -D
<device>`, so the route is independent of whatever the system default is.
Quick verification:

```bash
aplay -D plughw:CARD=A3301,DEV=0 /usr/share/sounds/alsa/Front_Center.wav
```

### 2.7 Get the Hailo-10H HEF models

Two HEFs go into `models/`:

- `models/scrfd_10g.hef`               — face detector
- `models/arcface_mobilefacenet.hef`   — face embedder

The Hailo Model Zoo CLI (`hailomz`) is **not** on PyPI and the standalone
`pip install -e .` of the GitHub repo fails on Pi (it expects a private
monorepo layout). Just download the HEFs manually:

1. From the Model Zoo releases: https://github.com/hailo-ai/hailo_model_zoo/releases
2. Or from the Developer Zone Model Zoo page (filter by Hailo-10H).
3. Or from the Hailo Application Code Examples repo:
   https://github.com/hailo-ai/Hailo-Application-Code-Examples

Drop the two files into `models/`:

```bash
mkdir -p /home/echo/Documents/code/pi5/models
mv ~/Downloads/scrfd_10g.hef /home/echo/Documents/code/pi5/models/
mv ~/Downloads/arcface_mobilefacenet.hef /home/echo/Documents/code/pi5/models/
```

**Verify the architecture before first run** — a HEF compiled for
`HAILO15H` will fail to load on the H10:

```bash
hailortcli parse-hef models/scrfd_10g.hef            | head -1
hailortcli parse-hef models/arcface_mobilefacenet.hef | head -1
# Both must say: HEF Compatible for: HAILO10H   (or "...HAILO15H, HAILO10H")
```

Smoke-test inference. Note H10 uses `run2`, not `run`:

```bash
hailortcli run2 -t 5 set-net models/scrfd_10g.hef
# expect something like: scrfd_10g: fps: 240+
```

### 2.8 Wake-word ("hello echo") setup

The wake word uses [Vosk](https://alphacephei.com/vosk/) — small, offline,
ARM-friendly. You need a microphone reachable as an ALSA input device, the
Vosk Python package + `sounddevice` (already in `requirements.txt`), and a
small acoustic model.

System dependencies:

```bash
sudo apt install -y libportaudio2 portaudio19-dev
arecord -l                                # confirm a capture device is listed
arecord -d 3 -f cd /tmp/test.wav && aplay /tmp/test.wav   # mic loopback test
```

Download the English small model (~40 MB) into `models/`:

```bash
cd /home/echo/Documents/code/pi5/models
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip
rm vosk-model-small-en-us-0.15.zip
ls vosk-model-small-en-us-0.15/           # should contain conf/, am/, graph/, ...
```

The wake phrase is in `config.py` (`WAKE_WORD = "hello echo"`). If you'd
rather pick a different short phrase, change it there and restart. Vosk
runs the recogniser with a tight grammar that only knows the keyword and
an "[unk]" sink, which keeps CPU usage minimal and reduces false matches.

If your mic isn't auto-selected, list devices and force one:

```bash
python -c "import sounddevice as sd; print(sd.query_devices())"
# pick the mic's index, then in your shell:
export SD_DEVICE=<index>     # picked up automatically by sounddevice
```

### 2.9 Better-sounding TTS with Piper (recommended)

The default voice uses `pyttsx3` + `espeak-ng` — fast but robotic.
**Piper** (via the [`piper1-gpl`](https://github.com/OHF-Voice/piper1-gpl)
rewrite) is a neural TTS engine that runs on-device and produces a much
more natural voice. It loads the ONNX voice once at startup so each
utterance is fast (~200–400 ms on Pi 5). Skip this section to keep the
espeak voice — the app falls back to `pyttsx3` automatically if Piper
isn't available.

#### 1. Install the Python package

```bash
cd /home/echo/Documents/code/pi5
source .venv/bin/activate
pip install piper-tts
```

Wheels are published for cp38–cp312. **On Python 3.13 (Trixie default)
the install will fail** with `No matching distribution found` — Piper
hasn't shipped cp313 wheels yet. Either pip will skip it (the
`requirements.txt` marks it conditional) or you can build a side venv
on Python 3.12 just for TTS. The app falls back to `pyttsx3` cleanly
either way.

#### 2. Download voice models

Use the bundled installer to grab a curated set of clear English voices
(~500 MB, 8 voices), or `--all` for every English voice listed in the
script (~2 GB).

```bash
./install_piper_voices.sh             # 8 curated voices (recommended)
./install_piper_voices.sh --all       # every English voice listed
./install_piper_voices.sh --list      # show what each option installs
```

The default voice (`config.PIPER_MODEL_PATH`) is
`en_US-hfc_female-medium`. Switch by dropping a different `.onnx` +
`.onnx.json` pair into `models/piper/` and updating `PIPER_MODEL_PATH`
in `config.py`.

| Voice                              | Style |
|------------------------------------|-------|
| `en_US-hfc_female-medium`          | US female, very clear  ← default |
| `en_US-amy-medium`                 | US female, slightly warmer |
| `en_US-lessac-medium`              | US male, news-anchor |
| `en_US-ryan-medium`                | US male, conversational |
| `en_US-libritts-high`              | Highest quality, slower |
| `en_GB-alan-medium`                | British male |
| `en_GB-jenny_dioco-medium`         | British female |

#### 3. Smoke test

```bash
python -c "
from piper import PiperVoice
import wave
v = PiperVoice.load('models/piper/en_US-hfc_female-medium.onnx')
with wave.open('/tmp/test.wav', 'wb') as f:
    v.synthesize_wav('Hello, this is the default voice.', f)
"
aplay -D plughw:CARD=PowerConf,DEV=0 /tmp/test.wav
```

**Smoke test** through the Anker (if you've pinned `AUDIO_OUTPUT_DEVICE`):

```bash
echo "Hello, this is Amy speaking." | tools/piper/piper \
    --model models/piper/en_US-amy-medium.onnx \
    --output_file /tmp/test.wav
aplay -D plughw:CARD=PowerConf,DEV=0 /tmp/test.wav
```

`./start.sh` will print one of these on launch:

- `[TTS] using Piper (Python): en_US-hfc_female-medium.onnx`  ← installed
- `[TTS] using pyttsx3 / espeak-ng`                            ← Piper unavailable

To force the espeak voice anyway, set `TTS_BACKEND = "pyttsx3"` in
`config.py`.

---

## 3. Using it

### 3.0 Welcome screen, tabs, and counter

**IDLE** is a full-screen blue panel with "Welcome to Echo AI" centred,
weather top-right, and the persistent interaction counter bottom-right.
The camera keeps running in the background — but it's **not shown** in
this state. Detection / silent learning don't run in IDLE either, so
CPU is mostly the wake-word listener.

Saying **"hello echo"** transitions to **ACTIVE**: the cv2 window
splits into camera (left) and a tabbed right panel (right). Three tabs:

| Tab        | Key | Purpose                                                            |
|------------|-----|--------------------------------------------------------------------|
| `PROJECTS` | `P` | Read-only list pulled from `projects` table; refreshes every 5 s.  |
| `REGISTER` | `R` | In-window text fields. Tab to switch field, Enter to submit, then voice-guided pose capture runs. Auto-opens after a quality unknown face streak. |
| `CHAT`     | `C` | Up to 5 questions per session via OpenAI / Ollama (see §3.6). `V` toggles voice / keyboard input. |

The header strip across each tab also shows
`this session: N` — the number of distinct people greeted since the
last wake-up.

After `IDLE_AFTER_LAST_INTERACTION_SEC` (default **30 s**) without a
new event the kiosk drops back to IDLE. New events are: a different
emp_id greeted, an unknown high-quality face in frame, a chat
exchange, or a registration.

### 3.1 Live recogniser with auto-registration

The fastest way to launch with mic + speaker pinned to the Anker is the
provided wrapper:

```bash
cd /home/echo/Documents/code/pi5
./start.sh
```

`start.sh` exports `SD_DEVICE` (mic input index for sounddevice) and
`AUDIO_OUTPUT_DEVICE` (ALSA name for the speaker), activates the venv,
and runs `python main.py --auto-register`. Edit the defaults at the top
of the script for your machine, or override per invocation:

```bash
SD_DEVICE=3 ./start.sh
EXTRA_ARGS="--no-wake-word --auto-register" ./start.sh
./start.sh --no-display      # extra args pass through to main.py
```

If you'd rather drive `python main.py` directly:

```bash
source .venv/bin/activate
SD_DEVICE=1 AUDIO_OUTPUT_DEVICE=plughw:CARD=PowerConf,DEV=0 \
    python main.py --auto-register
```

What happens:

- The system starts in **IDLE** state. Faces are still detected and drawn,
  but no embeddings are computed and no greetings are spoken. A banner
  across the top reads `Say 'hello echo' to start recognition`.
- When the wake word is heard, the system transitions to **ACTIVE**, says
  "Hello. I am ready.", and starts the full pipeline.
- Faces are tracked in real time. Bounding boxes are drawn green for
  recognised people, with `name (cosine_score)`.
- Detections that fail the **quality gate** (a hand near the face, a side
  profile, a face touching the edge of the frame, a tilted head, a face
  too far from the camera) are labelled `low quality: <reason>` and
  ignored. They will NOT trigger recognition or registration.
- The largest face is also fed to the **temporal liveness check** — it
  has to show facial micro-motion AND face-region pixel jitter over a
  sliding window before the system will recognise it. Other faces in
  the scene get the cheap **single-frame screen-attack gates** (specular
  highlights + texture variance). Failing faces show
  `checking liveness... (static (photo?))` or `liveness: glare/screen`.
- Once a recognised face passes both gates, the system runs a one-shot
  **active blink challenge** for that emp_id (configurable via
  `LIVENESS_REQUIRE_BLINK`). Speak prompt: "Please blink once to
  confirm." A clear range of eye-region pixel std across a 5-second
  window passes the challenge. Confirmation is held for the rest of the
  ACTIVE session and cleared on IDLE. This defeats the remaining attack
  vector — a high-quality video replay.
- **All recognised faces are greeted**, not just the largest one. Each
  emp_id is rate-limited to one greeting per `GREET_COOLDOWN_SEC` so
  someone walking back and forth doesn't trigger repeats.
- The window is split: **camera on the left, conversation panel on the
  right**. The panel reads like a chat — three roles only:
  - **You** — the wake word when matched.
  - **Echo** — what the system says (greetings, prompts).
  - **·** — short status events (`Recognised Echo`, `New face — getting
    ready to register`, `Going to sleep`).

  Vosk partials, internal state transitions, streak counters, etc. go
  to stdout for debugging — the on-screen panel stays readable. Disable
  with `SHOW_TRANSCRIPT_PANEL = False` if you want the camera-only
  window.
- After `IDLE_AFTER_LAST_INTERACTION_SEC` (10 s) **with no new event**
  the system drops back to IDLE. "New event" means a different person
  greeted, or an unknown face standing in front of the camera. A
  recognised person who keeps standing there does NOT keep the system
  awake — the timer counts down anyway. Saying "hello echo" again wakes
  it back up.
- On every confident match (score ≥ `SILENT_LEARN_MIN_SCORE`) the new
  embedding is silently appended to that person's gallery, so the
  recogniser gets more robust over time. Rate-limited to one new sample
  per person per `SILENT_LEARN_MIN_INTERVAL_SEC` (60 s default), and
  capped at `SILENT_LEARN_MAX_SAMPLES_PER_PERSON` (30 default — oldest
  drop first when over). Set `SILENT_LEARN_ENABLED = False` to turn off.
- An unknown but high-quality, **live** face must persist for
  `UNKNOWN_FRAMES_BEFORE_REGISTER` consecutive frames (~half a second)
  before registration is offered. The counter is shown on screen.

Skip the wake word during development:

```bash
python main.py --auto-register --no-wake-word
```
- Registration is voice-guided through 5 poses. For each prompt the
  capture only happens when:
  1. enough time has elapsed for the user to actually move (`POSE_HOLD_SEC`),
  2. the face has shifted in the asked direction by at least
     `POSE_MIN_SHIFT` × inter-eye distance (no shift required for the first
     "look straight" calibration pose), and
  3. the face has been still for `POSE_STABLE_SEC` (so we don't grab a
     motion-blurred frame).
- Greeter rule: the same person greeted twice in a row stays silent. A
  *different* person resets the gate.
- Press **`r`** to force a registration of the current largest face.
- Press **`q`** to quit.

Headless (no preview window):

```bash
python main.py --auto-register --no-display
```

### 3.2 Pre-enrol from the CLI (no live loop)

```bash
python enroll.py --emp-id E001 --name "Alice Kumar" --frames 5
```

### 3.3 Manage the database

The DB lives at `/home/echo/Documents/code/pi5/faces.db` (SQLite).
`emp_id` is the primary key.

#### CLI

```bash
python admin.py list                       # all employees + sample counts
python admin.py show E001                  # one employee + per-embedding info
python admin.py delete E001                # asks for confirmation
python admin.py delete E001 --yes          # skip confirmation
python admin.py export employees.csv       # CSV (no embedding bytes)
python admin.py path                       # absolute path to faces.db
```

#### Web UI

A minimal Flask app on port 8081 (configurable; 8080 is left for
Open WebUI / other services). Lets you list, rename, and delete
employees from a browser. SQLite handles concurrent access, so it can
run alongside `./start.sh` on the same Pi.

```bash
./start_admin.sh                           # http://0.0.0.0:8081
ADMIN_HOST=127.0.0.1 ADMIN_PORT=9000 ./start_admin.sh
```

JSON API for scripting:

```
GET  /api/employees                  -> list
GET  /api/employees/<emp_id>         -> details
POST /api/employees/<emp_id>/rename  -> body: {"name": "..."}
POST /api/employees/<emp_id>/delete  -> remove + cascade embeddings
```

#### Direct SQLite

```bash
sudo apt install -y sqlite3 sqlitebrowser
sqlite3 /home/echo/Documents/code/pi5/faces.db
sqlite> .tables
sqlite> SELECT emp_id, name FROM employees;
```

`sqlitebrowser` provides a GUI if you'd rather click around.

### 3.6 Chat (OpenAI → Ollama fallback)

Chat is opened from the right panel by pressing **`C`**. The header
shows which backend is live:

- `OpenAI · gpt-4o-mini` when `OPENAI_API_KEY` is set and
  `api.openai.com` is reachable.
- `Ollama (local) · qwen3:1.7b` when local Hailo-Ollama is running on
  `localhost:11434` and the model is pulled
  (see https://www.raspberrypi.com/documentation/computers/ai.html).
  `start.sh` will probe the API on launch and run `hailo-ollama serve`
  in the background if it's not already up.
- `Chat unavailable — no backend reachable` when neither works.

Each emp_id gets `CHAT_MAX_QUESTIONS_PER_SESSION` questions (default
**5**) per ACTIVE session; the budget resets on the next wake. The
**6th** question is rejected with `Question budget reached for this
session.`

Setup OpenAI:

```bash
echo 'export OPENAI_API_KEY=sk-...' >> ~/.bashrc
exec bash
./start.sh
```

Setup Hailo-Ollama (matches the Pi 5 docs above; runs LLMs on the
Hailo-10H instead of the CPU):

```bash
# Install per https://www.raspberrypi.com/documentation/computers/ai.html
hailo-ollama pull qwen3:1.7b      # recommended for kiosk chat
# (other models that fit Pi 5: qwen2.5:1.5b, llama3.2:1b. Avoid
#  deepseek_r1:* -- it emits visible <think> blocks. Avoid
#  qwen2.5-coder:* -- code-tuned, weak at chat.)
```

`start.sh` checks `localhost:11434/api/tags` on launch; if the daemon
isn't up it runs `hailo-ollama serve` in the background and waits up
to 10 s for the API to come up (log at `/tmp/hailo-ollama.log`). If
your build uses a different invocation, set `HAILO_OLLAMA_CMD` before
calling `start.sh`. Failure is non-fatal — chat will simply route to
OpenAI (or display "Chat unavailable" if neither is configured).

Switch backend preference order in `config.py`:

```python
CHAT_BACKEND_ORDER = ("ollama", "openai")   # local-first
```

Toggle voice / keyboard input with **`V`** while the chat tab is open;
voice mode listens via the same Vosk model used for the wake word
(no grammar) and submits when you stop speaking. Replies are spoken
through the configured TTS backend and shown in the panel.

### 3.7 Projects board

Projects live in the `projects` table (`id, title, description,
ordering, created_at`). The `PROJECTS` tab in ACTIVE state renders a
scrolling list, refreshed every 5 seconds. Edit them through the admin
UI (§3.3 *Web UI*) — the kiosk picks up changes without a restart.

### 3.8 Counter / metrics

Every distinct face recognised in an ACTIVE session inserts one row
into the `interactions` table (`UNIQUE(emp_id, session_id)` — so the
same person re-recognised in the same session doesn't double-count).
Total is shown bottom-right of the welcome screen and at
`http://<pi>:8081/api/metrics`. To reset:

```bash
sqlite3 /home/echo/Documents/code/pi5/faces.db "DELETE FROM interactions;"
```

### 3.4 Re-registration

Running registration with an `emp_id` that already exists triggers a
match check:

- The captured embeddings are compared to the existing ones for that
  `emp_id`.
- If the best cosine score ≥ `REREGISTER_MATCH_THRESHOLD` (0.35 by
  default) → embeddings are appended (more samples = better recognition).
- Otherwise → registration is **refused** with a spoken warning. This is
  what stops someone else "claiming" your `emp_id`.

### 3.5 Tuning knobs (in `config.py`)

Recognition / registration trigger:

| Setting | Effect |
|---------|--------|
| `COSINE_MATCH_THRESHOLD`         | Lower = looser match (more false accepts). Default 0.38. |
| `REREGISTER_MATCH_THRESHOLD`     | How strictly the re-register match must agree. Default 0.35. |
| `UNKNOWN_FRAMES_BEFORE_REGISTER` | Anti-flicker streak length before offering enrolment. |

Quality gate:

| Setting | Effect |
|---------|--------|
| `QUALITY_SCORE_THRESHOLD`     | Min detector confidence (0.70) |
| `QUALITY_MIN_FACE_PIXELS`     | Min bbox W and H (110) — rejects far-away faces |
| `QUALITY_FRAME_EDGE_MARGIN`   | Reject faces near the frame border |
| `QUALITY_MIN_EYE_DISTANCE`    | Reject too-small / occluded faces (28 px) |
| `QUALITY_MAX_EYE_TILT`        | Reject extreme head roll (0.45) |

Wake word + state machine:

| Setting | Effect |
|---------|--------|
| `WAKE_WORD`                          | Phrase that activates recognition. Default `"hello echo"`. |
| `IDLE_AFTER_LAST_INTERACTION_SEC`    | Drop back to IDLE after this many seconds with no new event (10) |
| `ACTIVE_SESSION_MAX_SEC`             | Hard cap on an ACTIVE session (10 min) |

Silent learning:

| Setting | Effect |
|---------|--------|
| `SILENT_LEARN_ENABLED`                | Master switch (True) |
| `SILENT_LEARN_MIN_SCORE`              | Only learn when match score ≥ this (0.55) |
| `SILENT_LEARN_MAX_SIMILARITY`         | Skip if new sample is ~ a duplicate of one already stored (0.92) |
| `SILENT_LEARN_MIN_INTERVAL_SEC`       | At most one new sample per person per this many seconds (60) |
| `SILENT_LEARN_MAX_SAMPLES_PER_PERSON` | Cap; oldest drop first when over (30) |

Liveness (passive):

| Setting | Effect |
|---------|--------|
| `LIVENESS_WINDOW_FRAMES`        | Sliding window length (24) |
| `LIVENESS_MAX_SPECULAR_RATIO`   | Reject if more than this fraction of the face crop is near-saturated white. Defeats screens with glare. (0.10) |
| `LIVENESS_MIN_TEXTURE_VAR`      | Reject if face crop is too smooth (Laplacian variance below this). Defeats flat phone/monitor displays. (60) |
| `LIVENESS_REL_MOTION_MIN`       | Min facial micro-motion in window. Lower = more permissive. |
| `LIVENESS_PIXEL_JITTER_MIN`     | Min face-region pixel jitter beyond camera read noise. |

Liveness (active blink):

| Setting | Effect |
|---------|--------|
| `LIVENESS_REQUIRE_BLINK`     | Master switch (True). Set False to skip the active challenge. |
| `LIVENESS_BLINK_PROMPT`      | Spoken prompt ("Please blink once to confirm.") |
| `LIVENESS_BLINK_TIMEOUT_SEC` | Max wait for a blink (5 s) |
| `LIVENESS_BLINK_PATCH_PX`    | Half-extent of eye-region patch sampled from each eye landmark (14) |
| `LIVENESS_BLINK_DELTA_MIN`   | Min std-dev range across the window for "blink seen" (6.0). Raise if too lax, lower if real blinks miss. |

TTS:

| Setting | Effect |
|---------|--------|
| `TTS_BACKEND`        | `"piper"` (preferred) or `"pyttsx3"` |
| `PIPER_MODEL_PATH`   | Path to the `.onnx` voice model |
| `TTS_PREBUFFER_MS`   | Silence padded at the start of each utterance so USB speakerphones don't clip the first word (500). Set to 0 to disable. |
| `AUDIO_OUTPUT_DEVICE` | ALSA name for the speaker (`"plughw:CARD=PowerConf,DEV=0"`). `None` = system default. |

HUD transcript panel:

| Setting | Effect |
|---------|--------|
| `SHOW_TRANSCRIPT_PANEL`   | Master switch. False keeps the camera-only window. |
| `TRANSCRIPT_PANEL_WIDTH`  | Pixels added to the right of the camera (420). |
| `TRANSCRIPT_MAX_EVENTS`   | Ring-buffer size; oldest events scroll out (16). |
| `DISPLAY_SCALE`           | Scales the whole composed window at display time only. Detection runs on full-res frames -- this just shrinks the cv2 window so it fits a small screen. (1.0) |
| `CAMERA_RESOLUTION`       | Source resolution from picamera2 (1280, 720). |

#### Fitting a small screen (e.g. a 10.1" Pi-mounted display)

The camera+panel composite is roughly `CAMERA_RESOLUTION[0] +
TRANSCRIPT_PANEL_WIDTH` wide × `CAMERA_RESOLUTION[1]` tall — by default
about **1700 × 720**, which overflows most 10.1" panels.

Two ways to make it fit. Pick whichever matches your priorities.

**Option A — display-only scale (recommended).** Detection runs on the
full-res frame so accuracy is unchanged; the cv2 window is shrunk just
before `imshow`. One knob:

```python
# config.py
DISPLAY_SCALE = 0.75    # for 1280 x 800 panels  -> ~1275 x 540
DISPLAY_SCALE = 0.60    # for 1024 x 600 panels  -> ~1020 x 432
```

**Option B — shrink at the source.** Lower-res camera capture (faster
and lighter) plus a narrower panel. Quality thresholds stay valid as
long as faces still occupy the same fraction of the frame — but if
you're standing far back, drop `QUALITY_MIN_FACE_PIXELS` proportionally.

```python
# config.py — for a 1280 x 800 panel
CAMERA_RESOLUTION = (800, 480)
TRANSCRIPT_PANEL_WIDTH = 320
# total: 1120 x 480, fits with headroom

# config.py — for a 1024 x 600 panel
CAMERA_RESOLUTION = (640, 480)
TRANSCRIPT_PANEL_WIDTH = 280
QUALITY_MIN_FACE_PIXELS = 70    # was 110; you're at half-res
# total: 920 x 480, fits 1024 x 600
```

You can also combine the two: keep the camera at native res for the
detector, then `DISPLAY_SCALE = 0.6` so the on-screen window fits.

The HUD prints every signal alongside its threshold while liveness is
failing, e.g. `motion=0.32/0.45  jitter=3.2/4.0  glare=18%/10%
tex=42/60` — the value before the slash is the live measurement, after
it is the threshold. Anything where measurement < threshold is the
reason the face is being rejected, so you tune that knob.

Voice-guided pose capture:

| Setting | Effect |
|---------|--------|
| `POSE_PROMPTS`            | List of (spoken text, expected direction). Edit freely. |
| `POSE_HOLD_SEC`           | Min time after the prompt before capture is even attempted (1.5) |
| `POSE_STABLE_SEC`         | Face must be still this long before capturing (0.5) |
| `POSE_STABLE_PIXEL_TOL`   | Max landmark drift inside the stability window (4 px) |
| `POSE_MIN_SHIFT`          | Min directional shift in eye-distance units (0.25) |
| `POSE_CAPTURE_TIMEOUT_SEC`| Skip the pose if not satisfied in this many seconds (8) |

---

## 4. Run on boot (optional)

`/etc/systemd/system/face-recog.service`:

```ini
[Unit]
Description=Pi5 Face Recognition
After=multi-user.target sound.target

[Service]
Type=simple
User=echo
WorkingDirectory=/home/echo/Documents/code/pi5
ExecStart=/home/echo/Documents/code/pi5/.venv/bin/python /home/echo/Documents/code/pi5/main.py --auto-register --no-display
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now face-recog.service
journalctl -u face-recog.service -f
```

Note: registration prompts read from stdin, which won't work under systemd.
For the headless service, pre-enrol via `enroll.py` instead and let the
service only do recognition + greeting.

---

## 5. Troubleshooting

| Symptom | Likely cause / fix |
|---------|--------------------|
| `lspci` shows no Hailo | M.2 not seated, or PCIe Gen 3 not enabled in `config.txt`. |
| `hailortcli scan` says "Hailo devices not found" | userspace HailoRT version skew (apt 4.23 vs H10 driver 5.x). Install HailoRT 5.x from the Developer Zone (§2.4). |
| `Hailo1X Devices are only supported in versions 5.0.0 and above` | Same as above — upgrade HailoRT userspace. |
| `HAILO_NOT_IMPLEMENTED` from `vdevice.configure(...)` | You're on the legacy 4.x Python API; this repo's `hailo_infer.py` already uses the 5.x `InferModel` API. Ensure you installed the 5.x cp313 wheel. |
| `HEF Compatible for: HAILO15H` only | Wrong HEF — re-download the `HAILO10H` build (§2.7). |
| `pip install picamera2` fails on `python-prctl` / `libcap` | Don't pip-install picamera2 — `sudo apt install python3-picamera2` and use `--system-site-packages` venv (§2.5). |
| Preview is purple / discoloured | Older code did a redundant RGB↔BGR swap. Pull latest `main.py`. |
| Always says "unknown" | Threshold too tight, or too few enrolment samples. Lower `COSINE_MATCH_THRESHOLD` or re-enrol with more poses. |
| Registration triggers when I bring my hand near my face | The quality gate should be filtering this; if not, raise `QUALITY_SCORE_THRESHOLD` or `QUALITY_MIN_EYE_DISTANCE`. |
| Greeting doesn't speak | No audio sink. `aplay -l` to check, then `raspi-config` → System → Audio. |
| Camera freezes for ~1 s during a greeting | Resolved in current code: TTS now runs on a worker thread (`async_tts.py`). If you're still seeing freezes, you may have an old checkout — `git pull`. |
| Idle screen says `weather: offline` | No internet at startup so the IP-geolocation lookup failed. Either bring the Pi online and restart, or set `WEATHER_LATITUDE` / `WEATHER_LONGITUDE` in `config.py` to skip the IP lookup. |
| Chat tab says `Chat unavailable — no backend reachable` | `OPENAI_API_KEY` is unset (or no internet) AND local Ollama isn't running. Either `export OPENAI_API_KEY=...` and restart, or `sudo systemctl start ollama && ollama pull llama3.2:1b`. |
| `QFontDatabase: Cannot find font directory ... cv2/qt/fonts` | Harmless — opencv-python's bundled Qt has no fonts. `main.py` already sets `QT_LOGGING_RULES` to silence it. To fix properly: `sudo apt install -y fonts-dejavu-core && cp /usr/share/fonts/truetype/dejavu/*.ttf .venv/lib/python3.13/site-packages/cv2/qt/fonts/`. |
| `qt.qpa.xcb: could not connect to display` / `Aborted` | You're running as root (or otherwise have no `DISPLAY`). Best fix: run as your normal user — `exit` the root shell and `./start.sh` again. If you must run as root, `start.sh` now auto-falls-back to `--no-display`; or set `export DISPLAY=:0; export XAUTHORITY=/home/echo/.Xauthority` first. To get full graphical preview as your user, also make sure you're in `video,audio,render` groups: `sudo usermod -aG video,audio,render echo`, then log out and back in. |
| Wake word never triggers | `arecord -l` to confirm a mic exists; `python -c "import sounddevice as sd; print(sd.query_devices())"` to see what `sounddevice` sees. Set the mic as the default ALSA capture device or export `SD_DEVICE=<index>`. |
| `[wake-word] disabled: ...` | Either `vosk` / `sounddevice` failed to import (re-run `pip install -r requirements.txt`) or the model dir is missing (re-run §2.8). The app falls back to ACTIVE mode automatically so you can still use it. |
| Liveness flags real people as "static" | Lighting too flat or face too far. Lower `LIVENESS_PIXEL_JITTER_MIN` and/or `LIVENESS_REL_MOTION_MIN`. Watch the HUD signals to see which one is actually failing. |
| Liveness flags real people as "glare/screen" | Glasses or strong forehead sheen. Raise `LIVENESS_MAX_SPECULAR_RATIO` (e.g. 0.15). |
| Liveness flags real people as "too smooth" | Camera out of focus or face too small. Lower `LIVENESS_MIN_TEXTURE_VAR` (e.g. 30). |
| A photo on a phone/monitor *passes* liveness | Tighten the screen-attack gates: lower `LIVENESS_MAX_SPECULAR_RATIO` (e.g. 0.06) and raise `LIVENESS_MIN_TEXTURE_VAR` (e.g. 100). Read the live signals off the HUD to find the right values for your camera + lighting. |

---

## 6. Liveness limitations

The full liveness stack now combines:

1. **Single-frame screen-attack gates** — specular highlight ratio +
   texture variance (Laplacian). Defeats printed photos and held-up
   phone/monitor screens with glare.
2. **Temporal micro-motion** — over a sliding window the largest face
   must show non-rigid landmark jitter and face-region pixel changes
   beyond camera noise.
3. **Active blink challenge** — when `LIVENESS_REQUIRE_BLINK` is on,
   each emp_id is asked to blink before its first greeting in an ACTIVE
   session.

This combination defeats:

- printed photos (still or waved),
- phone-screen still images (with or without glare),
- and even high-quality video replays where the played-back person
  doesn't blink in time, since the active challenge requires a blink
  on demand.

It does **not** defeat:

- A video replay where the played subject blinks within the timeout
  window (rare and requires the attacker to anticipate the prompt).
- A high-quality 3D mask.

For higher security, plug in a dedicated anti-spoofing model
(Silent-Face-Anti-Spoofing or similar) on the Hailo as a fourth signal.

---

## 7. Privacy

Face embeddings are biometric data. `faces.db` is gitignored. Store it on
an encrypted volume if this leaves a controlled environment, and only
enrol people who have given informed consent.
