# Project layout

```
pi5/
├── README.md                  Top-level entry; links into docs/
├── start.sh                   Launcher: sources .env, starts backend, opens Chromium
├── stop.sh                    Graceful shutdown
├── requirements.txt           Python dependencies
├── .env.example               Copy to .env and fill in secrets
├── .gitignore                 Excludes .env, faces.db, models/*.hef, etc.
│
├── main.py                    Camera worker + orchestrator
├── state.py                   StateBus + KioskState dataclass
├── messages.py                Greeting / error / fact copy (from ai_lab.txt)
├── fun_facts.py               Background rotator for AI Fun Fact / AI Tip card
├── frame_streamer.py          MJPEG ring buffer
│
├── chat.py                    LLM client + per-session budget tracking
├── chat_voice.py              ChatVoiceCapture (records mic, calls ASR)
├── asr.py                     ChatASR backends (Hailo / OpenAI / faster-whisper / Vosk)
├── async_tts.py               Threaded TTS queue with on_start callbacks
├── tts.py                     Piper / pyttsx3 backends
│
├── wake_word.py               Vosk wake-word listener (paused during ACTIVE)
├── hailo_infer.py             HailoFacePipeline (SCRFD + ArcFace + face alignment)
├── liveness.py                Anti-spoofing checks (optional)
├── blink.py                   Active blink challenge (optional)
├── quality.py                 Face size / landmark / direction helpers
├── weather.py                 IP-geolocation + open-meteo poller
│
├── database.py                FaceDB (employees, embeddings, interactions,
│                                       projects, sessions)
├── admin.py                   Legacy admin CLI (still works)
├── admin_web.py               Legacy admin Flask app on :8081 (still works alongside /admin)
├── enroll.py                  Pre-enrol from photos via CLI
├── seed_demo.py               Populate demo projects + a session
├── audio_utils.py             Linear resampling + device pickers
├── config.py                  All tunable knobs
│
├── web/                       Flask app + SPA assets
│   ├── __init__.py
│   ├── app.py                 Flask app factory + all routes
│   ├── templates/
│   │   ├── index.html         SPA shell (idle dashboard + active camera)
│   │   └── admin.html         Tabbed admin
│   └── static/
│       ├── echo.css           Glassmorphism + 1280x800 layout + scan line
│       └── echo.js            State subscriber + DOM mirror + overlay state machine
│
├── docs/                      You are here
│   ├── README.md              Index
│   ├── quickstart.md
│   ├── architecture.md
│   ├── installation.md
│   ├── configuration.md
│   ├── running.md
│   ├── speech.md
│   ├── hailo-whisper.md
│   ├── face-recognition.md
│   ├── admin.md
│   ├── state-machine.md
│   ├── tuning.md
│   ├── troubleshooting.md
│   ├── project-layout.md
│   ├── privacy.md
│   └── contributing.md
│
└── models/                    Runtime data, mostly git-ignored
    ├── scrfd_10g.hef
    ├── arcface_mobilefacenet.hef
    ├── vosk-model-small-en-us-0.15/
    ├── whisper-base-encoder.hef               (optional, Hailo Whisper)
    ├── whisper-base-decoder.hef               (optional, Hailo Whisper)
    ├── whisper-base-assets/                   (optional, Hailo Whisper)
    ├── whisper/                               (optional, faster-whisper cache)
    └── en_US-hfc_female-medium.onnx           (optional, Piper TTS)
```

## Module responsibilities

| Module | Owns | Talks to |
|---|---|---|
| `main.py` | Camera worker thread + main() orchestrator | All other modules |
| `state.py` | Thread-safe state bus + SSE pub/sub | Flask app, camera worker |
| `frame_streamer.py` | MJPEG ring buffer | Flask `/camera.mjpg`, camera worker |
| `wake_word.py` | Vosk grammar-locked listener | StateBus (via wake_event), audio device |
| `chat_voice.py` | Mic capture + VAD + ASR dispatch | wake_word (pause/resume), asr |
| `asr.py` | ASR backend factory + four implementations | Hailo NPU / OpenAI / CTranslate2 / Vosk |
| `chat.py` | OpenAI / Ollama client + budget | StateBus, AsyncTTS |
| `tts.py` + `async_tts.py` | Piper / pyttsx3 wrappers + queue | OS audio device |
| `hailo_infer.py` | SCRFD + ArcFace pipeline | Hailo NPU |
| `database.py` | SQLite schema + CRUD | All persistence callers |
| `web/app.py` | Flask routes, SSE, MJPEG | StateBus, FrameStreamer, FaceDB |

## Where state actually lives

| Surface | What persists |
|---|---|
| `faces.db` (SQLite) | Employees, face embeddings, interactions, projects, sessions |
| `models/` | All HEFs, voices, recogniser data |
| `/tmp/echo-backend.log` | Backend stdout/stderr (cleared on each `start.sh`) |
| `.env` | Secrets, per-deployment overrides |
| In-memory `KioskState` | Live UI state — never persisted, resets on every restart |
