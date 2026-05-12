# ECHO SCOPE documentation

A face-recognising kiosk for the Raspberry Pi 5 + Hailo-10H — wake it
with your voice, chat with an LLM, and the kiosk remembers returning
visitors.

## Contents

### Getting started
1. [Quickstart](quickstart.md) — three commands to a running kiosk.
2. [Architecture](architecture.md) — what the SPA looks like, where
   each piece runs.
3. [Installation](installation.md) — OS prep, HailoRT, venv, models,
   audio, browser.
4. [Configuration](configuration.md) — `.env` knobs and `config.py`
   defaults.
5. [Running the kiosk](running.md) — `start.sh`, `stop.sh`, live
   logs, auto-start on boot.

### Features
6. [Speech](speech.md) — wake word, chat-voice, chat LLM, TTS, and
   the auto-selection precedence.
7. [Hailo Whisper](hailo-whisper.md) — NPU speech-to-text setup.
8. [Face recognition](face-recognition.md) — pipeline, registration
   flows, silent learning, liveness.
9. [Admin UI](admin.md) — `/admin` tabs, HTTP API surface.

### Operations
10. [State machine](state-machine.md) — IDLE / ACTIVE / chat busy.
11. [Tuning](tuning.md) — every `config.py` knob in one place.
12. [Troubleshooting](troubleshooting.md) — symptom → fix table.
13. [Privacy and limitations](privacy.md) — what's stored, what's
    sent off-device.

### Project
14. [Project layout](project-layout.md) — annotated file tree.
15. [Contributing](contributing.md) — branch / commit conventions.

---

## What this is in one paragraph

A browser SPA presents a glassmorphic 1280×800 idle dashboard
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
| **Backends** | Chat: OpenAI / Hailo-Ollama · STT: Hailo / OpenAI / faster-whisper / Vosk · TTS: Piper / pyttsx3 |
| **UI** | Chromium kiosk pointed at a local Flask app on `:8090` |

See [Architecture](architecture.md) for the full picture.
