# Architecture

## Two top-level views

The browser SPA toggles between two main `<section>`s based on the
backend's `state` field.

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

## Module map

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

## Data flow

1. **Camera thread** grabs RGB frames from the IMX500 via picamera2.
2. Detection (SCRFD) + embedding (ArcFace) run on the Hailo NPU.
3. Recognition compares the 512-d embedding against the SQLite
   gallery via cosine similarity. Margin-gated silent learning may
   add the new sample.
4. Bounding boxes + names are drawn server-side onto each frame
   *before* MJPEG encode.
5. The annotated frame goes into a ring buffer in
   `frame_streamer.FrameStreamer`. Browsers pull via
   `/camera.mjpg` (multipart `image/jpeg` over HTTP).
6. State changes (recognised person, registration step, chat
   pending) push to all SSE subscribers via `state.StateBus`.
7. The SPA's vanilla JS controller mirrors state diffs into the DOM.
8. User actions (tap-to-wake, listening pill, register form, ✕
   close) POST to `/api/*` endpoints in `web/app.py`.
9. Endpoints push onto thread-safe queues that `CameraWorker` drains
   on its next iteration.

See [State machine](state-machine.md) for transitions and the
"chat busy" detection-skip rule.
