from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
DB_PATH = ROOT / "faces.db"

# Vosk model directory. We use the SMALL model only -- it's enough for
# the four-word wake phrase and uses ~200 MB instead of 2-3 GB. Chat
# free-form dictation goes through Whisper (see CHAT_ASR_BACKEND below)
# so the big Vosk model isn't needed.
VOSK_MODEL_DIR = MODELS_DIR / "vosk-model-small-en-us-0.15"

# Hailo-10H compiled models (HEF). Download via the install guide.
DETECTOR_HEF = MODELS_DIR / "scrfd_10g.hef"
EMBEDDER_HEF = MODELS_DIR / "arcface_mobilefacenet.hef"

# Detector
DETECTOR_INPUT = (640, 640)
DETECTOR_SCORE_THRESHOLD = 0.5
DETECTOR_NMS_IOU = 0.4

# Embedder
EMBEDDER_INPUT = (112, 112)
EMBEDDING_DIM = 512

# Recognition
COSINE_MATCH_THRESHOLD = 0.38  # ArcFace: 0.35-0.45 typical; lower => looser match
# Each emp_id is greeted at most once per this many seconds. Lets the
# system greet every recognised face in a multi-person scene without
# spamming when someone keeps stepping in and out of frame.
GREET_COOLDOWN_SEC = 60

# Camera
CAMERA_RESOLUTION = (1280, 720)
CAMERA_FRAMERATE = 30

# ---- HUD: side transcript panel ------------------------------------------
# When True, the OpenCV window is split: camera on the left, a live
# transcript of mic input + TTS output + recognition / state events on
# the right. Set False to keep the simple camera-only window.
SHOW_TRANSCRIPT_PANEL = True
TRANSCRIPT_PANEL_WIDTH = 420       # pixels added to the right of the camera
TRANSCRIPT_MAX_EVENTS = 16         # ring-buffer length

# Final display scale applied after compositing camera + transcript.
# Keep at 1.0 for full-resolution display. Lower values fit the window
# onto smaller screens (recommended values below). Detection / liveness
# / quality logic still runs on the full-resolution camera frame, so
# accuracy is unaffected -- only the displayed window shrinks.
#
#   1.0  -> native (camera + panel = ~1700 x 720)
#   0.75 -> ~1275 x 540   (fits 1280 x 800 panels)
#   0.6  -> ~1020 x 432   (fits 1024 x 600 panels)
#   0.5  -> ~850  x 360
DISPLAY_SCALE = 1.0

# ---- Face quality gate ---------------------------------------------------
# A "real" face must clear ALL of these before recognition / registration.
QUALITY_SCORE_THRESHOLD = 0.70    # min detector score (overrides DETECTOR_SCORE_THRESHOLD for gating)
QUALITY_MIN_FACE_PIXELS = 110     # min bbox width AND height in pixels
QUALITY_FRAME_EDGE_MARGIN = 12    # face must be at least this far from any frame edge
QUALITY_MIN_EYE_DISTANCE = 28     # min inter-pupil distance in pixels
QUALITY_MAX_EYE_TILT = 0.45       # max |dy/dx| between the two eyes (rejects extreme roll)

# Anti-flicker: the largest face must be of acceptable quality AND below the
# match threshold for this many consecutive frames before we offer to register.
UNKNOWN_FRAMES_BEFORE_REGISTER = 15

# ---- Registration: voice-guided poses ------------------------------------
# Each entry: (spoken prompt, expected landmark-shift direction relative to the
# previous capture). The shift is checked in normalised units of the inter-eye
# distance, so it works at any range. Use None for the first/calibration pose.
#   ("right",  +x shift in image)   ("left", -x)   ("up", -y)   ("down", +y)
POSE_PROMPTS = [
    ("Look straight at the camera.",   None),
    ("Slowly shift to your right.",   "right"),
    ("Now shift to your left.",       "left"),
    ("Tilt your head up.",            "up"),
    ("Tilt your head down.",          "down"),
]

POSE_HOLD_SEC = 1.5            # min time after prompt before we even *try* to capture
POSE_STABLE_SEC = 0.5          # face landmarks must stay still for this long before capture
POSE_STABLE_PIXEL_TOL = 4.0    # max landmark drift (px) within the stability window
POSE_MIN_SHIFT = 0.25          # min directional shift, in eye-distance units
POSE_CAPTURE_TIMEOUT_SEC = 8.0 # give up on this pose if user doesn't comply

# Re-register: the new face must self-match this strongly to count as the
# same person before we append samples to an existing emp_id.
REREGISTER_MATCH_THRESHOLD = 0.35

# ---- Active liveness (blink challenge) ------------------------------------
# When True, every emp_id must blink once per ACTIVE session before they
# get greeted. Clears on IDLE so the next session re-prompts. Defeats the
# remaining attack vector: a high-quality video replay on a screen.
LIVENESS_REQUIRE_BLINK = False
LIVENESS_BLINK_PROMPT = "Just a quick blink so I know it's you!"
LIVENESS_BLINK_TIMEOUT_SEC = 5
LIVENESS_BLINK_PATCH_PX = 14         # half-extent (px) around each eye landmark
LIVENESS_BLINK_DELTA_MIN = 6.0       # min std-dev range across the window
LIVENESS_BLINK_REVERIFY_SEC = 600    # within an ACTIVE session, this is huge
                                     # (1h); blinks are session-scoped anyway

# ---- TTS backend ----------------------------------------------------------
# "piper"   -> neural Piper via the piper1-gpl Python package
#              (pip install piper-tts). Falls back to pyttsx3 automatically
#              if the package or the voice model isn't installed yet.
# "pyttsx3" -> espeak-ng. Robotic but always available.
TTS_BACKEND = "piper"

# Default voice. Drop more .onnx + .onnx.json pairs into models/piper/
# and point this at any of them to switch voices. See README §2.9.
PIPER_MODEL_PATH = MODELS_DIR / "piper" / "en_US-hfc_female-medium.onnx"

# Pad the start of each TTS utterance with this much silence. USB speakers
# (Anker A3301 etc.) often clip the first ~300 ms while their amp wakes up.
# 0 disables. The silence is inserted *into the WAV*, not added as a
# blocking sleep, so total latency is the same either way.
TTS_PREBUFFER_MS = 500

# ---- Audio output for TTS -------------------------------------------------
# None  -> use system default audio sink (HDMI / 3.5 mm / whatever PipeWire
#          decides). To pin TTS to the Anker (or any specific output), set
#          this to an ALSA name from `aplay -L`, e.g.:
#            "plughw:CARD=PowerConf,DEV=0"
#            "plughw:2,0"
# Also overridable at runtime via the AUDIO_OUTPUT_DEVICE env var (the
# start.sh script uses this).
# When set, pyttsx3 synthesises to a temporary WAV and aplay plays it on
# the chosen device, so this works even when the system default is HDMI.
AUDIO_OUTPUT_DEVICE: str | None = os.environ.get("AUDIO_OUTPUT_DEVICE") or None

# ---- Wake-word / activation -----------------------------------------------
BRAND_NAME = "ECHO SCOPE"
CONTACT_EMAIL = "Calgary_AIClub@infosys.com"
WAKE_WORD = "hello echo scope"    # phrase that activates recognition
WAKE_WORD_SAMPLERATE = 16000
WAKE_WORD_BLOCKSIZE = 8000
# Drop back to IDLE this long after the last *interaction* (a new person
# greeted, an unknown face in frame, or a registration completing). A
# recognised person who keeps standing in front of the camera does NOT
# reset this timer -- 10 s after the last new event we sleep.
IDLE_AFTER_LAST_INTERACTION_SEC = 30
# Backwards-compat alias used by older code paths.
SLEEP_AFTER_NO_LIVE_FACE_SEC = IDLE_AFTER_LAST_INTERACTION_SEC
# Hard back-stop on any single ACTIVE session.
ACTIVE_SESSION_MAX_SEC = 600

# ---- Idle / Active screen layout -----------------------------------------
# Welcome banner colours, BGR triplets.
THEME = {
    "bg":     (180, 110, 30),   # blue background
    "fg":     (0, 0, 0),         # black text
    "accent": (0, 0, 0),
}
# Window size for the cv2 output. The camera frame is centred / scaled
# into the left half during ACTIVE; the right half is the panel. Pick a
# size that fits your display.
WINDOW_SIZE = (1280, 720)        # (width, height)

# Right-panel width in ACTIVE state. Camera left half = WINDOW_SIZE[0] - PANEL_WIDTH.
PANEL_WIDTH = 540

# ---- Weather widget -------------------------------------------------------
KIOSK_HOST = "127.0.0.1"
KIOSK_PORT = 8080
FUN_FACT_ROTATE_SEC = 15
MJPEG_QUALITY = 80                # JPEG quality for the camera stream
MJPEG_MAX_FPS = 20                # display only; detection runs at full rate

WEATHER_REFRESH_SEC = 1800        # 30 minutes
WEATHER_LATITUDE = None           # set both to skip IP geolocation
WEATHER_LONGITUDE = None
WEATHER_FALLBACK_CITY = None      # used as label when lat/lon are pinned

# ---- OpenAI / Ollama chat -------------------------------------------------
OPENAI_MODEL = "gpt-4o-mini"
OLLAMA_URL = "http://localhost:8000"  # Hailo-Ollama on this Pi binds to :8000
                                      # (`ss -tlnp` shows hailo-ollama owning :8000).
                                      # :8080 is Open WebUI; upstream Ollama uses 11434.
OLLAMA_MODEL = "qwen3:1.7b"       # qwen3 instruct on Hailo-Ollama; bigger
                                  # alternatives that fit on Pi 5: "qwen2.5:1.5b"
                                  # avoid "deepseek_r1:*" (emits <think> blocks)
                                  # and "qwen2.5-coder:*" (code-tuned, weak chat)
# Backend probe order. First available wins. Set to ("ollama", "openai")
# if you want to prefer the local model.
CHAT_BACKEND_ORDER = ("openai", "ollama")
CHAT_MAX_QUESTIONS_PER_SESSION = 5
CHAT_VOICE_MODE_DEFAULT = "voice"   # "voice" or "keyboard"

# ---- Chat-voice ASR -------------------------------------------------------
# Wake-word listener stays on the small Vosk model. Long-form chat
# dictation goes through this separate backend, lazy-loaded so the
# kiosk doesn't pay the memory bill until the user actually opens
# chat voice mode.
#
#   "faster-whisper" -> CTranslate2 + tiny.en, ~250 MB resident, ~3 s
#                       per 10 s of speech on Pi 5 CPU. Default.
#   "openai"         -> openai.audio.transcriptions.create (cloud). No
#                       local resources but a network round-trip per
#                       question and a paid API call.
#   "vosk"           -> legacy: reuse the small Vosk model in a
#                       no-grammar recognizer. Lower quality, kept as
#                       an offline fallback.
CHAT_ASR_BACKEND = "faster-whisper"
WHISPER_MODEL = "tiny.en"           # "tiny.en" | "base.en" | "small.en"
WHISPER_DEVICE = "cpu"              # Pi 5 has no GPU; keep "cpu".
WHISPER_COMPUTE_TYPE = "int8"       # 8-bit quantisation for memory.
WHISPER_CACHE_DIR = MODELS_DIR / "whisper"
# Voice capture timing knobs.
CHAT_VOICE_MAX_SEC = 12.0           # hard cap on a single utterance.
CHAT_VOICE_SILENCE_SEC = 1.5        # auto-finalise after this much silence.
CHAT_VOICE_SILENCE_RMS = 350        # int16 RMS threshold below which audio
                                    # counts as silence.

# ---- Silent learning ------------------------------------------------------
# When a confidently recognised face passes the quality + liveness gates,
# append the new embedding to that person's gallery so recognition gets
# more robust over time. Rate-limited and capped so the DB doesn't grow
# unbounded.
SILENT_LEARN_ENABLED = True
SILENT_LEARN_MIN_SCORE = 0.55          # only learn when match is comfortable
SILENT_LEARN_MAX_SIMILARITY = 0.92     # skip if new sample is ~ a duplicate of an existing one
SILENT_LEARN_MIN_INTERVAL_SEC = 60     # at most one new sample per person per minute
SILENT_LEARN_MAX_SAMPLES_PER_PERSON = 30  # cap; oldest non-enrolment samples drop first

# ---- Liveness (passive anti-spoofing) -------------------------------------
# Sliding window length over which we compute liveness signals.
LIVENESS_WINDOW_FRAMES = 24
# Min per-frame face crop size used by the pixel-jitter check.
LIVENESS_CROP_PX = 96

# Single-frame screen-attack defences (run BEFORE temporal checks; if any
# fails, the temporal window is cleared so glare/flat content can never
# accumulate enough frames to be considered live).
#
# Specular highlight ratio: fraction of pixels in the face crop whose HSV
# V channel is > 240. Real indoor faces produce 0-5% (glasses, forehead
# sheen, glints in eyes). Phone/monitor screens with glare typically
# light up 10-40% of the visible "face". Lower this if a screen still
# passes; raise if real faces are flagged.
LIVENESS_MAX_SPECULAR_RATIO = 0.10

# Texture variance: variance of the Laplacian of the face crop. Real
# skin has fine pores/wrinkles -> high variance. Phones/monitors smooth
# the face out -> low variance. Bump this up if a screen still passes.
LIVENESS_MIN_TEXTURE_VAR = 60.0

# Temporal signals (computed over the sliding window). Both must clear:
# 1. relative landmark motion (subtracting whole-face translation).
LIVENESS_REL_MOTION_MIN = 0.45    # pixels (std), in original-image scale
# 2. face-region pixel jitter beyond camera read noise.
LIVENESS_PIXEL_JITTER_MIN = 4.0   # mean abs frame-to-frame diff in [0..255]
