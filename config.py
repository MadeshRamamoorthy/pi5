from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
DB_PATH = ROOT / "faces.db"

# Vosk wake-word model directory (download via README §2.8).
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
GREET_COOLDOWN_SEC = 10        # not used today; kept for reference

# Camera
CAMERA_RESOLUTION = (1280, 720)
CAMERA_FRAMERATE = 30

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

# ---- TTS backend ----------------------------------------------------------
# "piper"   -> neural Piper (much more natural). Falls back to pyttsx3
#              automatically if the binary or voice model is missing.
# "pyttsx3" -> espeak-ng. Robotic but always available.
TTS_BACKEND = "piper"

# Project-local Piper install. README §2.9 covers downloading both.
PIPER_BIN = ROOT / "tools" / "piper" / "piper"
PIPER_MODEL_PATH = MODELS_DIR / "piper" / "en_US-amy-medium.onnx"

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
WAKE_WORD = "hello echo"          # phrase that activates recognition
WAKE_WORD_SAMPLERATE = 16000
WAKE_WORD_BLOCKSIZE = 8000
# Drop back to IDLE this long after the last *interaction* (a new person
# greeted, an unknown face in frame, or a registration completing). A
# recognised person who keeps standing in front of the camera does NOT
# reset this timer -- 10 s after the last new event we sleep.
IDLE_AFTER_LAST_INTERACTION_SEC = 10
# Backwards-compat alias used by older code paths.
SLEEP_AFTER_NO_LIVE_FACE_SEC = IDLE_AFTER_LAST_INTERACTION_SEC
# Hard back-stop on any single ACTIVE session.
ACTIVE_SESSION_MAX_SEC = 600

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
