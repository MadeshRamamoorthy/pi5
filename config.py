from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
DB_PATH = ROOT / "faces.db"

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
