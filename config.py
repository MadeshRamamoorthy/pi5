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
GREET_COOLDOWN_SEC = 10        # don't repeat greeting for the same person

# Registration: one prompt per pose, one embedding per prompt
POSE_PROMPTS = [
    "Look straight at the camera.",
    "Slowly shift to your right.",
    "Now shift to your left.",
    "Tilt your head up.",
    "Tilt your head down.",
]
POSE_HOLD_SEC = 1.2            # time given to settle in the new pose
POSE_CAPTURE_TIMEOUT_SEC = 4.0 # max wait per pose before giving up

# How permissive the re-registration "is this really the same person?" check is.
# Same scale as COSINE_MATCH_THRESHOLD.
REREGISTER_MATCH_THRESHOLD = 0.35

# Camera
CAMERA_RESOLUTION = (1280, 720)
CAMERA_FRAMERATE = 30
