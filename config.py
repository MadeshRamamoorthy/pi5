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

# Registration
REGISTRATION_FRAMES = 5        # how many embeddings to capture per new person
REGISTRATION_INTERVAL_MS = 400 # wait between capture frames (lets you change pose)

# Camera
CAMERA_RESOLUTION = (1280, 720)
CAMERA_FRAMERATE = 30
