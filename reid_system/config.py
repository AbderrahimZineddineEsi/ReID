"""
Central configuration for the ReID system.
Edit paths and thresholds here — everything else reads from this file.
"""

from pathlib import Path

# ---------------------
# PROJECT PATHS
# ---------------------
BASE_DIR = Path(__file__).resolve().parent  # reid_system/
MODELS_DIR = BASE_DIR / "models"
EXPERIMENTS_DIR = BASE_DIR / "experiments"
OUTPUT_DIR = BASE_DIR / "outputs"

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------
# MODELS
# ---------------------
YOLO_MODEL = str(MODELS_DIR / "yolov8s.pt")          # YOLOv8 nano (fast, low VRAM)
OSNET_MODEL = str(MODELS_DIR / "osnet_x1_0.onnx")    # OSNet ReID (will be used in step 2)
FACE_MODEL = str(MODELS_DIR / "arcface.onnx")        # Placeholder for later

# ---------------------
# DETECTION & TRACKING
# ---------------------
DETECTION_CONFIDENCE = 0.15          # Minimum confidence for person detection
TRACKER_CONFIG = "bytetrack.yaml"     # BOT‑SORT config file (uses ultralytics default)
DEVICE = "cuda"                     # Use 'cuda' if GPU available, else 'cpu'
IMG_SIZE = (640, 640)               # Inference size (smaller = faster, less accurate)

# ---------------------
# QUALITY GATES (from your original script)
# ---------------------
MIN_WIDTH = 10
MIN_HEIGHT = 20
MIN_AREA_RATIO = 0.0010             # Minimum bbox area relative to frame area for saving
MIN_SHARPNESS = 8.0                 # Laplacian variance threshold for saving
JPEG_QUALITY = 95                   # Crop saving quality
SAVE_EVERY_N = 1                    # Save one crop every N frames per track

# ---------------------
# INPUT / OUTPUT
# ---------------------
INPUT_VIDEO = "assets/store_cam3.mp4"      # Override in experiment scripts or command line
OUTPUT_VIDEO = "tracked_output.mp4" # Annotated video output
CROP_OUTPUT_DIR = str(OUTPUT_DIR / "step1_crops")

# ---------------------
# REID EMBEDDING (will be used in step 2)
# ---------------------
REID_SIZE = (256, 128)              # H, W – OSNet expects 256x128
EMBED_EVERY_N = 3                   # Re-compute embedding every N frames for active tracks


# ---------------------
# GLOBAL ID LINKING (Step 2)
# ---------------------
SHORT_GAP_SEC = 3.0              # seconds – use IoU bonus for recent tracks
SHORT_LINK_THRESHOLD = 0.74
LONG_LINK_THRESHOLD = 0.92       ## 82
IOU_WEIGHT = 0.12
COLOR_THRESHOLD = 0.55           # minimum colour similarity to even consider merging ##35
AMBIGUITY_MARGIN = 0.05
DRIFT_REID_THRESHOLD = 0.82      # below this, split identity  ##0.62
DRIFT_COLOR_THRESHOLD = 0.24
PROTOTYPE_ALPHA = 0.20           # embedding update rate
COLOR_ALPHA = 0.12               # colour histogram update rate
LOCAL_STALE_SEC = 8.0            # forget local->global mapping after this time