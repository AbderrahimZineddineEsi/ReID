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
DETECTION_CONFIDENCE = 0.4          # Minimum confidence for person detection
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



# ---------------------
# REID QUALITY GATES (separate from saving)
# ---------------------
MIN_REID_WIDTH = 10
MIN_REID_HEIGHT = 20
MIN_REID_AREA_RATIO = 0.0012      # stricter than saving
MIN_REID_SHARPNESS = 12.0         # stricter than saving

# ---------------------
# EMBEDDING DECIMATION
# ---------------------
EMBED_EVERY_N = 3                 # refresh embedding only every N frames per track

# ---------------------
# GLOBAL ID LINKING (same as original)
# ---------------------
SHORT_GAP_SEC = 3.0
SHORT_LINK_THRESHOLD = 0.74
LONG_LINK_THRESHOLD = 0.82
IOU_WEIGHT = 0.12
COLOR_THRESHOLD = 0.35
AMBIGUITY_MARGIN = 0.03
DRIFT_REID_THRESHOLD = 0.62
DRIFT_COLOR_THRESHOLD = 0.24
PROTOTYPE_ALPHA = 0.20
COLOR_ALPHA = 0.12
LOCAL_STALE_SEC = 8.0
INCLUDE_LOCAL_ID_ZERO = False    # we ignore ID 0 by 

# ---------------------
# OFFLINE REFINEMENT V2 (Pass 2)
# ---------------------
# Stage A – extra quality filter
REFINE_MIN_REID_AREA_RATIO = 0.0012   # stricter than saving
REFINE_MIN_REID_SHARPNESS = 12.0      # stricter than saving
REFINE_MAX_ASPECT_RATIO = 4.0         # reject crops taller than 3.5× wide (optional, 0 to disable)

# Stage B – intra-folder
MIN_TRACK_SIZE = 18                  # ignore tracks smaller than this
SPLIT_REID_THRESHOLD = 0.9           # below this → different person
SPLIT_COLOR_THRESHOLD = 0.65

# Stage C – cross-folder merging
MERGE_REID_THRESHOLD = 0.9
MERGE_COLOR_THRESHOLD = 0.65
MERGE_COLOR_WEIGHT = 0.20             # contribution of colour to merge score
COHERENCE_MIN = 0.80                  # cluster must have mean self-similarity ≥ this
MAX_SAME_FRAME_OVERLAP = 3            # always 0
MIN_FOLDER_IMAGES_FINAL = 18          # drop folders with fewer images after refinement


# Large folder bonus
LARGE_FOLDER_SIZE = 200             # clusters with ≥ this many images get relaxed thresholds
LARGE_MERGE_REID_THRESHOLD = 0.87   # lower than normal 0.85? You later set normal to 0.90? Adjust as you like
LARGE_MERGE_COLOR_THRESHOLD = 0.65  # optional relaxed colour, but keep same if it works