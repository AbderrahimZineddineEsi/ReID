"""
Central configuration for the ReID system.
All paths, model names, and thresholds live here.
"""

from pathlib import Path

# ---------- Base paths ----------
BASE_DIR = Path(__file__).resolve().parent      
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

MODELS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------- Video input (change per experiment) ----------
INPUT_VIDEO = "assets/store_cam3.mp4"            # <-- your test video

# ---------- Device ----------
DEVICE_YOLO = "cuda:0"      # GPU for detection + tracking (BoxMOT runs here)
# In the future we'll add a second GPU for ReID

# ---------- Detection model ----------
YOLO_MODEL = "yolov8x.pt"   # placeholder – we will use RT-DETRv2 through BoxMOT
# BoxMOT will load the actual detection model, configurable later

# ---------- Tracking ----------
TRACKER_CONFIG = "botsort.yaml"   # BoxMOT uses a YAML; BoT-SORT is default
TRACKER_MODEL = "osnet_x1_0.onnx" # ReID model inside BoT-SORT (will be replaced by CLIP-ReID later)
DET_CONFIDENCE = 0.25            # initial confidence threshold (will tune)

# ---------- Output video ----------
OUTPUT_VIDEO = "tracked_output.mp4"