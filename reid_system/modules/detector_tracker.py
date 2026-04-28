"""
YOLO + BOT‑SORT Person Tracker Module.

This module provides a clean, reusable class that:
- Loads YOLOv8 with a BOT‑SORT tracking configuration.
- Runs detection + tracking on a single frame (or batch).
- Returns per‑detection information: bounding box, local track ID, confidence,
  and a cropped image of the person.
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
import cv2
import torch
from ultralytics import YOLO

# We import config – if you run this module standalone, make sure config is importable.
try:
    import config
except ImportError:
    # Fallback if running as script
    class config:
        YOLO_MODEL = "yolov8n.pt"
        TRACKER_CONFIG = "botsort.yaml"
        DETECTION_CONFIDENCE = 0.4
        DEVICE = "cuda"
        IMG_SIZE = (640, 640)


class YOLOTracker:
    """
    Wrapper around YOLO for detection + tracking (BOT‑SORT).

    Usage:
        tracker = YOLOTracker(model_path='yolov8n.pt', tracker='botsort.yaml', device='cuda')
        results = tracker.track(frame)
        for det in results:
            print(det['local_id'], det['bbox'])
    """

    def __init__(
        self,
        model_path: str = config.YOLO_MODEL,
        tracker: str = config.TRACKER_CONFIG,
        device: str = config.DEVICE,
        conf: float = config.DETECTION_CONFIDENCE,
        img_size: Tuple[int, int] = config.IMG_SIZE,
    ):
        """
        Args:
            model_path: Path to YOLO weights (e.g., yolov8n.pt)
            tracker: Tracker config filename (BOT‑SORT yaml)
            device: 'cuda' or 'cpu' – where to run the model
            conf: Minimum confidence for detection
            img_size: Inference size (width, height)
        """
        self.device = device

        # Load YOLO model and handle CUDA init failures by falling back to CPU.
        self.model = YOLO(model_path)
        if str(device).startswith("cuda"):
            if not torch.cuda.is_available():
                print("WARNING: CUDA not available; using CPU instead.")
                self.device = "cpu"
            else:
                try:
                    self.model.to(device)
                except RuntimeError as exc:
                    print(f"WARNING: CUDA init failed ({exc}); using CPU instead.")
                    self.device = "cpu"
        if self.device == "cpu":
            self.model.to(self.device)

        # Store parameters for the track() call
        self.tracker_config = tracker
        self.conf = conf
        self.img_size = img_size

        # Track whether we are processing a continuous stream (persist=True if so)
        # This is used for video files; for per-frame processing in a loop, set to True.
        self._stream_persist = False  # We'll set it to True when tracking a video

    def track(self, frame: np.ndarray, persist: bool = True) -> List[Dict]:
        """
        Run YOLO detection + tracking on a single frame.

        Args:
            frame: BGR image (numpy array) from OpenCV.
            persist: If True, keeps tracking state across frames (must be True for correct tracking).

        Returns:
            List of dictionaries, each containing:
                'local_id': int        – BOT‑SORT track ID
                'bbox': (x1,y1,x2,y2)  – integer coordinates
                'crop': np.ndarray      – cropped person image (BGR)
                'conf': float           – detection confidence
        """
        # The model.track() method returns a list of Results objects (one per image if batch).
        # We feed a single frame, so results[0] is used.
        results = self.model.track(
            source=frame,
            persist=persist,            # keep tracking state across frames
            tracker=self.tracker_config,
            classes=[0],                # only persons (COCO class 0)
            conf=self.conf,
            device=self.device,
            imgsz=self.img_size,
            verbose=False,              # suppress console output
        )

        detections = []

        # Check if any detections were made
        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()   # shape (N,4)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)  # shape (N,)
            confs = results[0].boxes.conf.cpu().numpy()   # shape (N,)

            for box, tid, conf in zip(boxes, track_ids, confs):
                x1, y1, x2, y2 = map(int, box)
                # Skip invalid boxes
                if x2 <= x1 or y2 <= y1:
                    continue

                # Crop the person from the frame
                crop = self._safe_crop(frame, x1, y1, x2, y2)
                if crop is None or crop.size == 0:
                    continue

                detections.append({
                    "local_id": int(tid),
                    "bbox": (x1, y1, x2, y2),
                    "crop": crop,
                    "conf": float(conf),
                })

        return detections

    def _safe_crop(self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> Optional[np.ndarray]:
        """
        Crop the frame safely, clipping coordinates to image boundaries.

        Returns:
            Cropped image, or None if the region is invalid.
        """
        h, w = frame.shape[:2]
        x1c = max(0, min(x1, w - 1))
        y1c = max(0, min(y1, h - 1))
        x2c = max(0, min(x2, w))
        y2c = max(0, min(y2, h))
        if x2c <= x1c or y2c <= y1c:
            return None
        return frame[y1c:y2c, x1c:x2c]
