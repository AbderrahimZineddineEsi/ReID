"""
Detector + Tracker module – wraps BoxMOT (YOLO + BoT‑SORT).

Provides one simple function:
    track(frame) → list of dicts with keys 'local_id', 'bbox', 'crop', 'conf'
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2
from boxmot import BoTSORT
from ultralytics import YOLO
import torch
import config


class DetectorTracker:
    def __init__(
        self,
        model_path: str = config.YOLO_MODEL,
        tracker_name: str = "BoTSORT",     # BoxMOT tracker name
        device: str = config.DEVICE_YOLO,
        conf: float = config.DET_CONFIDENCE,
        img_size: Tuple[int, int] = (640, 640),
    ):
        # Load YOLO (or RT‑DETR) model
        self.detector = YOLO(model_path)
        self.detector.to(device)

        # Create BoT‑SORT tracker via BoxMOT
        self.tracker = BoTSORT(
            track_high_thresh=conf,   # high-confidence detection threshold
            track_low_thresh=0.1,     # low-confidence thresh (ByteTrack‑style)
            new_track_thresh=conf,    # threshold to start a new track
            track_buffer=30,          # frames to keep a lost track alive
            match_thresh=0.8,         # IoU threshold for matching
        )

        self.device = device
        self.conf = conf
        self.img_size = img_size

    def track(self, frame: np.ndarray) -> List[Dict]:
        """
        Run detection and tracking on a BGR frame.
        Returns a list of dicts:
            local_id: int, bbox: (x1,y1,x2,y2), crop: np.ndarray, conf: float
        """
        # 1. Run YOLO detection
        results = self.detector.track(
            frame,
            persist=True,
            tracker="",          # we use BoxMOT tracker separately
            classes=[0],         # only persons
            conf=self.conf,
            device=self.device,
            imgsz=self.img_size,
            verbose=False,
        )

        # 2. Extract detections in the format BoxMOT expects: (x1,y1,x2,y2, conf, class)
        dets = []
        if results[0].boxes is not None and len(results[0].boxes):
            boxes = results[0].boxes.xyxy.cpu().numpy()   # (N,4)
            confs = results[0].boxes.conf.cpu().numpy()   # (N,)
            cls_ids = results[0].boxes.cls.cpu().numpy()  # (N,)
            for box, conf, cls_id in zip(boxes, confs, cls_ids):
                if int(cls_id) != 0:   # keep only person class
                    continue
                x1, y1, x2, y2 = box
                dets.append([x1, y1, x2, y2, conf, 0])   # format: x1,y1,x2,y2,conf,class
        dets = np.array(dets) if dets else np.empty((0, 6))

        # 3. Update BoxMOT tracker with current detections
        # BoxMOT expects detections as numpy array (N,6)
        tracked_objects = self.tracker.update(dets, frame)

        # 4. Build output list
        output = []
        for t in tracked_objects:
            # BoxMOT returns: [x1,y1,x2,y2, track_id, class, conf]
            x1, y1, x2, y2, local_id = int(t[0]), int(t[1]), int(t[2]), int(t[3]), int(t[4])
            conf = float(t[6]) if len(t) > 6 else self.conf
            # Crop the person
            crop = self._safe_crop(frame, x1, y1, x2, y2)
            if crop is None:
                continue
            output.append({
                "local_id": local_id,
                "bbox": (x1, y1, x2, y2),
                "crop": crop,
                "conf": conf,
            })
        return output

    def _safe_crop(self, frame, x1, y1, x2, y2) -> Optional[np.ndarray]:
        h, w = frame.shape[:2]
        x1c = max(0, min(x1, w - 1))
        y1c = max(0, min(y1, h - 1))
        x2c = max(0, min(x2, w))
        y2c = max(0, min(y2, h))
        if x2c <= x1c or y2c <= y1c:
            return None
        return frame[y1c:y2c, x1c:x2c]