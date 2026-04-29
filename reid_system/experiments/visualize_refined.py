#!/usr/bin/env python3
"""
Visualize refined identities on the original video.

Usage:
    python experiments/visualize_refined.py --refined outputs/refined_v2_fixed --video assets/store_cam3.mp4 --output refined_video.mp4
"""

import argparse, re, sys
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from modules.detector_tracker import YOLOTracker

# ---------- Parse refined folders ----------
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FILENAME_RE = re.compile(r"frame_(\d+).*_l(\d+)")

def build_identity_map(refined_dir: Path) -> dict:
    """
    Returns dict: (frame_idx, local_id) -> person_id (int)
    by scanning person_* folders.
    """
    mapping = {}
    for folder in sorted(refined_dir.iterdir()):
        if not folder.is_dir() or not folder.name.startswith("person_"):
            continue
        # person_001 -> 1
        try:
            person_id = int(folder.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        for img_file in folder.iterdir():
            if img_file.suffix.lower() not in IMG_EXTENSIONS:
                continue
            m = FILENAME_RE.search(img_file.stem)
            if m:
                frame_idx = int(m.group(1))
                local_id  = int(m.group(2))
                mapping[(frame_idx, local_id)] = person_id
    return mapping

# ---------- Colour helper ----------
def id_to_bgr_color(id_val: int):
    hue = ((id_val * 0.61803398875) % 1.0) * 179.0
    hsv = np.array([[[hue, 220, 255]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refined", required=True, help="Path to refined person folders")
    parser.add_argument("--video", required=True, help="Original video file")
    parser.add_argument("--output", default="refined_visualization.mp4", help="Output video name")
    args = parser.parse_args()

    refined_dir = Path(args.refined)
    if not refined_dir.exists():
        raise FileNotFoundError(f"Refined folder not found: {refined_dir}")

    mapping = build_identity_map(refined_dir)
    print(f"Loaded identity map: {len(mapping)} crops mapped to {len(set(mapping.values()))} persons.")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    tracker = YOLOTracker()
    frame_idx = 0

    pbar = tqdm(total=total_frames, desc="Visualizing")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        detections = tracker.track(frame, persist=True)
        annotated = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            local_id = det["local_id"]
            # Look up the final identity
            person_id = mapping.get((frame_idx, local_id), None)

            if person_id is not None:
                color = id_to_bgr_color(person_id)
                label = f"P{person_id}"
            else:
                color = (128, 128, 128)  # grey for unknown
                label = "?"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, label, (x1, max(y1-10, 20)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, color, 1, cv2.LINE_AA)

        out.write(annotated)
        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()
    print(f"Video saved: {args.output}")

if __name__ == "__main__":
    main()


##
#python experiments/visualize_refined.py 
#--refined outputs/refined_v2_fixed --video assets/store_cam3.mp4 --output final_check.mp4