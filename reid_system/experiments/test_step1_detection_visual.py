"""
Experiment: Test YOLO detection on a single frame (no tracking) to tune confidence
and model choice.

Usage:
  python test_step1_detection_visual.py [--model yolov8s.pt] [--conf 0.3] [--frame N]
"""

import argparse
import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from modules.detector_tracker import YOLOTracker


def draw_detections(frame, detections, output_path):
    """Draw bounding boxes and confidence on a copy of the frame."""
    annotated = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        conf = det["conf"]
        # Green box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"person {conf:.2f}"
        cv2.putText(annotated, label, (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(output_path), annotated)
    print(f"Saved {len(detections)} detections to {output_path}")
    for i, det in enumerate(detections):
        print(f"  {i+1}: bbox={det['bbox']}, conf={det['conf']:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=config.YOLO_MODEL, help="Path to YOLO weights")
    parser.add_argument("--conf", type=float, default=0.3, help="Detection confidence threshold")
    parser.add_argument("--frame", type=int, default=0, help="Frame number from the video")
    parser.add_argument("--video", default=config.INPUT_VIDEO, help="Input video path")
    args = parser.parse_args()

    # Open video and seek to requested frame
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {args.video}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {args.video}  Frames:{total_frames}  FPS:{fps:.2f}")

    # Seek
    if args.frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError(f"Could not read frame {args.frame}")

    cap.release()

    # Create tracker with overridden confidence, no tracking needed, but we use the same class
    # persist=False because it's a single frame
    tracker = YOLOTracker(
        model_path=args.model,
        tracker=config.TRACKER_CONFIG,
        device=config.DEVICE,
        conf=args.conf,
    )

    # Run detection only – set persist=False to not maintain state
    detections = tracker.track(frame, persist=False)

    # Draw and save
    out_name = f"detections_frame{args.frame:04d}_conf{args.conf}.jpg"
    out_path = config.OUTPUT_DIR / out_name
    draw_detections(frame, detections, out_path)


if __name__ == "__main__":
    main()