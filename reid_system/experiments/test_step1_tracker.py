"""
Experiment: Step 1 – Tracker + Crop Saving

Reads a video, runs YOLO + BOT‑SORT, draws bounding boxes with local track IDs,
saves annotated video, and crops per track ID into folders.

This script tests the detector_tracker module without any ReID or global linking.
"""

import time
from pathlib import Path
from typing import Tuple
import cv2
import numpy as np
import torch
# Add parent directory to path so we can import config and modules
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from modules.detector_tracker import YOLOTracker


# ---------- Helper functions (copied from your original code for consistency) ----------
def id_to_bgr_color(track_id: int) -> Tuple[int, int, int]:
    """Deterministically map an ID to a vivid BGR color."""
    hue = ((int(track_id) * 0.61803398875) % 1.0) * 179.0
    hsv = np.array([[[hue, 220, 255]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_text_with_outline(
    frame, text: str, origin: Tuple[int, int], color: Tuple[int, int, int],
    font=cv2.FONT_HERSHEY_DUPLEX, font_scale=0.62, thickness=1
):
    """Draw text with a black outline for better readability."""
    x, y = origin
    cv2.putText(frame, text, (x, y), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def estimate_sharpness(crop: np.ndarray) -> float:
    """Compute Laplacian variance as a simple sharpness metric."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def should_save_crop(crop: np.ndarray, bbox: Tuple[int, int, int, int],
                     frame_width: int, frame_height: int) -> bool:
    """
    Apply quality gates to decide whether to save a crop.
    Reuses the thresholds from config.
    """
    bw = bbox[2] - bbox[0]
    bh = bbox[3] - bbox[1]
    if bw < config.MIN_WIDTH or bh < config.MIN_HEIGHT:
        return False

    area_ratio = (bw * bh) / float(frame_width * frame_height)
    if area_ratio < config.MIN_AREA_RATIO:
        return False

    if estimate_sharpness(crop) < config.MIN_SHARPNESS:
        return False

    return True


# ---------- Main experiment ----------
def main():
    # Load configuration
    video_path = config.INPUT_VIDEO
    output_video_path = config.OUTPUT_VIDEO
    crop_base_dir = Path(config.CROP_OUTPUT_DIR)

    # Create output directories
    crop_base_dir.mkdir(parents=True, exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Prepare video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    # Initialize tracker
    tracker = YOLOTracker(
        model_path=config.YOLO_MODEL,
        tracker=config.TRACKER_CONFIG,
        device=config.DEVICE,
    )
    
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device count: {torch.cuda.device_count()}")

    frame_idx = 0
    saved_total = 0
    saved_per_track = {}

    print(f"Processing {video_path} ...")
    print(f"Press 'q' to quit early. Output video: {output_video_path}")

    start_time = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Run detection + tracking
        detections = tracker.track(frame, persist=True)
        t0 = time.perf_counter()
        detections = tracker.track(frame, persist=True)
        track_time = time.perf_counter() - t0
        if frame_idx % 30 == 0:
            print(f"Frame {frame_idx}: track={track_time*1000:.1f}ms")
            
        # Annotate frame
        annotated = frame.copy()
        for det in detections:
            local_id = det["local_id"]
            x1, y1, x2, y2 = det["bbox"]
            crop = det["crop"]

            # Color for this track ID
            color = id_to_bgr_color(local_id)

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw track ID label
            text_y = y1 - 10 if y1 > 24 else y1 + 20
            draw_text_with_outline(annotated, f"ID:{local_id}", (x1, text_y), color)

            # Decide whether to save the crop
            if should_save_crop(crop, (x1, y1, x2, y2), width, height):
                if frame_idx % config.SAVE_EVERY_N == 0:
                    # Create folder for this track if not exists
                    track_dir = crop_base_dir / f"track_{local_id:03d}"
                    track_dir.mkdir(parents=True, exist_ok=True)

                    # Filename: frame_<idx>_t<sec>.jpg
                    timestamp_s = frame_idx / fps
                    img_name = f"frame_{frame_idx:06d}_t_{timestamp_s:.3f}.jpg"
                    img_path = track_dir / img_name

                    cv2.imwrite(
                        str(img_path),
                        crop,
                        [int(cv2.IMWRITE_JPEG_QUALITY), config.JPEG_QUALITY],
                    )
                    saved_total += 1
                    saved_per_track[local_id] = saved_per_track.get(local_id, 0) + 1

        # Write annotated frame
        out.write(annotated)

        # Display (optional)
        #cv2.imshow("Step 1 – Tracker", annotated)
        #if cv2.waitKey(1) & 0xFF == ord("q"):
        #    break

    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    elapsed = time.perf_counter() - start_time
    speed = frame_idx / elapsed if elapsed > 0 else 0

    print("\n--- Step 1 Experiment Summary ---")
    print(f"Frames processed: {frame_idx}")
    print(f"Time elapsed: {elapsed:.2f}s ({elapsed/60:.2f} min)")
    print(f"Effective speed: {speed:.2f} FPS")
    print(f"Total crops saved: {saved_total}")
    print(f"Unique track IDs: {len(saved_per_track)}")
    print(f"Crop folders: {crop_base_dir}")

    if saved_per_track:
        print("Crops per track (top 20):")
        for tid in sorted(saved_per_track, key=saved_per_track.get, reverse=True)[:20]:
            print(f"  track_{tid:03d}: {saved_per_track[tid]} crops")


if __name__ == "__main__":
    main()