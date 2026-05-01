"""
Step 1: Test detector + tracker (RT-DETRv2 + BoT-SORT through BoxMOT).

Saves crops per local track and an annotated video.
"""

import sys, time
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from modules.detector_tracker import DetectorTracker


def id_to_color(track_id):
    """Map an integer ID to a bright BGR color."""
    hue = ((track_id * 0.61803398875) % 1.0) * 179.0
    hsv = np.array([[[hue, 220, 255]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def main():
    video_path = config.INPUT_VIDEO
    out_video_path = config.OUTPUT_VIDEO
    crops_dir = config.OUTPUT_DIR / "step1_crops"
    crops_dir.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Cannot open {video_path}"
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        out_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    tracker = DetectorTracker()

    frame_idx = 0
    saved_total = 0
    saved_per_track = {}

    pbar = tqdm(total=total_frames, desc="Tracking")
    start_time = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        detections = tracker.track(frame)

        annotated = frame.copy()
        for det in detections:
            lid = det["local_id"]
            x1, y1, x2, y2 = det["bbox"]
            crop = det["crop"]

            color = id_to_color(lid)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, f"L{lid}", (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, color, 1, cv2.LINE_AA)

            # Save crop
            track_dir = crops_dir / f"track_{lid:03d}"
            track_dir.mkdir(exist_ok=True)
            ts = frame_idx / fps
            fname = f"frame_{frame_idx:06d}_t{ts:.3f}_l{lid}.jpg"
            cv2.imwrite(str(track_dir / fname), crop,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_total += 1
            saved_per_track[lid] = saved_per_track.get(lid, 0) + 1

        out.write(annotated)
        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()

    elapsed = time.perf_counter() - start_time
    print(f"\nDone in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Frames: {frame_idx}, FPS: {frame_idx/elapsed:.2f}")
    print(f"Total crops saved: {saved_total}")
    print(f"Unique track IDs: {len(saved_per_track)}")

if __name__ == "__main__":
    main()