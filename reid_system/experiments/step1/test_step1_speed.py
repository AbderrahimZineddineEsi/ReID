"""Speed test: YOLO tracking only, no saving, no display."""
import sys, time
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import cv2, config
from modules.detector_tracker import YOLOTracker

def main():
    video_path = config.INPUT_VIDEO
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise IOError(f"Cannot open {video_path}")

    tracker = YOLOTracker()
    frame_idx = 0
    start = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1
        t0 = time.perf_counter()
        _ = tracker.track(frame, persist=True)
        track_ms = (time.perf_counter() - t0) * 1000
        if frame_idx % 30 == 0:
            print(f"Frame {frame_idx}: track={track_ms:.1f} ms")

    elapsed = time.perf_counter() - start
    cap.release()
    print(f"\nTotal frames: {frame_idx}")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Pure tracking FPS: {frame_idx/elapsed:.2f}")

if __name__ == "__main__":
    main()