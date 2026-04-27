"""Speed test: compare YOLO detection vs YOLO+tracking (ByteTrack)"""
import sys, time
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import cv2, numpy as np
import config
from ultralytics import YOLO

def run_test(video_path, model, tracker_config=None, max_frames=200):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise IOError(f"Cannot open {video_path}")

    if tracker_config:
        print(f"Testing with tracker: {tracker_config}")
    else:
        print("Testing detection only (no tracker)")

    frame_times = []
    for i in range(max_frames):
        ret, frame = cap.read()
        if not ret: break
        t0 = time.perf_counter()
        if tracker_config:
            _ = model.track(frame, persist=True, tracker=tracker_config,
                            classes=[0], conf=config.DETECTION_CONFIDENCE,
                            device=0, imgsz=640, verbose=False)
        else:
            _ = model(frame, classes=[0], conf=config.DETECTION_CONFIDENCE,
                      device=0, imgsz=640, verbose=False)
        frame_times.append(time.perf_counter() - t0)

    cap.release()
    avg = np.mean(frame_times[10:])  # skip warm-up
    print(f"Average time per frame: {avg*1000:.1f} ms")
    print(f"FPS: {1.0/avg:.1f}")
    return 1.0/avg

if __name__ == "__main__":
    video = config.INPUT_VIDEO
    model = YOLO(config.YOLO_MODEL)

    # 1) Detection only
    fps_det = run_test(video, model, tracker_config=None)

    # 2) Detection + ByteTrack
    fps_byte = run_test(video, model, tracker_config="bytetrack.yaml")

    print(f"\nDetection only FPS: {fps_det:.1f}")
    print(f"Detection+ByteTrack FPS: {fps_byte:.1f}")