"""
Experiment: Step 2 – Global Identity Linking (Fixed Version)

- Uses strict ReID quality gates
- Uses embedding decimation (embed every N frames)
- Applies same‑frame uniqueness constraint
- Saves crops under global IDs
- Writes annotated video without showing it
- Shows progress bar with tqdm
"""

import sys, time
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from tqdm import tqdm

import config
from modules.detector_tracker import YOLOTracker
from modules.reid_embedder import ReIDEmbedder
from modules.global_id_manager import GlobalIDManager


# ---------- Quality gate (ReID strict thresholds) ----------
def reid_quality_ok(crop, bbox, fw, fh):
    bw = bbox[2] - bbox[0]
    bh = bbox[3] - bbox[1]
    if bw < config.MIN_REID_WIDTH or bh < config.MIN_REID_HEIGHT:
        return False
    area_ratio = (bw * bh) / (fw * fh)
    if area_ratio < config.MIN_REID_AREA_RATIO:
        return False
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if cv2.Laplacian(gray, cv2.CV_32F).var() < config.MIN_REID_SHARPNESS:
        return False
    return True


# ---------- Drawing helpers ----------
def id_to_bgr_color(id_val):
    hue = ((int(id_val) * 0.61803398875) % 1.0) * 179.0
    hsv = np.array([[[hue, 220, 255]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_ids(frame, det, global_id, local_id):
    x1, y1, x2, y2 = det["bbox"]
    color = id_to_bgr_color(global_id) if global_id else (0, 165, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text_y = y1 - 10 if y1 > 24 else y1 + 20
    cv2.putText(frame, f"L:{local_id}", (x1, text_y),
                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"G:{global_id}" if global_id else "G:?",
                (x1, text_y + 18), cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)


def main():
    video_path = config.INPUT_VIDEO
    output_video_path = "tracked_global_fixed.mp4"
    crop_base = Path(config.CROP_OUTPUT_DIR).parent / "step2_global_crops_fixed"
    crop_base.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    # Initialise modules
    tracker = YOLOTracker()
    embedder = ReIDEmbedder(model_path=config.OSNET_MODEL)
    manager = GlobalIDManager()
    manager.reset_for_video(fps)

    frame_idx = 0
    saved_total = 0
    saved_per_global = {}
    start_time = time.perf_counter()

    print(f"Processing {video_path} ({total_frames} frames) ...")
    pbar = tqdm(total=total_frames, unit="frame", desc="Step 2")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        manager.new_frame()   # reset same‑frame uniqueness set

        detections = tracker.track(frame, persist=True)
        annotated = frame.copy()

        for det in detections:
            local_id = det["local_id"]
            crop = det["crop"]
            bbox = det["bbox"]

            # Determine ReID quality
            ok = reid_quality_ok(crop, bbox, width, height)
            if ok:
                emb = embedder.extract_embedding(crop)
                color_sig = embedder.extract_color_signature(crop)
            else:
                emb = None
                color_sig = None

            gid, is_new, reason = manager.update(local_id, frame_idx,
                                                 emb, color_sig, bbox, reid_ok=ok)

            # Draw IDs on frame
            draw_ids(annotated, det, gid, local_id)

            # Save crop under global folder (if ReID quality ok OR we just want to save all)
            # We'll save any crop that passes the standard save quality gates (from config)
            if _should_save_crop(crop, bbox, width, height):
                if gid is not None:
                    global_dir = crop_base / f"person_{gid:03d}"
                    global_dir.mkdir(parents=True, exist_ok=True)
                    ts = frame_idx / fps
                    fname = f"frame_{frame_idx:06d}_t{ts:.3f}_l{local_id}.jpg"
                    cv2.imwrite(str(global_dir / fname), crop,
                                [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
                    saved_total += 1
                    saved_per_global[gid] = saved_per_global.get(gid, 0) + 1

        # Cleanup stale local mappings
        manager.cleanup_stale(frame_idx)

        out.write(annotated)
        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()

    elapsed = time.perf_counter() - start_time
    print("\n--- Step 2 Experiment Summary ---")
    print(f"Frames: {frame_idx}, Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Global identities created: {manager.next_id - 1}")
    print(f"Total crops saved: {saved_total}")
    if saved_per_global:
        print("Crops per global identity (top 10):")
        for gid in sorted(saved_per_global, key=saved_per_global.get, reverse=True)[:10]:
            print(f"  person_{gid:03d}: {saved_per_global[gid]} crops")
    print(f"Annotated video: {output_video_path}")
    print(f"Crop folders: {crop_base}")


def _should_save_crop(crop, bbox, fw, fh):
    """Standard save quality gates (from config)."""
    bw = bbox[2] - bbox[0]
    bh = bbox[3] - bbox[1]
    if bw < config.MIN_WIDTH or bh < config.MIN_HEIGHT:
        return False
    area_ratio = (bw * bh) / (fw * fh)
    if area_ratio < config.MIN_AREA_RATIO:
        return False
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if cv2.Laplacian(gray, cv2.CV_32F).var() < config.MIN_SHARPNESS:
        return False
    return True


if __name__ == "__main__":
    main()