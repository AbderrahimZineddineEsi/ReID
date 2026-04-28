#!/usr/bin/env python3
"""
Full pipeline: from input video to final refined-identity visualization.

Usage:
    python scripts/full_pipeline.py --input assets/my_video.mp4 --output-dir outputs/my_test --final-video final.mp4

If the intermediate step2 or refined folders already exist, they will be reused (add --force to redo).
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from modules.detector_tracker import YOLOTracker
from modules.reid_embedder import ReIDEmbedder
from modules.global_id_manager import GlobalIDManager

import cv2
import numpy as np
from tqdm import tqdm

# We'll also need offline refinement and visualization functions
from scripts.offline_refine_v2 import (
    load_osnet_onnx,
    is_good_crop,
    TrackCluster,
    split_folder_tracks,
    cross_folder_merge,
    merge_clusters,
    FILENAME_RE as REFINE_RE,
    extract_embedding as refine_extract_emb,
    extract_color_signature as refine_extract_color,
    color_similarity as refine_color_sim,
)
from experiments.visualize_refined import build_identity_map, id_to_bgr_color


def create_tracker_with_fallback() -> YOLOTracker:
    """Create YOLO tracker with graceful fallback to CPU when CUDA is unavailable."""
    try:
        return YOLOTracker()
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "cuda" not in msg:
            raise
        print("[WARN] CUDA tracker init failed. Falling back to CPU for YOLO tracking.")
        return YOLOTracker(device="cpu")

# ---------- Online Step2 helper ----------
def run_step2_online(video_path: Path, output_crops_dir: Path, force: bool = False):
    """
    Run YOLO + ByteTrack + GlobalIDManager to produce person_* folders.
    Returns the directory where crops were saved.
    """
    if output_crops_dir.exists() and any(output_crops_dir.iterdir()):
        if not force:
            print(f"Step2 output already exists at {output_crops_dir}, skipping. Use --force to re-run.")
            return output_crops_dir
        else:
            shutil.rmtree(output_crops_dir)
    output_crops_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tracker = create_tracker_with_fallback()
    embedder = ReIDEmbedder(model_path=config.OSNET_MODEL)
    manager = GlobalIDManager()
    manager.reset_for_video(fps)

    frame_idx = 0
    pbar = tqdm(total=total_frames, desc="Step2 online linking")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        manager.new_frame()

        detections = tracker.track(frame, persist=True)
        for det in detections:
            local_id = det["local_id"]
            crop = det["crop"]
            x1, y1, x2, y2 = det["bbox"]

            # Quality gate for ReID
            bw, bh = x2 - x1, y2 - y1
            area_ratio = (bw * bh) / (width * height)
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            sharp = cv2.Laplacian(gray, cv2.CV_32F).var()
            reid_ok = (bw >= config.MIN_REID_WIDTH and bh >= config.MIN_REID_HEIGHT and
                       area_ratio >= config.MIN_REID_AREA_RATIO and sharp >= config.MIN_REID_SHARPNESS)

            if reid_ok:
                emb = embedder.extract_embedding(crop)
                color_sig = embedder.extract_color_signature(crop)
                gid, _, _ = manager.update(local_id, frame_idx, emb, color_sig, (x1, y1, x2, y2), reid_ok=True)
            else:
                gid, _, _ = manager.update(local_id, frame_idx, None, None, (x1, y1, x2, y2), reid_ok=False)

            # Save crop if it passes save quality (using standard config)
            if (bw >= config.MIN_WIDTH and bh >= config.MIN_HEIGHT and
                area_ratio >= config.MIN_AREA_RATIO and sharp >= config.MIN_SHARPNESS):
                person_dir = output_crops_dir / f"person_{gid:03d}"
                person_dir.mkdir(parents=True, exist_ok=True)
                ts = frame_idx / fps
                fname = f"frame_{frame_idx:06d}_t{ts:.3f}_l{local_id}.jpg"
                cv2.imwrite(str(person_dir / fname), crop,
                            [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])

        manager.cleanup_stale(frame_idx)
        pbar.update(1)
    pbar.close()
    cap.release()
    print(f"Step2 done. {manager.next_id - 1} global identities, crops saved to {output_crops_dir}")
    return output_crops_dir


# ---------- Offline refinement wrapper ----------
def run_offline_refinement(input_dir: Path, output_dir: Path, force: bool = False):
    """
    Run the offline refinement script's main logic on the given input directory.
    """
    if output_dir.exists() and any(output_dir.iterdir()):
        if not force:
            print(f"Refined output already exists at {output_dir}, skipping. Use --force to re-run.")
            return output_dir
        else:
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load ONNX model
    reid_path = Path(config.OSNET_MODEL)
    session, input_name = load_osnet_onnx(reid_path)

    # Frame dimensions (dummy, adjust if you store metadata)
    frame_w, frame_h = 100000, 100000

    # Collect folders
    folders = sorted([d for d in input_dir.iterdir() if d.is_dir() and d.name.startswith("person_")],
                     key=lambda d: d.name)
    if not folders:
        print("No person folders found for refinement.")
        return output_dir

    # Stage A & B
    all_clusters = []
    for folder in tqdm(folders, desc="Intra-folder processing"):
        images = sorted([p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        if not images:
            continue
        tracks = {}
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            # approximate bbox size (crop size)
            if not is_good_crop(img, w, h, frame_w, frame_h):
                continue
            m = REFINE_RE.search(img_path.stem)
            local_id = int(m.group(2)) if m else -1
            frame = int(m.group(1)) if m else -1
            emb = refine_extract_emb(img, session, input_name, config.REID_SIZE)
            up, lo = refine_extract_color(img)
            if local_id not in tracks:
                tracks[local_id] = TrackCluster(local_id)
            tracks[local_id].add(img_path, emb, up, lo, frame)

        # Build prototypes
        for c in tracks.values():
            c.build_prototype()
        clusters_from_folder = split_folder_tracks(tracks, verbose=False)
        all_clusters.extend(clusters_from_folder)

    print(f"After intra-folder splitting: {len(all_clusters)} clusters. Cross-folder merging...")
    final_clusters = cross_folder_merge(all_clusters, verbose=False)

    # Write output
    final_clusters.sort(key=lambda c: c.size(), reverse=True)
    out_idx = 0
    for cluster in final_clusters:
        if cluster.size() < config.MIN_FOLDER_IMAGES_FINAL:
            continue
        out_idx += 1
        person_dir = output_dir / f"person_{out_idx:03d}"
        person_dir.mkdir()
        for src in cluster.paths:
            dest = person_dir / src.name
            if dest.exists():
                i = 1
                while (person_dir / f"{src.stem}_{i}{src.suffix}").exists():
                    i += 1
                dest = person_dir / f"{src.stem}_{i}{src.suffix}"
            shutil.copy2(src, dest)
    print(f"Refinement complete. {out_idx} final folders written to {output_dir}")
    return output_dir


# ---------- Visualization ----------
def visualize_refined(refined_dir: Path, video_path: Path, output_video: Path):
    mapping = build_identity_map(refined_dir)
    print(f"Loaded {len(set(mapping.values()))} refined persons for visualization.")

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    tracker = create_tracker_with_fallback()
    frame_idx = 0
    pbar = tqdm(total=total, desc="Visualizing")
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
            person_id = mapping.get((frame_idx, local_id))
            if person_id is not None:
                color = id_to_bgr_color(person_id)
                label = f"P{person_id}"
            else:
                color = (128, 128, 128)
                label = "?"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, label, (x1, max(y1-10, 20)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, color, 1, cv2.LINE_AA)
        out.write(annotated)
        pbar.update(1)
    pbar.close()
    cap.release()
    out.release()
    print(f"Visualization video saved to {output_video}")


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="Full ReID pipeline (video -> refined identities)")
    parser.add_argument("--input", required=True, help="Input video file")
    parser.add_argument("--output-dir", required=True, help="Base output directory")
    parser.add_argument("--final-video", default="final_pipeline_output.mp4", help="Output visualization video")
    parser.add_argument("--force", action="store_true", help="Rerun steps even if outputs exist")
    args = parser.parse_args()

    input_video = Path(args.input)
    if not input_video.exists():
        raise FileNotFoundError(f"Video not found: {input_video}")

    base_out = Path(args.output_dir)
    base_out.mkdir(parents=True, exist_ok=True)

    step2_dir = base_out / "step2_crops"
    refined_dir = base_out / "refined"
    final_video_path = base_out / args.final_video

    start = time.time()

    # Step 1: Online linking (Step2)
    print("="*60)
    print("STEP 2: Online tracking + global ID linking")
    step2_dir = run_step2_online(input_video, step2_dir, force=args.force)

    # Step 2: Offline refinement
    print("="*60)
    print("OFFLINE REFINEMENT (Pass 2)")
    refined_dir = run_offline_refinement(step2_dir, refined_dir, force=args.force)

    # Step 3: Visualization
    print("="*60)
    print("VISUALIZATION")
    visualize_refined(refined_dir, input_video, final_video_path)

    elapsed = time.time() - start
    print(f"\nFull pipeline completed in {elapsed/60:.1f} minutes.")
    print(f"Final video: {final_video_path}")


if __name__ == "__main__":
    main()