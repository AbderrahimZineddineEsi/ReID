#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("is_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))


# # Full Re‑ID Pipeline – Transparent Run
# This notebook runs the complete pipeline **step‑by‑step** and prints every important decision:  
# detections → track counting → crop quality filtering → online global linking → offline refinement (split/merge) → final visualisation.
# 
# **First, set your paths below** and then execute all cells in order.

# In[2]:


# ========== EDIT THESE TWO STRINGS ==========
INPUT_VIDEO = "../assets/store_cam3.mp4"          # <-- your video file
OUTPUT_DIR  = "./outputs/test3_large_200"        # <-- where to save everything
# ============================================


# In[3]:


import sys, time, warnings, shutil, os, re
from pathlib import Path
from collections import defaultdict
import numpy as np
import cv2
import torch
from tqdm.notebook import tqdm

# Project root
PROJECT_ROOT = Path(os.getcwd())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

def print_config_summary():
    print("=== Config values in use ===")
    print(f"OSNET_MODEL: {config.OSNET_MODEL}")
    print("Step2 (online) quality gates:")
    print(f"  MIN_REID_WIDTH={config.MIN_REID_WIDTH}, MIN_REID_HEIGHT={config.MIN_REID_HEIGHT}")
    print(f"  MIN_REID_AREA_RATIO={config.MIN_REID_AREA_RATIO}, MIN_REID_SHARPNESS={config.MIN_REID_SHARPNESS}")
    print("Save gates (crop saving):")
    print(f"  MIN_WIDTH={config.MIN_WIDTH}, MIN_HEIGHT={config.MIN_HEIGHT}")
    print(f"  MIN_AREA_RATIO={config.MIN_AREA_RATIO}, MIN_SHARPNESS={config.MIN_SHARPNESS}")
    print("Stage A (offline quality gates):")
    print(f"  REFINE_MIN_REID_AREA_RATIO={config.REFINE_MIN_REID_AREA_RATIO}")
    print(f"  REFINE_MIN_REID_SHARPNESS={config.REFINE_MIN_REID_SHARPNESS}")
    print(f"  REFINE_MAX_ASPECT_RATIO={config.REFINE_MAX_ASPECT_RATIO}")
    print("Stage B (split thresholds):")
    print(f"  MIN_TRACK_SIZE={config.MIN_TRACK_SIZE}")
    print(f"  SPLIT_REID_THRESHOLD={config.SPLIT_REID_THRESHOLD}")
    print(f"  SPLIT_COLOR_THRESHOLD={config.SPLIT_COLOR_THRESHOLD}")
    print("Stage C (merge thresholds):")
    print(f"  MERGE_REID_THRESHOLD={config.MERGE_REID_THRESHOLD}")
    print(f"  MERGE_COLOR_THRESHOLD={config.MERGE_COLOR_THRESHOLD}")
    print(f"  MERGE_COLOR_WEIGHT={config.MERGE_COLOR_WEIGHT}")
    print(f"  COHERENCE_MIN={config.COHERENCE_MIN}")
    print(f"  MAX_SAME_FRAME_OVERLAP={config.MAX_SAME_FRAME_OVERLAP}")
    print(f"  MIN_FOLDER_IMAGES_FINAL={config.MIN_FOLDER_IMAGES_FINAL}")
    print("Large-folder relaxations:")
    print(f"  LARGE_FOLDER_SIZE={config.LARGE_FOLDER_SIZE}")
    print(f"  LARGE_MERGE_REID_THRESHOLD={config.LARGE_MERGE_REID_THRESHOLD}")
    print(f"  LARGE_MERGE_COLOR_THRESHOLD={config.LARGE_MERGE_COLOR_THRESHOLD}")

print_config_summary()

from modules.detector_tracker import YOLOTracker
from modules.reid_embedder import ReIDEmbedder
from modules.global_id_manager import GlobalIDManager

# Notebook-specific output folders
INPUT_VIDEO_PATH = Path(INPUT_VIDEO)
OUTPUT_BASE = Path(OUTPUT_DIR)
STEP2_CROPS = OUTPUT_BASE / "step2_crops"
REFINED_DIR  = OUTPUT_BASE / "refined"
FINAL_VIDEO  = OUTPUT_BASE / "final_visualization.mp4"

# Create output directories
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
print(f"Input video  : {INPUT_VIDEO_PATH}")
print(f"Output folder: {OUTPUT_BASE}")


# In[4]:


# Print virtual environment info
venv_env = os.environ.get('VIRTUAL_ENV') or os.environ.get('CONDA_PREFIX')
sys_exec = sys.executable
sys_pref = sys.prefix
base_pref = getattr(sys, "base_prefix", None)
project_venv = PROJECT_ROOT / ".venv"

print("sys.executable:", sys_exec)
print("sys.prefix   :", sys_pref)
print("sys.base_prefix:", base_pref)
print("VIRTUAL_ENV / CONDA_PREFIX:", venv_env)
print("PROJECT_ROOT/.venv exists?:", project_venv.exists(), "->", str(project_venv.resolve()) if project_venv.exists() else str(project_venv))

# Is the current python executable located inside PROJECT_ROOT/.venv?
try:
    in_project_venv = project_venv.resolve() in Path(sys_exec).resolve().parents
except Exception:
    in_project_venv = False
print("sys.executable inside PROJECT_ROOT/.venv?:", in_project_venv)


# In[5]:


# Check GPU availability
print("PyTorch version:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("CUDA device count:", torch.cuda.device_count())
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
else:
    print("WARNING: GPU not available – everything will run on CPU.")
    print("Tip: if you changed CUDA_VISIBLE_DEVICES, restart the kernel.")


# In[6]:


# Load video info
cap = cv2.VideoCapture(str(INPUT_VIDEO_PATH))
if not cap.isOpened():
    raise IOError(f"Cannot open video: {INPUT_VIDEO_PATH}")
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

print(f"Video: {INPUT_VIDEO_PATH.name}")
print(f"Frames: {total_frames}, FPS: {fps:.2f}, Resolution: {width}x{height}")
print(f"Duration: {total_frames/fps:.1f}s")


# ## Step 1 – YOLO + ByteTrack (Local Tracking)
# We only run the tracker to see how many local tracks appear. No ReID yet.

# In[7]:


tracker = YOLOTracker()
cap = cv2.VideoCapture(str(INPUT_VIDEO_PATH))

local_tracks_all = defaultdict(list)   # local_id -> list of frame indices
frame_idx = 0
with tqdm(total=total_frames, desc="Tracking") as pbar:
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1
        dets = tracker.track(frame, persist=True)
        for d in dets:
            local_tracks_all[d['local_id']].append(frame_idx)
        pbar.update(1)
cap.release()

print(f"\nTotal frames processed: {frame_idx}")
print(f"Distinct local track IDs: {len(local_tracks_all)}")
# Print some track lengths
track_lens = sorted([(lid, len(fr)) for lid, fr in local_tracks_all.items()], key=lambda x: x[1], reverse=True)
print("Top 10 longest local tracks:")
for lid, cnt in track_lens[:10]:
    print(f"  L{lid}: {cnt} frames")


# ## Step 2 – Online Global ID Linking (with ReID and Colour)
# Now we run the full online pipeline.  
# **Quality gates** are applied: only crops passing `MIN_REID_AREA_RATIO` and `MIN_REID_SHARPNESS` are used for ReID.  
# The `GlobalIDManager` decides whether a detection gets a new global ID or is linked to an existing one.  
# We save the crops under `person_XXX` folders.

# In[8]:


STEP2_CROPS.mkdir(parents=True, exist_ok=True)

tracker2 = YOLOTracker()
embedder = ReIDEmbedder(model_path=config.OSNET_MODEL)
manager = GlobalIDManager()
manager.reset_for_video(fps)

cap = cv2.VideoCapture(str(INPUT_VIDEO_PATH))
frame_idx = 0
total_saved = 0
saved_per_global = defaultdict(int)
filtered_reid = 0  # count of crops that failed ReID quality gates

print("Processing video with online global linking ...")
pbar = tqdm(total=total_frames, desc="Step2 linking")

while True:
    ret, frame = cap.read()
    if not ret: break
    frame_idx += 1
    manager.new_frame()
    dets = tracker2.track(frame, persist=True)
    for det in dets:
        lid = det['local_id']
        crop = det['crop']
        x1,y1,x2,y2 = det['bbox']
        bw, bh = x2-x1, y2-y1
        area_ratio = (bw*bh)/(width*height)
        sharp = cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_32F).var()
        reid_ok = (bw >= config.MIN_REID_WIDTH and bh >= config.MIN_REID_HEIGHT and
                   area_ratio >= config.MIN_REID_AREA_RATIO and sharp >= config.MIN_REID_SHARPNESS)
        if reid_ok:
            emb = embedder.extract_embedding(crop)
            color_sig = embedder.extract_color_signature(crop)
            gid, is_new, reason = manager.update(lid, frame_idx, emb, color_sig, (x1,y1,x2,y2), reid_ok=True)
        else:
            filtered_reid += 1
            gid, is_new, reason = manager.update(lid, frame_idx, None, None, (x1,y1,x2,y2), reid_ok=False)
        # Save crop if it passes the standard save quality gates
        save_ok = (bw >= config.MIN_WIDTH and bh >= config.MIN_HEIGHT and
                   area_ratio >= config.MIN_AREA_RATIO and sharp >= config.MIN_SHARPNESS)
        if save_ok:
            person_dir = STEP2_CROPS / f"person_{gid:03d}"
            person_dir.mkdir(parents=True, exist_ok=True)
            ts = frame_idx / fps
            fname = f"frame_{frame_idx:06d}_t{ts:.3f}_l{lid}.jpg"
            cv2.imwrite(str(person_dir / fname), crop,
                        [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
            total_saved += 1
            saved_per_global[gid] += 1
    manager.cleanup_stale(frame_idx)
    pbar.update(1)

pbar.close()
cap.release()

# Summary
print(f"\n--- Step2 Summary ---")
print(f"Global identities created: {manager.next_id - 1}")
print(f"Total crops saved: {total_saved}")
print(f"Crops discarded by ReID quality gate: {filtered_reid}")
print("Crops per global ID (top 10):")
for gid in sorted(saved_per_global, key=saved_per_global.get, reverse=True)[:10]:
    print(f"  person_{gid:03d}: {saved_per_global[gid]} crops")


# ## Step 3 – Offline Refinement (Pass 2)
# This stage:
# 1. Filters bad crops again (Stage A).
# 2. Splits folders that contain multiple people (Stage B).
# 3. Merges folders that belong to the same person (Stage C).
# 
# Every decision is printed below.

# In[9]:


from scripts.offline_refine_v2 import (
    load_osnet_onnx, is_good_crop, TrackCluster,
    split_folder_tracks, cross_folder_merge, merge_clusters,
    extract_embedding as refine_extract_emb,
    extract_color_signature as refine_extract_color,
    FILENAME_RE as REFINE_RE,
)

# Prepare output
if REFINED_DIR.exists():
    shutil.rmtree(REFINED_DIR)
REFINED_DIR.mkdir()

# Load ReID model
reid_path = Path(config.OSNET_MODEL)
sess, iname = load_osnet_onnx(reid_path)

# We'll use dummy frame dims for area ratio (the other filters are enough)
frame_w, frame_h = 100000, 100000

# Gather folders
folders = sorted([d for d in STEP2_CROPS.iterdir() if d.is_dir() and d.name.startswith("person_")],
                 key=lambda d: d.name)
print(f"Input folders to refine: {len(folders)}")


# In[10]:


# ---- Stage A & B: Intra-folder processing with verbose output ----
all_clusters = []

for folder in tqdm(folders, desc="Intra-folder"):
    images = sorted([p for p in folder.iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png"}])
    original_count = len(images)
    if original_count == 0:
        continue

    # Stage A filter
    tracks = defaultdict(lambda: TrackCluster(-1))
    filtered_out = 0
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        if not is_good_crop(img, w, h, frame_w, frame_h):
            filtered_out += 1
            continue
        m = REFINE_RE.search(img_path.stem)
        local_id = int(m.group(2)) if m else -1
        frame_no = int(m.group(1)) if m else -1
        emb = refine_extract_emb(img, sess, iname, config.REID_SIZE)
        up, lo = refine_extract_color(img)
        tracks[local_id].add(img_path, emb, up, lo, frame_no)

    print(f"\n{'='*60}")
    print(f"Folder: {folder.name} | original images: {original_count} | kept after quality filter: {original_count - filtered_out}")
    print(f"Number of local tracks: {len(tracks)}")

    # Build prototypes and show track sizes
    for lid, tc in tracks.items():
        tc.build_prototype()
        coh = tc.coherence if tc.coherence is not None else float("nan")
        print(f"  L{lid}: {tc.size()} images, coherence {coh:.3f}")

    print("Split thresholds:")
    print(f"  MIN_TRACK_SIZE={config.MIN_TRACK_SIZE}")
    print(f"  SPLIT_REID_THRESHOLD={config.SPLIT_REID_THRESHOLD}")
    print(f"  SPLIT_COLOR_THRESHOLD={config.SPLIT_COLOR_THRESHOLD}")

    # Show which tracks are considered tiny vs large (tiny are discarded)
    large_ids = [lid for lid, tc in tracks.items() if tc.size() >= config.MIN_TRACK_SIZE]
    tiny_ids = [lid for lid, tc in tracks.items() if tc.size() < config.MIN_TRACK_SIZE]
    if tiny_ids:
        preview = tiny_ids[:20]
        suffix = " ..." if len(tiny_ids) > 20 else ""
        print(f"  Tiny tracks (<{config.MIN_TRACK_SIZE}) discarded: {preview}{suffix}")
    else:
        print(f"  Tiny tracks (<{config.MIN_TRACK_SIZE}) discarded: none")

    # Pairwise best-match summary to explain split decisions
    if len(tracks) > 1:
        track_items = list(tracks.items())
        for lid, tc in track_items:
            best = None
            for other_lid, other in track_items:
                if other_lid == lid:
                    continue
                reid, col, _ = tc.similarity_to(other)
                if best is None or reid + col > best[0]:
                    best = (reid + col, other_lid, reid, col)
            if best is not None:
                _, other_lid, reid, col = best
                passes = (reid >= config.SPLIT_REID_THRESHOLD and col >= config.SPLIT_COLOR_THRESHOLD)
                print(f"  Best match for L{lid} -> L{other_lid}: reid={reid:.3f}, color={col:.3f}, pass={passes}")

    # Stage B – intra-folder splitting (verbose)
    clusters_from_folder = split_folder_tracks(dict(tracks), verbose=True)
    print(f"Result: {len(clusters_from_folder)} cluster(s) after intra-folder split.")
    all_clusters.extend(clusters_from_folder)

print(f"\nTotal clusters before cross-folder merge: {len(all_clusters)}")


# In[11]:


# ---- Stage C: Cross-folder merging with verbose output ----

def debug_cross_folder_merge(all_clusters, verbose=True, max_compare_print=5):
    # Sort by size so largest, most reliable clusters act as anchors
    sorted_clusters = sorted(all_clusters, key=lambda c: c.size(), reverse=True)
    final_clusters = []

    for cluster in tqdm(sorted_clusters, desc="Cross-folder merging"):
        coh = cluster.coherence
        if coh is not None and coh < config.COHERENCE_MIN:
            if verbose:
                print(f"\nCluster size {cluster.size()} coherence {coh:.3f} < {config.COHERENCE_MIN} -> kept separate")
            final_clusters.append(cluster)
            continue

        compare_records = []
        best_idx = -1
        best_score = -1.0
        best_detail = None
        best_fail = None

        for i, final in enumerate(final_clusters):
            overlap = len(cluster.frame_set().intersection(final.frame_set()))
            if overlap > config.MAX_SAME_FRAME_OVERLAP:
                compare_records.append({
                    "i": i,
                    "size": final.size(),
                    "overlap": overlap,
                    "status": "overlap"
                })
                continue

            reid, col, score = cluster.similarity_to(final)
            size_a = cluster.size()
            size_b = final.size()
            if size_a >= config.LARGE_FOLDER_SIZE or size_b >= config.LARGE_FOLDER_SIZE:
                req_reid = config.LARGE_MERGE_REID_THRESHOLD
                req_col = config.LARGE_MERGE_COLOR_THRESHOLD
                mode = "LARGE"
            else:
                req_reid = config.MERGE_REID_THRESHOLD
                req_col = config.MERGE_COLOR_THRESHOLD
                mode = "NORMAL"

            passed = (reid >= req_reid and col >= req_col)
            compare_records.append({
                "i": i,
                "size": final.size(),
                "overlap": overlap,
                "reid": reid,
                "col": col,
                "score": score,
                "req_reid": req_reid,
                "req_col": req_col,
                "mode": mode,
                "passed": passed
            })
            if best_fail is None or score > best_fail["score"]:
                best_fail = compare_records[-1]
            if passed and score > best_score:
                best_score = score
                best_idx = i
                best_detail = compare_records[-1]

        if verbose:
            coh_str = f"{coh:.3f}" if coh is not None else "n/a"
            print(f"\nCluster size {cluster.size()} coherence {coh_str} | candidates: {len(final_clusters)}")
            if compare_records:
                def sort_key(r):
                    return r.get("score", -1.0)
                for r in sorted(compare_records, key=sort_key, reverse=True)[:max_compare_print]:
                    if r.get("status") == "overlap":
                        print(f"  vs #{r['i']} size={r['size']} overlap={r['overlap']} > {config.MAX_SAME_FRAME_OVERLAP} -> skip")
                    else:
                        print(
                            f"  vs #{r['i']} size={r['size']} overlap={r['overlap']} "
                            f"reid={r['reid']:.3f} col={r['col']:.3f} score={r['score']:.3f} "
                            f"req_reid={r['req_reid']:.3f} req_col={r['req_col']:.3f} "
                            f"mode={r['mode']} pass={r['passed']}"
                        )

        if best_idx >= 0:
            final_clusters[best_idx] = merge_clusters(final_clusters[best_idx], cluster)
            if verbose and best_detail:
                print(
                    f"  -> MERGED into #{best_idx} | mode={best_detail['mode']} "
                    f"reid={best_detail['reid']:.3f} col={best_detail['col']:.3f} "
                    f"req_reid={best_detail['req_reid']:.3f} req_col={best_detail['req_col']:.3f} "
                    f"score={best_detail['score']:.3f}"
                )
        else:
            final_clusters.append(cluster)
            if verbose:
                if best_fail:
                    reasons = []
                    if best_fail["reid"] < best_fail["req_reid"]:
                        reasons.append("reid<req_reid")
                    if best_fail["col"] < best_fail["req_col"]:
                        reasons.append("col<req_col")
                    reason_text = ",".join(reasons) if reasons else "no_threshold_failure"
                    print(
                        f"  -> NEW cluster | best candidate #{best_fail['i']} mode={best_fail['mode']} "
                        f"reid={best_fail['reid']:.3f} col={best_fail['col']:.3f} "
                        f"req_reid={best_fail['req_reid']:.3f} req_col={best_fail['req_col']:.3f} "
                        f"fail={reason_text}"
                    )
                else:
                    print("  -> NEW cluster | reason: no candidates (all overlap or none yet)")

    return final_clusters

final_clusters = debug_cross_folder_merge(all_clusters, verbose=True)
print(f"\nClusters after merging: {len(final_clusters)}")


# In[12]:


# ---- Write final refined folders ----
final_clusters.sort(key=lambda c: c.size(), reverse=True)
out_idx = 0
for cluster in final_clusters:
    if cluster.size() < config.MIN_FOLDER_IMAGES_FINAL:
        print(f"Dropping cluster of size {cluster.size()} (min required: {config.MIN_FOLDER_IMAGES_FINAL})")
        continue
    out_idx += 1
    person_dir = REFINED_DIR / f"person_{out_idx:03d}"
    person_dir.mkdir()
    for src in cluster.paths:
        dest = person_dir / src.name
        if dest.exists():
            i = 1
            while (person_dir / f"{src.stem}_{i}{src.suffix}").exists():
                i += 1
            dest = person_dir / f"{src.stem}_{i}{src.suffix}"
        shutil.copy2(src, dest)

print(f"Final refined persons: {out_idx}")


# ## Final Visualisation
# We replay the video and draw bounding boxes with the **refined person IDs**.

# In[13]:


# Build identity map from refined folders
identity_map = {}
for folder in sorted(REFINED_DIR.iterdir()):
    if not folder.is_dir() or not folder.name.startswith("person_"):
        continue
    pid = int(folder.name.split("_")[1])
    for img_file in folder.iterdir():
        if img_file.suffix.lower() not in {".jpg",".jpeg",".png"}:
            continue
        m = REFINE_RE.search(img_file.stem)
        if m:
            frame_no = int(m.group(1))
            local_id = int(m.group(2))
            identity_map[(frame_no, local_id)] = pid

print(f"Loaded {len(set(identity_map.values()))} distinct refined identities for visualisation.")


# In[ ]:


# Visualization
cap = cv2.VideoCapture(str(INPUT_VIDEO_PATH))
out_video = cv2.VideoWriter(str(FINAL_VIDEO), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
tracker_vis = YOLOTracker()

def id_to_color(id_val):
    hue = ((id_val * 0.61803398875) % 1.0) * 179.0
    hsv = np.array([[[hue, 220, 255]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0,0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])

frame_idx = 0
pbar = tqdm(total=total_frames, desc="Visualizing")
while True:
    ret, frame = cap.read()
    if not ret: break
    frame_idx += 1
    dets = tracker_vis.track(frame, persist=True)
    annotated = frame.copy()
    for d in dets:
        x1,y1,x2,y2 = d['bbox']
        lid = d['local_id']
        pid = identity_map.get((frame_idx, lid))
        if pid is not None:
            color = id_to_color(pid)
            label = f"P{pid}"
        else:
            color = (128,128,128)
            label = "?"
        cv2.rectangle(annotated, (x1,y1), (x2,y2), color, 2)
        cv2.putText(annotated, label, (x1, max(y1-10, 20)),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, color, 1, cv2.LINE_AA)
    out_video.write(annotated)
    pbar.update(1)
pbar.close()
cap.release()
out_video.release()
print(f"Visualisation saved to {FINAL_VIDEO}")


# ## Total Pipeline Time
# *(Times printed above are per stage; total is displayed after all cells.)* 
