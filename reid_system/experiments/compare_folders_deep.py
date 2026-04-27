#!/usr/bin/env python3
"""
Deep folder comparison – shows exactly why two folders were (or were not) merged.

Usage:
    python experiments/compare_folders_deep.py --folder1 outputs/step2_global_crops_fixed/person_001 --folder2 person_002
"""

import argparse, re, sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import cv2
import numpy as np
import onnxruntime as ort

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FILENAME_RE = re.compile(r"frame_(\d+).*_l(\d+)")

def load_osnet():
    sess = ort.InferenceSession(str(config.OSNET_MODEL), providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    return sess, iname

def preprocess_img(img):
    h, w = (256, 128)
    img = cv2.resize(img, (w, h))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = np.transpose(rgb, (2, 0, 1))
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3,1,1)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3,1,1)
    return (rgb - mean) / std

def extract_emb(img, sess, iname):
    tensor = preprocess_img(img)[np.newaxis, ...]
    emb = sess.run(None, {iname: tensor})[0][0]
    return emb / (np.linalg.norm(emb)+1e-12)

def extract_color_sig(img):
    resized = cv2.resize(img, (64, 128))
    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
    split = lab.shape[0] // 2
    upper = lab[:split, :, :]
    lower = lab[split:, :, :]
    hist_u = cv2.calcHist([upper], [0,1,2], None, [8,8,8],
                          [0,256,0,256,0,256]).flatten().astype(np.float32)
    hist_l = cv2.calcHist([lower], [0,1,2], None, [8,8,8],
                          [0,256,0,256,0,256]).flatten().astype(np.float32)
    hist_u /= (hist_u.sum()+1e-12)
    hist_l /= (hist_l.sum()+1e-12)
    return hist_u, hist_l

def color_similarity(u1, l1, u2, l2):
    up = float(np.minimum(u1, u2).sum())
    lo = float(np.minimum(l1, l2).sum())
    return 0.4*up + 0.6*lo

def summarize_tracks(track_buckets):
    track_protos = {}
    track_stats = {}
    for tid, data in track_buckets.items():
        if not data["embs"]:
            continue
        embs = np.vstack(data["embs"])
        proto_emb = embs.mean(axis=0)
        proto_emb /= (np.linalg.norm(proto_emb) + 1e-12)
        proto_u = np.mean(data["uhist"], axis=0)
        proto_l = np.mean(data["lhist"], axis=0)
        proto_u /= (proto_u.sum() + 1e-12)
        proto_l /= (proto_l.sum() + 1e-12)
        sims = np.dot(embs, proto_emb)
        track_protos[tid] = (proto_emb, proto_u, proto_l)
        track_stats[tid] = {
            "count": len(data["embs"]),
            "mean": float(sims.mean()),
            "min": float(sims.min()),
            "max": float(sims.max()),
            "frame_min": int(min(data["frames"])) if data["frames"] else -1,
            "frame_max": int(max(data["frames"])) if data["frames"] else -1,
            "frames": set(data["frames"]),
        }
    return track_protos, track_stats

def load_folder(folder_path, sess, iname):
    folder = Path(folder_path)
    images = sorted([p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT])
    paths = []
    embs = []
    u_hist = []
    l_hist = []
    frames = []
    track_buckets: Dict[int, dict] = defaultdict(
        lambda: {"embs": [], "uhist": [], "lhist": [], "frames": []}
    )
    for imp in images:
        m = FILENAME_RE.search(imp.stem)
        f = int(m.group(1)) if m else -1
        local_id = int(m.group(2)) if m else -1
        img = cv2.imread(str(imp))
        if img is None:
            continue
        emb = extract_emb(img, sess, iname)
        u, l = extract_color_sig(img)
        paths.append(imp)
        embs.append(emb)
        u_hist.append(u)
        l_hist.append(l)
        frames.append(f)
        track_buckets[local_id]["embs"].append(emb)
        track_buckets[local_id]["uhist"].append(u)
        track_buckets[local_id]["lhist"].append(l)
        track_buckets[local_id]["frames"].append(f)
    if embs:
        embs = np.vstack(embs)
        proto_emb = embs.mean(axis=0)
        proto_emb /= (np.linalg.norm(proto_emb)+1e-12)
        proto_u = np.mean(u_hist, axis=0)
        proto_l = np.mean(l_hist, axis=0)
        proto_u /= (proto_u.sum()+1e-12)
        proto_l /= (proto_l.sum()+1e-12)
    else:
        return None, None, None, None, None, None, None
    track_protos, track_stats = summarize_tracks(track_buckets)
    return paths, embs, (proto_emb, proto_u, proto_l), set(frames), (u_hist, l_hist), track_protos, track_stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder1", required=True)
    parser.add_argument("--folder2", required=True)
    args = parser.parse_args()

    sess, iname = load_osnet()
    paths1, embs1, proto1, frames1, _, track_protos1, track_stats1 = load_folder(args.folder1, sess, iname)
    paths2, embs2, proto2, frames2, _, track_protos2, track_stats2 = load_folder(args.folder2, sess, iname)
    if embs1 is None or embs2 is None:
        print("One of the folders has no valid images.")
        return

    # Compute metrics
    reid_sim = float(np.dot(proto1[0], proto2[0]))
    col_sim = color_similarity(proto1[1], proto1[2], proto2[1], proto2[2])
    overlap = len(frames1 & frames2)

    print(f"\nFolder 1: {len(embs1)} images")
    print(f"Folder 2: {len(embs2)} images")
    print(f"Prototype ReID similarity: {reid_sim:.3f}")
    print(f"Prototype Colour similarity: {col_sim:.3f}")
    print(f"Frame overlap: {overlap} (>0 forbids merge)")

    print("\nPer-track consistency in Folder 1:")
    for tid in sorted(track_stats1.keys()):
        s = track_stats1[tid]
        print(f"  L{tid}: {s['count']} images, mean similarity to its own prototype: {s['mean']:.3f} "
              f"(min: {s['min']:.3f}, max: {s['max']:.3f}), frames {s['frame_min']}-{s['frame_max']}")

    print("\nPer-track consistency in Folder 2:")
    for tid in sorted(track_stats2.keys()):
        s = track_stats2[tid]
        print(f"  L{tid}: {s['count']} images, mean similarity to its own prototype: {s['mean']:.3f} "
              f"(min: {s['min']:.3f}, max: {s['max']:.3f}), frames {s['frame_min']}-{s['frame_max']}")

    # Thresholds (from the offline second pass defaults)
    merge_reid_thresh = 0.85
    merge_color_thresh = 0.40
    max_overlap = 0

    # Check merge conditions
    can_merge = True
    if overlap > max_overlap:
        can_merge = False
        print("Merge BLOCKED: frame overlap exceeded (max=0).")
    if reid_sim < merge_reid_thresh:
        can_merge = False
        print(f"Merge BLOCKED: ReID {reid_sim:.3f} < threshold {merge_reid_thresh}")
    if col_sim < merge_color_thresh:
        can_merge = False
        print(f"Merge BLOCKED: Colour {col_sim:.3f} < threshold {merge_color_thresh}")
    if can_merge:
        print("All conditions satisfied → would MERGE.")

    # Track-by-track cross-folder comparison
    merge_reid_thresh = 0.85
    merge_color_thresh = 0.40
    print("\nTrack-by-track cross-folder similarity:")
    print(f"{'Track pair':<18s} {'ReID sim':>8s} {'Color sim':>9s} {'Frame ov':>8s} {'Verdict':>10s}")
    print("-" * 62)
    track_ids_1 = sorted(track_protos1.keys())
    track_ids_2 = sorted(track_protos2.keys())
    for t1 in track_ids_1:
        emb1, u1, l1 = track_protos1[t1]
        frames_t1 = track_stats1[t1]["frames"]
        for t2 in track_ids_2:
            emb2, u2, l2 = track_protos2[t2]
            frames_t2 = track_stats2[t2]["frames"]
            reid_t = float(np.dot(emb1, emb2))
            col_t = color_similarity(u1, l1, u2, l2)
            ov_t = len(frames_t1 & frames_t2)
            verdict = "MATCH" if (reid_t >= merge_reid_thresh and col_t >= merge_color_thresh) else "NO"
            print(f"L{t1} vs L{t2}      {reid_t:8.3f} {col_t:9.3f} {ov_t:8d} {verdict:>10s}")

    # Top image pairs
    cos_matrix = np.dot(embs1, embs2.T)
    flat_idx = np.argsort(cos_matrix.ravel())[::-1]
    print("\nTop 5 most similar image pairs:")
    for rank, idx in enumerate(flat_idx[:5]):
        i = idx // cos_matrix.shape[1]
        j = idx % cos_matrix.shape[1]
        sim = cos_matrix[i][j]
        print(f"  {paths1[i].name} <-> {paths2[j].name} : {sim:.3f}")

    # Also show range of similarities to get intuition
    print(f"\nPairwise similarities range: min={cos_matrix.min():.3f}, max={cos_matrix.max():.3f}, mean={cos_matrix.mean():.3f}")

if __name__ == "__main__":
    main()