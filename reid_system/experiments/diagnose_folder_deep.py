#!/usr/bin/env python3
"""
Deep folder diagnosis – shows exactly why tracks inside a folder were merged or split.

Usage:
    python experiments/diagnose_folder_deep.py --input outputs/step2_global_crops_fixed/person_001
"""

import argparse, re, sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FILENAME_RE = re.compile(r"frame_(\d+).*_l(\d+)")

# ---------- ReID & colour helpers ----------
def load_osnet():
    sess = ort.InferenceSession(str(config.OSNET_MODEL), providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    return sess, iname

def preprocess_img(img, size=(256, 128)):
    h, w = size
    img = cv2.resize(img, (w, h))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = np.transpose(rgb, (2, 0, 1))
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    return (rgb - mean) / std

def extract_emb(img, sess, iname):
    tensor = preprocess_img(img)[np.newaxis, ...]
    emb = sess.run(None, {iname: tensor})[0][0]
    return emb / (np.linalg.norm(emb) + 1e-12)

def extract_color_sig(img):
    resized = cv2.resize(img, (64, 128))
    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
    split = lab.shape[0] // 2
    upper = lab[:split, :, :]
    lower = lab[split:, :, :]
    hist_u = cv2.calcHist([upper], [0, 1, 2], None, [8, 8, 8],
                          [0, 256, 0, 256, 0, 256]).flatten().astype(np.float32)
    hist_l = cv2.calcHist([lower], [0, 1, 2], None, [8, 8, 8],
                          [0, 256, 0, 256, 0, 256]).flatten().astype(np.float32)
    hist_u /= (hist_u.sum() + 1e-12)
    hist_l /= (hist_l.sum() + 1e-12)
    return hist_u, hist_l

def color_similarity(u1, l1, u2, l2):
    up = float(np.minimum(u1, u2).sum())
    lo = float(np.minimum(l1, l2).sum())
    return 0.4 * up + 0.6 * lo

# ---------- Analysis ----------
def analyze_folder(folder_path, sess, iname):
    folder = Path(folder_path)
    images = sorted([p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT])
    if not images:
        print("No images found.")
        return

    tracks: Dict[int, dict] = defaultdict(lambda: {"paths": [], "embs": [], "uhist": [], "lhist": [], "frames": []})
    for imp in images:
        m = FILENAME_RE.search(imp.stem)
        local_id = int(m.group(2)) if m else -1
        frame = int(m.group(1)) if m else -1
        img = cv2.imread(str(imp))
        if img is None:
            continue
        emb = extract_emb(img, sess, iname)
        u, l = extract_color_sig(img)
        tracks[local_id]["paths"].append(imp)
        tracks[local_id]["embs"].append(emb)
        tracks[local_id]["uhist"].append(u)
        tracks[local_id]["lhist"].append(l)
        tracks[local_id]["frames"].append(frame)

    print(f"\n{'='*60}")
    print(f"Folder: {folder.name}")
    print(f"{'='*60}")
    print(f"Total images: {len(images)}")
    print(f"Distinct local tracks: {len(tracks)}")
    print()

    # Build track prototypes
    track_protos = {}
    for tid, data in tracks.items():
        embs = np.vstack(data["embs"])
        proto_emb = embs.mean(axis=0)
        proto_emb /= (np.linalg.norm(proto_emb) + 1e-12)
        proto_u = np.mean(data["uhist"], axis=0)
        proto_l = np.mean(data["lhist"], axis=0)
        proto_u /= (proto_u.sum() + 1e-12)
        proto_l /= (proto_l.sum() + 1e-12)
        track_protos[tid] = (proto_emb, proto_u, proto_l)
        # internal consistency
        sims = np.dot(embs, proto_emb)
        print(f"Track L{tid}: {len(data['embs'])} images, "
              f"mean similarity to its own prototype: {sims.mean():.3f} "
              f"(min: {sims.min():.3f}, max: {sims.max():.3f})")
    print()

    # Show pairwise comparisons between tracks
    track_ids = sorted(track_protos.keys())
    if len(track_ids) <= 1:
        print("Only one track – no intra‑folder merging decisions needed.")
    else:
        print("Intra‑folder track similarity analysis (would they be merged?):")
        split_reid_thresh = config.SPLIT_REID_THRESHOLD
        split_color_thresh = config.SPLIT_COLOR_THRESHOLD
        print(f"Using config thresholds: SPLIT_REID_THRESHOLD={split_reid_thresh}, SPLIT_COLOR_THRESHOLD={split_color_thresh}")
        print(f"{'Track pair':<15s} {'ReID sim':>8s} {'Color sim':>8s} {'Thresholds (ReID/Color)':>30s} {'Verdict':>20s}")
        print("-" * 80)
        for i in range(len(track_ids)):
            for j in range(i+1, len(track_ids)):
                tid1, tid2 = track_ids[i], track_ids[j]
                emb1, u1, l1 = track_protos[tid1]
                emb2, u2, l2 = track_protos[tid2]
                reid = float(np.dot(emb1, emb2))
                col = color_similarity(u1, l1, u2, l2)
                merged = (reid >= split_reid_thresh and col >= split_color_thresh)
                verdict = "MERGED" if merged else "SPLIT"
                print(f"L{tid1} vs L{tid2}     {reid:8.3f}  {col:8.3f}  "
                      f"ReID ≥ {split_reid_thresh}, Color ≥ {split_color_thresh}        {verdict}")

    print()
    # Overall folder prototype (average of all images)
    all_embs = []
    all_u = []
    all_l = []
    for data in tracks.values():
        all_embs.extend(data["embs"])
        all_u.extend(data["uhist"])
        all_l.extend(data["lhist"])
    all_embs = np.vstack(all_embs)
    proto_all = all_embs.mean(axis=0)
    proto_all /= (np.linalg.norm(proto_all) + 1e-12)
    folder_sims = np.dot(all_embs, proto_all)
    print("Folder-level embedding consistency:")
    print(f"  All images vs folder prototype: mean {folder_sims.mean():.3f} "
          f"(min: {folder_sims.min():.3f}, max: {folder_sims.max():.3f})")
    # similarity of each track prototype to the overall folder prototype
    print("Track similarity to overall folder prototype:")
    for tid in track_ids:
        emb, _, _ = track_protos[tid]
        sim = float(np.dot(emb, proto_all))
        print(f"  L{tid}: {sim:.3f}")
    # Frame overlaps between tracks
    print("\nFrame overlap between tracks:")
    for tid in track_ids:
        frames = set(tracks[tid]["frames"])
        print(f"  L{tid}: frames {min(frames)}-{max(frames)} (count {len(frames)})")
    for i in range(len(track_ids)):
        for j in range(i+1, len(track_ids)):
            f1 = set(tracks[track_ids[i]]["frames"])
            f2 = set(tracks[track_ids[j]]["frames"])
            ov = len(f1 & f2)
            if ov > 0:
                print(f"  L{track_ids[i]} & L{track_ids[j]} share {ov} frames (possible conflict)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to folder")
    args = parser.parse_args()
    sess, iname = load_osnet()
    analyze_folder(args.input, sess, iname)

if __name__ == "__main__":
    main()