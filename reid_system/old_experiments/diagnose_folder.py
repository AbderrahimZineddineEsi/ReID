#!/usr/bin/env python3
"""
Analyse a person folder (or all folders) to understand mixing.
Usage: python experiments/diagnose_folder.py --input outputs/step2_global_crops_fixed/person_001
"""

import argparse, re, sys
from pathlib import Path
import cv2, numpy as np
import onnxruntime as ort

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FILENAME_RE = re.compile(r"frame_(\d+).*_l(\d+)")

def load_osnet():
    sess = ort.InferenceSession(config.OSNET_MODEL, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    return sess, iname

def preprocess(img, size=(256,128)):
    h,w = size
    img = cv2.resize(img, (w,h))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    rgb = np.transpose(rgb, (2, 0, 1))  # HWC -> CHW
    mean = np.array([0.485,0.456,0.406], dtype=np.float32).reshape(3,1,1)
    std = np.array([0.229,0.224,0.225], dtype=np.float32).reshape(3,1,1)
    return (rgb - mean) / std

def extract_emb(img, sess, iname):
    tensor = preprocess(img)[np.newaxis, ...]
    emb = sess.run(None, {iname: tensor})[0][0]
    return emb / (np.linalg.norm(emb)+1e-12)

def analyze_folder(folder_path, sess, iname):
    folder = Path(folder_path)
    images = sorted([p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT])
    if not images:
        print(f"{folder.name}: empty")
        return
    embs = []
    frames = []
    locals_ = []
    for imp in images:
        m = FILENAME_RE.search(imp.stem)
        if m:
            frames.append(int(m.group(1)))
            locals_.append(int(m.group(2)))
        else:
            frames.append(-1)
            locals_.append(-1)
        img = cv2.imread(str(imp))
        if img is None: continue
        emb = extract_emb(img, sess, iname)
        embs.append(emb)
    if not embs:
        print(f"{folder.name}: no valid images")
        return
    embs = np.vstack(embs)
    proto = embs.mean(axis=0); proto /= (np.linalg.norm(proto)+1e-12)
    sims_to_proto = np.dot(embs, proto)
    pairwise = np.dot(embs, embs.T)
    mask = ~np.eye(len(embs), dtype=bool)
    mean_pair = pairwise[mask].mean() if len(embs)>1 else float('nan')

    # frame duplicates
    frame_counts = {}
    for f in frames:
        if f>=0: frame_counts[f] = frame_counts.get(f,0)+1
    dup_frames = {f:c for f,c in frame_counts.items() if c>1}

    distinct_tracks = set(l for l in locals_ if l>=0)
    print(f"\n=== {folder.name} ===")
    print(f"Images: {len(embs)}")
    print(f"Distinct local tracks: {len(distinct_tracks)}")
    print(f"Proto similarity: mean={sims_to_proto.mean():.3f}, min={sims_to_proto.min():.3f}, max={sims_to_proto.max():.3f}")
    print(f"Mean pairwise cos sim: {mean_pair:.3f}")
    if dup_frames:
        print(f"WARNING: same frame multiple crops: {dup_frames}")
    else:
        print("No frame duplicates.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    sess, iname = load_osnet()
    if args.all:
        base = Path(args.input)
        for d in sorted(base.iterdir()):
            if d.is_dir() and d.name.startswith("person_"):
                analyze_folder(d, sess, iname)
    else:
        analyze_folder(args.input, sess, iname)

if __name__ == "__main__":
    main()