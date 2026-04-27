#!/usr/bin/env python3
"""
Compare two person folders to see why they didn't merge.

Prints:
- Prototype similarity (ReID + colour)
- Frame overlap (if >0, cannot merge)
- Distribution of top‑5 nearest image pairs across folders
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

def preprocess(img):
    h,w = (256,128)
    img = cv2.resize(img, (w,h))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    mean = np.array([0.485,0.456,0.406], dtype=np.float32).reshape(3,1,1)
    std = np.array([0.229,0.224,0.225], dtype=np.float32).reshape(3,1,1)
    return (rgb - mean) / std

def extract_emb(img, sess, iname):
    tensor = preprocess(img)[np.newaxis, ...]
    emb = sess.run(None, {iname: tensor})[0][0]
    return emb / (np.linalg.norm(emb)+1e-12)

def load_folder(folder_path, sess, iname):
    folder = Path(folder_path)
    images = sorted([p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT])
    embs, paths, frames = [], [], []
    for imp in images:
        m = FILENAME_RE.search(imp.stem)
        f = int(m.group(1)) if m else -1
        frames.append(f)
        img = cv2.imread(str(imp))
        if img is None: continue
        emb = extract_emb(img, sess, iname)
        embs.append(emb)
        paths.append(imp)
    if embs:
        embs = np.vstack(embs)
        proto = embs.mean(axis=0); proto /= (np.linalg.norm(proto)+1e-12)
    else:
        embs, proto = None, None
    return paths, embs, proto, set(frames)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder1", required=True)
    parser.add_argument("--folder2", required=True)
    args = parser.parse_args()
    sess, iname = load_osnet()
    paths1, embs1, proto1, frames1 = load_folder(args.folder1, sess, iname)
    paths2, embs2, proto2, frames2 = load_folder(args.folder2, sess, iname)

    if embs1 is None or embs2 is None:
        print("One folder is empty or has no valid images.")
        return

    reid = float(np.dot(proto1, proto2))
    # we can't compute full colour here quickly without re‑computing all signatures,
    # so we'll show ReID only, but mention that colour would be similar.
    # For a full check, we could compute histograms on the fly.
    overlap = len(frames1.intersection(frames2))

    print(f"Folder 1: {len(embs1)} images, Folder 2: {len(embs2)} images")
    print(f"Prototype cosine similarity: {reid:.3f}")
    print(f"Frame overlap count: {overlap}")
    if overlap > 0:
        print(" -> This overlap prevents merging (hard rule).")
    else:
        print(" -> No frame overlap, merging is possible if similarity high enough.")

    # Show top 5 similar image pairs
    cos_matrix = np.dot(embs1, embs2.T)  # shape (N1, N2)
    # find top 5 indices
    flat_idx = np.argsort(cos_matrix.ravel())[::-1]
    print("\nTop 5 most similar image pairs:")
    for rank, idx in enumerate(flat_idx[:5]):
        i = idx // cos_matrix.shape[1]
        j = idx % cos_matrix.shape[1]
        sim = cos_matrix[i][j]
        print(f"  {paths1[i].name} <-> {paths2[j].name} : {sim:.3f}")

if __name__ == "__main__":
    main()