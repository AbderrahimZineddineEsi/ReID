#!/usr/bin/env python3
"""
Offline Refinement Pass (Pass 2+3) for Global Identity Folders.

Reads the crop folders produced by the live Step 2 (or any similar structure),
splits mixed folders and merges fragmented ones to produce a cleaner set.

Usage:
    python scripts/offline_refine.py --input outputs/step2_global_crops_fixed --output outputs/refined_final
"""

import argparse
import json
import re
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort
from tqdm import tqdm

# We'll import config from the project root
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FRAME_RE = re.compile(r"frame_(\d+)")


# ---------- ReID & colour helpers (mirrored from modules) ----------
def load_osnet_onnx(model_path):
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    return session, input_name

def preprocess_crop(crop, target_size=(256, 128)):
    h, w = target_size
    resized = cv2.resize(crop, (w, h))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # HWC -> CHW
    rgb = np.transpose(rgb, (2, 0, 1))

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    rgb = (rgb - mean) / std
    return rgb[np.newaxis, ...]  # (1, 3, H, W)

def extract_embedding(crop, session, input_name, target_size):
    tensor = preprocess_crop(crop, target_size)
    emb = session.run(None, {input_name: tensor})[0][0]
    norm = np.linalg.norm(emb)
    if norm > 1e-12:
        emb /= norm
    return emb

def extract_color_signature(crop):
    resized = cv2.resize(crop, (64, 128))
    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
    split = lab.shape[0] // 2
    upper = lab[:split, :, :]
    lower = lab[split:, :, :]
    hist_size = (8, 8, 8)
    upper_hist = cv2.calcHist([upper], [0, 1, 2], None, hist_size,
                              [0, 256, 0, 256, 0, 256]).flatten().astype(np.float32)
    lower_hist = cv2.calcHist([lower], [0, 1, 2], None, hist_size,
                              [0, 256, 0, 256, 0, 256]).flatten().astype(np.float32)
    upper_hist /= (upper_hist.sum() + 1e-12)
    lower_hist /= (lower_hist.sum() + 1e-12)
    return upper_hist, lower_hist

def color_similarity(u1, l1, u2, l2):
    return 0.4 * np.minimum(u1, u2).sum() + 0.6 * np.minimum(l1, l2).sum()

# ---------- Cluster representation ----------
class Cluster:
    def __init__(self, name):
        self.name = name
        self.paths: List[Path] = []          # image file paths
        self.embeddings: List[np.ndarray] = []
        self.upper_hists: List[np.ndarray] = []
        self.lower_hists: List[np.ndarray] = []
        self.frame_indices: List[int] = []   # extracted from filename
        self.proto_emb: Optional[np.ndarray] = None
        self.proto_upper: Optional[np.ndarray] = None
        self.proto_lower: Optional[np.ndarray] = None

    def add_crop(self, img_path, emb, uhist, lhist, frame_idx):
        self.paths.append(img_path)
        self.embeddings.append(emb)
        self.upper_hists.append(uhist)
        self.lower_hists.append(lhist)
        self.frame_indices.append(frame_idx)

    def build_prototype(self):
        if not self.embeddings:
            return
        embs = np.vstack(self.embeddings)
        self.proto_emb = embs.mean(axis=0)
        norm = np.linalg.norm(self.proto_emb)
        if norm > 1e-12:
            self.proto_emb /= norm
        self.proto_upper = np.mean(self.upper_hists, axis=0)
        self.proto_lower = np.mean(self.lower_hists, axis=0)
        s_up = self.proto_upper.sum()
        s_lo = self.proto_lower.sum()
        if s_up > 0: self.proto_upper /= s_up
        if s_lo > 0: self.proto_lower /= s_lo

    def size(self):
        return len(self.paths)

    def frame_set(self):
        return set(self.frame_indices)

def clone_cluster(cluster, name):
    newc = Cluster(name)
    newc.paths = cluster.paths.copy()
    newc.embeddings = cluster.embeddings.copy()
    newc.upper_hists = cluster.upper_hists.copy()
    newc.lower_hists = cluster.lower_hists.copy()
    newc.frame_indices = cluster.frame_indices.copy()
    newc.build_prototype()
    return newc

# ---------- Split one cluster (intra-folder) ----------
def split_cluster(cluster: Cluster, args) -> List[Cluster]:
    """Split a cluster if it contains visually dissimilar subgroups."""
    if cluster.size() < args.split_min_samples:
        return [cluster]

    # Order by frame index (temporal)
    order = sorted(range(cluster.size()), key=lambda i: cluster.frame_indices[i])
    subgroups: List[Cluster] = []
    for idx in order:
        emb = cluster.embeddings[idx]
        upper = cluster.upper_hists[idx]
        lower = cluster.lower_hists[idx]
        best_sub = -1
        best_score = -1.0
        for si, sub in enumerate(subgroups):
            reid_sim = float(np.dot(emb, sub.proto_emb))
            col_sim = color_similarity(upper, lower, sub.proto_upper, sub.proto_lower)
            if reid_sim < args.split_reid_threshold or col_sim < args.split_color_threshold:
                continue
            combined = reid_sim + args.split_color_weight * col_sim
            if combined > best_score:
                best_score = combined
                best_sub = si
        if best_sub >= 0:
            sub = subgroups[best_sub]
            sub.add_crop(cluster.paths[idx], emb, upper, lower, cluster.frame_indices[idx])
            sub.build_prototype()
        else:
            new_sub = Cluster(f"{cluster.name}_split{len(subgroups)}")
            new_sub.add_crop(cluster.paths[idx], emb, upper, lower, cluster.frame_indices[idx])
            new_sub.build_prototype()
            subgroups.append(new_sub)

    # Reattach tiny subgroups to larger ones if strong evidence
    if args.split_reattach_min_size > 1 and len(subgroups) > 1:
        big = [sub for sub in subgroups if sub.size() >= args.split_reattach_min_size]
        tiny = [sub for sub in subgroups if sub.size() < args.split_reattach_min_size]
        if big and tiny:
            for small in tiny:
                best_big = -1
                best_score = -1.0
                for bi, large in enumerate(big):
                    reid_sim = float(np.dot(small.proto_emb, large.proto_emb))
                    col_sim = color_similarity(small.proto_upper, small.proto_lower,
                                               large.proto_upper, large.proto_lower)
                    if reid_sim < args.reattach_reid_threshold or col_sim < args.reattach_color_threshold:
                        continue
                    combined = reid_sim + args.split_color_weight * col_sim
                    if combined > best_score:
                        best_score = combined
                        best_big = bi
                if best_big >= 0:
                    for pi in range(small.size()):
                        big[best_big].add_crop(small.paths[pi], small.embeddings[pi],
                                               small.upper_hists[pi], small.lower_hists[pi],
                                               small.frame_indices[pi])
                    big[best_big].build_prototype()
                else:
                    big.append(small)
            subgroups = big
    return subgroups


# ---------- Merge across clusters ----------
def same_frame_overlap(a: Cluster, b: Cluster) -> int:
    fs_a = a.frame_set()
    fs_b = b.frame_set()
    if not fs_a or not fs_b:
        return 0
    return len(fs_a.intersection(fs_b))

def cluster_similarity(a: Cluster, b: Cluster, args) -> Tuple[float, float, float]:
    reid = float(np.dot(a.proto_emb, b.proto_emb))
    col = color_similarity(a.proto_upper, a.proto_lower,
                           b.proto_upper, b.proto_lower)
    combined = reid + args.merge_color_weight * col
    return combined, reid, col

def greedy_merge(clusters: List[Cluster], args) -> Tuple[List[Cluster], int]:
    work = clusters[:]
    merge_count = 0
    while True:
        best_i, best_j = -1, -1
        best_score = -1.0
        for i in range(len(work)):
            for j in range(i+1, len(work)):
                a, b = work[i], work[j]
                overlap = same_frame_overlap(a, b)
                if overlap > args.max_same_frame_overlap:
                    continue
                score, reid, col = cluster_similarity(a, b, args)
                if reid < args.merge_reid_threshold or col < args.merge_color_threshold:
                    continue
                if score > best_score:
                    best_score = score
                    best_i, best_j = i, j
        if best_i < 0:
            break
        # merge
        merged = Cluster(f"merged_{merge_count}")
        merged.paths = work[best_i].paths + work[best_j].paths
        merged.embeddings = work[best_i].embeddings + work[best_j].embeddings
        merged.upper_hists = work[best_i].upper_hists + work[best_j].upper_hists
        merged.lower_hists = work[best_i].lower_hists + work[best_j].lower_hists
        merged.frame_indices = work[best_i].frame_indices + work[best_j].frame_indices
        merged.build_prototype()
        # Remove in reverse order
        for idx in sorted([best_i, best_j], reverse=True):
            work.pop(idx)
        work.append(merged)
        merge_count += 1
    return work, merge_count


# ---------- Main pipeline ----------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Folder with person_* directories")
    p.add_argument("--output", required=True, help="Where to write refined folders")
    p.add_argument("--overwrite", action="store_true")
    # Split thresholds
    p.add_argument("--split-reid-threshold", type=float, default=0.72)
    p.add_argument("--split-color-threshold", type=float, default=0.28)
    p.add_argument("--split-color-weight", type=float, default=0.15)
    p.add_argument("--split-min-samples", type=int, default=3)
    p.add_argument("--split-reattach-min-size", type=int, default=3)
    p.add_argument("--reattach-reid-threshold", type=float, default=0.80)
    p.add_argument("--reattach-color-threshold", type=float, default=0.34)
    # Merge thresholds
    p.add_argument("--merge-reid-threshold", type=float, default=0.85)
    p.add_argument("--merge-color-threshold", type=float, default=0.40)
    p.add_argument("--merge-color-weight", type=float, default=0.20)
    p.add_argument("--max-same-frame-overlap", type=int, default=0)
    p.add_argument("--min-folder-images", type=int, default=3, help="Drop final folders with fewer images")
    return p.parse_args()

def main():
    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    if output_dir.exists():
        if args.overwrite:
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(f"{output_dir} exists, use --overwrite")
    output_dir.mkdir(parents=True)

    # Load ReID model
    reid_path = Path(config.OSNET_MODEL)
    if not reid_path.exists():
        raise FileNotFoundError(f"ReID model not found: {reid_path}")
    session, input_name = load_osnet_onnx(reid_path)

    # Scan folders, load all crops into clusters
    folders = sorted([d for d in input_dir.iterdir() if d.is_dir() and d.name.startswith("person_")])
    clusters = []
    for folder in tqdm(folders, desc="Loading folders"):
        images = sorted([p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTENSIONS])
        if not images:
            continue
        cluster = Cluster(folder.name)
        for img_path in images:
            # parse frame index from filename (pattern: frame_XXXXXX...)
            m = FRAME_RE.search(img_path.stem)
            frame_idx = int(m.group(1)) if m else -1
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            emb = extract_embedding(img, session, input_name, config.REID_SIZE)
            uhist, lhist = extract_color_signature(img)
            cluster.add_crop(img_path, emb, uhist, lhist, frame_idx)
        cluster.build_prototype()
        if cluster.size() > 0:
            clusters.append(cluster)

    print(f"Loaded {len(clusters)} folders, now splitting each...")
    all_split = []
    for c in tqdm(clusters, desc="Splitting"):
        splitted = split_cluster(c, args)
        all_split.extend(splitted)

    print(f"After split: {len(all_split)} clusters. Merging...")
    merged, merge_ops = greedy_merge(all_split, args)

    # Sort by size descending and assign new person_001, person_002 ...
    merged.sort(key=lambda c: c.size(), reverse=True)
    final_idx = 0
    for c in merged:
        if c.size() < args.min_folder_images:
            continue
        final_idx += 1
        new_name = f"person_{final_idx:03d}"
        new_dir = output_dir / new_name
        new_dir.mkdir()
        for src in c.paths:
            # preserve original filename (unique)
            dest = new_dir / src.name
            # handle duplicates
            if dest.exists():
                i = 1
                while (new_dir / f"{src.stem}_{i}{src.suffix}").exists():
                    i += 1
                dest = new_dir / f"{src.stem}_{i}{src.suffix}"
            shutil.copy2(src, dest)

    print(f"Final refined folders written to {output_dir} ({final_idx} folders)")


if __name__ == "__main__":
    main()