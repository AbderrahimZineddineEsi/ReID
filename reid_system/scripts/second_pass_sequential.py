#!/usr/bin/env python3
"""
Second‑pass refinement (sequential, track‑aware).

For each person folder:
  - Group crops by local track ID (extracted from filename _l<id>).
  - Check similarity between track prototypes: if two tracks are dissimilar, split them.
  - For each cluster (track group), try to merge with already processed folders.
    If match found, merge; else, it becomes a new person.
"""

import argparse, re, shutil, time
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2, numpy as np
import onnxruntime as ort
from tqdm import tqdm

# Import config from project root
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# Pattern for frame_<frame>_t<...>_l<local_id>.jpg
FILENAME_RE = re.compile(r"frame_(\d+).*_l(\d+)")

# ---------------- ReID & colour helpers ----------------
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

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3,1,1)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3,1,1)
    rgb = (rgb - mean) / std
    return rgb[np.newaxis, ...]

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
    hist_upper = cv2.calcHist([upper], [0,1,2], None, [8,8,8],
                               [0,256,0,256,0,256]).flatten().astype(np.float32)
    hist_lower = cv2.calcHist([lower], [0,1,2], None, [8,8,8],
                               [0,256,0,256,0,256]).flatten().astype(np.float32)
    hist_upper /= (hist_upper.sum() + 1e-12)
    hist_lower /= (hist_lower.sum() + 1e-12)
    return hist_upper, hist_lower

def color_similarity(u1, l1, u2, l2):
    return 0.4 * np.minimum(u1, u2).sum() + 0.6 * np.minimum(l1, l2).sum()

# ---------------- Cluster representation ----------------
class TrackCluster:
    """A group of crops belonging to the same local track."""
    def __init__(self, local_id):
        self.local_id = local_id
        self.paths: List[Path] = []
        self.embeddings: List[np.ndarray] = []
        self.upper: List[np.ndarray] = []
        self.lower: List[np.ndarray] = []
        self.frames: List[int] = []
        self.proto_emb = None
        self.proto_up = None
        self.proto_lo = None

    def add(self, path, emb, up, lo, frame):
        self.paths.append(path)
        self.embeddings.append(emb)
        self.upper.append(up)
        self.lower.append(lo)
        self.frames.append(frame)

    def build_prototype(self):
        if not self.embeddings:
            return
        embs = np.vstack(self.embeddings)
        self.proto_emb = embs.mean(axis=0)
        norm = np.linalg.norm(self.proto_emb)
        if norm > 1e-12:
            self.proto_emb /= norm
        self.proto_up = np.mean(self.upper, axis=0)
        self.proto_lo = np.mean(self.lower, axis=0)
        s_up = self.proto_up.sum()
        s_lo = self.proto_lo.sum()
        if s_up > 0: self.proto_up /= s_up
        if s_lo > 0: self.proto_lo /= s_lo

    def size(self):
        return len(self.paths)

    def frame_set(self):
        return set(self.frames)

    def similarity_to(self, other: "TrackCluster", merge_color_weight: float):
        """Return (reid_sim, color_sim, combined) with current prototypes."""
        reid = float(np.dot(self.proto_emb, other.proto_emb))
        col = color_similarity(self.proto_up, self.proto_lo,
                               other.proto_up, other.proto_lo)
        combined = reid + merge_color_weight * col
        return reid, col, combined

def merge_clusters(a: TrackCluster, b: TrackCluster, new_name: str) -> TrackCluster:
    """Combine two clusters and recompute prototype."""
    merged = TrackCluster(new_name)
    merged.paths = a.paths + b.paths
    merged.embeddings = a.embeddings + b.embeddings
    merged.upper = a.upper + b.upper
    merged.lower = a.lower + b.lower
    merged.frames = a.frames + b.frames
    merged.build_prototype()
    return merged

# ---------------- Main algorithm ----------------
def parse_args():
    p = argparse.ArgumentParser(description="Sequential second‑pass refinement")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--split-reid-threshold", type=float, default=0.72,
                   help="If track similarity < this, split")
    p.add_argument("--split-color-threshold", type=float, default=0.28,
                   help="If track colour similarity < this, split")
    p.add_argument("--merge-reid-threshold", type=float, default=0.85,
                   help="ReID threshold for merging across folders")
    p.add_argument("--merge-color-threshold", type=float, default=0.40,
                   help="Colour threshold for merging across folders")
    p.add_argument("--merge-color-weight", type=float, default=0.20,
                   help="Contribution of colour to merge score")
    p.add_argument("--max-same-frame-overlap", type=int, default=0)
    p.add_argument("--min-folder-images", type=int, default=3)
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

    # Load ONNX
    reid_path = Path(config.OSNET_MODEL)
    if not reid_path.exists():
        raise FileNotFoundError(f"ReID model not found: {reid_path}")
    session, input_name = load_osnet_onnx(reid_path)

    # Get sorted folder list
    folders = sorted([d for d in input_dir.iterdir()
                      if d.is_dir() and d.name.startswith("person_")],
                     key=lambda d: d.name)
    # We'll store final clusters here
    final_clusters: List[TrackCluster] = []
    # Track global name counter
    cluster_counter = 0

    for folder in tqdm(folders, desc="Processing folders"):
        images = sorted([p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTENSIONS])
        if not images:
            continue

        # Group by local track ID
        tracks: Dict[int, TrackCluster] = defaultdict(lambda: TrackCluster(None))
        for img_path in images:
            m = FILENAME_RE.search(img_path.stem)
            if m:
                frame = int(m.group(1))
                local = int(m.group(2))
            else:
                frame = -1
                local = -1
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            emb = extract_embedding(img, session, input_name, config.REID_SIZE)
            up, lo = extract_color_signature(img)
            tracks[local].add(img_path, emb, up, lo, frame)
            tracks[local].local_id = local

        # Build prototypes for each track group
        for tc in tracks.values():
            tc.build_prototype()

        # Now merge track groups within this folder if they are the same person
        track_list = list(tracks.values())
        merged_groups = []         # contains merged clusters for this folder
        used = set()
        for i, tc1 in enumerate(track_list):
            if i in used:
                continue
            current = tc1
            for j in range(i+1, len(track_list)):
                if j in used:
                    continue
                tc2 = track_list[j]
                # Check if they could be the same person
                reid, col, _ = current.similarity_to(tc2, args.merge_color_weight)
                if reid >= args.split_reid_threshold and col >= args.split_color_threshold:
                    # Merge them
                    current = merge_clusters(current, tc2, f"{folder.name}_merged")
                    used.add(j)
            used.add(i)
            merged_groups.append(current)

        # Now for each group in this folder, try to merge with existing final clusters
        for group in merged_groups:
            best_match = None
            best_score = -1.0
            for final in final_clusters:
                # Check frame overlap
                overlap = len(group.frame_set().intersection(final.frame_set()))
                if overlap > args.max_same_frame_overlap:
                    continue
                reid, col, score = group.similarity_to(final, args.merge_color_weight)
                if reid < args.merge_reid_threshold or col < args.merge_color_threshold:
                    continue
                if score > best_score:
                    best_score = score
                    best_match = final
            if best_match is not None:
                # Merge into existing final cluster
                best_match.paths.extend(group.paths)
                best_match.embeddings.extend(group.embeddings)
                best_match.upper.extend(group.upper)
                best_match.lower.extend(group.lower)
                best_match.frames.extend(group.frames)
                best_match.build_prototype()
            else:
                # New person
                final_clusters.append(group)

    # Sort final clusters by size descending, then assign person_001, person_002, ...
    final_clusters.sort(key=lambda c: c.size(), reverse=True)
    out_idx = 0
    for cluster in final_clusters:
        if cluster.size() < args.min_folder_images:
            continue
        out_idx += 1
        person_dir = output_dir / f"person_{out_idx:03d}"
        person_dir.mkdir()
        for src in cluster.paths:
            dest = person_dir / src.name
            # avoid name collisions
            if dest.exists():
                i = 1
                while (person_dir / f"{src.stem}_{i}{src.suffix}").exists():
                    i += 1
                dest = person_dir / f"{src.stem}_{i}{src.suffix}"
            shutil.copy2(src, dest)

    print(f"Final refined folders written to {output_dir} ({out_idx} folders)")

if __name__ == "__main__":
    main()