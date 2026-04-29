#!/usr/bin/env python3
"""
Deep folder comparison – shows exactly why two folders were (or were not) merged.

Single pair usage:
    python experiments/compare_folders_deep.py --folder1 outputs/step2_global_crops_fixed/person_001 --folder2 person_002

Batch usage:
    python experiments/compare_folders_deep.py \
        --root outputs/step2_global_crops_fixed \
        --compare "4 is 3" --compare "20 is 7" --compare "34,21 is 16" \
        --output outputs/step2_global_crops_fixed/compare_report.txt
"""

import argparse, re, sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import onnxruntime as ort

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FILENAME_RE = re.compile(r"frame_(\d+).*_l(\d+)")
PAIR_SPEC_RE = re.compile(r"\s*(.*?)\s*(?:\bis\b|:|=)\s*(.*?)\s*$", re.IGNORECASE)


def load_osnet():
    sess = ort.InferenceSession(str(config.OSNET_MODEL), providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    return sess, iname


def emit(lines, text=""):
    lines.append(text)


def resolve_folder_token(root: Path, token: str) -> Path:
    token = token.strip()
    if not token:
        raise ValueError("Empty folder token in comparison spec")
    path = Path(token)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    if token.isdigit():
        return root / f"person_{int(token):03d}"
    if token.startswith("person_"):
        return root / token
    return root / token


def parse_pair_spec(root: Path, spec: str) -> List[Tuple[Path, Path]]:
    match = PAIR_SPEC_RE.match(spec)
    if not match:
        raise ValueError(f"Could not parse comparison spec: {spec!r}")
    left_text, right_text = match.group(1), match.group(2)
    left_tokens = [token.strip() for token in left_text.split(",") if token.strip()]
    right_tokens = [token.strip() for token in right_text.split(",") if token.strip()]
    if len(right_tokens) != 1:
        raise ValueError(f"Right-hand side must contain exactly one folder: {spec!r}")
    right_folder = resolve_folder_token(root, right_tokens[0])
    return [(resolve_folder_token(root, token), right_folder) for token in left_tokens]


def preprocess_img(img):
    h, w = (256, 128)
    img = cv2.resize(img, (w, h))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = np.transpose(rgb, (2, 0, 1))
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
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
    hist_u = cv2.calcHist([upper], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]).flatten().astype(np.float32)
    hist_l = cv2.calcHist([lower], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]).flatten().astype(np.float32)
    hist_u /= (hist_u.sum() + 1e-12)
    hist_l /= (hist_l.sum() + 1e-12)
    return hist_u, hist_l


def color_similarity(u1, l1, u2, l2):
    up = float(np.minimum(u1, u2).sum())
    lo = float(np.minimum(l1, l2).sum())
    return 0.4 * up + 0.6 * lo


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
    track_buckets: Dict[int, dict] = defaultdict(lambda: {"embs": [], "uhist": [], "lhist": [], "frames": []})
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
        proto_emb /= (np.linalg.norm(proto_emb) + 1e-12)
        proto_u = np.mean(u_hist, axis=0)
        proto_l = np.mean(l_hist, axis=0)
        proto_u /= (proto_u.sum() + 1e-12)
        proto_l /= (proto_l.sum() + 1e-12)
    else:
        return None, None, None, None, None, None, None
    track_protos, track_stats = summarize_tracks(track_buckets)
    return paths, embs, (proto_emb, proto_u, proto_l), set(frames), (u_hist, l_hist), track_protos, track_stats


def analyze_pair(folder1_path, folder2_path, sess, iname):
    paths1, embs1, proto1, frames1, _, track_protos1, track_stats1 = load_folder(folder1_path, sess, iname)
    paths2, embs2, proto2, frames2, _, track_protos2, track_stats2 = load_folder(folder2_path, sess, iname)
    if embs1 is None or embs2 is None:
        return [f"One of the folders has no valid images: {folder1_path} | {folder2_path}"]

    lines: List[str] = []

    reid_sim = float(np.dot(proto1[0], proto2[0]))
    col_sim = color_similarity(proto1[1], proto1[2], proto2[1], proto2[2])
    overlap = len(frames1 & frames2)

    merge_reid_thresh = config.MERGE_REID_THRESHOLD
    merge_color_thresh = config.MERGE_COLOR_THRESHOLD
    max_overlap = config.MAX_SAME_FRAME_OVERLAP

    emit(lines, f"Folder 1: {folder1_path} ({len(embs1)} images)")
    emit(lines, f"Folder 2: {folder2_path} ({len(embs2)} images)")
    emit(lines, f"Prototype ReID similarity: {reid_sim:.3f}")
    emit(lines, f"Prototype Colour similarity: {col_sim:.3f}")
    emit(lines, f"Frame overlap: {overlap} (>{max_overlap} forbids merge)")

    emit(lines, "")
    emit(lines, "Per-track consistency in Folder 1:")
    for tid in sorted(track_stats1.keys()):
        s = track_stats1[tid]
        emit(lines, f"  L{tid}: {s['count']} images, mean similarity to its own prototype: {s['mean']:.3f} "
                    f"(min: {s['min']:.3f}, max: {s['max']:.3f}), frames {s['frame_min']}-{s['frame_max']}")

    emit(lines, "")
    emit(lines, "Per-track consistency in Folder 2:")
    for tid in sorted(track_stats2.keys()):
        s = track_stats2[tid]
        emit(lines, f"  L{tid}: {s['count']} images, mean similarity to its own prototype: {s['mean']:.3f} "
                    f"(min: {s['min']:.3f}, max: {s['max']:.3f}), frames {s['frame_min']}-{s['frame_max']}")

    

    can_merge = True
    if overlap > max_overlap:
        can_merge = False
        emit(lines, f"Merge BLOCKED: frame overlap exceeded (max={max_overlap}).")    
    if reid_sim < merge_reid_thresh:
        can_merge = False
        emit(lines, f"Merge BLOCKED: ReID {reid_sim:.3f} < threshold {merge_reid_thresh}")
    if col_sim < merge_color_thresh:
        can_merge = False
        emit(lines, f"Merge BLOCKED: Colour {col_sim:.3f} < threshold {merge_color_thresh}")
    if can_merge:
        emit(lines, "All conditions satisfied -> would MERGE.")

    emit(lines, "")
    emit(lines, "Track-by-track cross-folder similarity:")
    emit(lines, f"{'Track pair':<18s} {'ReID sim':>8s} {'Color sim':>9s} {'Frame ov':>8s} {'Verdict':>10s}")
    emit(lines, "-" * 62)
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
            emit(lines, f"L{t1} vs L{t2}      {reid_t:8.3f} {col_t:9.3f} {ov_t:8d} {verdict:>10s}")

    cos_matrix = np.dot(embs1, embs2.T)
    flat_idx = np.argsort(cos_matrix.ravel())[::-1]
    emit(lines, "")
    emit(lines, "Top 5 most similar image pairs:")
    for idx in flat_idx[:5]:
        i = idx // cos_matrix.shape[1]
        j = idx % cos_matrix.shape[1]
        sim = cos_matrix[i][j]
        emit(lines, f"  {paths1[i].name} <-> {paths2[j].name} : {sim:.3f}")

    emit(lines, f"")
    emit(lines, f"Pairwise similarities range: min={cos_matrix.min():.3f}, max={cos_matrix.max():.3f}, mean={cos_matrix.mean():.3f}")
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder1", required=False)
    parser.add_argument("--folder2", required=False)
    parser.add_argument("--root", default="outputs/step2_global_crops_fixed", help="Root folder used to resolve numeric person ids")
    parser.add_argument("--compare", action="append", default=[], help='Batch spec like "4 is 3" or "34,21 is 16"')
    parser.add_argument("--output", help="Write the report to this txt file")
    args = parser.parse_args()

    sess, iname = load_osnet()
    root = Path(args.root)
    lines: List[str] = []

    if args.compare:
        for spec in args.compare:
            for folder1_path, folder2_path in parse_pair_spec(root, spec):
                emit(lines, "=" * 80)
                emit(lines, f"Comparison spec: {spec}")
                emit(lines, f"Resolved: {folder1_path} -> {folder2_path}")
                emit(lines, "=" * 80)
                lines.extend(analyze_pair(folder1_path, folder2_path, sess, iname))
                emit(lines, "")
    else:
        if not args.folder1 or not args.folder2:
            parser.error("Provide either --folder1/--folder2 or one or more --compare specs")
        lines.extend(analyze_pair(args.folder1, args.folder2, sess, iname))

    report = "\n".join(lines).rstrip() + "\n"
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()