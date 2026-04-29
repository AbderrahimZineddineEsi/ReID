#!/usr/bin/env python3
"""Render a refined-v2 review video with colored bounding boxes.

The refined output folders only contain crop images, but the original video is still
available. This script does two things:
- runs person detection/tracking on the original video
- uses the refined crops from `person_*` folders to assign the refined person color
  to each box on the matching frame

This gives you a visual review of the refined identities on the same video.

Usage:
    python scripts/render_refined_v2_video.py --refined-dir outputs/refined_v2_fixed_better

Defaults:
- input video: config.INPUT_VIDEO
- output video: outputs/<refined-folder-name>_review.mp4
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from modules.detector_tracker import YOLOTracker
from modules.reid_embedder import ReIDEmbedder

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PERSON_RE = re.compile(r"person_(\d+)")
FRAME_RE = re.compile(r"frame_(\d+).*_l(\d+)")


@dataclass
class RefinedSample:
    person_id: int
    local_id: int
    emb: np.ndarray
    color_upper: np.ndarray
    color_lower: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a review video from refined_v2 folders")
    parser.add_argument("--video", default=None, help="Input video path (defaults to config.INPUT_VIDEO)")
    parser.add_argument("--refined-dir", required=True, help="Folder containing refined person_* directories")
    parser.add_argument("--output", default=None, help="Output mp4 path (defaults to outputs/<refined-folder>_review.mp4)")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional cap for quick tests (0 = full video)")
    return parser.parse_args()


def resolve_path(path_str: str | None, default: str) -> Path:
    path = Path(path_str) if path_str else Path(default)
    if path.is_absolute():
        return path

    repo_root = Path(__file__).resolve().parent.parent
    project_root = repo_root.parent
    for base in (Path.cwd(), repo_root, project_root):
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return (Path.cwd() / path).resolve()


def build_refined_index(refined_dir: Path,
                        embedder: ReIDEmbedder) -> Dict[int, List[RefinedSample]]:
    if not refined_dir.exists() or not refined_dir.is_dir():
        raise FileNotFoundError(f"Refined directory not found: {refined_dir}")

    frame_map: Dict[int, List[RefinedSample]] = {}
    person_dirs = sorted(
        [d for d in refined_dir.iterdir() if d.is_dir() and d.name.startswith("person_")],
        key=lambda p: p.name,
    )

    for person_dir in person_dirs:
        person_match = PERSON_RE.fullmatch(person_dir.name)
        if not person_match:
            continue
        person_id = int(person_match.group(1))

        for img_path in sorted(person_dir.iterdir()):
            if not img_path.is_file() or img_path.suffix.lower() not in IMG_EXT:
                continue
            frame_match = FRAME_RE.search(img_path.stem)
            if not frame_match:
                continue
            frame_idx = int(frame_match.group(1))
            local_id = int(frame_match.group(2))

            crop = cv2.imread(str(img_path))
            if crop is None:
                continue

            emb = embedder.extract_embedding(crop)
            color_sig = embedder.extract_color_signature(crop)
            frame_map.setdefault(frame_idx, []).append(
                RefinedSample(
                    person_id=person_id,
                    local_id=local_id,
                    emb=emb,
                    color_upper=color_sig["upper"],
                    color_lower=color_sig["lower"],
                )
            )

    for frame_idx in frame_map:
        frame_map[frame_idx].sort(key=lambda s: (s.person_id, s.local_id))

    return frame_map


def person_color(person_id: int) -> tuple[int, int, int]:
    hue = ((person_id * 0.61803398875) % 1.0) * 179.0
    hsv = np.array([[[hue, 220, 255]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def color_similarity(u1: np.ndarray, l1: np.ndarray, u2: np.ndarray, l2: np.ndarray) -> float:
    up = float(np.minimum(u1, u2).sum())
    lo = float(np.minimum(l1, l2).sum())
    return 0.4 * up + 0.6 * lo


def match_detections_to_refined(detections: List[dict], refined_samples: List[RefinedSample], embedder: ReIDEmbedder):
    """Greedy one-to-one matching between current detections and refined samples for a frame."""
    if not detections or not refined_samples:
        return {}

    det_features = []
    for det in detections:
        crop = det["crop"]
        emb = embedder.extract_embedding(crop)
        color_sig = embedder.extract_color_signature(crop)
        det_features.append({
            "emb": emb,
            "upper": color_sig["upper"],
            "lower": color_sig["lower"],
        })

    pairs = []
    for d_idx, det in enumerate(det_features):
        for s_idx, sample in enumerate(refined_samples):
            reid = float(np.dot(det["emb"], sample.emb))
            col = color_similarity(det["upper"], det["lower"], sample.color_upper, sample.color_lower)
            combined = reid + config.MERGE_COLOR_WEIGHT * col
            pairs.append((combined, reid, col, d_idx, s_idx))

    pairs.sort(reverse=True, key=lambda item: item[0])

    det_used = set()
    sample_used = set()
    assignment = {}
    for combined, reid, col, d_idx, s_idx in pairs:
        if d_idx in det_used or s_idx in sample_used:
            continue
        if reid < config.MERGE_REID_THRESHOLD or col < config.MERGE_COLOR_THRESHOLD:
            continue
        det_used.add(d_idx)
        sample_used.add(s_idx)
        assignment[d_idx] = {
            "sample": refined_samples[s_idx],
            "reid": reid,
            "color": col,
            "combined": combined,
        }

    return assignment


def draw_frame_boxes(frame: np.ndarray, detections: List[dict], assignment: dict) -> np.ndarray:
    annotated = frame.copy()
    for d_idx, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox"]
        local_id = det["local_id"]
        if d_idx in assignment:
            sample = assignment[d_idx]["sample"]
            color = person_color(sample.person_id)
            label = f"P{sample.person_id:03d} L{local_id}"
        else:
            color = (0, 165, 255)
            label = f"L{local_id} ?"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        text_y = y1 - 10 if y1 > 24 else y1 + 20
        cv2.putText(annotated, label, (x1, text_y), cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1, cv2.LINE_AA)

    return annotated


def resolve_output_path(refined_dir: Path, output_arg: Optional[str]) -> Path:
    if output_arg:
        return resolve_path(output_arg, output_arg)
    return (Path(config.OUTPUT_DIR) / f"{refined_dir.name}_review.mp4").resolve()


def main() -> None:
    args = parse_args()
    video_path = resolve_path(args.video, config.INPUT_VIDEO)
    refined_dir = resolve_path(args.refined_dir, args.refined_dir)
    output_path = resolve_output_path(refined_dir, args.output)

    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    embedder = ReIDEmbedder(model_path=config.OSNET_MODEL)
    refined_index = build_refined_index(refined_dir, embedder)
    print(f"Indexed {len(refined_index)} frames from {refined_dir}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_w, frame_h),
    )

    tracker = YOLOTracker()

    max_frames = args.max_frames if args.max_frames > 0 else total_frames
    pbar = tqdm(total=min(total_frames, max_frames), desc="Rendering refined review", unit="frame")

    frame_idx = 0
    matched_boxes = 0
    detected_boxes = 0

    while frame_idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        detections = tracker.track(frame, persist=True)
        detected_boxes += len(detections)

        samples = refined_index.get(frame_idx, [])
        assignment = match_detections_to_refined(detections, samples, embedder)
        matched_boxes += len(assignment)

        annotated = draw_frame_boxes(frame, detections, assignment)
        out.write(annotated)
        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()

    print(f"Done. Output video: {output_path}")
    print(f"Frames processed: {frame_idx}")
    print(f"Detected boxes: {detected_boxes}")
    print(f"Matched to refined identities: {matched_boxes}")


if __name__ == "__main__":
    main()
