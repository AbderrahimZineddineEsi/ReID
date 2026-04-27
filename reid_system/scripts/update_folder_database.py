#!/usr/bin/env python3
"""Synchronize a folder-level ReID database/manifest with the images currently on disk.

This script is useful when you delete or add images inside a `person_*` folder and want
any cached folder statistics to reflect the current contents again.

It scans the folder, recomputes per-track and folder-level appearance statistics, and
writes a JSON manifest next to the folder (default: `.folder_db.json`).

Example:
    python scripts/update_folder_database.py --folder outputs/step2_global_crops_fixed/person_001
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import onnxruntime as ort

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FILENAME_RE = re.compile(r"frame_(\d+).*_l(\d+)")


def load_osnet() -> tuple[ort.InferenceSession, str]:
    session = ort.InferenceSession(str(config.OSNET_MODEL), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    return session, input_name


def preprocess_img(img: np.ndarray, size=(256, 128)) -> np.ndarray:
    h, w = size
    img = cv2.resize(img, (w, h))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = np.transpose(rgb, (2, 0, 1))
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    return (rgb - mean) / std


def extract_emb(img: np.ndarray, session: ort.InferenceSession, input_name: str) -> np.ndarray:
    tensor = preprocess_img(img)[np.newaxis, ...]
    emb = session.run(None, {input_name: tensor})[0][0]
    return emb / (np.linalg.norm(emb) + 1e-12)


def extract_color_sig(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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


def color_similarity(u1: np.ndarray, l1: np.ndarray, u2: np.ndarray, l2: np.ndarray) -> float:
    up = float(np.minimum(u1, u2).sum())
    lo = float(np.minimum(l1, l2).sum())
    return 0.4 * up + 0.6 * lo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update the manifest/database for a single person folder")
    parser.add_argument("--folder", required=True, help="Path to the person_* folder to update")
    parser.add_argument("--db", default=None, help="Path to the JSON database file (default: .folder_db.json inside the folder)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the JSON database if it exists")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    folder = Path(args.folder)
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")

    db_path = Path(args.db) if args.db else folder / ".folder_db.json"
    if db_path.exists() and not args.overwrite:
        # We still update the file in-place; this flag only exists so the CLI is explicit
        # about being allowed to replace an existing snapshot.
        pass

    images = sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXT])
    if not images:
        raise ValueError(f"No images found in {folder}")

    session, input_name = load_osnet()

    tracks: Dict[int, dict] = defaultdict(lambda: {
        "paths": [],
        "embs": [],
        "uhist": [],
        "lhist": [],
        "frames": [],
    })
    image_entries: List[dict] = []
    for img_path in images:
        match = FILENAME_RE.search(img_path.stem)
        frame = int(match.group(1)) if match else -1
        local_id = int(match.group(2)) if match else -1
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        emb = extract_emb(img, session, input_name)
        u_hist, l_hist = extract_color_sig(img)
        tracks[local_id]["paths"].append(img_path.name)
        tracks[local_id]["embs"].append(emb)
        tracks[local_id]["uhist"].append(u_hist)
        tracks[local_id]["lhist"].append(l_hist)
        tracks[local_id]["frames"].append(frame)
        image_entries.append({
            "file": img_path.name,
            "frame": frame,
            "local_id": local_id,
            "size": [int(img.shape[1]), int(img.shape[0])],
            "mtime": img_path.stat().st_mtime,
        })

    track_entries: Dict[str, dict] = {}
    all_embs: List[np.ndarray] = []
    all_uhist: List[np.ndarray] = []
    all_lhist: List[np.ndarray] = []

    for local_id, data in sorted(tracks.items(), key=lambda item: item[0]):
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
        track_entries[str(local_id)] = {
            "image_count": len(data["paths"]),
            "frame_min": int(min(data["frames"])),
            "frame_max": int(max(data["frames"])),
            "mean_similarity_to_track_proto": float(sims.mean()),
            "min_similarity_to_track_proto": float(sims.min()),
            "max_similarity_to_track_proto": float(sims.max()),
            "prototype_similarity_to_folder": None,
            "files": data["paths"],
        }
        all_embs.extend(data["embs"])
        all_uhist.extend(data["uhist"])
        all_lhist.extend(data["lhist"])
        track_entries[str(local_id)]["_proto_emb"] = proto_emb.tolist()
        track_entries[str(local_id)]["_proto_u"] = proto_u.tolist()
        track_entries[str(local_id)]["_proto_l"] = proto_l.tolist()

    all_embs_arr = np.vstack(all_embs)
    folder_proto = all_embs_arr.mean(axis=0)
    folder_proto /= (np.linalg.norm(folder_proto) + 1e-12)
    folder_sims = np.dot(all_embs_arr, folder_proto)
    folder_proto_u = np.mean(all_uhist, axis=0)
    folder_proto_l = np.mean(all_lhist, axis=0)
    folder_proto_u /= (folder_proto_u.sum() + 1e-12)
    folder_proto_l /= (folder_proto_l.sum() + 1e-12)

    for local_id, data in track_entries.items():
        proto_emb = np.array(data.pop("_proto_emb"), dtype=np.float32)
        proto_u = np.array(data.pop("_proto_u"), dtype=np.float32)
        proto_l = np.array(data.pop("_proto_l"), dtype=np.float32)
        data["prototype_similarity_to_folder"] = float(np.dot(proto_emb, folder_proto))
        data["prototype_color_similarity_to_folder"] = float(
            color_similarity(proto_u, proto_l, folder_proto_u, folder_proto_l)
        )

    existing_db = None
    if db_path.exists():
        try:
            existing_db = json.loads(db_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing_db = None

    previous_files = set(existing_db.get("files", [])) if isinstance(existing_db, dict) else set()
    current_files = {entry["file"] for entry in image_entries}

    payload = {
        "folder": str(folder.resolve()),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "image_count": len(image_entries),
        "files": sorted(current_files),
        "added_files": sorted(current_files - previous_files),
        "removed_files": sorted(previous_files - current_files),
        "folder_summary": {
            "folder_proto_mean_similarity": float(folder_sims.mean()),
            "folder_proto_min_similarity": float(folder_sims.min()),
            "folder_proto_max_similarity": float(folder_sims.max()),
        },
        "tracks": track_entries,
        "images": image_entries,
    }

    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Updated folder database: {db_path}")
    print(f"Folder: {folder.name}")
    print(f"Images indexed: {len(image_entries)}")
    print(f"Tracks indexed: {len(track_entries)}")
    print(
        f"Folder prototype similarity: mean {folder_sims.mean():.3f} "
        f"(min: {folder_sims.min():.3f}, max: {folder_sims.max():.3f})"
    )
    if payload["added_files"]:
        print(f"Added files: {len(payload['added_files'])}")
    if payload["removed_files"]:
        print(f"Removed files: {len(payload['removed_files'])}")


if __name__ == "__main__":
    main()
