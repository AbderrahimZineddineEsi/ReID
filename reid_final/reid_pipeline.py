"""
Person Re-ID Pipeline — Kaggle Script
Stack : RT-DETRv2-L · BoT-SORT · CLIP-ReID (via BoxMOT) · Offline Greedy Merge
GPU   : Detection on GPU-0, ReID on GPU-1 (falls back to GPU-0)

Outputs (all inside OUTPUT_DIR):
  persons/person_XXX/   — per-person crop folders
  reid_report.txt       — per-person stats + intra/inter-track similarities
  results.zip           — zip of persons/ + reid_report.txt
"""

# ── std / kaggle env ──────────────────────────────────────────────────────────
import os, sys, shutil, warnings, zipfile
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION  — edit these
# ══════════════════════════════════════════════════════════════════════════════
INPUT_VIDEO   = "/kaggle/input/datasets/almightyj/oxford-town-centre/TownCentreXVID.mp4"
OUTPUT_DIR    = Path("/kaggle/working/outputs")

MIN_CROP_W    = 40
MIN_CROP_H    = 80
MIN_SHARPNESS = 30.0

MERGE_THRESHOLD   = 0.85
MAX_FRAME_OVERLAP = 5
MIN_CROPS_FINAL   = 15

REID_WEIGHTS  = Path("clip_market1501.pt")
DET_MODEL     = "rtdetr-l.pt"

EMBED_BATCH   = 128        # larger batch → faster offline embedding
JPEG_Q        = 90
NUM_IO_THREADS = 4         # parallel disk I/O for crop saving
# ══════════════════════════════════════════════════════════════════════════════

def setup_dirs():
    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        raise RuntimeError("No GPU found. Enable GPU: Settings → Accelerator → GPU T4 x2")
    det_device  = torch.device("cuda:0")
    reid_device = torch.device("cuda:1" if n_gpus > 1 else "cuda:0")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    crops_dir   = OUTPUT_DIR / "track_crops"
    persons_dir = OUTPUT_DIR / "persons"
    for d in [crops_dir, persons_dir]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir()
    return det_device, reid_device, crops_dir, persons_dir


# ── Fast sharpness check (inline, no function-call overhead) ─────────────────
_LAP_THRESH = MIN_SHARPNESS

def is_sharp(gray: np.ndarray) -> bool:
    return float(cv2.Laplacian(gray, cv2.CV_32F).var()) >= _LAP_THRESH


# ── Step 1 : Detection + Tracking + Crop extraction ──────────────────────────
def step1_track_and_crop(det_device, reid_device, crops_dir):
    from ultralytics import RTDETR
    from boxmot.trackers.botsort.botsort import BotSort

    detector = RTDETR(DET_MODEL)
    detector.to(det_device)

    tracker = BotSort(
        reid_weights=REID_WEIGHTS,
        device=reid_device,
        half=False,
        with_reid=True,
    )

    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {INPUT_VIDEO}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_idx       = 0
    crops_per_track = defaultdict(int)
    save_queue      = []   # (path, img, quality) buffered for batch I/O

    def flush_saves(queue):
        for path, img, q in queue:
            cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, q])
        queue.clear()

    # Pre-create track dirs on demand — avoid repeated mkdir
    known_track_dirs = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        results = detector(frame, classes=[0], verbose=False)
        boxes   = results[0].boxes

        if boxes is not None and len(boxes) > 0:
            dets = torch.cat(
                [boxes.xyxy, boxes.conf.unsqueeze(1), boxes.cls.unsqueeze(1)], dim=1
            ).cpu().numpy().astype(np.float32)
        else:
            dets = np.empty((0, 6), dtype=np.float32)

        tracks = tracker.update(dets, frame)
        if tracks is None or len(tracks) == 0:
            continue

        for t in tracks:
            x1, y1, x2, y2 = int(t[0]), int(t[1]), int(t[2]), int(t[3])
            track_id        = int(t[4])

            x1 = max(0, x1);  y1 = max(0, y1)
            x2 = min(frame_w, x2);  y2 = min(frame_h, y2)
            bw, bh = x2 - x1, y2 - y1

            if bw < MIN_CROP_W or bh < MIN_CROP_H:
                continue

            crop = frame[y1:y2, x1:x2]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            if not is_sharp(gray):
                continue

            tid_key = f"track_{track_id:04d}"
            if tid_key not in known_track_dirs:
                (crops_dir / tid_key).mkdir(exist_ok=True)
                known_track_dirs.add(tid_key)

            fname = crops_dir / tid_key / f"{frame_idx:06d}_{track_id:04d}.jpg"
            save_queue.append((fname, crop, JPEG_Q))
            crops_per_track[track_id] += 1

        # Flush writes every 500 frames to avoid memory blow-up
        if frame_idx % 500 == 0:
            flush_saves(save_queue)

    flush_saves(save_queue)
    cap.release()
    return crops_per_track, frame_w, frame_h


# ── Step 2 : Offline embedding + greedy merge ─────────────────────────────────
def step2_embed_and_merge(reid_device, crops_dir):
    from boxmot.reid.core.reid import ReID

    reid_model = ReID(
        weights=REID_WEIGHTS,
        device=reid_device,
        half=False,
        path=None,
    )

    def embed_images(bgr_list):
        all_embs = []
        for i in range(0, len(bgr_list), EMBED_BATCH):
            batch = bgr_list[i: i + EMBED_BATCH]
            with torch.no_grad():
                embs = reid_model(batch)
            norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
            all_embs.append(embs / norms)
        return np.vstack(all_embs) if all_embs else np.empty((0, 512))

    # ── parallel image loading ─────────────────────────────────────────────
    def load_track(tdir):
        imgs    = sorted([p for p in tdir.iterdir() if p.suffix == ".jpg"])
        if not imgs:
            return None
        frame_nums = []
        bgr_crops  = []
        for p in imgs:
            parts = p.stem.split("_")
            try:
                frame_nums.append(int(parts[0]))
            except (ValueError, IndexError):
                pass
            img = cv2.imread(str(p))
            if img is not None:
                bgr_crops.append(img)
        if not bgr_crops:
            return None
        return tdir.name, imgs, frame_nums, bgr_crops

    track_dirs = sorted([d for d in crops_dir.iterdir() if d.is_dir()], key=lambda d: d.name)

    track_data = {}
    for tdir in track_dirs:
        result = load_track(tdir)
        if result is None:
            continue
        name, imgs, frame_nums, bgr_crops = result
        embs  = embed_images(bgr_crops)
        proto = embs.mean(axis=0)
        proto = proto / (np.linalg.norm(proto) + 1e-8)
        track_data[name] = {
            "prototype": proto,
            "frame_set": set(frame_nums),
            "paths"    : imgs,
            "size"     : len(imgs),
            # store per-crop embeddings for intra-track similarity report
            "embs"     : embs,
        }

    # ── Greedy merge ──────────────────────────────────────────────────────
    track_names = list(track_data.keys())
    n           = len(track_names)

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b    = track_names[i], track_names[j]
            overlap = len(track_data[a]["frame_set"] & track_data[b]["frame_set"])
            if overlap > MAX_FRAME_OVERLAP:
                continue
            sim = float(track_data[a]["prototype"] @ track_data[b]["prototype"])
            if sim >= MERGE_THRESHOLD:
                pairs.append((sim, a, b))
    pairs.sort(key=lambda x: -x[0])

    parent = {n: n for n in track_names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return
        if track_data[px]["size"] < track_data[py]["size"]:
            px, py = py, px
        parent[py] = px
        track_data[px]["frame_set"] |= track_data[py]["frame_set"]
        track_data[px]["paths"]     += track_data[py]["paths"]
        track_data[px]["size"]      += track_data[py]["size"]
        # merge embs for similarity reporting
        track_data[px]["embs"] = np.vstack([track_data[px]["embs"], track_data[py]["embs"]])

    for sim, a, b in pairs:
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        overlap = len(track_data[ra]["frame_set"] & track_data[rb]["frame_set"])
        if overlap > MAX_FRAME_OVERLAP:
            continue
        union(ra, rb)

    groups = defaultdict(list)
    for name in track_names:
        groups[find(name)].append(name)

    return track_data, groups, pairs


# ── Step 3 : Write person folders ─────────────────────────────────────────────
def step3_write_persons(track_data, groups, persons_dir):
    sorted_roots = sorted(groups.keys(), key=lambda r: -track_data[r]["size"])

    person_idx   = 0
    identity_map = {}   # (frame_idx, track_id) → person_id
    person_info  = {}   # person_id → metadata

    def find(x):
        root = track_data  # closure trick: parent dict is embedded in track_data
        # re-implement mini find without the outer parent dict
        # We know groups already resolved roots, so just return x
        return x

    def copy_file(args):
        src, dest = args
        shutil.copy2(src, dest)

    for root in sorted_roots:
        all_paths = track_data[root]["paths"]
        track_members = groups[root]

        if len(all_paths) < MIN_CROPS_FINAL:
            continue

        person_idx += 1
        pid  = person_idx
        pdir = persons_dir / f"person_{pid:03d}"
        pdir.mkdir(exist_ok=True)

        copy_args = []
        for src in all_paths:
            parts = src.stem.split("_")
            try:
                frame_no = int(parts[0])
                track_id = int(parts[1])
                identity_map[(frame_no, track_id)] = pid
            except (ValueError, IndexError):
                pass
            dest = pdir / src.name
            if dest.exists():
                k = 1
                while (pdir / f"{src.stem}_{k}{src.suffix}").exists():
                    k += 1
                dest = pdir / f"{src.stem}_{k}{src.suffix}"
            copy_args.append((src, dest))

        with ThreadPoolExecutor(max_workers=NUM_IO_THREADS) as ex:
            list(ex.map(copy_file, copy_args))

        # Intra-track similarity (mean pairwise cosine between all crops)
        embs = track_data[root]["embs"]
        if len(embs) > 1:
            # sample at most 200 crops to keep it fast
            idx_sample = np.random.choice(len(embs), min(200, len(embs)), replace=False)
            e = embs[idx_sample]
            sim_matrix  = e @ e.T
            mask        = np.triu(np.ones_like(sim_matrix, dtype=bool), k=1)
            intra_sim   = float(sim_matrix[mask].mean()) if mask.any() else 1.0
        else:
            intra_sim = 1.0

        person_info[pid] = {
            "num_tracks" : len(track_members),
            "track_ids"  : track_members,
            "num_crops"  : len(all_paths),
            "intra_sim"  : intra_sim,
        }

    return identity_map, person_info


# ── Step 4 : Generate reid_report.txt ─────────────────────────────────────────
def step4_write_report(person_info, pairs, track_data, output_dir):
    report_path = output_dir / "reid_report.txt"

    # Build a track → person lookup
    # We need the resolved groups mapping
    # person_info already has track_ids per person
    track_to_person = {}
    for pid, info in person_info.items():
        for tid in info["track_ids"]:
            track_to_person[tid] = pid

    lines = []
    lines.append("=" * 70)
    lines.append("  PERSON RE-ID REPORT")
    lines.append("=" * 70)
    lines.append(f"  Total persons : {len(person_info)}")
    lines.append(f"  MERGE_THRESHOLD   = {MERGE_THRESHOLD}")
    lines.append(f"  MAX_FRAME_OVERLAP = {MAX_FRAME_OVERLAP}")
    lines.append(f"  MIN_CROPS_FINAL   = {MIN_CROPS_FINAL}")
    lines.append("")

    for pid in sorted(person_info.keys()):
        info = person_info[pid]
        lines.append(f"person_{pid:03d}")
        lines.append(f"  num_tracks      : {info['num_tracks']}")
        lines.append(f"  track_ids       : {', '.join(info['track_ids'])}")
        lines.append(f"  num_crops       : {info['num_crops']}")
        lines.append(f"  intra_sim (avg) : {info['intra_sim']:.4f}   "
                     f"(mean pairwise cosine over sampled crops)")

        # Inter-track similarities for this person's tracks
        tids = info["track_ids"]
        if len(tids) > 1:
            lines.append("  inter-track similarities:")
            for i in range(len(tids)):
                for j in range(i + 1, len(tids)):
                    a, b = tids[i], tids[j]
                    if a in track_data and b in track_data:
                        sim = float(track_data[a]["prototype"] @ track_data[b]["prototype"])
                        lines.append(f"    {a} ↔ {b} : {sim:.4f}")
        lines.append("")

    report_path.write_text("\n".join(lines))
    return report_path


# ── Step 5 : Person-grid visualization ──────────────────────────────────────
def step5_visualize(persons_dir, person_info, output_dir):
    """
    For every person, build one row per track, each row showing
    N_COLS evenly-spaced crops on a pure-white background.
    All person images are stacked vertically into persons_overview.jpg.
    """
    import matplotlib
    matplotlib.use("Agg")            # no display needed on Kaggle
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    N_COLS      = 5    # crops per track
    THUMB_W     = 80   # thumbnail width  (px, used for aspect hint)
    THUMB_H     = 160  # thumbnail height
    PAD         = 8    # whitespace between cells (pt)
    LABEL_W     = 1.4  # inches for the row label column
    CELL_W      = 1.1  # inches per crop column
    CELL_H      = 2.0  # inches per row
    FONT_PERSON = 11
    FONT_TRACK  = 7
    BG          = "white"
    LABEL_COLOR = "#333333"
    SEP_COLOR   = "#cccccc"

    def pick_n(paths, n):
        if len(paths) <= n:
            return paths
        idxs = [round(i * (len(paths) - 1) / (n - 1)) for i in range(n)]
        return [paths[i] for i in sorted(set(idxs))]

    def load_rgb(p):
        img = cv2.imread(str(p))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None

    person_dirs = sorted(
        [d for d in persons_dir.iterdir() if d.is_dir()],
        key=lambda d: d.name
    )
    if not person_dirs:
        return None

    saved_paths = []

    for pdir in person_dirs:
        pid_str = pdir.name          # e.g. "person_001"
        pid     = int(pid_str.split("_")[1])
        info    = person_info.get(pid)
        if info is None:
            continue

        track_ids = sorted(info["track_ids"])
        n_rows    = len(track_ids)
        n_cols    = N_COLS

        fig_w = LABEL_W + n_cols * CELL_W
        fig_h = n_rows * CELL_H

        fig = plt.figure(figsize=(fig_w, fig_h), facecolor=BG)

        # title
        fig.suptitle(
            f"{pid_str}  ·  {info['num_tracks']} track(s)  ·  "
            f"{info['num_crops']} crops  ·  "
            f"intra-sim {info['intra_sim']:.3f}",
            fontsize=FONT_PERSON,
            color=LABEL_COLOR,
            x=0.5,
            y=1.0,
            va="bottom",
            fontweight="bold",
        )

        outer = gridspec.GridSpec(
            n_rows, 1, figure=fig,
            hspace=0.55, top=0.92, bottom=0.02, left=0.0, right=1.0
        )

        for row_i, tid in enumerate(track_ids):
            # find crops that belong to this track (filename contains the track id)
            tid_num = int(tid.split("_")[1])
            all_crops = sorted(
                [p for p in pdir.glob(f"*_{tid_num:04d}.jpg")],
                key=lambda p: p.name
            )
            # fallback: if person was merged from a single track, all crops
            if not all_crops:
                all_crops = sorted(pdir.glob("*.jpg"), key=lambda p: p.name)

            samples = pick_n(all_crops, n_cols)

            inner = gridspec.GridSpecFromSubplotSpec(
                1, n_cols + 1,           # +1 for the label column
                subplot_spec=outer[row_i],
                wspace=0.04,
                width_ratios=[LABEL_W] + [CELL_W] * n_cols,
            )

            # ── row label ─────────────────────────────────────────────────
            ax_lbl = fig.add_subplot(inner[0])
            ax_lbl.set_facecolor(BG)
            ax_lbl.set_xticks([]); ax_lbl.set_yticks([])
            for sp in ax_lbl.spines.values():
                sp.set_visible(False)
            ax_lbl.text(
                0.92, 0.5, tid,
                color=LABEL_COLOR, fontsize=FONT_TRACK,
                ha="right", va="center",
                transform=ax_lbl.transAxes,
            )

            # ── crop cells ────────────────────────────────────────────────
            for col_i in range(n_cols):
                ax = fig.add_subplot(inner[col_i + 1])
                ax.set_facecolor(BG)
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_edgecolor(SEP_COLOR)
                    sp.set_linewidth(0.5)

                if col_i < len(samples):
                    rgb = load_rgb(samples[col_i])
                    if rgb is not None:
                        ax.imshow(rgb, aspect="auto")

        out_path = output_dir / f"{pid_str}_overview.jpg"
        fig.savefig(
            str(out_path), dpi=110,
            bbox_inches="tight",
            facecolor=BG,
        )
        plt.close(fig)
        saved_paths.append(out_path)

    return saved_paths


# ── Step 6 : Zip everything ───────────────────────────────────────────────────
def step6_zip(persons_dir, report_path, overview_paths, output_dir):
    zip_path = output_dir / "results.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        zf.write(report_path, arcname="reid_report.txt")
        for f in sorted(persons_dir.rglob("*")):
            if f.is_file():
                zf.write(f, arcname=f.relative_to(output_dir))
        if overview_paths:
            for op in overview_paths:
                zf.write(op, arcname=op.name)
    return zip_path


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    det_device, reid_device, crops_dir, persons_dir = setup_dirs()

    # Step 1 — detect, track, save crops
    crops_per_track, frame_w, frame_h = step1_track_and_crop(det_device, reid_device, crops_dir)

    # Step 2 — embed + greedy merge
    track_data, groups, pairs = step2_embed_and_merge(reid_device, crops_dir)

    # Step 3 — write person folders
    identity_map, person_info = step3_write_persons(track_data, groups, persons_dir)

    # Step 4 — text report
    report_path = step4_write_report(person_info, pairs, track_data, OUTPUT_DIR)

    # Step 5 — per-person grid images
    overview_paths = step5_visualize(persons_dir, person_info, OUTPUT_DIR)

    # Step 6 — zip everything
    zip_path = step6_zip(persons_dir, report_path, overview_paths, OUTPUT_DIR)

    print(f"Done. Results at: {zip_path}")


if __name__ == "__main__":
    main()
