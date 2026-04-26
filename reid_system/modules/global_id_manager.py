"""
Global Identity Manager – links local track IDs to persistent person identities.

This module implements the online re‑identification logic:
- Prototype embedding + colour signature per global identity.
- Short‑term re‑link using IoU bonus, long‑term using appearance only.
- Drift detection: if a track diverges from its assigned global identity, it splits.
- Stale mapping cleanup.

All thresholds are read from config.py for easy tuning.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import config


class GlobalIDManager:
    def __init__(self):
        self.next_id = 1
        self.globals: Dict[int, dict] = {}  # global_id -> prototype info
        self.local_to_global: Dict[int, int] = {}  # local_id -> global_id
        self.local_last_seen: Dict[int, int] = {}  # local_id -> frame_idx for staleness
        self.local_last_embed: Dict[int, int] = {}  # local_id -> last frame when embedding was updated

        # Frame‑dependent parameters (set once per camera based on fps)
        self.short_gap_frames: int = 90   # will be recalculated in reset_for_video()

    def reset_for_video(self, fps: float):
        """Set time‑dependent parameters using the video's frame rate."""
        self.fps = fps
        self.short_gap_frames = max(1, int(round(config.SHORT_GAP_SEC * fps)))
        self.local_stale_frames = max(
            self.short_gap_frames * 2,
            int(round(config.LOCAL_STALE_SEC * fps)),
        )

    def update(
        self,
        local_id: int,
        frame_idx: int,
        emb: np.ndarray,
        color_sig: Dict[str, np.ndarray],
        bbox: Tuple[int, int, int, int],
    ) -> Tuple[int, bool, str]:
        """
        Process a new detection from a local track and return its global identity.

        Args:
            local_id: ByteTrack's track ID.
            frame_idx: Current frame number.
            emb: L2‑normalised ReID embedding for this crop.
            color_sig: Dict with 'upper' and 'lower' normalised LAB histograms.
            bbox: (x1,y1,x2,y2) bounding box.

        Returns:
            global_id: The assigned global identity.
            is_new: True if a new global ID was created.
            reason: Description of what happened (for logging).
        """
        # If we already have a mapping and it hasn't drifted, keep it (Pass 1)
        if local_id in self.local_to_global:
            gid = self.local_to_global[local_id]
            info = self.globals.get(gid)
            if info is not None:
                # Check for drift
                if self._check_drift(gid, emb, color_sig):
                    new_gid = self._create_global(emb, color_sig, bbox, frame_idx)
                    self.local_to_global[local_id] = new_gid
                    self.local_last_embed[local_id] = frame_idx
                    self.local_last_seen[local_id] = frame_idx
                    return new_gid, True, f"split from G{gid} due to drift"
                else:
                    # Update existing prototype
                    self._update_global(info, emb, color_sig, bbox, frame_idx)
                    self.local_last_seen[local_id] = frame_idx
                    self.local_last_embed[local_id] = frame_idx
                    return gid, False, f"updated G{gid}"

        # No prior mapping or mapping lost – search for a matching global identity
        matched_gid = self._match_global(emb, color_sig, bbox, frame_idx, local_id)
        if matched_gid is not None:
            self.local_to_global[local_id] = matched_gid
            self.local_last_seen[local_id] = frame_idx
            self.local_last_embed[local_id] = frame_idx
            return matched_gid, False, f"re‑linked to G{matched_gid}"
        else:
            gid = self._create_global(emb, color_sig, bbox, frame_idx)
            self.local_to_global[local_id] = gid
            self.local_last_seen[local_id] = frame_idx
            self.local_last_embed[local_id] = frame_idx
            return gid, True, f"new G{gid}"

    def _create_global(self, emb, color_sig, bbox, frame_idx) -> int:
        gid = self.next_id
        self.next_id += 1
        self.globals[gid] = {
            "prototype": emb.copy(),
            "color_upper": color_sig["upper"].copy(),
            "color_lower": color_sig["lower"].copy(),
            "samples": 1,
            "last_seen_frame": frame_idx,
            "last_bbox": bbox,
        }
        return gid

    def _update_global(self, info, emb, color_sig, bbox, frame_idx):
        alpha = config.PROTOTYPE_ALPHA
        color_alpha = config.COLOR_ALPHA
        # Update embedding prototype as moving average
        blended = (1.0 - alpha) * info["prototype"] + alpha * emb
        info["prototype"] = blended / (np.linalg.norm(blended) + 1e-12)
        # Update colour histograms
        info["color_upper"] = self._blend_hist(info["color_upper"], color_sig["upper"], color_alpha)
        info["color_lower"] = self._blend_hist(info["color_lower"], color_sig["lower"], color_alpha)
        info["samples"] += 1
        info["last_seen_frame"] = frame_idx
        info["last_bbox"] = bbox

    def _check_drift(self, gid, emb, color_sig) -> bool:
        info = self.globals[gid]
        reid_sim = float(np.dot(emb, info["prototype"]))
        color_sim = self._color_similarity(
            color_sig["upper"], color_sig["lower"],
            info["color_upper"], info["color_lower"]
        )
        if reid_sim < config.DRIFT_REID_THRESHOLD or color_sim < config.DRIFT_COLOR_THRESHOLD:
            return True
        return False

    def _match_global(self, emb, color_sig, bbox, frame_idx, exclude_local=None) -> Optional[int]:
        candidates = []
        for gid, info in self.globals.items():
            # Skip if assigned to the same local track we are processing (shouldn't happen)
            # Compute appearance similarity
            reid_sim = float(np.dot(emb, info["prototype"]))
            color_sim = self._color_similarity(
                color_sig["upper"], color_sig["lower"],
                info["color_upper"], info["color_lower"]
            )
            if color_sim < config.COLOR_THRESHOLD:
                continue

            age_frames = frame_idx - info["last_seen_frame"]
            if age_frames <= self.short_gap_frames:
                # Add IoU bonus for recent tracks
                iou = self._bbox_iou(bbox, info["last_bbox"])
                score = reid_sim + config.IOU_WEIGHT * iou
                threshold = config.SHORT_LINK_THRESHOLD
            else:
                score = reid_sim
                threshold = config.LONG_LINK_THRESHOLD

            if score >= threshold:
                candidates.append((gid, score, reid_sim, color_sim))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[1], reverse=True)
        best = candidates[0]
        # Ambiguity check: if second best is too close, reject
        if len(candidates) > 1:
            second = candidates[1]
            if (best[1] - second[1]) < config.AMBIGUITY_MARGIN:
                return None
        return best[0]

    def _color_similarity(self, u1, l1, u2, l2) -> float:
        up = float(np.minimum(u1, u2).sum())
        lo = float(np.minimum(l1, l2).sum())
        return 0.4 * up + 0.6 * lo  # lower body is usually more distinctive

    def _bbox_iou(self, a, b) -> float:
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _blend_hist(self, a, b, alpha):
        mixed = (1.0 - alpha) * a + alpha * b
        s = mixed.sum()
        if s > 0:
            mixed /= s
        return mixed

    def cleanup_stale_local_ids(self, frame_idx):
        """Remove local IDs not seen for a long time to prevent memory growth."""
        stale = [
            lid for lid, last_seen in self.local_last_seen.items()
            if frame_idx - last_seen > self.local_stale_frames
        ]
        for lid in stale:
            self.local_last_seen.pop(lid, None)
            self.local_to_global.pop(lid, None)
            self.local_last_embed.pop(lid, None)