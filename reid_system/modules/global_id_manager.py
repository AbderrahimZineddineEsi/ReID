"""
Global Identity Manager – faithful port of robust_track_linked.py logic.

Uses:
- Strict ReID quality gates (config.MIN_REID_AREA_RATIO, MIN_REID_SHARPNESS)
- Embedding decimation (config.EMBED_EVERY_N)
- Prototype + colour drift detection
- Short‑gap IoU bonus
- Same‑frame uniqueness constraint
"""

from typing import Dict, Optional, Tuple
import numpy as np
import config


class GlobalIDManager:
    def __init__(self):
        self.next_id = 1
        self.globals: Dict[int, dict] = {}          # global_id -> info
        self.local_to_global: Dict[int, int] = {}    # local_id -> global_id
        self.local_last_seen: Dict[int, int] = {}    # local_id -> frame_idx
        self.local_last_embed_frame: Dict[int, int] = {}  # last frame when embedding was updated
        self.per_frame_used_globals: set = set()
        self._fps = 30.0

    def reset_for_video(self, fps: float):
        self._fps = fps
        self.short_gap_frames = max(1, int(round(config.SHORT_GAP_SEC * fps)))
        self.local_stale_frames = max(self.short_gap_frames * 2,
                                      int(round(config.LOCAL_STALE_SEC * fps)))

    def new_frame(self):
        self.per_frame_used_globals.clear()

    def update(self, local_id: int, frame_idx: int,
               emb: np.ndarray, color_sig: Dict[str, np.ndarray],
               bbox: Tuple[int, int, int, int],
               reid_ok: bool) -> Tuple[int, bool, str]:
        """
        Args:
            reid_ok: Whether this crop passes the strict ReID quality gates.
        """
        # ---- If we already know this local track ----
        if local_id in self.local_to_global:
            gid = self.local_to_global[local_id]
            info = self.globals.get(gid)
            if info is None:
                # Global lost? remap
                pass
            else:
                # Check if we should refresh the embedding (decimation)
                should_refresh = (frame_idx - self.local_last_embed_frame.get(local_id, -10**9)
                                  >= config.EMBED_EVERY_N and reid_ok)
                if should_refresh:
                    # Drift check against current global
                    current_reid = float(np.dot(emb, info["prototype"]))
                    current_colour = self._color_similarity(
                        color_sig["upper"], color_sig["lower"],
                        info["color_upper"], info["color_lower"]
                    )
                    if (current_reid < config.DRIFT_REID_THRESHOLD or
                        current_colour < config.DRIFT_COLOR_THRESHOLD):
                        # Split: create new global and reassign local to it
                        new_gid = self._create_global(emb, color_sig, bbox, frame_idx)
                        self.local_to_global[local_id] = new_gid
                        self.local_last_embed_frame[local_id] = frame_idx
                        self.local_last_seen[local_id] = frame_idx
                        return new_gid, True, f"split from G{gid}"
                    else:
                        # Update existing prototype
                        self._update_prototype(info, emb, color_sig, bbox, frame_idx)
                else:
                    # Not refreshing, just mark seen and update last bbox
                    info["last_seen_frame"] = frame_idx
                    info["last_bbox"] = bbox

                self.local_last_seen[local_id] = frame_idx
                # Same-frame uniqueness: if this global already used, force split
                if gid in self.per_frame_used_globals:
                    # Conflict – should not happen if mapping is consistent, but safeguard
                    new_gid = self._create_global(emb, color_sig, bbox, frame_idx)
                    self.local_to_global[local_id] = new_gid
                    self.local_last_embed_frame[local_id] = frame_idx
                    return new_gid, True, f"same-frame conflict, split from G{gid}"
                self.per_frame_used_globals.add(gid)
                return gid, False, f"updated G{gid}"

        # ---- No prior mapping, try to match existing global ----
        matched_gid = self._match_global(emb, color_sig, bbox, frame_idx)
        if matched_gid is not None:
            # Same-frame uniqueness check
            if matched_gid in self.per_frame_used_globals:
                # Conflict, reject and create new
                print(f"[Frame {frame_idx}] CONFLICT: L{local_id} wants G{matched_gid} but already used")
                matched_gid = None
            else:
                self.local_to_global[local_id] = matched_gid
                self.local_last_seen[local_id] = frame_idx
                if reid_ok:
                    # Update prototype with this sample
                    self._update_prototype(self.globals[matched_gid], emb, color_sig, bbox, frame_idx)
                    self.local_last_embed_frame[local_id] = frame_idx
                self.per_frame_used_globals.add(matched_gid)
                return matched_gid, False, f"re‑linked to G{matched_gid}"

        # ---- No match -> new global ----
        gid = self._create_global(emb, color_sig, bbox, frame_idx)
        self.local_to_global[local_id] = gid
        self.local_last_seen[local_id] = frame_idx
        if reid_ok:
            self.local_last_embed_frame[local_id] = frame_idx
        self.per_frame_used_globals.add(gid)
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

    def _update_prototype(self, info, emb, color_sig, bbox, frame_idx):
        alpha = config.PROTOTYPE_ALPHA
        c_alpha = config.COLOR_ALPHA
        # Embedding
        blended = (1.0 - alpha) * info["prototype"] + alpha * emb
        info["prototype"] = blended / (np.linalg.norm(blended) + 1e-12)
        # Colour
        info["color_upper"] = self._blend_hist(info["color_upper"], color_sig["upper"], c_alpha)
        info["color_lower"] = self._blend_hist(info["color_lower"], color_sig["lower"], c_alpha)
        info["samples"] += 1
        info["last_seen_frame"] = frame_idx
        info["last_bbox"] = bbox

    def _match_global(self, emb, color_sig, bbox, frame_idx) -> Optional[int]:
        best_gid = None
        best_score = -1.0
        second_score = -1.0

        for gid, info in self.globals.items():
            if gid in self.per_frame_used_globals:
                continue
            sim = float(np.dot(emb, info["prototype"]))
            col_sim = self._color_similarity(
                color_sig["upper"], color_sig["lower"],
                info["color_upper"], info["color_lower"]
            )
            if col_sim < config.COLOR_THRESHOLD:
                continue

            age = frame_idx - info["last_seen_frame"]
            if age <= self.short_gap_frames:
                iou = self._bbox_iou(bbox, info["last_bbox"])
                score = sim + config.IOU_WEIGHT * iou
                threshold = config.SHORT_LINK_THRESHOLD
            else:
                score = sim
                threshold = config.LONG_LINK_THRESHOLD

            if score >= threshold:
                if score > best_score:
                    second_score = best_score
                    best_score = score
                    best_gid = gid
                elif score > second_score:
                    second_score = score

        if best_gid is not None and (best_score - second_score) >= config.AMBIGUITY_MARGIN:
            return best_gid
        return None

    def _color_similarity(self, u1, l1, u2, l2) -> float:
        up = float(np.minimum(u1, u2).sum())
        lo = float(np.minimum(l1, l2).sum())
        return 0.4 * up + 0.6 * lo

    def _bbox_iou(self, a, b) -> float:
        x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
        x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        area_a = (a[2]-a[0]) * (a[3]-a[1])
        area_b = (b[2]-b[0]) * (b[3]-b[1])
        union = area_a + area_b - inter
        return inter/union if union>0 else 0.0

    def _blend_hist(self, a, b, alpha):
        mixed = (1.0 - alpha)*a + alpha*b
        s = mixed.sum()
        if s>0: mixed /= s
        return mixed

    def cleanup_stale(self, frame_idx):
        stale = [lid for lid, last in self.local_last_seen.items()
                 if frame_idx - last > self.local_stale_frames]
        for lid in stale:
            self.local_last_seen.pop(lid, None)
            self.local_to_global.pop(lid, None)
            self.local_last_embed_frame.pop(lid, None)