"""
ReID Embedding Module – extracts appearance vectors from person crops using OSNet (ONNX).

OSNet (Omni‑Scale Network) is a lightweight CNN designed specifically for person re‑id.
We use a pre‑trained ONNX model that runs on CPU (fast enough for real‑time if we
embed only every few frames).

Public API:
    embedder = ReIDEmbedder(model_path, target_size=(256,128))
    emb = embedder.extract_embedding(crop)   # returns L2‑normalized 512-D numpy array
    color_sig = embedder.extract_color_signature(crop)   # dict with 'upper','lower' histograms
"""

from typing import Dict, Tuple
import numpy as np
import cv2
import onnxruntime as ort


class ReIDEmbedder:
    def __init__(
        self,
        model_path: str = "./models/osnet_x1_0.onnx",
        target_size: Tuple[int, int] = (256, 128),  # H x W
    ):
        self.target_size = target_size
        # Load ONNX session on CPU – OSNet is small and GPU‑to‑CPU transfer would be slower
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        print(f"ReID embedder loaded (ONNX, CPU). Input size: {target_size[1]}x{target_size[0]}")

    def extract_embedding(self, crop_bgr: np.ndarray) -> np.ndarray:
        """
        Extract OSNet embedding from a person crop (BGR image).

        Returns:
            A 1‑D float32 array of length 512, L2‑normalised.
        """
        # Pre‑process: resize, BGR->RGB, normalize with ImageNet mean/std, add batch dim
        h, w = self.target_size
        resized = cv2.resize(crop_bgr, (w, h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # HWC -> CHW for OSNet
        rgb = np.transpose(rgb, (2, 0, 1))

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        rgb = (rgb - mean) / std

        tensor = rgb[np.newaxis, ...]  # (1, 3, H, W)

        # Run ONNX inference
        emb = self.session.run(None, {self.input_name: tensor})[0][0]  # shape (512,)
        # L2 normalise
        norm = np.linalg.norm(emb)
        if norm > 1e-12:
            emb = emb / norm
        return emb.astype(np.float32)

    def extract_color_signature(self, crop_bgr: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute a coarse clothing colour descriptor:
          - Resize crop to 64x128
          - Split into upper/lower body halves
          - Compute 8‑bin LAB histograms for each half

        Returns a dict with keys 'upper' and 'lower', each a float32 array of length 8³=512.
        """
        resized = cv2.resize(crop_bgr, (64, 128))
        lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
        split = lab.shape[0] // 2
        upper = lab[:split, :, :]
        lower = lab[split:, :, :]
        return {
            "upper": self._compute_lab_hist(upper),
            "lower": self._compute_lab_hist(lower),
        }

    def _compute_lab_hist(self, region: np.ndarray) -> np.ndarray:
        """Compute 8x8x8 LAB histogram and normalise to sum=1."""
        hist = cv2.calcHist([region], [0, 1, 2], None, [8, 8, 8],
                            [0, 256, 0, 256, 0, 256]).flatten().astype(np.float32)
        s = float(hist.sum())
        if s > 0:
            hist /= s
        return hist