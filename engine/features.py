"""Shared per-pixel feature extraction for the land-cover classifier.

Used identically at training time (engine/train_model.py) and inference time
(engine/segmentation.py) so the two never drift out of sync.
"""

import cv2
import numpy as np

FEATURE_NAMES = ["H", "S", "V", "R", "G", "B", "exg", "vari", "tex"]
N_FEATURES = len(FEATURE_NAMES)


def _local_std(gray, ksize=11):
    mean = cv2.blur(gray, (ksize, ksize))
    sq = cv2.blur((gray.astype(np.float32) - mean) ** 2, (ksize, ksize))
    return np.sqrt(np.maximum(sq, 0.0))


def extract_features(bgr):
    """bgr: HxWx3 uint8 image.

    Returns (feats, smooth): feats is HxWx9 float32 (H, S, V, R, G, B, ExG,
    VARI, local texture std); smooth is the bilateral-filtered BGR image,
    reused downstream for superpixel aggregation.
    """
    smooth = cv2.bilateralFilter(bgr, d=7, sigmaColor=45, sigmaSpace=9)

    rgbf = cv2.cvtColor(smooth, cv2.COLOR_BGR2RGB).astype(np.float32)
    R, G, B = rgbf[..., 0], rgbf[..., 1], rgbf[..., 2]

    hsv = cv2.cvtColor(smooth, cv2.COLOR_BGR2HSV).astype(np.float32)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
    tex = _local_std(gray, 11)

    eps = 1e-3
    exg = 2.0 * G - R - B
    vari = (G - R) / np.maximum(G + R - B, eps)

    feats = np.stack([H, S, V, R, G, B, exg, vari, tex], axis=-1).astype(np.float32)
    return feats, smooth
