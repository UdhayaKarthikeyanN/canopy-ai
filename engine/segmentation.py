"""Fully local pixel-level land-cover segmentation engine (deterministic).

Pipeline: bilateral edge-preserving prefilter -> per-pixel classification by a
locally-trained Random Forest (engine/train_model.py, engine/model/) on
HSV/RGB/vegetation-index/texture features -> SLIC superpixel majority
aggregation -> mode filtering -> small-region removal -> shadow reassignment ->
shape-based building/road disambiguation -> final consolidation.
"""

import os

import cv2
import joblib
import numpy as np

from engine.features import N_FEATURES, extract_features

OPEN = 0
TREES = 1
GRASS = 2
BUILDINGS = 3
ROADS = 4
WATER = 5
BARE_SOIL = 6

CLASS_NAMES = {
    OPEN: "Plantable / Open",
    TREES: "Trees / Canopy",
    GRASS: "Grass / Low Vegetation",
    BUILDINGS: "Buildings",
    ROADS: "Roads / Paved",
    WATER: "Water",
    BARE_SOIL: "Bare Soil",
}

CLASS_COLORS_RGB = {
    OPEN: (255, 238, 88),
    TREES: (27, 122, 61),
    GRASS: (156, 204, 101),
    BUILDINGS: (230, 81, 0),
    ROADS: (97, 97, 97),
    WATER: (30, 136, 229),
    BARE_SOIL: (164, 134, 116),
}

ALL_CLASSES = [OPEN, TREES, GRASS, BUILDINGS, ROADS, WATER, BARE_SOIL]
PLANTABLE_CLASSES = (OPEN, GRASS, BARE_SOIL)

_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "land_cover_rf.joblib")
_MODEL = joblib.load(_MODEL_PATH)


def _disk(radius):
    r = int(max(1, round(radius)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _mode_filter(labels, ksize):
    if ksize < 3 or ksize % 2 == 0:
        ksize = 3
    best_count = np.full(labels.shape, -1.0, dtype=np.float32)
    best_class = np.zeros(labels.shape, dtype=np.uint8)
    for cls in ALL_CLASSES:
        onehot = (labels == cls).astype(np.float32)
        v = cv2.blur(onehot, (ksize, ksize))
        upd = v > best_count
        best_count[upd] = v[upd]
        best_class[upd] = cls
    return best_class


def _remove_small_regions(labels, min_area):
    out = labels.copy()
    for cls in ALL_CLASSES:
        if cls == OPEN:
            continue
        binm = (labels == cls).astype(np.uint8)
        n, cc, stats, _ = cv2.connectedComponentsWithStats(binm, connectivity=8)
        if n <= 1:
            continue
        drop_ids = np.nonzero(stats[1:, cv2.CC_STAT_AREA] < min_area)[0] + 1
        if drop_ids.size == 0:
            continue
        out[np.isin(cc, drop_ids)] = OPEN
    return out


def _slic_superpixels(smooth_bgr, total_px):
    try:
        region = int(np.clip(np.sqrt(total_px) / 38.0, 14, 48))
        slic = cv2.ximgproc.createSuperpixelSLIC(smooth_bgr, algorithm=cv2.ximgproc.SLICO,
                                                 region_size=region, ruler=12.0)
        slic.iterate(10)
        if slic.getNumberOfSuperpixels() < 8:
            return None
        return slic.getLabels().astype(np.int32)
    except Exception:
        return None


def _aggregate_superpixels(labels, sp_labels):
    out = labels.copy()
    n = int(sp_labels.max()) + 1
    combo = (sp_labels.astype(np.int64) * 8 + labels.astype(np.int64))
    counts = np.bincount(combo.ravel(), minlength=n * 8).reshape(n, 8)
    winner = np.argmax(counts[:, :7], axis=1).astype(np.uint8)
    out = winner[sp_labels]
    return out


def _reassign_shadows(labels, V, S, H, veg, water):
    shadow = ((((V < 58) & (S < 95)) | ((V < 110) & (S >= 100)))
              & ~water & ~veg)
    if shadow.sum() < 40:
        return labels
    out = labels.copy()
    k = 41
    eps = 1e-3
    remaining = shadow.copy()
    for _ in range(5):
        if not remaining.any():
            break
        valid = (~remaining).astype(np.float32)
        denom = cv2.blur(valid, (k, k))
        best_score = np.full(labels.shape, 0.10, dtype=np.float32)
        best_cls = np.zeros(labels.shape, dtype=np.uint8)
        for cls in ALL_CLASSES:
            if cls == OPEN:
                continue
            m = ((out == cls) & ~remaining).astype(np.float32)
            cnt = cv2.blur(m, (k, k))
            frac = cnt / np.maximum(denom, eps)
            s_mean = cv2.blur(m * S, (k, k)) / np.maximum(cnt, eps)
            sim = np.exp(-((S - s_mean) ** 2) / (2.0 * 30.0 ** 2))
            if H is not None:
                h_mean = cv2.blur(m * H, (k, k)) / np.maximum(cnt, eps)
                dh = np.abs(H - h_mean)
                dh = np.minimum(dh, 180.0 - dh)
                sim = sim * np.exp(-(dh ** 2) / (2.0 * 15.0 ** 2))
            score = frac * sim
            upd = (score > best_score) & remaining & (frac > 0.10) & (denom > 0.05)
            best_score[upd] = score[upd]
            best_cls[upd] = cls
        changed = best_cls > 0
        out[changed] = best_cls[changed]
        remaining = shadow & (out == labels)
    return out


def _context_reclass(labels, k=31, thresh=0.45):
    b = (labels == BUILDINGS).astype(np.float32)
    dens = cv2.blur(b, (k, k))
    out = labels.copy()
    fix = (labels == BARE_SOIL) & (dens >= thresh)
    out[fix] = BUILDINGS

    b21 = cv2.blur((out == BUILDINGS).astype(np.float32), (21, 21))
    r21 = cv2.blur((out == ROADS).astype(np.float32), (21, 21))
    fringe_b = (out == OPEN) & (b21 >= 0.50)
    fringe_r = (out == OPEN) & (r21 >= 0.60) & (b21 < 0.50)
    out[fringe_b] = BUILDINGS
    out[fringe_r] = ROADS
    return out


def _shape_reclass(labels, min_area=380, max_frac=0.02):
    total = labels.size
    out = labels.copy()

    binm = (labels == ROADS).astype(np.uint8)
    opened = cv2.morphologyEx(binm, cv2.MORPH_OPEN, _disk(12))
    tree_mask = (labels == TREES)
    n, cc, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area // 2 or area > max_frac * total:
            continue
        bw = max(stats[i, cv2.CC_STAT_WIDTH], 1)
        bh = max(stats[i, cv2.CC_STAT_HEIGHT], 1)
        aspect = bw / bh
        fill = area / float(bw * bh)
        if (0.30 <= aspect <= 3.4) and fill >= 0.55:
            core = (cc == i)
            touch_trees = cv2.dilate(core.astype(np.uint8), _disk(3)) > 0
            if (touch_trees & tree_mask).any():
                continue
            grow = cv2.dilate(core.astype(np.uint8), _disk(5)) > 0
            out[grow & (labels == ROADS)] = BUILDINGS

    for cls in (ROADS, BUILDINGS):
        binm = (labels == cls).astype(np.uint8)
        n, cc, stats, _ = cv2.connectedComponentsWithStats(binm, connectivity=8)
        if n <= 1:
            continue
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area or area > max_frac * total:
                continue
            bw = max(stats[i, cv2.CC_STAT_WIDTH], 1)
            bh = max(stats[i, cv2.CC_STAT_HEIGHT], 1)
            aspect = bw / bh
            fill = area / float(bw * bh)
            elongated = aspect >= 3.4 or aspect <= 0.30
            compact = (0.30 <= aspect <= 3.4) and fill >= 0.55
            if cls == ROADS and compact:
                out[cc == i] = BUILDINGS
            elif cls == BUILDINGS and elongated:
                out[cc == i] = ROADS
    return out


def _consolidate_open(labels, max_area, k=15):
    binm = (labels == OPEN).astype(np.uint8)
    n, cc, stats, _ = cv2.connectedComponentsWithStats(binm, connectivity=8)
    if n <= 1:
        return labels
    small = np.zeros(labels.shape, dtype=bool)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < max_area:
            small |= (cc == i)
    if not small.any():
        return labels
    neigh = _mode_filter(labels, k)
    out = labels.copy()
    repl = small & (neigh != OPEN)
    out[repl] = neigh[repl]
    return out


def segment_image(bgr):
    h, w = bgr.shape[:2]
    total_px = float(h * w)

    feats, smooth = extract_features(bgr)
    flat = feats.reshape(-1, N_FEATURES)
    labels = _MODEL.predict(flat).reshape(h, w).astype(np.uint8)

    Hc, S, V = feats[..., 0], feats[..., 1], feats[..., 2]
    veg = np.isin(labels, [TREES, GRASS])
    water = labels == WATER

    sp = _slic_superpixels(smooth, total_px)
    if sp is not None:
        labels = _aggregate_superpixels(labels, sp)

    min_area = int(max(24, total_px * 0.00020))
    labels = _mode_filter(labels, 5)
    labels = _remove_small_regions(labels, min_area)
    labels = _reassign_shadows(labels, V, S, Hc, veg, water)
    labels = _context_reclass(labels)
    labels = _shape_reclass(labels)
    labels = _mode_filter(labels, 5)
    labels = _remove_small_regions(labels, min_area // 2)
    labels = _consolidate_open(labels, int(max(64, total_px * 0.0006)), k=15)

    return labels


def colorize_labels(labels):
    palette = np.zeros((256, 3), dtype=np.uint8)
    for cls, color in CLASS_COLORS_RGB.items():
        palette[cls] = color
    return palette[labels]


def colorize_labels_overlay(original_bgr, labels, alpha=0.5):
    """Class-color map alpha-blended over the source image (BGR in/out) so the
    underlying satellite imagery stays visible through the segmentation colors."""
    seg_bgr = cv2.cvtColor(colorize_labels(labels), cv2.COLOR_RGB2BGR).astype(np.float32)
    out = original_bgr.astype(np.float32) * (1.0 - alpha) + seg_bgr * alpha
    return out.astype(np.uint8)


def class_fractions(labels):
    total = labels.size
    counts = {cls: int(np.count_nonzero(labels == cls)) for cls in ALL_CLASSES}
    return {cls: counts[cls] / total for cls in ALL_CLASSES}, counts
