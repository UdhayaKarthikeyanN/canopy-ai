"""Plantable-area detection, deterministic Poisson-disk tree placement, metrics and projections."""

import math

import cv2
import numpy as np

from engine.segmentation import (
    BARE_SOIL,
    BUILDINGS,
    GRASS,
    OPEN,
    ROADS,
    TREES,
    WATER,
)

COOLING_PER_CANOPY_PCT = 0.06
CO2_KG_PER_TREE_YR = 21.0


def _disk(radius):
    r = int(max(1, round(radius)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def compute_plantable_mask(labels, gsd_m):
    total = labels.size
    candidates = np.isin(labels, [OPEN, GRASS, BARE_SOIL])

    obstacles = ~np.isin(labels, [OPEN, GRASS, BARE_SOIL])
    water = labels == WATER

    def dist_to(mask):
        return cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)

    d_obstacle = dist_to(obstacles)
    d_water = dist_to(water)

    buffer_px = 1.5 / max(gsd_m, 1e-6)
    water_buffer_px = 2.5 / max(gsd_m, 1e-6)

    plantable = candidates & (d_obstacle > buffer_px) & (d_water > water_buffer_px)

    close_r = int(max(2.0 / max(gsd_m, 1e-6), 1))
    plantable = cv2.morphologyEx(plantable.astype(np.uint8), cv2.MORPH_CLOSE, _disk(close_r))
    plantable = (plantable.astype(bool) & candidates
                 & (d_obstacle > buffer_px) & (d_water > water_buffer_px)).astype(np.uint8)

    n, cc, stats, _ = cv2.connectedComponentsWithStats(plantable, connectivity=8)
    min_area = max(int(total * 0.0004), 120)
    keep = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[i] = True
    if n > 1:
        plantable = keep[cc].astype(np.uint8)

    return plantable.astype(bool)


def poisson_disk_sample(mask, min_dist_px, seed=42, k=24):
    """Bridson Poisson-disk sampling restricted to boolean mask. Deterministic given seed.

    Plantable masks are routinely fragmented into many disconnected pockets
    (isolated by buildings, roads, water and their buffers). A single random
    start point only ever grows within whichever pocket it happens to land
    in, so once that pocket's frontier is exhausted the whole sampler stops -
    leaving every other pocket, however large, with zero points. To fill the
    full mask, a new start point is (re)seeded from a pixel guaranteed to be
    at least min_dist_px from every point placed so far, each time the
    active frontier empties, until no such pixel remains anywhere in the mask.
    """
    h, w = mask.shape
    md = float(min_dist_px)
    if md < 1:
        md = 1.0
    cell = md / math.sqrt(2)
    gw = int(math.ceil(w / cell))
    gh = int(math.ceil(h / cell))
    grid = -np.ones((gh, gw), dtype=np.int32)
    points = []
    rng = np.random.default_rng(seed)

    if not mask.any():
        return []

    def fits(px, py):
        gx, gy = int(px / cell), int(py / cell)
        y0, y1 = max(gy - 2, 0), min(gy + 3, gh)
        x0, x1 = max(gx - 2, 0), min(gx + 3, gw)
        for ny in range(y0, y1):
            for nx in range(x0, x1):
                j = grid[ny, nx]
                if j != -1:
                    qx, qy = points[j]
                    if (px - qx) ** 2 + (py - qy) ** 2 < md * md:
                        return False
        return True

    def add_point(px, py):
        points.append((px, py))
        grid[int(py / cell), int(px / cell)] = len(points) - 1
        return len(points) - 1

    placed_raster = np.zeros((h, w), dtype=np.uint8)

    def find_new_seed():
        """A pixel raster distance transform only locates *candidates* cheaply
        (with rounding slack) - fits() against the exact sub-pixel point
        coordinates is what actually guarantees the minimum spacing
        invariant, so every candidate is re-checked with it before being
        accepted. Margin shrinks in tiers rather than falling straight back
        to an unfiltered full-mask scan, which is what made this pathologically
        slow once the mask was mostly packed and few valid pixels remained."""
        if not points:
            cy, cx = np.nonzero(mask)
            if cy.size == 0:
                return None
            i = int(rng.integers(0, cy.size))
            return float(cx[i]), float(cy[i])

        dist = cv2.distanceTransform(1 - placed_raster, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        for margin in (1.5, 0.75, 0.0):
            cy, cx = np.nonzero(mask & (dist >= md + margin))
            if cy.size == 0:
                continue
            for i in rng.permutation(cy.size):
                px, py = float(cx[i]), float(cy[i])
                if fits(px, py):
                    return px, py
        return None

    active = []
    while len(points) < 20000:
        if not active:
            seed_pt = find_new_seed()
            if seed_pt is None:
                break
            px0, py0 = seed_pt
            idx = add_point(px0, py0)
            placed_raster[int(round(py0)), int(round(px0))] = 1
            active.append(idx)
            continue

        aidx_pos = int(rng.integers(0, len(active)))
        aidx = active[aidx_pos]
        ax, ay = points[aidx]
        placed = False
        for _ in range(k):
            ang = float(rng.uniform(0, 2 * math.pi))
            rad = md * math.sqrt(float(rng.uniform(1.0, 4.0)))
            nx_, ny_ = ax + rad * math.cos(ang), ay + rad * math.sin(ang)
            ix, iy = int(nx_), int(ny_)
            if not (0 <= ix < w and 0 <= iy < h):
                continue
            if not mask[iy, ix]:
                continue
            if fits(nx_, ny_):
                idx = add_point(nx_, ny_)
                placed_raster[iy, ix] = 1
                active.append(idx)
                placed = True
        if not placed:
            active.pop(aidx_pos)

    return [(float(x), float(y)) for x, y in points]


def _blend_blue(original_bgr, plantable_mask, bgr_color, alpha):
    out = original_bgr.astype(np.float32).copy()
    color = np.array(bgr_color, dtype=np.float32)
    sel = plantable_mask.astype(bool)
    out[sel] = out[sel] * (1.0 - alpha) + color * alpha
    return out.astype(np.uint8)


def draw_recommendation(original_bgr, plantable_mask, markers, crown_radius_px):
    overlay_img = _blend_blue(original_bgr, plantable_mask, (233, 118, 34), 0.42)

    r_marker = int(max(5, round(crown_radius_px * 0.55)))
    for (x, y) in markers:
        cx, cy = int(round(x)), int(round(y))
        cv2.circle(overlay_img, (cx, cy), r_marker + 2, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(overlay_img, (cx, cy), r_marker, (46, 164, 79), -1, cv2.LINE_AA)
        cv2.circle(overlay_img, (cx, cy), max(1, r_marker // 4), (40, 40, 40), -1, cv2.LINE_AA)
    return overlay_img


def draw_plantable_overlay(original_bgr, plantable_mask):
    return _blend_blue(original_bgr, plantable_mask, (222, 135, 25), 0.45)


def place_markers(plantable_mask, params):
    clearance_px = max(2.0, params["marker_clearance_m"] / params["gsd_m"])
    eroded = cv2.erode(plantable_mask.astype(np.uint8), _disk(clearance_px)) > 0
    min_dist_px = max(params["spacing_m"] / params["gsd_m"], 2.0)
    return poisson_disk_sample(eroded, min_dist_px, seed=params["seed"])


def compute_metrics(labels, plantable_mask, params, markers):
    total = labels.size
    pixel_area = params["gsd_m"] ** 2
    total_area_m2 = total * pixel_area

    counts = {c: int(np.count_nonzero(labels == c)) for c in
              [OPEN, TREES, GRASS, BUILDINGS, ROADS, WATER, BARE_SOIL]}
    frac = {c: counts[c] / total for c in counts}

    canopy_pct = frac[TREES] * 100.0
    veg_pct = (counts[TREES] + counts[GRASS]) / total * 100.0
    imperv_pct = (counts[BUILDINGS] + counts[ROADS]) / total * 100.0
    water_pct = frac[WATER] * 100.0
    bare_pct = counts[BARE_SOIL] / total * 100.0

    heat_current = clamp(
        18.0 + 0.90 * imperv_pct + 0.40 * bare_pct - 0.70 * canopy_pct - 0.20 * (frac[GRASS] * 100.0),
        0.0, 100.0)

    t_base = params["base_temp_c"]
    t_current = t_base - COOLING_PER_CANOPY_PCT * canopy_pct

    plantable_px = int(np.count_nonzero(plantable_mask))
    plantable_pct = plantable_px / total * 100.0
    plantable_area_m2 = plantable_px * pixel_area

    spacing_m = params["spacing_m"]

    crown_radius_m = params["crown_radius_m"]
    crown_area_m2 = math.pi * crown_radius_m ** 2
    n_new = len(markers)
    new_canopy_m2 = n_new * crown_area_m2

    rec_pct_raw = new_canopy_m2 / total_area_m2 * 100.0
    rec_pct = min(rec_pct_raw, plantable_pct, 60.0)

    proj_canopy_pct = min(canopy_pct + rec_pct, 95.0)
    heat_proj = clamp(
        18.0 + 0.90 * imperv_pct + 0.40 * bare_pct - 0.70 * proj_canopy_pct - 0.20 * (frac[GRASS] * 100.0),
        0.0, 100.0)
    t_projected = t_base - COOLING_PER_CANOPY_PCT * proj_canopy_pct

    cooling_reduction = t_current - t_projected
    shade_gain_m2 = new_canopy_m2

    return {
        "image_size": [int(labels.shape[1]), int(labels.shape[0])],
        "gsd_m": params["gsd_m"],
        "total_area_m2": round(total_area_m2, 1),
        "canopy_pct": round(canopy_pct, 2),
        "canopy_area_m2": round(counts[TREES] * pixel_area, 1),
        "grass_pct": round(frac[GRASS] * 100.0, 2),
        "buildings_pct": round(frac[BUILDINGS] * 100.0, 2),
        "roads_pct": round(frac[ROADS] * 100.0, 2),
        "water_pct": round(water_pct, 2),
        "bare_soil_pct": round(bare_pct, 2),
        "open_pct": round(frac[OPEN] * 100.0, 2),
        "impervious_pct": round(imperv_pct, 2),
        "vegetation_pct": round(veg_pct, 2),
        "heat_score_current": round(heat_current, 1),
        "temp_current_c": round(t_current, 2),
        "base_temp_c": t_base,
        "plantable_pct": round(plantable_pct, 2),
        "plantable_area_m2": round(plantable_area_m2, 1),
        "recommended_trees": n_new,
        "recommended_planting_pct": round(rec_pct, 2),
        "new_canopy_m2": round(new_canopy_m2, 1),
        "projected_canopy_pct": round(proj_canopy_pct, 2),
        "projected_heat_score": round(heat_proj, 1),
        "projected_temp_c": round(t_projected, 2),
        "cooling_reduction_c": round(cooling_reduction, 2),
        "shade_improvement_pct_points": round(rec_pct, 2),
        "shade_improvement_m2": round(shade_gain_m2, 1),
        "co2_offset_kg_yr": round(n_new * CO2_KG_PER_TREE_YR, 1),
        "tree_spacing_m": spacing_m,
        "crown_radius_m": crown_radius_m,
        "class_counts": {str(k): v for k, v in counts.items()},
    }


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def crown_radius_px(params):
    return params["crown_radius_m"] / params["gsd_m"]


def default_params(base_temp_c=33.0, gsd_m=0.5, spacing_m=7.0, crown_radius_m=3.5,
                   marker_clearance_m=1.5, seed=42):
    return {
        "base_temp_c": float(base_temp_c),
        "gsd_m": float(gsd_m),
        "spacing_m": float(spacing_m),
        "crown_radius_m": float(crown_radius_m),
        "marker_clearance_m": float(marker_clearance_m),
        "seed": int(seed),
    }
