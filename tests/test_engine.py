"""End-to-end validation of the Canopy AI engine with ground-truth accuracy checks."""

import math
import os
import sys

import cv2
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from tests.make_test_image import make_test_image
from engine.planting import (compute_metrics, compute_plantable_mask,
                             default_params, draw_plantable_overlay,
                             draw_recommendation, place_markers)
from engine.segmentation import (BARE_SOIL, BUILDINGS, GRASS, OPEN, ROADS,
                                 TREES, WATER, colorize_labels_overlay, segment_image)
from engine.report import generate_report

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)


def region_accuracy(labels, mask, expected):
    sel = labels[mask]
    if sel.size == 0:
        return 0.0
    return float((sel == expected).sum()) / sel.size


def disk_mask(shape, cx, cy, r):
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r


def box_mask(shape, x0, y0, x1, y1, inset=0):
    m = np.zeros(shape, dtype=bool)
    m[y0 + inset:y1 - inset, x0 + inset:x1 - inset] = True
    return m


def main():
    pil, meta = make_test_image()
    rgb = np.asarray(pil)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    print(f"[1] test image {w}x{h}")

    labels = segment_image(bgr)
    assert labels.shape == (h, w), "label map must match input resolution"

    counts = {c: int((labels == c).sum()) for c in range(7)}
    total = labels.size
    pct = {c: 100.0 * v / total for c, v in counts.items()}
    print("[2] class fractions %:", {k: round(v, 1) for k, v in pct.items()})

    assert pct[TREES] > 3.0
    assert pct[ROADS] > 2.0
    assert pct[BUILDINGS] > 1.0
    assert pct[WATER] > 0.5
    assert pct[GRASS] + pct[OPEN] + pct[BARE_SOIL] > 30.0

    accs = {}

    for i, box in enumerate(meta["buildings"]):
        m = box_mask((h, w), *box, inset=14)
        accs[f"building{i}"] = region_accuracy(labels, m, BUILDINGS)
    b_acc = np.mean([v for k, v in accs.items() if k.startswith("building")])
    print(f"[3] building interior accuracy: {b_acc:.1%}")
    assert b_acc >= 0.80, f"building accuracy {b_acc:.1%} below 80%"

    rh = meta["road_h"]
    m = box_mask((h, w), 30, rh[0] + 8, w - 30, rh[1] - 8)
    r_acc = region_accuracy(labels, m, ROADS)
    rv = meta["road_v"]
    m2 = box_mask((h, w), rv[0] + 8, 30, rv[1] - 8, h - 30)
    r_acc = (r_acc + region_accuracy(labels, m2, ROADS)) / 2
    print(f"[4] road accuracy: {r_acc:.1%}")
    assert r_acc >= 0.85, f"road accuracy {r_acc:.1%} below 85%"

    px0, py0, px1, py1 = meta["pond"]
    m = box_mask((h, w), px0 + 30, py0 + 25, px1 - 30, py1 - 25)
    w_acc = region_accuracy(labels, m, WATER)
    print(f"[5] pond accuracy: {w_acc:.1%}")
    assert w_acc >= 0.90, f"pond accuracy {w_acc:.1%} below 90%"

    sx0, sy0, sx1, sy1 = meta["soil"]
    m = box_mask((h, w), sx0 + 25, sy0 + 12, sx1 - 25, sy1 - 12)
    s_acc = region_accuracy(labels, m, BARE_SOIL)
    print(f"[6] soil accuracy: {s_acc:.1%}")
    assert s_acc >= 0.70, f"soil accuracy {s_acc:.1%} below 70%"

    t_accs = []
    for (tx, ty), r in zip(meta["tree_centers"], meta["tree_radii"]):
        if r < 26:
            continue
        m = disk_mask((h, w), tx, ty, int(r * 0.45))
        t_accs.append(region_accuracy(labels, m, TREES))
    t_acc = float(np.mean(t_accs))
    print(f"[7] tree-core accuracy ({len(t_accs)} canopies): {t_acc:.1%}")
    assert t_acc >= 0.70, f"tree accuracy {t_acc:.1%} below 70%"

    shadow_accs = []
    for box in meta["buildings"]:
        x0, y0, x1, y1 = box
        sx0, sy0 = min(x0 + 20, w - 2), min(y0 + 20, h - 2)
        sx1, sy1 = min(x1 + 16, w), min(y1 + 16, h)
        if sx1 - sx0 < 12 or sy1 - sy0 < 12:
            continue
        m = box_mask((h, w), sx0, sy0, sx1, sy1)
        sel = labels[m]
        shadow_accs.append(float((sel == WATER).sum()) / sel.size)
    sh_water = float(np.mean(shadow_accs))
    print(f"[8] shadow misread as water: {sh_water:.1%}")
    assert sh_water < 0.05, f"shadows leaking to water: {sh_water:.1%}"

    params = default_params(base_temp_c=33.0, gsd_m=0.5, spacing_m=7.0)
    plantable = compute_plantable_mask(labels, params["gsd_m"])
    assert plantable.shape == (h, w)
    overlap_bad = int(np.count_nonzero(plantable & np.isin(labels, [TREES, BUILDINGS, ROADS, WATER])))
    assert overlap_bad == 0, f"plantable overlaps excluded classes on {overlap_bad}px"
    p_pct = 100.0 * plantable.sum() / total
    print(f"[9] plantable {p_pct:.1f}% ({int(plantable.sum())} px)")
    assert p_pct > 10.0

    grass_mask = (labels == GRASS)
    grass_far = grass_mask & (cv2.distanceTransform(
        (~np.isin(labels, [TREES, BUILDINGS, ROADS, WATER])).astype(np.uint8),
        cv2.DIST_L2, 3) > 4.0)
    if grass_far.sum() > 500:
        g_cov = float(np.count_nonzero(plantable & grass_far)) / grass_far.sum()
        print(f"[10] far-from-obstacle grass captured as plantable: {g_cov:.1%}")
        assert g_cov >= 0.80, f"plantable misses open grass: {g_cov:.1%}"

    markers = place_markers(plantable, params)
    print(f"[11] markers placed: {len(markers)}")
    assert len(markers) > 3
    min_dist_px = params["spacing_m"] / params["gsd_m"]
    pts = np.array(markers)
    for i in range(len(pts)):
        x, y = int(pts[i, 0]), int(pts[i, 1])
        assert plantable[y, x], f"marker {i} not inside plantable mask"
        d = np.sqrt(((pts - pts[i]) ** 2).sum(axis=1))
        d[i] = 1e9
        assert d.min() >= min_dist_px - 0.5

    metrics = compute_metrics(labels, plantable, params, markers)
    rec = metrics["recommended_planting_pct"]
    crown_area = math.pi * params["crown_radius_m"] ** 2
    expect_rec = len(markers) * crown_area / metrics["total_area_m2"] * 100
    assert abs(rec - min(expect_rec, metrics["plantable_pct"])) <= 0.05
    assert abs(metrics["projected_canopy_pct"] - min(metrics["canopy_pct"] + rec, 95.0)) <= 0.02
    assert abs(metrics["cooling_reduction_c"] -
               0.06 * (metrics["projected_canopy_pct"] - metrics["canopy_pct"])) <= 0.01
    assert metrics["cooling_reduction_c"] >= 0
    assert 0 <= metrics["heat_score_current"] <= 100
    print(f"[12] metrics ok: canopy={metrics['canopy_pct']}% heat={metrics['heat_score_current']} "
          f"temp={metrics['temp_current_c']}C -> proj {metrics['projected_temp_c']}C")

    seg_png = os.path.join(OUT, "segmentation.png")
    cv2.imwrite(seg_png, colorize_labels_overlay(bgr, labels))
    cv2.imwrite(os.path.join(OUT, "plantable.png"), draw_plantable_overlay(bgr, plantable))
    cv2.imwrite(os.path.join(OUT, "recommendation.png"),
                draw_recommendation(bgr, plantable, markers,
                                    params["crown_radius_m"] / params["gsd_m"]))
    orig_png = os.path.join(OUT, "original.png")
    cv2.imwrite(orig_png, bgr)
    print("[13] output images written at original resolution")

    pdf_path = os.path.join(OUT, "report.pdf")
    generate_report(pdf_path, orig_png, seg_png, os.path.join(OUT, "plantable.png"),
                    os.path.join(OUT, "recommendation.png"),
                    {**metrics, "analysis_id": "test0001", "generated_at": "test"}, params)
    assert os.path.getsize(pdf_path) > 20000
    print(f"[14] PDF generated ({os.path.getsize(pdf_path) / 1024:.0f} KB)")

    overall = np.mean([b_acc, r_acc, w_acc, s_acc, t_acc])
    print(f"[15] overall ground-truth accuracy: {overall:.1%}")
    assert overall >= 0.80, f"overall accuracy {overall:.1%} below 80%"

    print("ALL ENGINE TESTS PASSED")


if __name__ == "__main__":
    main()
