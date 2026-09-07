"""Generate a synthetic satellite/aerial test image with ground-truth metadata (deterministic).

When return_labels=True, a pixel-perfect ground-truth label mask is produced
alongside the image by mirroring every ImageDraw call onto a parallel
single-channel label canvas — used to train and validate the land-cover
classifier in engine/train_model.py. Class ids mirror engine.segmentation's
enum: OPEN=0, TREES=1, GRASS=2, BUILDINGS=3, ROADS=4, WATER=5, BARE_SOIL=6.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw

_TREES, _GRASS, _BUILDINGS, _ROADS, _WATER, _BARE_SOIL = 1, 2, 3, 4, 5, 6


def make_test_image(w=1024, h=1024, seed=7, jitter=False, return_labels=False):
    rng = np.random.default_rng(seed)

    grass = np.array([106, 168, 79], dtype=np.float32)
    img = np.ones((h, w, 3), dtype=np.float32) * grass

    noise = rng.normal(0, 9, size=(h, w, 1))
    img += noise
    patch_noise = rng.normal(0, 14, size=(h // 8 + 1, w // 8 + 1, 1))
    patch_up = np.kron(patch_noise, np.ones((8, 8, 1)))
    img += patch_up[:h, :w]

    pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(pil)

    lbl = Image.new("L", (w, h), color=_GRASS)
    ld = ImageDraw.Draw(lbl)

    d.rectangle([0, 430, w, 530], fill=(88, 88, 92))
    ld.rectangle([0, 430, w, 530], fill=_ROADS)
    d.rectangle([0, 434, w, 438], fill=(200, 195, 90))
    d.rectangle([0, 522, w, 526], fill=(200, 195, 90))
    d.rectangle([610, 0, 690, h], fill=(84, 84, 88))
    ld.rectangle([610, 0, 690, h], fill=_ROADS)
    d.rectangle([614, 0, 618, h], fill=(200, 195, 90))
    d.rectangle([682, 0, 686, h], fill=(200, 195, 90))

    buildings = [
        [60, 60, 190, 170], [210, 70, 330, 160], [360, 50, 470, 180],
        [80, 220, 240, 340], [270, 230, 420, 350], [450, 240, 560, 330],
        [720, 60, 850, 190], [870, 80, 980, 200], [730, 240, 900, 360],
        [120, 600, 260, 700], [300, 620, 430, 720], [480, 600, 600, 690],
        [750, 600, 880, 710], [900, 590, 1000, 680],
    ]
    roof_colors = [(178, 96, 62), (150, 150, 155), (120, 130, 140), (190, 110, 80),
                   (165, 165, 170), (105, 115, 125)]
    for i, box in enumerate(buildings):
        c = roof_colors[i % len(roof_colors)]
        d.rectangle(box, fill=c)
        ld.rectangle(box, fill=_BUILDINGS)
        d.rectangle([box[0] + 6, box[1] + 6, box[2] - 6, box[3] - 6],
                    outline=tuple(int(v * 0.75) for v in c), width=3)

    arr = np.asarray(pil).astype(np.float32)
    for box in buildings:
        x0, y0, x1, y1 = box
        sx0, sy0 = min(x0 + 18, w - 1), min(y0 + 18, h - 1)
        sx1, sy1 = min(x1 + 18, w), min(y1 + 18, h)
        arr[sy0:sy1, sx0:sx1] *= 0.52
    img_arr = np.clip(arr, 0, 255).astype(np.uint8)
    pil = Image.fromarray(img_arr)
    d = ImageDraw.Draw(pil)

    tree_centers = []
    for _ in range(26):
        tx = int(rng.integers(40, w - 40))
        ty = int(rng.integers(40, h - 40))
        if 400 < ty < 545 and not (tx > 680 or tx < 590):
            continue
        if 595 < ty < 730 and 60 < tx < 1010:
            continue
        tree_centers.append((tx, ty))
    tree_radii = []
    for (tx, ty) in tree_centers:
        r = int(rng.integers(22, 42))
        tree_radii.append(r)
        # Ground truth is the whole crown disc, not the individual dot strokes
        # below: the dots are a rendering texture (with gaps showing grass
        # underneath), but the canopy they depict is a solid, continuous mass
        # when photographed from above - matching how a real classifier should
        # read it and how tests/test_engine.py already scores tree accuracy.
        cr = int(r * 0.8)
        ld.ellipse([tx - cr, ty - cr, tx + cr, ty + cr], fill=_TREES)
        for _ in range(int(r * 1.4)):
            ang = rng.uniform(0, 6.283)
            rad = abs(rng.normal(0, r / 2.2))
            px, py = int(tx + rad * np.cos(ang)), int(ty + rad * np.sin(ang))
            if 0 <= px < w and 0 <= py < h:
                shade = rng.normal(-12, 10)
                col = (int(max(30, 52 + shade)), int(max(70, 118 + shade)), int(max(25, 48 + shade)))
                pr = max(2, int(abs(rng.normal(4, 2))))
                d.ellipse([px - pr, py - pr, px + pr, py + pr], fill=col)
        d.ellipse([tx - 4, ty - 4, tx + 4, ty + 4], fill=(38, 66, 32))

    d.ellipse([800, 800, 990, 950], fill=(58, 108, 148))
    ld.ellipse([800, 800, 990, 950], fill=_WATER)
    for _ in range(60):
        px = int(rng.uniform(815, 975))
        py = int(rng.uniform(815, 935))
        if (px - 895) ** 2 / 90 ** 2 + (py - 875) ** 2 / 67 ** 2 < 0.85:
            d.line([px, py, px + int(rng.integers(-14, 14)), py], fill=(78, 128, 168), width=1)

    soil_pts = [(430, 800), (500, 830), (465, 870), (540, 790)]
    for i, (sx, sy) in enumerate(soil_pts):
        r = 55 + i * 8
        box = [sx - r, sy - int(r * .7), sx + r, sy + int(r * .7)]
        d.ellipse(box, fill=(139, 112, 90))
        ld.ellipse(box, fill=_BARE_SOIL)
    for _ in range(400):
        px = int(rng.uniform(380, 610))
        py = int(rng.uniform(760, 930))
        if ((px - 485) / 135) ** 2 + ((py - 845) / 85) ** 2 < 1:
            v = int(rng.uniform(-16, 16))
            d.point((px, py), fill=(139 + v, 112 + v, 90 + v))
            ld.point((px, py), fill=_BARE_SOIL)

    if jitter:
        hsv_img = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2HSV).astype(np.float32)
        hue_shift = rng.uniform(-8, 8)
        sat_scale = rng.uniform(0.85, 1.15)
        val_scale = rng.uniform(0.85, 1.15)
        hsv_img[..., 0] = (hsv_img[..., 0] + hue_shift) % 180
        hsv_img[..., 1] = np.clip(hsv_img[..., 1] * sat_scale, 0, 255)
        hsv_img[..., 2] = np.clip(hsv_img[..., 2] * val_scale, 0, 255)
        pil = Image.fromarray(cv2.cvtColor(hsv_img.astype(np.uint8), cv2.COLOR_HSV2RGB))

    out = np.clip(np.asarray(pil).astype(np.float32) + rng.normal(0, 4, size=(h, w, 3)), 0, 255)

    meta = {
        "buildings": buildings,
        "tree_centers": tree_centers,
        "tree_radii": tree_radii,
        "road_h": (440, 520),
        "road_v": (618, 682),
        "pond": (800, 800, 990, 950),
        "soil": (430, 795, 545, 860),
    }
    image = Image.fromarray(out.astype(np.uint8))
    if return_labels:
        return image, meta, np.asarray(lbl, dtype=np.uint8)
    return image, meta


if __name__ == "__main__":
    import os
    out_dir = os.path.dirname(os.path.abspath(__file__))
    im, _ = make_test_image()
    path = os.path.join(out_dir, "test_satellite.jpg")
    im.save(path, quality=92)
    print("saved", path, im.size)
