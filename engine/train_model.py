"""Train the local Random Forest land-cover classifier on synthetic scenes.

No real imagery ships with this repo (Canopy AI is fully local/offline by
design), so training data is generated on the fly by
tests/make_test_image.py: many seeds, each with randomized hue/saturation/
brightness jitter, using the exact per-pixel ground truth the generator
draws (mirrored onto a label canvas as it renders each building/road/tree/
water/soil primitive). Held-out seeds (not used for training) measure
generalization to unseen lighting/color conditions.

Run:
    python engine/train_model.py

This overwrites engine/model/land_cover_rf.joblib, which app.py and the
tests load at import time.
"""

import os
import sys
import time

import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from engine.features import N_FEATURES, extract_features
from tests.make_test_image import make_test_image

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "land_cover_rf.joblib")

TRAIN_SEEDS = list(range(1, 17))
VAL_SEEDS = list(range(101, 106))
PER_CLASS_SAMPLES = 6000


def _scene_features(seed):
    pil, _meta, labels = make_test_image(seed=seed, jitter=True, return_labels=True)
    bgr = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    feats, _smooth = extract_features(bgr)
    return feats, labels


def _sample_scene(seed):
    feats, labels = _scene_features(seed)
    flat_feats = feats.reshape(-1, N_FEATURES)
    flat_labels = labels.reshape(-1)

    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for cls in np.unique(flat_labels):
        idx = np.nonzero(flat_labels == cls)[0]
        if idx.size > PER_CLASS_SAMPLES:
            idx = rng.choice(idx, size=PER_CLASS_SAMPLES, replace=False)
        xs.append(flat_feats[idx])
        ys.append(flat_labels[idx])
    return np.concatenate(xs), np.concatenate(ys)


def main():
    t0 = time.time()
    xs, ys = [], []
    for seed in TRAIN_SEEDS:
        x, y = _sample_scene(seed)
        xs.append(x)
        ys.append(y)
        print(f"  scene seed={seed}: {x.shape[0]} labeled px sampled")
    X = np.concatenate(xs)
    Y = np.concatenate(ys)
    print(f"[1] training set: {X.shape[0]} pixels x {X.shape[1]} features "
          f"from {len(TRAIN_SEEDS)} scenes ({time.time() - t0:.1f}s)")

    clf = RandomForestClassifier(
        n_estimators=80, max_depth=14, min_samples_leaf=6,
        n_jobs=-1, class_weight="balanced_subsample", random_state=0,
    )
    clf.fit(X, Y)
    print(f"[2] trained Random Forest ({time.time() - t0:.1f}s elapsed)")

    accs = []
    for seed in VAL_SEEDS:
        feats, labels = _scene_features(seed)
        h, w = labels.shape
        pred = clf.predict(feats.reshape(-1, N_FEATURES)).reshape(h, w)
        acc = float((pred == labels).mean())
        accs.append(acc)
        print(f"[3] held-out seed={seed}: pixel accuracy {acc:.1%}")
    print(f"[4] mean held-out pixel accuracy: {np.mean(accs):.1%}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(clf, MODEL_PATH, compress=9)
    size_kb = os.path.getsize(MODEL_PATH) / 1024
    print(f"[5] model saved to {MODEL_PATH} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
