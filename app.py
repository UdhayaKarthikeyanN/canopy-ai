"""Canopy AI - fully local urban tree & heat analysis server."""

import os
import time
import uuid

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

from engine.palette import legend_entries
from engine.planting import (compute_metrics, compute_plantable_mask,
                             default_params, draw_plantable_overlay,
                             draw_recommendation, place_markers)
from engine.report import generate_report
from engine.segmentation import colorize_labels_overlay, segment_image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALLOWED_EXT = {".jpg", ".jpeg", ".png"}
MAX_CONTENT_LENGTH = 48 * 1024 * 1024

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


def _parse_params():
    def fget(name, default):
        try:
            v = float(request.form.get(name, default))
            return v
        except (TypeError, ValueError):
            return default
    p = default_params(
        base_temp_c=fget("base_temp_c", 33.0),
        gsd_m=fget("gsd_m", 0.5),
        spacing_m=fget("spacing_m", 7.0),
        crown_radius_m=fget("crown_radius_m", 3.5),
    )
    if not (0.02 <= p["gsd_m"] <= 5.0):
        p["gsd_m"] = 0.5
    if not (2.0 <= p["spacing_m"] <= 30.0):
        p["spacing_m"] = 7.0
    if not (10 <= p["base_temp_c"] <= 55):
        p["base_temp_c"] = 33.0
    if not (1.0 <= p["crown_radius_m"] <= 8.0):
        p["crown_radius_m"] = 3.5
    return p


def _load_bgr(file_storage):
    data = file_storage.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/legend")
def api_legend():
    return jsonify({"legend": legend_entries()})


@app.post("/api/analyze")
def api_analyze():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400
    fs = request.files["file"]
    filename = secure_filename(fs.filename or "image.png")
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXT:
        return jsonify({"error": "Only JPG and PNG images are supported."}), 400

    bgr = _load_bgr(fs)
    if bgr is None:
        return jsonify({"error": "Could not decode image. Please upload a valid JPG/PNG."}), 400

    h, w = bgr.shape[:2]
    if max(h, w) < 64:
        return jsonify({"error": "Image too small for analysis (min 64 px)."}), 400

    analysis_id = uuid.uuid4().hex[:12]
    out_dir = os.path.join(OUTPUT_DIR, analysis_id)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    original_path = os.path.join(out_dir, "original.png")
    cv2.imwrite(original_path, bgr)

    labels = segment_image(bgr)
    seg_overlay_bgr = colorize_labels_overlay(bgr, labels)
    segmentation_path = os.path.join(out_dir, "segmentation.png")
    cv2.imwrite(segmentation_path, seg_overlay_bgr)

    params = _parse_params()
    plantable = compute_plantable_mask(labels, params["gsd_m"])
    plantable_img = draw_plantable_overlay(bgr, plantable)
    plantable_path = os.path.join(out_dir, "plantable.png")
    cv2.imwrite(plantable_path, plantable_img)

    markers = place_markers(plantable, params)
    metrics = compute_metrics(labels, plantable, params, markers)

    rec_img = draw_recommendation(bgr, plantable, markers,
                                  params["crown_radius_m"] / params["gsd_m"])
    recommendation_path = os.path.join(out_dir, "recommendation.png")
    cv2.imwrite(recommendation_path, rec_img)

    pdf_path = os.path.join(out_dir, "report.pdf")
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    metrics.update({
        "analysis_id": analysis_id,
        "generated_at": stamp,
        "processing_seconds": round(time.time() - t0, 2),
    })
    generate_report(pdf_path, original_path, segmentation_path, plantable_path,
                    recommendation_path, metrics, params)

    resp = {
        "id": analysis_id,
        "images": {
            "original": f"/outputs/{analysis_id}/original.png",
            "segmentation": f"/outputs/{analysis_id}/segmentation.png",
            "plantable": f"/outputs/{analysis_id}/plantable.png",
            "recommendation": f"/outputs/{analysis_id}/recommendation.png",
        },
        "metrics": metrics,
        "legend": legend_entries(),
        "params": params,
        "report_url": f"/api/report/{analysis_id}",
    }
    return jsonify(resp)


@app.get("/api/report/<analysis_id>")
def api_report(analysis_id):
    safe = "".join(ch for ch in analysis_id if ch.isalnum())[:32]
    pdf_path = os.path.join(OUTPUT_DIR, safe, "report.pdf")
    if not os.path.exists(pdf_path):
        return jsonify({"error": "Report not found. Please run an analysis first."}), 404
    return send_file(pdf_path, mimetype="application/pdf", as_attachment=True,
                     download_name=f"CanopyAI_Report_{safe}.pdf")


@app.get("/outputs/<analysis_id>/<imgname>")
def serve_output(analysis_id, imgname):
    safe_dir = "".join(ch for ch in analysis_id if ch.isalnum())[:32]
    safe_name = os.path.basename(imgname)
    path = os.path.join(OUTPUT_DIR, safe_dir, safe_name)
    if not os.path.exists(path):
        return jsonify({"error": "Not found"}), 404
    return send_file(path)


@app.errorhandler(413)
def too_large(_e):
    return jsonify({"error": "File exceeds the 48 MB limit."}), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=7860, debug=False)
