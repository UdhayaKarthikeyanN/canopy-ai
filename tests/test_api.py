"""HTTP end-to-end test: start server context, upload image, verify outputs and PDF."""

import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

IMG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_satellite.jpg")


def main():
    from app import app
    client = app.test_client()

    with open(IMG, "rb") as f:
        r = client.post("/api/analyze", data={
            "file": (f, "test_satellite.jpg"),
            "base_temp_c": "34",
            "gsd_m": "0.5",
            "spacing_m": "8",
            "crown_radius_m": "3.5",
        }, content_type="multipart/form-data")
    assert r.status_code == 200, f"analyze failed: {r.status_code} {r.data[:300]}"
    data = r.get_json()
    m = data["metrics"]
    print("[1] analyze ok:", data["id"])
    print("    canopy=%.2f%% plantable=%.2f%% heat=%.1f temp=%.2fC" %
          (m["canopy_pct"], m["plantable_pct"], m["heat_score_current"], m["temp_current_c"]))
    print("    trees=%d rec=%.2f%% projCanopy=%.2f%% projHeat=%.1f projTemp=%.2fC cool=%.2fC" %
          (m["recommended_trees"], m["recommended_planting_pct"], m["projected_canopy_pct"],
           m["projected_heat_score"], m["projected_temp_c"], m["cooling_reduction_c"]))

    required = ["canopy_pct", "plantable_pct", "heat_score_current", "temp_current_c",
                "recommended_planting_pct", "projected_canopy_pct", "projected_heat_score",
                "projected_temp_c", "cooling_reduction_c", "shade_improvement_pct_points",
                "shade_improvement_m2", "recommended_trees", "co2_offset_kg_yr"]
    missing = [k for k in required if k not in m]
    assert not missing, f"missing metrics: {missing}"
    print("[2] all required metrics present")

    for name, url in data["images"].items():
        ir = client.get(url)
        assert ir.status_code == 200 and len(ir.data) > 1000, f"image {name} failed"
    print("[3] all 4 output images served")

    rr = client.get(data["report_url"])
    assert rr.status_code == 200, "report failed"
    assert rr.data[:4] == b"%PDF", "not a PDF"
    pdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "api_report.pdf")
    with open(pdf_path, "wb") as f:
        f.write(rr.data)
    print(f"[4] PDF report downloaded ({len(rr.data) / 1024:.0f} KB)")

    bad = client.post("/api/analyze", data={"file": (b"notanimage", "x.jpg")},
                      content_type="multipart/form-data")
    assert bad.status_code == 400, "invalid image must return 400"
    txt = client.post("/api/analyze", data={"file": (b"hello", "x.txt")},
                      content_type="multipart/form-data")
    assert txt.status_code == 400, "txt upload must return 400"
    print("[5] error handling ok")

    print("ALL API TESTS PASSED")


if __name__ == "__main__":
    main()
