"""Downloadable PDF report generation (fully local, ReportLab)."""

import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

from engine.palette import legend_entries

PAGE_W, PAGE_H = A4
MARGIN = 16 * mm
CONTENT_W = PAGE_W - 2 * MARGIN

ACCENT = colors.HexColor("#1B5E20")
DARK = colors.HexColor("#17301c")
LIGHT_BG = colors.HexColor("#F1F7F2")
BLUE = colors.HexColor("#1565C0")
ORANGE = colors.HexColor("#E65100")


def _styles():
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("TitleX", parent=ss["Title"], fontSize=22, textColor=ACCENT,
                        spaceAfter=2 * mm)
    sub = ParagraphStyle("SubX", parent=ss["Normal"], fontSize=10.5,
                         alignment=TA_CENTER, textColor=colors.HexColor("#555555"))
    h2 = ParagraphStyle("H2X", parent=ss["Heading2"], fontSize=13.5, textColor=DARK,
                        spaceBefore=4 * mm, spaceAfter=2 * mm,
                        borderPadding=(2, 4, 2, 4), backColor=LIGHT_BG)
    body = ParagraphStyle("BodyX", parent=ss["Normal"], fontSize=9.3, leading=13)
    small = ParagraphStyle("SmallX", parent=body, fontSize=8.2, leading=11,
                           textColor=colors.HexColor("#444444"))
    return h1, sub, h2, body, small


def _img_flow(path, max_h=95 * mm):
    if not path or not os.path.exists(path):
        return Paragraph("Image unavailable", _styles()[4])
    from PIL import Image as PILImage
    with PILImage.open(path) as im:
        iw, ih = im.size
    scale = min(CONTENT_W / iw, max_h / ih, 1.0 if iw <= CONTENT_W else CONTENT_W / iw)
    w = iw * scale
    h = ih * scale
    return Image(path, width=w, height=h)


def _metric_table(rows):
    data = [["Metric", "Value"]] + rows
    t = Table(data, colWidths=[0.62 * CONTENT_W, 0.38 * CONTENT_W])
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B9CDBD")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("ALIGN", (1, 0), (1, -1), "RIGHT"),
    ]
    t.setStyle(TableStyle(style))
    return t


def generate_report(pdf_path, original_png, segmentation_png, plantable_png,
                    recommendation_png, metrics, params):
    h1, sub, h2, body, small = _styles()

    doc = SimpleDocTemplate(
        pdf_path, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN,
        title="Canopy AI - Urban Tree & Heat Analysis Report",
        author="Canopy AI (local CV engine)")

    story = []
    story.append(Paragraph("CANOPY AI", h1))
    story.append(Paragraph("Urban Tree Canopy & Heat Analysis Report - 100% Local Computer-Vision Analysis",
                           sub))
    story.append(Spacer(1, 4 * mm))

    meta_rows = [
        ["Analysis ID", str(metrics.get("analysis_id", "-"))],
        ["Generated", str(metrics.get("generated_at", "-"))],
        ["Image size (px)", f"{metrics['image_size'][0]} x {metrics['image_size'][1]}"],
        ["Ground resolution (m/px)", f"{params['gsd_m']}"],
        ["Study area", f"{metrics['total_area_m2']:,.0f} m^2"],
        ["Engine", "Local trained classifier + classical CV (no cloud / no external AI APIs)"],
    ]
    story.append(_metric_table(meta_rows))

    story.append(PageBreak())

    story.append(Paragraph("1. Input Imagery & Land-Cover Segmentation", h2))
    story.append(_img_flow(original_png, max_h=88 * mm))
    story.append(Spacer(1, 2 * mm))
    story.append(_img_flow(segmentation_png, max_h=88 * mm))
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph("<b>Legend</b>", small))
    leg_cells = []
    for entry in legend_entries():
        sw = Table([[""]], colWidths=[6 * mm], rowHeights=[4 * mm])
        sw.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, 0), colors.HexColor(entry["hex"]))]))
        leg_cells.append([sw, Paragraph(entry["name"], small)])
    leg_tbl = Table(leg_cells, colWidths=[7 * mm, 70 * mm], hAlign="LEFT")
    leg_tbl.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 1),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
    ]))
    story.append(leg_tbl)

    story.append(PageBreak())
    story.append(Paragraph("2. Plantable Areas & Recommended Planting Plan", h2))
    story.append(Paragraph(
        "Blue pixels mark the actual detected plantable ground (grass / bare soil) after excluding "
        "existing trees, buildings, roads, water and safety buffers. Green markers are suggested tree "
        f"positions placed with minimum spacing of {params['spacing_m']} m.", body))
    story.append(Spacer(1, 2 * mm))
    story.append(_img_flow(plantable_png, max_h=85 * mm))
    story.append(Spacer(1, 2 * mm))
    story.append(_img_flow(recommendation_png, max_h=85 * mm))

    story.append(PageBreak())
    story.append(Paragraph("3. Current Urban State - Metrics", h2))
    current_rows = [
        ["Existing canopy cover", f"{metrics['canopy_pct']} % ({metrics['canopy_area_m2']:,.0f} m^2)"],
        ["Grass / low vegetation", f"{metrics['grass_pct']} %"],
        ["Buildings", f"{metrics['buildings_pct']} %"],
        ["Roads / paved surfaces", f"{metrics['roads_pct']} %"],
        ["Water bodies", f"{metrics['water_pct']} %"],
        ["Bare soil", f"{metrics['bare_soil_pct']} %"],
        ["Total impervious surface", f"{metrics['impervious_pct']} %"],
        ["Plantable area available", f"{metrics['plantable_pct']} % ({metrics['plantable_area_m2']:,.0f} m^2)"],
        ["Current heat score (0-100)", f"{metrics['heat_score_current']}"],
        ["Estimated current temperature", f"{metrics['temp_current_c']} °C"],
    ]
    story.append(_metric_table(current_rows))

    story.append(Spacer(1, 5 * mm))
    story.append(Paragraph("4. Before vs After - Recommended Planting Scenario", h2))
    ba_rows = [
        ["Recommended new trees", f"{metrics['recommended_trees']} trees "
                                  f"({metrics['tree_spacing_m']} m min. spacing)"],
        ["Recommended planting coverage", f"+{metrics['recommended_planting_pct']} % of total area"],
        ["New canopy added", f"{metrics['new_canopy_m2']:,.0f} m^2"],
        ["Canopy cover", f"{metrics['canopy_pct']} %  ->  {metrics['projected_canopy_pct']} %"],
        ["Heat score", f"{metrics['heat_score_current']}  ->  {metrics['projected_heat_score']}"],
        ["Estimated temperature", f"{metrics['temp_current_c']} °C  ->  {metrics['projected_temp_c']} °C"],
        ["Cooling reduction", f"-{metrics['cooling_reduction_c']} °C"],
        ["Shade improvement", f"+{metrics['shade_improvement_pct_points']} pct points "
                              f"(+{metrics['shade_improvement_m2']:,.0f} m^2 shade)"],
    ]
    t = _metric_table(ba_rows)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), BLUE)]))
    story.append(t)

    story.append(PageBreak())
    story.append(Paragraph("5. Methodology Summary", h2))
    methodology = (
        "Canopy AI performs all analysis locally - no external AI APIs, cloud services or map APIs "
        "are involved, making results fully reproducible.<br/><br/>"
        "<b>Segmentation.</b> An edge-preserving bilateral filter first stabilises spectral estimates. "
        "Each pixel is then described by vegetation indices (ExG = 2G - R - B, VARI), HSV "
        "hue/saturation/value, RGB, and an 11x11 local texture statistic. A Random Forest classifier, "
        "trained locally on synthetic labeled scenes (engine/train_model.py - no real imagery, no cloud "
        "training), assigns trees, grass, buildings, roads/paved, water or bare/open soil from those "
        "9 features. SLIC superpixels aggregate labels into perceptually uniform regions, a majority "
        "(mode) filter removes salt-and-pepper noise, and connected-component analysis deletes regions "
        "below 0.02% of the image. Dark or high-saturation shadow spectra are re-assigned by "
        "hue/saturation-weighted neighbourhood voting, and compact versus elongated shape analysis "
        "separates buildings from road segments. All masks keep the original image resolution and "
        "align 1:1 with input pixels.<br/><br/>"
        "<b>Plantable mask.</b> Grass and bare-soil pixels are kept, then an exact Euclidean "
        "distance transform enforces a 1.5 m buffer around buildings, roads and trees and a 2.5 m "
        "buffer around water. Regions smaller than 0.04% of the scene are discarded as noise."
        "<br/><br/>"
        "<b>Tree placement.</b> Bridson Poisson-disk sampling (deterministic seed {seed}) drops "
        f"candidate points inside the eroded plantable mask enforcing a strict minimum spacing of "
        f"{params['spacing_m']} m, so markers never overlap existing trees, buildings or roads. Each "
        f"marker assumes a mature crown radius of {params['crown_radius_m']} m.<br/><br/>"
        "<b>Heat model.</b> Heat score = clamp(18 + 0.90*impervious% + 0.40*bare% - 0.70*canopy% "
        "- 0.20*grass%, 0, 100). Estimated temperature uses a baseline of "
        f"{params['base_temp_c']} °C minus {0.06:.2f} °C per percentage point of canopy cover, "
        "consistent with published urban-forestry cooling coefficients (0.04-0.10 °C per canopy % point)."
    ).replace("{seed}", str(params.get("seed", 42)))
    story.append(Paragraph(methodology, body))

    story.append(Spacer(1, 5 * mm))
    story.append(Paragraph("6. Environmental Impact Results", h2))
    impact_rows = [
        ["CO2 sequestration added", f"approx. {metrics['co2_offset_kg_yr']:,.0f} kg CO2 / year "
                                    f"(21 kg per mature tree per year)"],
        ["New shaded ground", f"{metrics['shade_improvement_m2']:,.0f} m^2"],
        ["Projected canopy cover", f"{metrics['projected_canopy_pct']} %"],
        ["Peak-temperature reduction", f"{metrics['cooling_reduction_c']} °C (estimated local air temperature)"],
    ]
    ti = _metric_table(impact_rows)
    ti.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), ORANGE)]))
    story.append(ti)

    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph(
        "Disclaimer: heat score and temperatures are engineering estimates derived from land-cover "
        "fractions and literature-based coefficients; they indicate relative change rather than measured "
        "meteorological values.", small))

    doc.build(story)
    return pdf_path
