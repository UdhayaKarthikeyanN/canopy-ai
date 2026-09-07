"""Shared class palette helpers (hex colors for UI/report)."""

from engine.segmentation import ALL_CLASSES, CLASS_COLORS_RGB, CLASS_NAMES


def hex_for(cls):
    r, g, b = CLASS_COLORS_RGB[cls]
    return "#{:02X}{:02X}{:02X}".format(r, g, b)


def legend_entries():
    return [{"id": int(c), "name": CLASS_NAMES[c], "hex": hex_for(c)} for c in ALL_CLASSES]
