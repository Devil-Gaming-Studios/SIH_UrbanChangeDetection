"""
Builds the visual output images (mask, overlay, NDVI/NDWI maps + their
overlays) and a downloadable PDF report summarizing urban growth.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle,
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors


def _colorize(normalized: np.ndarray, cmap_name: str) -> np.ndarray:
    cmap = plt.get_cmap(cmap_name)
    colored = cmap(np.clip(normalized, 0.0, 1.0))[:, :, :3]
    return (colored * 255).astype(np.uint8)


def _overlay(base_rgb: np.ndarray, mask: np.ndarray, color=(255, 0, 0), alpha=0.5) -> np.ndarray:
    out = base_rgb.astype(np.float32).copy()
    color_arr = np.array(color, dtype=np.float32)
    out[mask] = (1 - alpha) * out[mask] + alpha * color_arr
    return np.clip(out, 0, 255).astype(np.uint8)


def _resize_to(base_img: Image.Image, hw) -> np.ndarray:
    if base_img.size != (hw[1], hw[0]):
        base_img = base_img.resize((hw[1], hw[0]))
    return np.array(base_img.convert("RGB"))


def save_outputs(result: dict, earlier_image_path, later_image_path, output_dir, stem, threshold=0.5):
    """Save all output images to output_dir. Returns a dict of file paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    probability = result["probability"]
    change_mask = result["change_mask"]
    orig_hw = result["orig_hw"]

    later_img = Image.open(later_image_path)
    later_rgb = _resize_to(later_img, orig_hw)
    earlier_img = Image.open(earlier_image_path)
    earlier_rgb = _resize_to(earlier_img, orig_hw)

    paths = {}

    # 0. Before / after side-by-side comparison
    side_by_side = np.concatenate([earlier_rgb, later_rgb], axis=1)
    p = output_dir / f"{stem}_before_after.png"
    Image.fromarray(side_by_side).save(p)
    paths["before_after"] = p

    # 1. Binary change bitmap
    mask_img = (change_mask.astype(np.uint8) * 255)
    p = output_dir / f"{stem}_change_mask.png"
    Image.fromarray(mask_img).save(p)
    paths["change_mask"] = p

    # 2. Change probability heatmap
    heat = _colorize(probability, "inferno")
    p = output_dir / f"{stem}_change_prob.png"
    Image.fromarray(heat).save(p)
    paths["change_prob"] = p

    # 3. Change bitmap overlaid on the newer image
    overlay_newer = _overlay(later_rgb, change_mask, color=(255, 0, 0), alpha=0.5)
    p = output_dir / f"{stem}_overlay_on_newer.png"
    Image.fromarray(overlay_newer).save(p)
    paths["overlay_on_newer"] = p

    # 4. NDVI maps (earlier / later) + change delta
    ndvi_earlier = _colorize(np.clip((result["ndvi_earlier"] + 1) / 2, 0, 1), "RdYlGn")
    p = output_dir / f"{stem}_ndvi_earlier.png"
    Image.fromarray(ndvi_earlier).save(p)
    paths["ndvi_earlier"] = p

    ndvi_later = _colorize(np.clip((result["ndvi_later"] + 1) / 2, 0, 1), "RdYlGn")
    p = output_dir / f"{stem}_ndvi_later.png"
    Image.fromarray(ndvi_later).save(p)
    paths["ndvi_later"] = p

    ndvi_delta_img = _colorize(np.clip(result["ndvi_delta"] / 2.0, 0, 1), "inferno")
    p = output_dir / f"{stem}_ndvi_change.png"
    Image.fromarray(ndvi_delta_img).save(p)
    paths["ndvi_change"] = p

    # 5. NDVI overlay on newer image (highlighting vegetation loss areas)
    veg_loss_mask = (result["ndvi_earlier"] - result["ndvi_later"]) > 0.15
    ndvi_overlay = _overlay(later_rgb, veg_loss_mask, color=(255, 140, 0), alpha=0.5)
    p = output_dir / f"{stem}_ndvi_overlay_on_newer.png"
    Image.fromarray(ndvi_overlay).save(p)
    paths["ndvi_overlay_on_newer"] = p

    # 6. NDWI maps (earlier / later) + change delta
    ndwi_earlier = _colorize(np.clip((result["ndwi_earlier"] + 1) / 2, 0, 1), "Blues")
    p = output_dir / f"{stem}_ndwi_earlier.png"
    Image.fromarray(ndwi_earlier).save(p)
    paths["ndwi_earlier"] = p

    ndwi_later = _colorize(np.clip((result["ndwi_later"] + 1) / 2, 0, 1), "Blues")
    p = output_dir / f"{stem}_ndwi_later.png"
    Image.fromarray(ndwi_later).save(p)
    paths["ndwi_later"] = p

    ndwi_delta_img = _colorize(np.clip(result["ndwi_delta"] / 2.0, 0, 1), "inferno")
    p = output_dir / f"{stem}_ndwi_change.png"
    Image.fromarray(ndwi_delta_img).save(p)
    paths["ndwi_change"] = p

    # 7. NDWI overlay on newer image (highlighting water-body change)
    water_change_mask = np.abs(result["ndwi_later"] - result["ndwi_earlier"]) > 0.15
    ndwi_overlay = _overlay(later_rgb, water_change_mask, color=(0, 120, 255), alpha=0.5)
    p = output_dir / f"{stem}_ndwi_overlay_on_newer.png"
    Image.fromarray(ndwi_overlay).save(p)
    paths["ndwi_overlay_on_newer"] = p

    # 8. Hotspot map — grid-cell change density, to show where change is concentrated
    p = output_dir / f"{stem}_hotspot.png"
    make_hotspot_map(change_mask, p)
    paths["hotspot"] = p

    # 9. Confidence / uncertainty map (from QA validity, if available)
    if "qa_later" in result:
        conf_img = _colorize(result["qa_later"], "Greens")
        p = output_dir / f"{stem}_confidence.png"
        Image.fromarray(conf_img).save(p)
        paths["confidence"] = p

    return paths


def make_hotspot_map(change_mask: np.ndarray, output_path, grid_size=20):
    """Grid-based heatmap of where change is concentrated."""
    h, w = change_mask.shape
    gh = max(1, h // grid_size)
    gw = max(1, w // grid_size)
    rows = h // gh
    cols = w // gw

    density = np.zeros((rows, cols), dtype=np.float32)
    for r in range(rows):
        for c in range(cols):
            cell = change_mask[r * gh:(r + 1) * gh, c * gw:(c + 1) * gw]
            density[r, c] = cell.mean() if cell.size else 0.0

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    im = ax.imshow(density, cmap="hot", vmin=0, vmax=max(density.max(), 1e-6))
    ax.set_title("Change hotspot density (by grid cell)")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, label="Fraction of cell changed")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def compute_change_direction(change_mask: np.ndarray):
    """Estimate the dominant compass direction of expansion, via the
    centroid of changed pixels relative to the image center."""
    ys, xs = np.nonzero(change_mask)
    if len(xs) == 0:
        return "No significant directional trend detected."

    h, w = change_mask.shape
    center_y, center_x = h / 2.0, w / 2.0
    mean_y, mean_x = ys.mean(), xs.mean()

    dy = mean_y - center_y  # positive -> south
    dx = mean_x - center_x  # positive -> east

    ns = "south" if dy > h * 0.03 else ("north" if dy < -h * 0.03 else None)
    ew = "east" if dx > w * 0.03 else ("west" if dx < -w * 0.03 else None)

    if ns and ew:
        return f"Change is concentrated toward the {ns}-{ew} of the scene."
    elif ns:
        return f"Change is concentrated toward the {ns} of the scene."
    elif ew:
        return f"Change is concentrated toward the {ew} of the scene."
    return "Change is fairly evenly distributed across the scene, with no strong directional trend."


def compute_growth_stats(result: dict, year_earlier: int, year_later: int, pixel_resolution_m: float = None):
    """Compute urban-growth style statistics from the change mask.

    'New urban/changed area' = fraction of pixels flagged as changed.
    Growth rate = change fraction annualized over the gap between the two
    supplied years (defaults to 1 year apart if years are equal/unknown).

    pixel_resolution_m: ground sampling distance in meters/pixel, if known
    (e.g. 10 for Sentinel-2, 5 for LISS-4). Enables real-world area units.
    """
    change_mask = result["change_mask"]
    total_pixels = change_mask.size
    changed_pixels = int(change_mask.sum())
    changed_fraction = changed_pixels / total_pixels

    year_gap = max(1, abs(year_later - year_earlier))
    annual_rate = changed_fraction / year_gap

    # Simple linear projection: assume the same annual rate continues.
    projection_years = list(range(year_earlier, year_earlier + year_gap * 6, year_gap if year_gap else 1))
    cumulative = []
    for i, _ in enumerate(projection_years):
        running = min(1.0, annual_rate * (i * year_gap if year_gap else i))
        cumulative.append(running)

    stats = {
        "total_pixels": total_pixels,
        "changed_pixels": changed_pixels,
        "changed_fraction": changed_fraction,
        "changed_percentage": changed_fraction * 100,
        "year_gap": year_gap,
        "annual_growth_rate_percentage": annual_rate * 100,
        "projection_years": projection_years,
        "projection_cumulative_percentage": [c * 100 for c in cumulative],
    }

    # Land-use breakdown: split "changed" pixels into vegetation-loss,
    # water-change, and other/built-up buckets using the NDVI/NDWI deltas.
    veg_loss_mask = change_mask & ((result["ndvi_earlier"] - result["ndvi_later"]) > 0.15)
    water_change_mask = change_mask & (np.abs(result["ndwi_later"] - result["ndwi_earlier"]) > 0.15)
    builtup_mask = change_mask & ~veg_loss_mask & ~water_change_mask

    stats["landuse_breakdown"] = {
        "built_up_percentage": float(builtup_mask.mean() * 100),
        "vegetation_loss_percentage": float(veg_loss_mask.mean() * 100),
        "water_change_percentage": float(water_change_mask.mean() * 100),
    }

    # Real-world area units, if a ground sampling distance was provided.
    if pixel_resolution_m:
        pixel_area_m2 = pixel_resolution_m ** 2
        changed_area_m2 = changed_pixels * pixel_area_m2
        stats["pixel_resolution_m"] = pixel_resolution_m
        stats["changed_area_hectares"] = changed_area_m2 / 10_000
        stats["changed_area_km2"] = changed_area_m2 / 1_000_000
        stats["annual_growth_area_hectares"] = stats["changed_area_hectares"] / year_gap

    # Confidence / uncertainty summary from the QA validity maps, if present.
    if "qa_later" in result:
        low_confidence_fraction = float((result["qa_later"] < 0.5).mean())
        stats["low_confidence_percentage"] = low_confidence_fraction * 100
        stats["mean_confidence_percentage"] = float(result["qa_later"].mean() * 100)

    # Direction of expansion.
    stats["direction_summary"] = compute_change_direction(change_mask)

    # Auto-generated executive summary paragraph.
    area_clause = ""
    if pixel_resolution_m:
        area_clause = f" (~{stats['changed_area_hectares']:.1f} hectares)"
    summary = (
        f"Between {year_earlier} and {year_later}, approximately "
        f"{stats['changed_percentage']:.1f}% of the analyzed area changed"
        f"{area_clause}, an estimated annual growth rate of "
        f"{stats['annual_growth_rate_percentage']:.2f}% per year. "
        f"{stats['direction_summary']} Of the changed area, "
        f"{stats['landuse_breakdown']['built_up_percentage']:.1f}% appears "
        f"built-up/other change, {stats['landuse_breakdown']['vegetation_loss_percentage']:.1f}% "
        f"shows vegetation loss, and {stats['landuse_breakdown']['water_change_percentage']:.1f}% "
        f"shows water-body change."
    )
    if "low_confidence_percentage" in stats:
        summary += (
            f" {stats['low_confidence_percentage']:.1f}% of the newer image was flagged "
            f"as low-confidence (cloud/shadow) and should be interpreted cautiously."
        )
    stats["executive_summary"] = summary

    return stats


def make_growth_chart(stats: dict, output_path):
    fig, ax = plt.subplots(figsize=(6, 3.2), dpi=150)
    ax.plot(
        stats["projection_years"],
        stats["projection_cumulative_percentage"],
        marker="o", color="#c0392b",
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Cumulative changed area (%)")
    ax.set_title("Projected urban change growth")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def build_pdf_report(
    output_pdf_path,
    image_paths: dict,
    stats: dict,
    growth_chart_path,
    year_earlier,
    year_later,
    title="Urban Change Detection Report",
):
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(output_pdf_path), pagesize=A4,
                             topMargin=1.5 * cm, bottomMargin=1.5 * cm)
    story = []

    story.append(Paragraph(title, styles["Title"]))
    story.append(Paragraph(f"Comparing {year_earlier} vs {year_later}", styles["Normal"]))
    story.append(Spacer(1, 0.4 * cm))

    # Executive summary
    if "executive_summary" in stats:
        story.append(Paragraph("Executive summary", styles["Heading2"]))
        story.append(Paragraph(stats["executive_summary"], styles["BodyText"]))
        story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("Summary statistics", styles["Heading2"]))
    table_data = [
        ["Metric", "Value"],
        ["Total pixels analyzed", f"{stats['total_pixels']:,}"],
        ["Changed pixels", f"{stats['changed_pixels']:,}"],
        ["New / changed area", f"{stats['changed_percentage']:.2f}%"],
        ["Time span", f"{stats['year_gap']} year(s)"],
        ["Estimated annual growth rate", f"{stats['annual_growth_rate_percentage']:.2f}% / year"],
    ]
    if "changed_area_km2" in stats:
        table_data.append(["Changed area", f"{stats['changed_area_km2']:.3f} km\u00b2 ({stats['changed_area_hectares']:.1f} ha)"])
        table_data.append(["Annual growth area", f"{stats['annual_growth_area_hectares']:.2f} ha / year"])
    if "mean_confidence_percentage" in stats:
        table_data.append(["Mean data confidence", f"{stats['mean_confidence_percentage']:.1f}%"])
        table_data.append(["Low-confidence area (cloud/shadow)", f"{stats['low_confidence_percentage']:.1f}%"])
    table = Table(table_data, colWidths=[8 * cm, 7 * cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
    ]))
    story.append(table)
    story.append(Spacer(1, 0.5 * cm))

    # Land-use breakdown table
    if "landuse_breakdown" in stats:
        story.append(Paragraph("Land-use change breakdown", styles["Heading2"]))
        lb = stats["landuse_breakdown"]
        breakdown_data = [
            ["Category", "% of scene"],
            ["Built-up / other change", f"{lb['built_up_percentage']:.2f}%"],
            ["Vegetation loss", f"{lb['vegetation_loss_percentage']:.2f}%"],
            ["Water-body change", f"{lb['water_change_percentage']:.2f}%"],
        ]
        breakdown_table = Table(breakdown_data, colWidths=[8 * cm, 7 * cm])
        breakdown_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
        ]))
        story.append(breakdown_table)
        story.append(Spacer(1, 0.5 * cm))

    # Direction of expansion
    if "direction_summary" in stats:
        story.append(Paragraph("Direction of change", styles["Heading2"]))
        story.append(Paragraph(stats["direction_summary"], styles["BodyText"]))
        story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("Projected growth", styles["Heading2"]))
    story.append(RLImage(str(growth_chart_path), width=15 * cm, height=8 * cm))
    story.append(Spacer(1, 0.5 * cm))

    def add_image_row(section_title, keys_and_labels, width=14 * cm, height=9 * cm):
        story.append(Paragraph(section_title, styles["Heading2"]))
        for key, label in keys_and_labels:
            if key in image_paths:
                story.append(Paragraph(label, styles["Normal"]))
                story.append(RLImage(str(image_paths[key]), width=width, height=height))
                story.append(Spacer(1, 0.3 * cm))

    add_image_row("Before / after comparison", [
        ("before_after", f"{year_earlier} (left) vs {year_later} (right)"),
    ], width=16 * cm, height=6 * cm)

    add_image_row("Change hotspots", [
        ("hotspot", "Grid-based change density — shows where change is concentrated"),
    ])

    if "confidence" in image_paths:
        add_image_row("Data confidence", [
            ("confidence", "QA validity map for the newer image (darker = lower confidence)"),
        ])

    add_image_row("Change detection outputs", [
        ("change_mask", "Binary change bitmap"),
        ("change_prob", "Change probability heatmap"),
        ("overlay_on_newer", "Change bitmap overlaid on the newer image"),
    ])

    add_image_row("NDVI (vegetation index)", [
        ("ndvi_earlier", f"NDVI — {year_earlier}"),
        ("ndvi_later", f"NDVI — {year_later}"),
        ("ndvi_change", "NDVI change magnitude"),
        ("ndvi_overlay_on_newer", "NDVI change overlaid on the newer image"),
    ])

    add_image_row("NDWI (water index)", [
        ("ndwi_earlier", f"NDWI — {year_earlier}"),
        ("ndwi_later", f"NDWI — {year_later}"),
        ("ndwi_change", "NDWI change magnitude"),
        ("ndwi_overlay_on_newer", "NDWI change overlaid on the newer image"),
    ])

    doc.build(story)
    return output_pdf_path
