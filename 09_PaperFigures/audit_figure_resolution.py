# -*- coding: utf-8 -*-
"""Audit exported raster figure resolution for manuscript assembly.

This does not replace visual QC.  It checks PNG/TIFF pixel dimensions and
calculates the effective horizontal DPI if a figure is placed at a requested
journal width (default 190 mm, approximately full page width).  A matching PDF
or SVG is also reported because vector output should be preferred for journal
submission whenever possible.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--figures-root", type=Path, default=Path("results/09_PaperFigures"))
    p.add_argument("--report", type=Path, default=None)
    p.add_argument("--target-width-mm", type=float, default=190.0)
    p.add_argument("--single-column-width-mm", type=float, default=90.0)
    p.add_argument("--min-effective-dpi", type=float, default=500.0)
    return p.parse_args()


def _dpi_for_width(width_px: int, width_mm: float) -> float:
    return float(width_px) / (float(width_mm) / 25.4)


def main() -> None:
    args = parse_args()
    root = args.figures_root.resolve()
    report = args.report or (root / "figure_resolution_audit.csv")
    report = report.resolve()

    rows: list[dict[str, object]] = []
    raster_files = sorted(
        [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".png", ".tif", ".tiff"}]
    )

    for path in raster_files:
        try:
            with Image.open(path) as im:
                width_px, height_px = im.size
                stored_dpi = im.info.get("dpi")
        except Exception as exc:
            rows.append({
                "file": str(path.relative_to(root)),
                "status": "READ_ERROR",
                "error": f"{type(exc).__name__}: {exc}",
            })
            continue

        full_dpi = _dpi_for_width(width_px, args.target_width_mm)
        single_dpi = _dpi_for_width(width_px, args.single_column_width_mm)
        base = path.with_suffix("")
        pdf = base.with_suffix(".pdf")
        svg = base.with_suffix(".svg")
        vector_available = pdf.exists() or svg.exists()
        if full_dpi >= args.min_effective_dpi:
            status = "PASS"
        elif vector_available:
            status = "RASTER_LOW_BUT_VECTOR_AVAILABLE"
        else:
            status = "LOW_RESOLUTION"

        if isinstance(stored_dpi, tuple) and len(stored_dpi) >= 2:
            stored_dpi_x = float(stored_dpi[0])
            stored_dpi_y = float(stored_dpi[1])
        else:
            stored_dpi_x = stored_dpi_y = None

        rows.append({
            "file": str(path.relative_to(root)),
            "width_px": int(width_px),
            "height_px": int(height_px),
            "stored_dpi_x": stored_dpi_x,
            "stored_dpi_y": stored_dpi_y,
            f"effective_dpi_at_{args.target_width_mm:g}mm": round(full_dpi, 1),
            f"effective_dpi_at_{args.single_column_width_mm:g}mm": round(single_dpi, 1),
            "pdf_available": pdf.exists(),
            "svg_available": svg.exists(),
            "status": status,
            "error": "",
        })

    df = pd.DataFrame(rows)
    report.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(report, index=False, encoding="utf-8-sig")

    print(f"[AUDIT] figure root: {root}")
    print(f"[AUDIT] raster files: {len(df)}")
    if len(df):
        counts = df["status"].value_counts(dropna=False)
        for status, n in counts.items():
            print(f"[AUDIT] {status}: {int(n)}")
    print(f"[AUDIT] report: {report}")
    print(
        "[NOTE] For submission, prefer the generated PDF files for plots. "
        "Use the high-resolution PNG files for Word/WPS manuscript assembly."
    )


if __name__ == "__main__":
    main()
