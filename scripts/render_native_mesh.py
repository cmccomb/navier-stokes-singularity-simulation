"""Draw active cells from the completed run's fixed refinement geometry."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COLORS = ["#42677e", "#328daf", "#55becf", "#9bddd9", "#ffe8a3"]


def hierarchy(record: dict) -> list[dict]:
    n = record["parameters"]["base_n"]
    widths = [record["profile"]["half_domain"], *record["parameters"]["widths"]]
    if n < 4 or n % 4 or any(b != a / 2 for a, b in itertools.pairwise(widths)):
        raise ValueError("Expected centered 2:1 refinement with aligned cell faces")
    levels = [
        {"level": i, "half": h, "n": n, "dx": 2 * h / n, "color": COLORS[i]}
        for i, h in enumerate(widths)
    ]
    stored = len(levels) * n**3
    active = stored - (len(levels) - 1) * (n // 2) ** 3
    if (stored, active) != (record["stored_cells"], record["active_cells"]):
        raise ValueError("Mesh geometry disagrees with the run's recorded cell counts")
    return levels


def plane_lines(levels: list[dict]) -> list[list]:
    """Every active face edge in a center section; covered interiors are absent."""
    lines = []
    for i, level in enumerate(levels):
        h, dx, n = level["half"], level["dx"], level["n"]
        hole = levels[i + 1]["half"] if i + 1 < len(levels) else 0
        for j in range(n + 1):
            v = -h + j * dx
            spans = [(-h, -hole), (hole, h)] if hole and abs(v) < hole else [(-h, h)]
            for lo, hi in spans:
                lines.extend([[i, v, lo, v, hi], [i, lo, v, hi, v]])
    return lines


def transition_cells(levels: list[dict]) -> tuple[list, list]:
    """An aligned block across the positive-x boundary of the finest cube."""
    fine, coarse = levels[-1], levels[-2]
    h, d = fine["half"], coarse["dx"]
    bounds = [[h - 2 * d, h + 2 * d], [-2 * d, 2 * d], [-2 * d, 2 * d]]
    cells = []
    for level in (coarse, fine):
        spacing, half = level["dx"], level["half"]
        indices = [
            range(
                max(0, round((lo + half) / spacing)),
                min(level["n"], round((hi + half) / spacing)),
            )
            for lo, hi in bounds
        ]
        for index in itertools.product(*indices):
            low = [-half + j * spacing for j in index]
            center = [v + spacing / 2 for v in low]
            if level is coarse and all(abs(v) < h for v in center):
                continue
            cells.append([level["level"], *low, spacing])
    return cells, bounds


def render_svg(levels: list[dict], lines: list[list]) -> str:
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="430" viewBox="0 0 720 430" role="img" aria-labelledby="title desc">',
        '<title id="title">Active nested solver cells</title>',
        '<desc id="desc">Actual cell edges in the central x-z section, with a close-up of the innermost refinement interface. Covered coarse cells are excluded.</desc>',
        '<rect width="720" height="430" fill="#07111f"/>',
        "<style>text{font-family:Arial,Helvetica,sans-serif;fill:#9fb3c2;font-size:14px}.heading{fill:#e9f1f5;font-size:16px}</style>",
    ]
    fine = levels[-1]["half"]
    views = [
        (42, 57, levels[0]["half"], "Whole domain"),
        (396, 57, fine * 1.5, "Core and refinement interface"),
    ]
    for k, (left, top, half, title) in enumerate(views):
        size = 278

        def xy(x, y, left=left, top=top, half=half, size=size):
            return left + (x + half) * size / (2 * half), top + (half - y) * size / (
                2 * half
            )

        parts.extend(
            [
                f'<text class="heading" x="{left}" y="30">{title}</text>',
                f'<defs><clipPath id="mesh-clip-{k}"><rect x="{left}" y="{top}" width="{size}" height="{size}"/></clipPath></defs>',
                f'<g clip-path="url(#mesh-clip-{k})">',
            ]
        )
        for i, x0, y0, x1, y1 in lines:
            a, b = xy(x0, y0)
            c, d = xy(x1, y1)
            parts.append(
                f'<path d="M{a:.4f},{b:.4f}L{c:.4f},{d:.4f}" stroke="{COLORS[i]}" stroke-width="0.55"/>'
            )
        parts.append("</g>")
        parts.append(
            f'<rect x="{left}" y="{top}" width="{size}" height="{size}" fill="none" stroke="#42677e"/>'
        )
        for value in (-half, 0, half):
            x, y = xy(value, value)
            parts.append(
                f'<text x="{x}" y="{top + size + 22}" text-anchor="middle">{value:g}</text>'
            )
            parts.append(
                f'<text x="{left - 7}" y="{y + 5}" text-anchor="end">{value:g}</text>'
            )
        parts.extend(
            [
                f'<text x="{left + size / 2}" y="{top + size + 46}" text-anchor="middle">x</text>',
                f'<text x="{left - 26}" y="{top + size / 2}" text-anchor="middle">z</text>',
            ]
        )
    parts.append(
        '<text x="42" y="414">y = 0 · active cell boundaries · covered coarse cells excluded</text></svg>'
    )
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, default=ROOT / "site/data/best.json")
    args = parser.parse_args()
    record = json.loads(args.record.read_text())
    levels = hierarchy(record)
    lines = plane_lines(levels)
    cells, bounds = transition_cells(levels)
    data = {
        "run_id": record["id"],
        "levels": levels,
        "planes": lines,
        "cells": cells,
        "detail_bounds": bounds,
        "stored_cells": record["stored_cells"],
        "active_cells": record["active_cells"],
    }
    (ROOT / "site/data/native-mesh.json").write_text(
        json.dumps(data, separators=(",", ":")) + "\n"
    )
    (ROOT / "site/media/native-mesh.svg").write_text(render_svg(levels, lines))
    print(
        json.dumps(
            {
                "levels": len(levels),
                "plane_segments": len(lines),
                "detail_cells": len(cells),
                "active_cells": data["active_cells"],
            }
        )
    )


if __name__ == "__main__":
    main()
