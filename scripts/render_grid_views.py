"""Actual fixed native cell edges: XY, XZ, isometric, in that order.

The isometric panel shows three central grid sections, not an opaque pile of
all voxel edges. Covered coarse interiors are excluded. Same world scale in
every view. Center-section construction follows render_native_mesh.py from
the separately reviewed native-mesh UI work.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

COLORS = ("#42677e", "#328daf", "#55becf", "#9bddd9", "#ffe8a3")


def hierarchy(record):
    n = record["parameters"]["base_n"]
    widths = [record["profile"]["half_domain"], *record["parameters"]["widths"]]
    if n % 4 or any(b != a / 2 for a, b in itertools.pairwise(widths)):
        raise ValueError("Expected centered, aligned 2:1 cubes")
    if len(widths) * n**3 - (len(widths) - 1) * (n // 2) ** 3 != record["active_cells"]:
        raise ValueError("Native active cell count disagrees")
    return [{"half": half, "n": n, "spacing": 2 * half / n} for half in widths]


def plane_edges(levels):
    edges = []
    for level, spec in enumerate(levels):
        h, n, dx = spec["half"], spec["n"], spec["spacing"]
        hole = levels[level + 1]["half"] if level + 1 < len(levels) else 0
        for j in range(n + 1):
            value = -h + j * dx
            spans = (
                [(-h, -hole), (hole, h)] if hole and abs(value) < hole else [(-h, h)]
            )
            for low, high in spans:
                edges.extend(
                    (
                        (level, (value, low), (value, high)),
                        (level, (low, value), (high, value)),
                    )
                )
    return edges


def render(record):
    levels = hierarchy(record)
    edges = plane_edges(levels)
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="2400" height="960" viewBox="0 0 2400 960" role="img" aria-labelledby="title desc">',
        '<title id="title">Native grid: x–y, x–z, isometric</title>',
        '<desc id="desc">Actual active cell boundaries in fixed central sections. Left: x-y at z=0. Middle: x-z at y=0. Right: isometric projection of three perpendicular central sections. Covered coarse interiors are omitted. The static mesh does not change with time.</desc>',
        '<rect width="2400" height="960" fill="#07111f"/>',
        "<style>text{font-family:Arial,Helvetica,sans-serif;fill:#9fb3c2;font-size:27px}.title{fill:#e9f1f5;font-size:42px;font-weight:bold}.heading{fill:#e9f1f5;font-size:34px}</style>",
        '<text class="title" x="108" y="75">Native grid</text>',
        '<text x="1490" y="75">Fixed geometry · all 280 saved states</text>',
    ]
    right = np.array([1, 1, 0]) / np.sqrt(2)
    up = np.array([-1, 1, 2]) / np.sqrt(6)
    # One physical scale for both planes and the orthographic 3D section view.
    scale = 220
    for view, label in enumerate(
        ("x–y slice · z = 0", "x–z slice · y = 0", "Isometric · central grid sections")
    ):
        center = np.array([400 + 800 * view, 460])
        parts.append(
            f'<text class="heading" x="{center[0]}" y="160" text-anchor="middle">{label}</text>'
        )
        for level, a, b in edges:
            planes = ((0, 1), (0, 2), (1, 2)) if view == 2 else ((0, 1),)
            for h, v in planes:
                points = []
                for end in (a, b):
                    if view == 2:
                        world = np.zeros(3)
                        world[h], world[v] = end
                        projected = np.array([world @ right, -(world @ up)])
                    else:
                        projected = np.array([end[0], -end[1]])
                    points.append(center + scale * projected)
                p, q = points
                parts.append(
                    f'<path d="M{p[0]:.3f},{p[1]:.3f}L{q[0]:.3f},{q[1]:.3f}" fill="none" stroke="{COLORS[level]}" stroke-width="0.8"/>'
                )
        if view < 2:
            for value in (-1, 0, 1):
                parts.append(
                    f'<text x="{center[0] + scale * value}" y="723" text-anchor="middle">{value}</text>'
                )
                parts.append(
                    f'<text x="{center[0] - scale - 16}" y="{center[1] - scale * value + 9}" text-anchor="end">{value}</text>'
                )
            parts.append(f'<text x="{center[0]}" y="767" text-anchor="middle">x</text>')
            parts.append(
                f'<text x="{center[0] - scale - 60}" y="460">{"y" if view == 0 else "z"}</text>'
            )
        else:
            for d, name in enumerate("xyz"):
                end = center + 1.16 * scale * np.array([right[d], -up[d]])
                parts.append(
                    f'<text x="{end[0]:.2f}" y="{end[1]:.2f}" text-anchor="middle">{name}</text>'
                )
    for i, spec in enumerate(levels):
        parts.append(
            f'<text x="{108 + i * 450}" y="858" style="fill:{COLORS[i]}">Level {i} · Δ = {spec["spacing"]:g}</text>'
        )
    parts.append(
        '<text x="108" y="923">Actual native cell edges · covered coarse interiors excluded · same physical scale · model length units · static grid</text></svg>'
    )
    return "\n".join(parts) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, default=Path("site/data/best.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(render(json.loads(args.record.read_text())))


if __name__ == "__main__":
    main()
