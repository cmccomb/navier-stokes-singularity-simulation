"""Render actual active native cell edges for a fixed irregular AMR hierarchy."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.ragged_native import bounding_levels, diagnostic_masks
from scripts.render_native_3d import sha
from scripts.render_native_mesh import COLORS, transition_cells


def runs(mask):
    changes = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(int)))
    return zip(changes[::2], changes[1::2], strict=True)


def plane_edges(blocks, masks, fixed):
    h, v = [d for d in range(3) if d != fixed]
    edges = []
    for level in bounding_levels(blocks):
        low = np.array(level["index_lo"])
        size = level["shape"]
        dx = level["spacing"][0]
        grid = np.zeros((size[h], size[v]), dtype=bool)
        for block, mask in zip(blocks, masks, strict=True):
            if block["level"] != level["level"]:
                continue
            at = round(1 / dx) - block["index_lo"][fixed]
            if not 0 <= at < block["shape"][fixed]:
                continue
            offset = np.array(block["index_lo"]) - low
            grid[offset[h]:offset[h] + block["shape"][h],
                 offset[v]:offset[v] + block["shape"][v]] = np.take(mask, at, axis=fixed)
        vertical = np.zeros((size[h] + 1, size[v]), dtype=bool)
        horizontal = np.zeros((size[h], size[v] + 1), dtype=bool)
        vertical[:-1] |= grid
        vertical[1:] |= grid
        horizontal[:, :-1] |= grid
        horizontal[:, 1:] |= grid
        for index, row in enumerate(vertical):
            for a, b in runs(row):
                start, end = [0.0] * 3, [0.0] * 3
                start[h] = end[h] = -1 + (low[h] + index) * dx
                start[v], end[v] = -1 + (low[v] + a) * dx, -1 + (low[v] + b) * dx
                edges.append([level["level"], *start, *end])
        for index, row in enumerate(horizontal.T):
            for a, b in runs(row):
                start, end = [0.0] * 3, [0.0] * 3
                start[v] = end[v] = -1 + (low[v] + index) * dx
                start[h], end[h] = -1 + (low[h] + a) * dx, -1 + (low[h] + b) * dx
                edges.append([level["level"], *start, *end])
    return edges


def mesh_record(run, source_hash, blocks):
    final = run["history"][-1]
    masks = diagnostic_masks(blocks)
    stored = sum(int(np.prod(b["shape"][:3])) for b in blocks)
    active = sum(int(m.sum()) for m in masks)
    if (stored, active) != (final["stored_cells"], final["active_cells"]):
        raise ValueError("Native mesh cell counts disagree with solver")
    volume = sum(int(m.sum()) * b["spacing"][0]**3 for b, m in zip(blocks, masks, strict=True))
    if volume != 8:
        raise ValueError("Native active mesh does not cover the domain once")
    n = run["parameters"]["base_n"]
    widths = [1, *run["parameters"]["widths"]]
    levels = [{"level": i, "half": h, "n": n, "dx": 2 * h / n, "color": COLORS[i]}
              for i, h in enumerate(widths)]
    cells, bounds = transition_cells(levels)
    for index, x, y, z, dx in cells:
        center = np.array([x, y, z]) + dx / 2
        owners = [b["level"] for b in blocks if np.all(center >= b["origin"]) and
                  np.all(center < np.array(b["origin"]) + np.array(b["shape"][:3]) * b["spacing"])]
        if max(owners, default=-1) != index:
            raise ValueError("Core close-up does not match actual native ownership")
    planes = {str(fixed): plane_edges(blocks, masks, fixed) for fixed in range(3)}
    return {"schema_version": 2, "run_id": "outer-action-n128-l5-rest-t0995",
            "source_record_sha256": source_hash, "levels": levels,
            "planes_by_fixed_axis": planes, "planes3d": [edge for plane in planes.values() for edge in plane],
            "cells": cells, "detail_bounds": bounds, "native_blocks": blocks,
            "stored_cells": stored, "active_cells": active, "composite_volume": volume,
            "scope": "Active native cell edges in each distinct central section; irregular outer bands included. Cube outlines identify central refinement regions only."}


def render_panel(mesh, view):
    iso = view == "isometric"
    right, up = np.array([1, 1, 0]) / np.sqrt(2), np.array([-1, 1, 2]) / np.sqrt(6)
    edges = mesh["planes3d"] if iso else mesh["planes_by_fixed_axis"]["2"]
    result = ['<svg xmlns="http://www.w3.org/2000/svg" width="800" height="800" viewBox="0 0 800 800" role="img" aria-labelledby="title desc">',
              f'<title id="title">Oliver outer-refinement mesh: {view}</title>',
              '<desc id="desc">Actual active native cell edges, including outer bands. Covered coarse interiors are excluded.</desc>',
              '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#9fb3c2;font-size:27px}</style>']
    for level, *points in edges:
        a, b = np.array(points[:3]), np.array(points[3:])
        project = (lambda p: [400 + 220 * (p @ right), 400 - 220 * (p @ up)]) if iso else (lambda p: [400 + 220 * p[0], 400 - 220 * p[1]])
        x, y = project(a), project(b)
        result.append(f'<path d="M{x[0]:.3f},{x[1]:.3f}L{y[0]:.3f},{y[1]:.3f}" fill="none" stroke="{COLORS[level]}" stroke-width="0.8"/>')
    if iso:
        for d, name in enumerate("xyz"):
            p = [400 + 255 * right[d], 400 - 255 * up[d]]
            result.append(f'<text x="{p[0]:.2f}" y="{p[1]:.2f}" text-anchor="middle">{name}</text>')
    else:
        for value in (-1, 0, 1):
            result.append(f'<text x="{400 + 220 * value}" y="663" text-anchor="middle">{value}</text>')
            result.append(f'<text x="164" y="{409 - 220 * value}" text-anchor="end">{value}</text>')
        result.extend(['<text x="400" y="707" text-anchor="middle">x</text>', '<text x="120" y="400">y</text>'])
    return "\n".join([*result, "</svg>"]) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "prepared", "site"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run = json.loads(args.run.read_text())
    frame = json.loads(args.prepared.read_text())
    if frame["source_record_sha256"] != sha(args.run):
        raise ValueError("Prepared mesh has the wrong source")
    mesh = mesh_record(run, sha(args.run), frame["blocks"])
    (args.site / "data/native-mesh.json").write_text(json.dumps(mesh, indent=2) + "\n")
    for view in ("xy", "isometric"):
        (args.site / f"media/mesh-{view}.svg").write_text(render_panel(mesh, view))


if __name__ == "__main__":
    main()
