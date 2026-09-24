"""Display reconstruction and diagnostics for fixed, irregular native AMR boxes.

Native magnitudes are retained at every cell center. Trilinear reconstruction
crosses same-level box boundaries; uncovered fine-level slots use the coarser
reconstruction for the display stencil, but never take ownership of a sample.
"""

from __future__ import annotations

import numpy as np

from scripts.compare_mesh_snapshots import active_mask
from scripts.render_native_3d import interpolate_scalar


def bounding_levels(blocks):
    levels = []
    for index in sorted({b["level"] for b in blocks}):
        selected = [b for b in blocks if b["level"] == index]
        low = np.min([b["index_lo"] for b in selected], axis=0)
        high = np.max(
            [np.array(b["index_lo"]) + b["shape"][:3] for b in selected], axis=0
        )
        dx = np.array(selected[0]["spacing"])
        levels.append({
            "level": index, "index_lo": low.tolist(),
            "origin": (-1 + low * dx).tolist(), "spacing": dx.tolist(),
            "shape": [*map(int, high - low), 6],
            "stored_cells": sum(int(np.prod(b["shape"][:3])) for b in selected),
        })
    return levels


def diagnostic_masks(blocks):
    return [active_mask(b, blocks) for b in blocks]


def diagnostics(fields, blocks, masks, rest=False):
    volume = energy = peak = 0.0
    for values, block, mask in zip(fields, blocks, masks, strict=True):
        if not np.isfinite(values).all() or (rest and np.any(values)):
            raise ValueError("Nonfinite native block or nonzero rest")
        speed2 = np.sum(values[..., :3] ** 2, axis=-1)[mask]
        dv = float(np.prod(block["spacing"]))
        volume += int(mask.sum()) * dv
        energy += float(speed2.sum()) * dv / 2
        if speed2.size:
            peak = max(peak, float(np.sqrt(speed2.max())))
    return {"volume": volume, "energy": energy, "peak_speed": peak}


def reconstruct(scalars, coverage, levels, axes):
    result = np.full(tuple(len(a) for a in axes), np.nan)
    for scalar, valid, level in zip(scalars, coverage, levels, strict=True):
        ids, coordinates, owners = [], [], []
        for d, axis in enumerate(axes):
            lo, dx, n = level["origin"][d], level["spacing"][d], level["shape"][d]
            take = np.flatnonzero((axis >= lo) & (axis < lo + dx * n))
            ids.append(take)
            relative = (axis[take] - lo) / dx
            coordinates.append(relative - 0.5)
            owners.append(np.minimum(np.floor(relative).astype(int), n - 1))
        if any(not len(i) for i in ids):
            continue
        target = np.ix_(*ids)
        use = valid[np.ix_(*owners)]
        sampled = interpolate_scalar(scalar, coordinates)
        result[target] = np.where(use, sampled, result[target])
    if not np.isfinite(result).all():
        raise ValueError("Native hierarchy does not cover all display points")
    return result


def sample_speed(fields, blocks, levels, axes, quantity="velocity"):
    if quantity not in ("velocity", "force"):
        raise ValueError("Unknown quantity")
    components = slice(0, 3) if quantity == "velocity" else slice(3, 6)
    scalars, coverage = [], []
    for level in levels:
        shape = tuple(level["shape"][:3])
        if np.prod(shape) > 256**3:
            raise ValueError("Native level bounding allocation exceeds render budget")
        scalar, valid = np.zeros(shape), np.zeros(shape, dtype=bool)
        for values, block in zip(fields, blocks, strict=True):
            if block["level"] != level["level"]:
                continue
            low = np.array(block["index_lo"]) - level["index_lo"]
            target = tuple(slice(int(i), int(i + n)) for i, n in zip(
                low, block["shape"][:3], strict=True
            ))
            if valid[target].any():
                raise ValueError("Overlapping native boxes")
            scalar[target] = np.linalg.norm(values[..., components], axis=-1)
            valid[target] = True
        if not valid.all():
            centers = [o + (np.arange(n) + 0.5) * d for o, n, d in zip(
                level["origin"], shape, level["spacing"], strict=True
            )]
            coarse = reconstruct(scalars, coverage, levels[:len(scalars)], centers)
            scalar[~valid] = coarse[~valid]
            del coarse
        scalars.append(scalar)
        coverage.append(valid)
    return reconstruct(scalars, coverage, levels, axes)
