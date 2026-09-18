"""Design fixed revolved bands from a hash-verified native trajectory.

Activity and per-cell variation are design indicators, not truncation errors.
The full 3D solver, original core cubes, and periodic domain are retained.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.paper_run import sha, write


def runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(
        zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True)
    )


def selected_bins(frame, amplitude, variation):
    selected = np.zeros(frame["speed"].shape, dtype=bool)
    for name, gradient in (("speed", "velocity_gradient"), ("force", "force_gradient")):
        peak = frame[name].max()
        if peak:
            selected |= (frame[name] >= amplitude * peak) & (
                frame[gradient] * frame["native_spacing"] >= variation * peak
            )
    return selected


def bands_from_masks(masks, r, z, base_n, buffer_cells=2):
    """Expand each selected bin physically; add parent bands for proper nesting.

    AMReX still supplies its own proper-nesting and box-generation machinery.
    These bands express requested coverage, not an exact allocation prediction.
    """
    bands = []
    for parent, mask in enumerate(masks):
        dx = 2 / (base_n * 2**parent)
        for j in range(mask.shape[1]):
            for start, end in runs(mask[:, j]):
                for ancestor in range(parent + 1):
                    pad = buffer_cells * dx + sum(
                        2 * 2 / (base_n * 2**lev) for lev in range(ancestor, parent)
                    )
                    lo, hi = max(0.0, r[start] - pad), min(0.999, r[end] + pad)
                    bottom, top = max(-0.999, z[j] - pad), min(0.999, z[j + 1] + pad)
                    if hi > lo and top > bottom:
                        bands.append((ancestor, lo, hi, bottom, top))
    # Exact duplicates can arise from overlapping bins; preserve all other coverage.
    return sorted(set(bands))


def angular_cube_fraction(radius, half):
    fraction = np.ones_like(radius)
    outside = radius > half
    fraction[outside] = np.clip(1 - 4 / np.pi * np.arccos(half / radius[outside]), 0, 1)
    return fraction


def geometric_cells(bands, base_n, widths, n=1024):
    """Midpoint cylindrical quadrature, before box padding, ghosts, or scratch."""
    r = (np.arange(n) + 0.5) / n
    z = -1 + (np.arange(2 * n) + 0.5) / n
    result = [float(base_n**3)]
    for level, half in enumerate(widths, start=1):
        mask = np.zeros((n, 2 * n), dtype=bool)
        for parent, lo, hi, bottom, top in bands:
            if parent != level - 1:
                continue
            a, b = np.searchsorted(r, (lo, hi), side="left")
            c, d = np.searchsorted(z, (bottom, top), side="left")
            mask[a:b, c:d] = True
        cube = angular_cube_fraction(r, half)[:, None] * (np.abs(z) < half)
        extra_volume = np.sum(mask * (1 - cube) * (2 * np.pi * r[:, None])) / n**2
        # Integrate only the added region; preserve exact baseline cube volume.
        volume = (2 * half) ** 3 + extra_volume
        result.append(float(volume / (2 / (base_n * 2**level)) ** 3))
    return result


def design(args):
    audit = json.loads((args.audit / "manifest.json").read_text())
    source = json.loads(args.source.read_text())
    if (
        not audit["validated"]
        or not audit["complete_history"]
        or audit["source_record_sha256"] != sha(args.source)
        or len(audit["records"]) != len(source["native_frames"])
        or [r["time"] for r in audit["records"]]
        != [r["time"] for r in source["native_frames"]]
    ):
        raise ValueError("Complete matching native audit required")
    n = source["parameters"]["base_n"]
    widths = source["parameters"]["widths"]
    max_parent = (
        len(widths) - 1 if args.max_parent_level is None else args.max_parent_level
    )
    if not 0 <= max_parent < len(widths):
        raise ValueError("Parent level outside the existing hierarchy")
    r, z = np.array(audit["r_edges"]), np.array(audit["z_edges"])
    masks = [np.zeros((len(r) - 1, len(z) - 1), dtype=bool) for _ in widths]
    coverage = []
    for row in audit["records"]:
        path = args.audit / row["path"]
        if sha(path) != row["sha256"]:
            raise ValueError("Action map hash differs")
        with np.load(path) as frame:
            selected = selected_bins(frame, args.amplitude, args.variation)
            for parent in range(max_parent + 1):
                masks[parent] |= selected & np.isclose(
                    frame["native_spacing"], 2 / (n * 2**parent), rtol=1e-12, atol=0
                )
            coverage.append(
                {
                    "index": row["index"],
                    "time": row["time"],
                    **{
                        name + "_squared_fraction_in_selected_bins": float(
                            frame[name + "_squared_integral"][selected].sum()
                            / row["squared_integrals"][name]
                        )
                        if row["squared_integrals"][name] > 0
                        else None
                        for name in ("speed", "force")
                    },
                }
            )
    bands = bands_from_masks(masks, r, z, n, args.buffer_cells)
    if not bands or len(bands) > 100000:
        raise ValueError("No useful bands or excessive band count")
    args.output.mkdir(parents=True, exist_ok=False)
    band_path = args.output / "refinement.bands"
    band_path.write_text(
        "NS_RZ_BANDS_V1\n"
        + str(len(bands))
        + "\n"
        + "\n".join(
            f"{p} {a:.17g} {b:.17g} {c:.17g} {d:.17g}" for p, a, b, c, d in bands
        )
        + "\n"
    )
    np.savez_compressed(
        args.output / "selected.npz", **{f"parent_{p}": v for p, v in enumerate(masks)}
    )
    estimates = {str(q): geometric_cells(bands, n, widths, q) for q in (512, 1024)}
    report = {
        "schema_version": 1,
        "kind": "trajectory-informed-fixed-revolved-refinement-design",
        "source_record_sha256": sha(args.source),
        "audit_manifest_sha256": sha(args.audit / "manifest.json"),
        "designer_sha256": sha(Path(__file__)),
        "band_sha256": sha(band_path),
        "parameters": {
            "amplitude_fraction": args.amplitude,
            "variation_fraction": args.variation,
            "buffer_parent_cells": args.buffer_cells,
            "base_n": n,
            "preserved_core_widths": widths,
            "maximum_parent_level_to_extend": max_parent,
        },
        "band_count": len(bands),
        "geometric_stored_cells_by_quadrature": estimates,
        "estimated_stored_cells": sum(estimates["1024"]),
        "baseline_stored_cells": source["history"][-1]["stored_cells"],
        "coverage": coverage,
        "scope": "One additional level where resolved amplitude and per-cell variation exceed design thresholds; union over all saved times and maximum over azimuth. Core cubes retained. Full 3D equations retained. Physical buffers and ancestor coverage added.",
        "limits": [
            "Thresholds are design choices, not error tolerances or convergence claims.",
            "Histograms can mix source levels; their maximum cell spacing is conservative for selection but may not request another level for every finer cell in that bin.",
            "Observed activity cannot reveal structures missed by the source mesh; analytic phase and direct source-refinement checks remain required.",
            "Geometric estimates exclude AMReX box padding, ghosts, solver and forcing scratch; a measured capacity probe is mandatory.",
            "Buffers clip at r=0.999 and z=+-0.999; the periodic outer domain and original cubes are retained.",
            "The finite surrogate force depends on mesh spacing, so this changes both forcing discretization and evolved solution.",
            "This design covers only the audited interval; it does not authorize extrapolation past its endpoint.",
        ],
    }
    write(args.output / "design.json", report)
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "band_count",
                    "estimated_stored_cells",
                    "baseline_stored_cells",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--amplitude", type=float, default=0.25)
    parser.add_argument("--variation", type=float, default=0.1)
    parser.add_argument("--buffer-cells", type=int, default=2)
    parser.add_argument(
        "--max-parent-level",
        type=int,
        help="restrict the experiment to outer levels; existing core cubes stay unchanged",
    )
    args = parser.parse_args()
    if not (0 < args.amplitude <= 1 and 0 < args.variation and args.buffer_cells >= 2):
        parser.error(
            "Require amplitude in (0,1], positive variation, and at least two buffer cells"
        )
    design(args)
