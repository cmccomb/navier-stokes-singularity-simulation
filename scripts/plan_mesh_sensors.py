"""Choose reproducible discrepancy and prospective phase-support source sensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from navier_stokes_sim.config import SimulationConfig
from navier_stokes_sim.refinement_design import phase_geometry
from scripts.paper_run import write


def containing_level(point, blocks):
    levels = [
        b["level"]
        for b in blocks
        if np.all(np.array(point) >= b["origin"])
        and np.all(
            np.array(point)
            < np.array(b["origin"]) + np.array(b["spacing"]) * b["shape"][:3]
        )
    ]
    if not levels:
        raise ValueError("Sensor outside native mesh")
    return max(levels)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--table", type=Path, required=True)
    p.add_argument("--comparison", type=Path)
    p.add_argument("--discrepancies-only", action="store_true")
    p.add_argument("--times", type=float, nargs="+", default=[0.937, 0.985, 0.995])
    p.add_argument("--similarity-x", type=float, nargs="+", default=[0.62, 1.025, 1.43])
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    snapshot = json.loads(args.snapshot.read_text())
    cfg = SimulationConfig(
        **json.loads(args.table.with_suffix(args.table.suffix + ".json").read_text())[
            "config"
        ]
    )
    blocks = snapshot["frames"][-1]["export"]["blocks"]
    rows = []
    if args.discrepancies_only and not args.comparison:
        p.error("--discrepancies-only requires --comparison")
    if args.comparison:
        comparison = json.loads(args.comparison.read_text())
        if not comparison["complete"]:
            raise ValueError("Require completed native comparison")
        for frame in comparison["frames"]:
            for name in ("velocity", "force"):
                top = frame["largest_block_discrepancies"][name][0]
                level = containing_level(top["xyz"], blocks)
                rows.append(
                    {
                        "label": f"discrepancy-{frame['step']}-{name}",
                        "time": frame["time"],
                        "xyz": top["xyz"],
                        "dx": 2 / (snapshot["parameters"]["base_n"] * 2**level),
                        "candidate_level": level,
                        "selection": "Largest block maximum in conservatively restricted native comparison",
                    }
                )
    times = [] if args.discrepancies_only else args.times
    for t, similarity_x in ((t, x) for t in times for x in args.similarity_x):
        for eta in (0.0, 0.5, 0.8256):
            for theta in (0.0, np.pi / 4):
                xyz, wave = phase_geometry(
                    t, np.array(similarity_x), np.array(eta), np.array(theta), cfg
                )
                level = containing_level(xyz, blocks)
                dx = 2 / (snapshot["parameters"]["base_n"] * 2**level)
                rows.append(
                    {
                        "label": f"phase-{t}-{similarity_x}-{eta}-{theta:.4f}",
                        "time": t,
                        "xyz": xyz.tolist(),
                        "dx": dx,
                        "candidate_level": level,
                        "sampled_doubled_phase_points": float(np.pi / (dx * wave)),
                        "similarity_x": similarity_x,
                        "selection": "Analytic support sensor, not necessarily evolved maximum; center spacing only",
                    }
                )
    if args.output.exists():
        raise FileExistsError(args.output)
    write(args.output, rows)


if __name__ == "__main__":
    main()
