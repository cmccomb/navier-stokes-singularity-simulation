"""Compare bounded fine-grid patches to the existing Python/PhiFlow stencils.

Full-domain coordinates are preserved. A padded local array supplies identical
derivatives in the interior; its artificial periodic edges are discarded.
No full 16,384-cubed array is allocated. This checks implementation agreement,
not continuum accuracy or an accepted production mesh.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from phi.flow import math

from navier_stokes_sim.config import SimulationConfig
from navier_stokes_sim.profile import target_velocity
from navier_stokes_sim.solver import _as_numpy, _manufactured_force, _to_field
from scripts.refinement_pilot import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--executable", type=Path, default=Path("build/amrex/ns_force_probe")
    )
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    table, executable = (
        args.table.resolve(strict=True),
        args.executable.resolve(strict=True),
    )
    cfg = SimulationConfig(
        **json.loads(table.with_suffix(table.suffix + ".json").read_text())["config"]
    )
    width = 12
    padding = cfg.pulse_correction_passes + 4
    report = {
        "schema_version": 1,
        "kind": "fine-local-patch-PhiFlow-force-crosscheck",
        "scope": __doc__,
        "table_sha256": hashlib.sha256(table.read_bytes()).hexdigest(),
        "binary_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "width": width,
        "padding": padding,
        "cases": [],
        "passed": False,
    }
    math.set_global_precision(64)
    for n in (1024, 16384):
        dx = 2 * cfg.half_domain / n
        local_n = width + 2 * padding
        # PhiFlow's box sets the derivative spacing only. The patched sampler
        # evaluates the unchanged profile at the original full-domain points.
        local = replace(cfg, resolution=local_n, half_domain=local_n * dx / 2)
        for t in (0.85, 0.985, 0.99983872):
            tau = cfg.t_star - t
            for index, (eta, theta) in enumerate(((0, 0), (0.5, 0.7), (0.8, 2.2))):
                q = tau / (1 - eta**2)
                radius = cfg.base_radius * np.sqrt(2 * q)
                z = cfg.base_height * eta * q ** (0.5 - cfg.h)
                center = np.array([radius * np.cos(theta), radius * np.sin(theta), z])
                start = (
                    np.floor((center + cfg.half_domain) / dx).astype(int) - width // 2
                )
                coords = tuple(
                    np.meshgrid(
                        *[
                            (
                                -cfg.half_domain
                                + (np.arange(s - padding, s + width + padding) + 0.5)
                                * dx
                            )
                            for s in start
                        ],
                        indexing="ij",
                    )
                )
                name = f"n{n}-t{t}-p{index}"
                path = args.output / (name + ".bin")
                process = subprocess.run(
                    [
                        str(executable),
                        f"table={table}",
                        f"output={path.resolve()}",
                        f"n_cell={n}",
                        f"patch_width={width}",
                        "patch_start=" + " ".join(str(s) for s in start),
                        f"time={t:.17g}",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                )
                path.with_suffix(".log").write_text(process.stdout + process.stderr)
                if process.returncode:
                    raise ValueError(process.stdout + process.stderr)
                actual = np.fromfile(path, dtype="<f8").reshape(
                    (width, width, width, 6)
                )
                with patch(
                    "navier_stokes_sim.profile.cell_centers", return_value=coords
                ):
                    velocity = target_velocity(t, local)
                    force = _as_numpy(
                        _manufactured_force(t, _to_field(velocity, local), local)
                    )
                core = (slice(padding, -padding),) * 3
                expected = np.concatenate((velocity[core], force[core]), axis=-1)
                errors = np.max(np.abs(actual - expected), axis=(0, 1, 2))
                scales = np.max(np.abs(expected), axis=(0, 1, 2))
                row = {
                    "n": n,
                    "dx": dx,
                    "time": t,
                    "eta": eta,
                    "theta": theta,
                    "start": start.tolist(),
                    "component_linf_errors": errors.tolist(),
                    "component_linf_scales": scales.tolist(),
                    "passed": bool(
                        np.isfinite(actual).all()
                        and np.all(errors[:3] < 1e-9 + 1e-9 * scales[:3])
                        and np.all(errors[3:] < 2e-6 + 2e-6 * scales[3:])
                    ),
                }
                report["cases"].append(row)
                print(json.dumps(row), flush=True)
                write_json(args.output / "summary.json", report)
    report["passed"] = all(row["passed"] for row in report["cases"])
    write_json(args.output / "summary.json", report)
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
