"""Cross-check C++ discrete velocity and force against the PhiFlow implementation."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path

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
    table = args.table.resolve(strict=True)
    cfg = SimulationConfig(
        **json.loads(table.with_suffix(table.suffix + ".json").read_text())["config"]
    )
    executable = args.executable.resolve(strict=True)
    report = {
        "schema_version": 1,
        "kind": "full-vector-velocity-and-PhiFlow-force-crosscheck",
        "profile_revision": cfg.paper_profile_revision,
        "table_sha256": hashlib.sha256(table.read_bytes()).hexdigest(),
        "binary_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "cases": [],
        "passed": False,
    }
    math.set_global_precision(64)
    for n in (16, 32):
        config = replace(cfg, resolution=n)
        for t in (0, 0.55, 0.56, 0.62, 0.775, 0.85, 0.985):
            output = args.output / f"n{n}-t{t}.bin"
            run = subprocess.run(
                [
                    str(executable),
                    f"table={table}",
                    f"output={output.resolve()}",
                    f"n_cell={n}",
                    f"time={t:.17g}",
                ],
                text=True,
                capture_output=True,
                timeout=120,
                check=False,
            )
            output.with_suffix(".log").write_text(run.stdout + run.stderr)
            if run.returncode:
                raise ValueError(run.stdout + run.stderr)
            actual = np.fromfile(output, dtype="<f8").reshape((n, n, n, 6))
            u = target_velocity(t, config)
            f = _as_numpy(_manufactured_force(t, _to_field(u, config), config))
            ue = float(np.max(np.abs(actual[..., :3] - u)))
            fe = float(np.max(np.abs(actual[..., 3:] - f)))
            us = float(np.max(np.abs(u)))
            fs = float(np.max(np.abs(f)))
            row = {
                "n": n,
                "time": t,
                "velocity_linf_error": ue,
                "force_linf_error": fe,
                "velocity_linf": us,
                "force_linf": fs,
                "passed": bool(
                    np.isfinite(actual).all()
                    and ue < 1e-10 + 1e-10 * us
                    and fe < 1e-7 + 2e-7 * fs
                ),
            }
            report["cases"].append(row)
            print(row, flush=True)
            write_json(args.output / "summary.json", report)
    report["passed"] = all(r["passed"] for r in report["cases"])
    write_json(args.output / "summary.json", report)
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
