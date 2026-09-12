"""Cross-check the C++ potential evaluator against the unchanged Python profile."""

from __future__ import annotations

import argparse
import io
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from unittest.mock import PropertyMock, patch

import numpy as np

from navier_stokes_sim.config import SimulationConfig
from navier_stokes_sim.profile import _paper_vector_potentials
from scripts.refinement_pilot import write_json


def compare(executable: Path, table: Path, cfg: SimulationConfig) -> dict:
    rng = np.random.default_rng(9162026)
    cfg = replace(cfg, resolution=8)
    cases = []
    for t in [0, 0.55, 0.56, 0.62, 0.775, 0.82, 0.9014, 0.94, 0.985, 0.99983872]:
        for dx in [2 / 32, 2 / 192, 2 / 1024, 2 / 16384]:
            xyz = rng.uniform(-1, 1, (8, 8, 8, 3))
            scale = cfg.base_radius * np.sqrt(cfg.t_star - t)
            xyz[:4] *= 3 * scale
            xyz[0, 0, 0] = [0, 0, 0]
            coords = tuple(xyz[..., c] for c in range(3))
            # Only the sampling geometry and spacing are replaced. The original
            # profile algebra, tables and potential construction are unchanged.
            with (
                patch("navier_stokes_sim.profile.cell_centers", return_value=coords),
                patch.object(
                    SimulationConfig, "dx", new_callable=PropertyMock, return_value=dx
                ),
            ):
                background, primary, _ = _paper_vector_potentials(t, cfg)
            expected = np.stack((*background, *primary), axis=-1).reshape(-1, 6)
            points = np.column_stack(
                (np.full(512, t), np.full(512, dx), xyz.reshape(-1, 3))
            )
            stream = io.StringIO()
            np.savetxt(stream, points, fmt="%.17g")
            result = subprocess.run(
                [str(executable), str(table)],
                input=stream.getvalue(),
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
            if result.returncode:
                raise ValueError(result.stderr)
            actual = np.loadtxt(io.StringIO(result.stdout))
            if actual.shape != expected.shape or not np.isfinite(actual).all():
                raise ValueError("invalid C++ potential output")
            error = float(np.max(np.abs(actual - expected)))
            scale = max(float(np.max(np.abs(expected))), 1e-12)
            cases.append(
                {
                    "time": t,
                    "dx": dx,
                    "absolute_linf_error": error,
                    "relative_linf_error": error / scale,
                    "passed": error <= 1e-12 + 2e-11 * scale,
                }
            )
    return {
        "schema_version": 1,
        "kind": "raw-potential-language-crosscheck",
        "points_per_case": 512,
        "cases": cases,
        "passed": all(r["passed"] for r in cases),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--executable", type=Path, default=Path("build/amrex/ns_profile_probe")
    )
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(
        args.table.with_suffix(args.table.suffix + ".json").read_text()
    )["config"]
    result = compare(
        args.executable.resolve(), args.table.resolve(), SimulationConfig(**config)
    )
    write_json(args.output, result)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 2)


if __name__ == "__main__":
    main()
