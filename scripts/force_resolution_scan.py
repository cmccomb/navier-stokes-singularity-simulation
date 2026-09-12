"""Sample local spatial and derivative-window sensitivity of the actual source.

These overlapping patches are sensors, not a global norm, an evolved solution,
or a mesh certificate. All six components are sampled on aligned dx, dx/2, and
dx/4 patches; finer cell values are volume-averaged to the coarse patch. Tiny
pulse envelopes are not discarded: background force is included throughout.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from navier_stokes_sim.config import SimulationConfig
from navier_stokes_sim.refinement_design import phase_geometry
from scripts.paper_run import sha, thread_environment, write


def average(field: np.ndarray, ratio: int) -> np.ndarray:
    n = field.shape[0] // ratio
    return field.reshape(n, ratio, n, ratio, n, ratio, 6).mean(axis=(1, 3, 5))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--executable", type=Path, default=Path("build/amrex-omp/ns_force_probe")
    )
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--base-n", type=int, required=True)
    parser.add_argument("--widths", type=float, nargs="*", default=[])
    parser.add_argument(
        "--times", type=float, nargs="*", default=[0.6, 0.775, 0.85, 0.95, 0.985, 0.998]
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    executable, table = (
        args.executable.resolve(strict=True),
        args.table.resolve(strict=True),
    )
    manifest = json.loads(table.with_suffix(table.suffix + ".json").read_text())
    if sha(table) != manifest["sha256"]:
        raise ValueError("table hash mismatch")
    cfg = SimulationConfig(**manifest["config"])
    cfg.validate()
    if (
        args.base_n < 16
        or args.base_n % 8
        or args.base_n * 2 ** (len(args.widths) + 2) > 32768
    ):
        raise ValueError("unsupported bounded sensor grid")
    if any(
        not 0 < w < (args.widths[i - 1] if i else cfg.half_domain)
        for i, w in enumerate(args.widths)
    ):
        raise ValueError("widths must be positive and decreasing")
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output.resolve()
    report = {
        "schema_version": 1,
        "scope": __doc__,
        "binary_sha256": sha(executable),
        "table_sha256": sha(table),
        "parameters": {
            "base_n": args.base_n,
            "widths": args.widths,
            "times": args.times,
        },
        "sensors": [],
        "completed": False,
    }
    environment = {**os.environ, **thread_environment(1, 1)}
    width = 4

    def probe(
        n: int, start: np.ndarray, size: int, t: float, name: str, ratio: float = 0
    ) -> np.ndarray:
        path = output / (name + ".bin")
        command = [
            str(executable),
            f"table={table}",
            f"output={path}",
            f"n_cell={n}",
            f"patch_width={size}",
            "patch_start=" + " ".join(str(int(i)) for i in start),
            f"time={t:.17g}",
            f"epsilon_tau_ratio={ratio:.17g}",
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env=environment,
            timeout=120,
            check=False,
        )
        (output / (name + ".log")).write_text(result.stdout + result.stderr)
        if result.returncode:
            raise ValueError(f"source sensor failed: {name}")
        field = np.fromfile(path, dtype="<f8").reshape(size, size, size, 6)
        if not np.isfinite(field).all():
            raise ValueError("nonfinite source sensor")
        return field

    for t in args.times:
        for X in (0.2, 0.62, 1.025, 1.43):
            for eta in (0, 0.5, 0.8256):
                for angle in (0, np.pi / 4):
                    xyz, wave = phase_geometry(
                        t, np.array(X), np.array(eta), np.array(angle), cfg
                    )
                    if (
                        np.hypot(*xyz[:2]) >= cfg.localization_outer
                        or abs(xyz[2]) >= cfg.localization_outer
                    ):
                        continue
                    level = 0
                    for lev, w in enumerate(args.widths, start=1):
                        fine_dx = 2 * cfg.half_domain / (args.base_n * 2**lev)
                        if max(abs(xyz)) + 2 * fine_dx < w:
                            level = lev
                    n = args.base_n * 2**level
                    dx = 2 * cfg.half_domain / n
                    start = (
                        np.floor((xyz + cfg.half_domain) / dx).astype(int) - width // 2
                    )
                    if np.any(start < 0) or np.any(start + width > n):
                        continue
                    name = f"sensor-{len(report['sensors']):03d}"
                    fields = [
                        average(
                            probe(n * r, start * r, width * r, t, name + f"-r{r}"), r
                        )
                        for r in (1, 2, 4)
                    ]
                    epsilon = min(
                        cfg.derivative_epsilon,
                        0.2 * (cfg.t_star - t),
                        0.1
                        * -(cfg.t_star - t)
                        * np.expm1(
                            -cfg.forcing_phase_step
                            / manifest["parameters"]["forcing_log_rate_bound"]
                        ),
                    )
                    smaller = average(
                        probe(
                            n * 4,
                            start * 4,
                            width * 4,
                            t,
                            name + "-epsilon-half",
                            epsilon / 2 / (cfg.t_star - t),
                        ),
                        4,
                    )

                    def rms(a: np.ndarray) -> list[float]:
                        return [
                            float(np.sqrt(np.mean(np.sum(a[..., sl] ** 2, axis=-1))))
                            for sl in (slice(0, 3), slice(3, 6))
                        ]

                    row = {
                        "time": t,
                        "X": X,
                        "eta": eta,
                        "theta": float(angle),
                        "xyz": xyz.tolist(),
                        "selected_level": level,
                        "effective_n": n,
                        "patch_start": start.tolist(),
                        "dx": dx,
                        "doubled_phase_points": float(np.pi / (dx * wave)),
                        "velocity_force_rms_reference": rms(fields[2]),
                        "velocity_force_rms_delta_h_half": rms(fields[0] - fields[1]),
                        "velocity_force_rms_delta_half_quarter": rms(
                            fields[1] - fields[2]
                        ),
                        "velocity_force_rms_epsilon_half_delta": rms(
                            fields[2] - smaller
                        ),
                        "epsilon": float(epsilon),
                    }
                    report["sensors"].append(row)
                    write(output / "summary.json", report)
        print(json.dumps({"time": t, "sensors": len(report["sensors"])}), flush=True)
    report["completed"] = True
    write(output / "summary.json", report)


if __name__ == "__main__":
    main()
