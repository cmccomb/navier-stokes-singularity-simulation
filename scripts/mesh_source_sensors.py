"""Bounded h/h2/h4 source probes with directional variation indicators.

Directions are derivatives of Cartesian vectors along local r, theta, and z,
not an anisotropic refinement experiment or a continuum-error estimate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from navier_stokes_sim.config import SimulationConfig
from scripts.export_mesh_snapshot import bounded, sha
from scripts.force_resolution_scan import average
from scripts.paper_run import write


def vector_rms(values):
    return float(np.sqrt(np.mean(np.sum(values**2, axis=-1))))


def directional_variation(values, origin, dx, coarse_dx):
    """Interior-only centered differences avoid one-sided patch boundary bias."""
    axes = [
        o + (np.arange(n) + 0.5) * dx
        for o, n in zip(origin, values.shape[:3], strict=True)
    ]
    x, y, _ = np.meshgrid(*axes, indexing="ij")
    radius = np.hypot(x, y)
    cos = np.divide(x, radius, out=np.zeros_like(x), where=radius > 0)[1:-1, 1:-1, 1:-1]
    sin = np.divide(y, radius, out=np.zeros_like(y), where=radius > 0)[1:-1, 1:-1, 1:-1]
    interior = (slice(1, -1),) * 3
    result = {}
    for name, sl in (("velocity", slice(0, 3)), ("force", slice(3, 6))):
        gx, gy, gz = [
            np.gradient(values[..., sl], dx, axis=d, edge_order=2)[interior]
            for d in range(3)
        ]
        normalizer = vector_rms(values[..., sl][interior])
        terms = {
            "radial": cos[..., None] * gx + sin[..., None] * gy,
            "azimuthal": -sin[..., None] * gx + cos[..., None] * gy,
            "axial": gz,
        }
        result[name] = {
            key: {
                "gradient_rms": vector_rms(v),
                "variation_per_coarse_cell_over_local_rms": coarse_dx
                * vector_rms(v)
                / normalizer
                if normalizer
                else None,
            }
            for key, v in terms.items()
        }
    return result


def metrics(coarse, half, quarter, epsilon_half):
    result = {}
    for name, sl in (("velocity", slice(0, 3)), ("force", slice(3, 6))):
        ref = vector_rms(quarter[..., sl])
        first, second = (
            vector_rms(coarse[..., sl] - half[..., sl]),
            vector_rms(half[..., sl] - quarter[..., sl]),
        )
        result[name] = {
            "reference_rms": ref,
            "h_to_half_rms": first,
            "half_to_quarter_rms": second,
            "h_to_half_relative": first / ref if ref else None,
            "half_to_quarter_relative": second / ref if ref else None,
            "difference_ratio": first / second if second else None,
            "epsilon_half_relative": vector_rms(
                quarter[..., sl] - epsilon_half[..., sl]
            )
            / ref
            if ref
            else None,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("executable", "table", "requests", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    executable, table = args.executable.resolve(), args.table.resolve()
    manifest = json.loads(table.with_suffix(table.suffix + ".json").read_text())
    if sha(table) != manifest["sha256"]:
        raise ValueError("Profile hash differs")
    cfg = SimulationConfig(**manifest["config"])
    cfg.validate()
    requests = json.loads(args.requests.read_text())
    if not 0 < len(requests) <= 64:
        raise ValueError("Require 1..64 bounded sensors")
    output = args.output.resolve()
    output.mkdir(exist_ok=False)
    report = {
        "schema_version": 1,
        "complete": False,
        "table_sha256": sha(table),
        "probe_sha256": sha(executable),
        "requests_sha256": sha(args.requests),
        "script_sha256": sha(__file__),
        "sensors": [],
        "scope": __doc__,
    }
    for index, request in enumerate(requests):
        t, dx, center = request["time"], request["dx"], np.array(request["xyz"])
        if not 0 <= t < cfg.t_star or not np.isfinite(center).all():
            raise ValueError("Invalid sensor coordinates/time")
        n = round(2 * cfg.half_domain / dx)
        if not np.isclose(n * dx, 2 * cfg.half_domain) or not 8 <= n <= 8192:
            raise ValueError("Invalid sensor grid")
        start = np.floor((center + cfg.half_domain) / dx).astype(int) - 2
        if np.any(start < 0) or np.any(start + 4 > n):
            raise ValueError("Sensor would cross periodic domain boundary")
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
        fields, hashes, usages = [], {}, []
        for label, ratio, window in (
            ("h", 1, 0),
            ("half", 2, 0),
            ("quarter", 4, 0),
            ("epsilon_half", 4, epsilon / (2 * (cfg.t_star - t))),
        ):
            raw = output / f"sensor-{index:02d}-{label}.bin"
            command = [
                str(executable),
                f"table={table}",
                f"output={raw}",
                f"n_cell={n * ratio}",
                f"patch_width={4 * ratio}",
                "patch_start=" + " ".join(str(v * ratio) for v in start),
                f"time={t:.17g}",
                f"epsilon_tau_ratio={window:.17g}",
            ]
            usages.append(
                bounded(
                    command, output, raw.with_suffix(".log"), seconds=120, rss_mib=512
                )
            )
            field = np.fromfile(raw, dtype="<f8").reshape(
                4 * ratio, 4 * ratio, 4 * ratio, 6
            )
            if not np.isfinite(field).all():
                raise ValueError("Nonfinite source")
            hashes[raw.name] = sha(raw)
            fields.append(average(field, ratio))
            if label == "quarter":
                directions = directional_variation(
                    field, -cfg.half_domain + start * dx, dx / 4, dx
                )
        report["sensors"].append(
            {
                "request": request,
                "patch_start": start.tolist(),
                "metrics": metrics(*fields),
                "directions": directions,
                "raw_sha256": hashes,
                "resource_usage": usages,
            }
        )
        write(output / "progress.json", report)
        print(
            json.dumps(
                {
                    "index": index,
                    "label": request["label"],
                    "force": report["sensors"][-1]["metrics"]["force"],
                }
            ),
            flush=True,
        )
    report["complete"] = True
    write(output / "sensors.json", report)


if __name__ == "__main__":
    main()
