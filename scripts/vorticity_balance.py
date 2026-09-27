"""Sample a pressure-free vorticity balance from five native AMR snapshots.

The sampled cells and their two-cell same-block neighborhoods must all be
active. The diagnostic excludes patch edges, AMR interfaces, and boundaries.
It is a finite-difference check of saved fields, not the solver's discrete
momentum residual or a continuum error estimate.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from scripts.compare_mesh_snapshots import active_mask, values
from scripts.compare_mixed_native import load_snapshot
from scripts.export_mesh_snapshot import sha


def derivative_weights(times: list[float], center: int) -> np.ndarray:
    """Polynomial first-derivative weights at one of the sample times."""
    offsets = np.asarray(times, float) - times[center]
    if len(set(times)) != len(times) or not 0 <= center < len(times):
        raise ValueError("time samples must be distinct")
    scale = float(np.max(np.abs(offsets)))
    if not scale:
        raise ValueError("time samples have no span")
    normalized = offsets / scale
    matrix = np.vstack([normalized**order for order in range(len(times))])
    rhs = np.zeros(len(times))
    rhs[1] = 1
    return np.linalg.solve(matrix, rhs) / scale


def curl(field: np.ndarray, h: float) -> np.ndarray:
    """Centered curl, defined only one cell inside each block face."""
    center = (slice(1, -1),) * 3

    def d(component: int, axis: int) -> np.ndarray:
        plus, minus = list(center), list(center)
        plus[axis], minus[axis] = slice(2, None), slice(None, -2)
        return (field[(*plus, component)] - field[(*minus, component)]) / (2 * h)

    return np.stack((d(2, 1) - d(1, 2), d(0, 2) - d(2, 0), d(1, 0) - d(0, 1)), axis=-1)


def laplacian(field: np.ndarray, h: float) -> np.ndarray:
    center = (slice(1, -1),) * 3
    result = np.zeros_like(field[center])
    for axis in range(3):
        plus, minus = list(center), list(center)
        plus[axis], minus[axis] = slice(2, None), slice(None, -2)
        result += (field[tuple(plus)] - 2 * field[center] + field[tuple(minus)]) / h**2
    return result


def advection(velocity: np.ndarray, h: float) -> np.ndarray:
    center = (slice(1, -1),) * 3
    result = np.zeros_like(velocity[center])
    for axis in range(3):
        plus, minus = list(center), list(center)
        plus[axis], minus[axis] = slice(2, None), slice(None, -2)
        result += velocity[(*center, axis, None)] * (
            velocity[tuple(plus)] - velocity[tuple(minus)]
        ) / (2 * h)
    return result


def balance(fields: list[np.ndarray], times: list[float], h: float, viscosity: float) -> dict[str, np.ndarray]:
    """Return five/three-point curl-momentum residuals on block interior."""
    if len(fields) != 5 or len(times) != 5 or times != sorted(times) or h <= 0 or viscosity < 0:
        raise ValueError("require five ordered fields and valid physical scales")
    shape = fields[0].shape
    if len(shape) != 4 or min(shape[:3]) < 5 or shape[-1] != 6 or any(f.shape != shape for f in fields):
        raise ValueError("incompatible six-component native blocks")
    velocity = fields[2][..., :3]
    force = fields[2][..., 3:]
    omega = curl(velocity, h)
    advective_curl = curl(advection(velocity, h), h)
    viscous_curl = viscosity * laplacian(omega, h)
    force_curl = curl(force, h)[1:-1, 1:-1, 1:-1]

    def time_curl(indices: list[int], weights: np.ndarray) -> np.ndarray:
        change = np.zeros_like(velocity)
        for index, weight in zip(indices, weights, strict=True):
            change += weight * fields[index][..., :3]
        return curl(change, h)[1:-1, 1:-1, 1:-1]

    five = time_curl(list(range(5)), derivative_weights(times, 2))
    three = time_curl([1, 2, 3], derivative_weights(times[1:4], 1))
    return {
        "time_curl_five": five,
        "time_curl_three": three,
        "advection_curl": advective_curl,
        "viscous_curl": viscous_curl,
        "force_curl": force_curl,
        "residual_five": five + advective_curl - viscous_curl - force_curl,
        "residual_three": three + advective_curl - viscous_curl - force_curl,
        "time_stencil_difference": five - three,
    }


def eroded_active(active: np.ndarray) -> np.ndarray:
    """Retain cells with an active two-step Manhattan stencil in one block."""
    valid = active.copy()
    for _ in range(2):
        padded = np.pad(valid, 1, constant_values=False)
        next_valid = padded[1:-1, 1:-1, 1:-1].copy()
        for axis in range(3):
            plus, minus = [slice(1, -1)] * 3, [slice(1, -1)] * 3
            plus[axis], minus[axis] = slice(2, None), slice(None, -2)
            next_valid &= padded[tuple(plus)] & padded[tuple(minus)]
        valid = next_valid
    return valid[2:-2, 2:-2, 2:-2]


def measure(paths: list[Path]) -> dict:
    if len(paths) != 5:
        raise ValueError("require exactly five native snapshots")
    loaded = [load_snapshot(path) for path in paths]
    manifests, runs = zip(*loaded, strict=True)
    frames = [manifest["frames"][0] for manifest in manifests]
    times = [frame["export"]["time"] for frame in frames]
    if times != sorted(set(times)) or any(len(manifest["frames"]) != 1 for manifest in manifests):
        raise ValueError("require five increasing one-frame snapshots")
    blocks = frames[0]["export"]["blocks"]
    for manifest, run, frame in zip(manifests, runs, frames, strict=True):
        if (run["binary_sha256"] != runs[0]["binary_sha256"]
                or run["adapter_sha256"] != runs[0]["adapter_sha256"]
                or manifest["profile_sha256"] != manifests[0]["profile_sha256"]
                or manifest["parameters"] != manifests[0]["parameters"]
                or frame["export"]["blocks"] != blocks):
            raise ValueError("snapshot source or AMR geometry changed across times")
    viscosity = float(runs[0]["profile_manifest"]["parameters"]["viscosity"])
    core_width = float(manifests[0]["parameters"]["widths"][-1])
    max_level = max(block["level"] for block in blocks)
    totals: dict[str, dict] = {}
    names = ("residual_five", "residual_three", "time_stencil_difference", "time_curl_five", "advection_curl", "viscous_curl", "force_curl")
    raw = [path / frame["path"] for path, frame in zip(paths, frames, strict=True)]
    for block in blocks:
        h = float(block["spacing"][0])
        active = active_mask(block, blocks)
        mask = eroded_active(active)
        if not mask.any():
            continue
        fields = [values(path, block) for path in raw]
        terms = balance(fields, times, h, viscosity)
        core = np.zeros(mask.shape, bool)
        if block["level"] == max_level:
            coordinates = [
                block["origin"][axis] + (np.arange(2, block["shape"][axis] - 2) + 0.5) * h
                for axis in range(3)
            ]
            core = mask.copy()
            for axis, points in enumerate(coordinates):
                shape = [1, 1, 1]
                shape[axis] = len(points)
                core &= (np.abs(points) <= core_width + 1e-14).reshape(shape)
        for region, selected in (("all_interior", mask), ("finest_core_interior", core)):
            if not selected.any():
                continue
            row = totals.setdefault(region, {"cells": 0, "volume": 0.0, "squared_integrals": {name: 0.0 for name in names}})
            dv = h**3
            row["cells"] += int(selected.sum())
            row["volume"] += int(selected.sum()) * dv
            for name in names:
                row["squared_integrals"][name] += float(np.square(terms[name][selected]).sum()) * dv
    for row in totals.values():
        row["rms"] = {name: math.sqrt(value / row["volume"]) for name, value in row.pop("squared_integrals").items()}
        denominator = sum(row["rms"][name] for name in ("time_curl_five", "advection_curl", "viscous_curl", "force_curl"))
        row["relative_residual_five"] = row["rms"]["residual_five"] / denominator if denominator else None
    if not totals.get("all_interior") or not totals.get("finest_core_interior"):
        raise ValueError("no valid interior samples in the domain or finest core")
    return {
        "schema_version": 1,
        "kind": "sampled-native-vorticity-balance",
        "source_snapshot_sha256": [sha(path / "manifest.json") for path in paths],
        "source_run_state": [{"status": run["status"], "validated": run["validated"]} for run in runs],
        "source_binary_sha256": runs[0]["binary_sha256"],
        "source_profile_sha256": manifests[0]["profile_sha256"],
        "analyzer_sha256": sha(Path(__file__)),
        "times": times,
        "center_time": times[2],
        "viscosity": viscosity,
        "core_half_width": core_width,
        "regions": totals,
        "scope": __doc__.strip(),
        "validated": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = measure([path.resolve(strict=True) for path in args.snapshot])
    if args.output.exists():
        raise ValueError("output already exists")
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"time": result["center_time"], "regions": {name: region["relative_residual_five"] for name, region in result["regions"].items()}}))


if __name__ == "__main__":
    main()
