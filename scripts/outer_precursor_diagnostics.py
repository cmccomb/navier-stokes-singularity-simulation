"""Measure finite precursor diagnostics from selected native outer-band frames.

Every exported block is hash-checked against its source snapshot. Derivatives
use only active cells with six same-block neighbors; patch edges and AMR
interfaces are excluded. A running source trajectory remains unaudited beyond
the selected immutable frames.
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
from scripts.paper_precursor_diagnostics import interior_derivatives


def measure_raw(raw: Path, blocks: list[dict], expected: dict, core_width: float) -> dict:
    peak = energy = force_sq = volume = 0.0
    peak_level = None
    peak_in_core = False
    by_level: dict[int, dict] = {}
    for block in blocks:
        active = active_mask(block, blocks)
        if not active.any():
            continue
        field = values(raw, block)
        if not np.isfinite(field).all():
            raise ValueError("nonfinite native field")
        velocity = field[..., :3]
        speed_sq = np.square(velocity).sum(axis=-1)
        h = float(block["spacing"][0])
        dv = h**3
        selected_speed = speed_sq[active]
        local_peak = float(np.sqrt(selected_speed.max()))
        if local_peak > peak:
            peak, peak_level = local_peak, block["level"]
            index = np.unravel_index(np.argmax(np.where(active, speed_sq, -1)), active.shape)
            position = [block["origin"][axis] + (index[axis] + 0.5) * h for axis in range(3)]
            peak_in_core = max(abs(x) for x in position) <= core_width + 1e-14
        count = int(active.sum())
        volume += count * dv
        energy += float(selected_speed.sum()) * dv / 2
        force_sq += float(np.square(field[..., 3:][active]).sum()) * dv
        level = by_level.setdefault(block["level"], {"active_cells": 0, "derivative_cells": 0, "derivative_volume": 0.0, "vorticity_squared_integral": 0.0, "divergence_squared_integral": 0.0, "peak_vorticity": 0.0, "divergence_linf": 0.0})
        level["active_cells"] += count
        try:
            derivatives = interior_derivatives(velocity, active, block["spacing"])
        except ValueError as exc:
            if str(exc) != "no valid interior derivative cells":
                raise
            continue
        level["derivative_cells"] += derivatives["cells"]
        level["derivative_volume"] += derivatives["cells"] * dv
        level["vorticity_squared_integral"] += derivatives["vorticity_squared_sum"] * dv
        level["divergence_squared_integral"] += derivatives["divergence_squared_sum"] * dv
        level["peak_vorticity"] = max(level["peak_vorticity"], derivatives["peak_vorticity"])
        level["divergence_linf"] = max(level["divergence_linf"], derivatives["divergence_linf"])
    if not math.isclose(volume, 8, rel_tol=0, abs_tol=1e-10):
        raise ValueError("native active volume differs from full domain")
    if not math.isclose(peak, expected["peak_speed"], rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("native peak differs from solver diagnostic")
    if not math.isclose(energy, expected["energy"], rel_tol=1e-9, abs_tol=1e-11):
        raise ValueError("native energy differs from solver diagnostic")
    support_volume = core_volume = 0.0
    for block in blocks:
        active = active_mask(block, blocks)
        if not active.any():
            continue
        field = values(raw, block)
        speed_sq = np.square(field[..., :3]).sum(axis=-1)
        h = float(block["spacing"][0])
        selected = active & (speed_sq >= (peak / 2) ** 2)
        support_volume += int(selected.sum()) * h**3
        if block["level"] == max(by_level):
            coordinates = [block["origin"][axis] + (np.arange(block["shape"][axis]) + 0.5) * h for axis in range(3)]
            for axis, points in enumerate(coordinates):
                shape = [1, 1, 1]
                shape[axis] = len(points)
                selected &= (np.abs(points) <= core_width + 1e-14).reshape(shape)
            core_volume += int(selected.sum()) * h**3
    derivative_volume = sum(level["derivative_volume"] for level in by_level.values())
    if derivative_volume <= 0:
        raise ValueError("no valid derivative volume")
    return {
        "peak_speed": peak,
        "peak_level": peak_level,
        "peak_in_finest_core": bool(peak_level == max(by_level) and peak_in_core),
        "kinetic_energy": energy,
        "force_l2": math.sqrt(force_sq / volume),
        "half_peak_support_volume": support_volume,
        "half_peak_support_equivalent_radius": (3 * support_volume / (4 * math.pi)) ** (1 / 3) if support_volume else 0.0,
        "finest_core_half_peak_volume": core_volume,
        "finest_core_half_peak_equivalent_radius": (3 * core_volume / (4 * math.pi)) ** (1 / 3) if peak_level == max(by_level) and peak_in_core and core_volume else None,
        "derivative_interior_volume": derivative_volume,
        "peak_vorticity": max(level["peak_vorticity"] for level in by_level.values()),
        "vorticity_rms": math.sqrt(sum(level["vorticity_squared_integral"] for level in by_level.values()) / derivative_volume),
        "divergence_rms": math.sqrt(sum(level["divergence_squared_integral"] for level in by_level.values()) / derivative_volume),
        "divergence_linf": max(level["divergence_linf"] for level in by_level.values()),
        "levels": [{"level": index, **data} for index, data in sorted(by_level.items())],
    }


def measure(paths: list[Path]) -> dict:
    loaded = [load_snapshot(path) for path in paths]
    manifests, runs = zip(*loaded, strict=True)
    if not manifests or any(len(manifest["frames"]) != 1 for manifest in manifests):
        raise ValueError("require one native frame per snapshot")
    times = [manifest["frames"][0]["export"]["time"] for manifest in manifests]
    if times != sorted(set(times)):
        raise ValueError("snapshot times must increase")
    for manifest, run in loaded:
        if (run["binary_sha256"] != runs[0]["binary_sha256"]
                or run["adapter_sha256"] != runs[0]["adapter_sha256"]
                or manifest["profile_sha256"] != manifests[0]["profile_sha256"]
                or manifest["parameters"] != manifests[0]["parameters"]):
            raise ValueError("native source configuration changes across frames")
    core_width = float(manifests[0]["parameters"]["widths"][-1])
    frames = []
    for path, manifest in zip(paths, manifests, strict=True):
        frame = manifest["frames"][0]
        result = measure_raw(path / frame["path"], frame["export"]["blocks"], frame["diagnostic"], core_width)
        frames.append({"time": frame["export"]["time"], "step": frame["step"], "native_sha256": frame["sha256"], "target_l2_error": frame["diagnostic"]["l2_error"], "target_linf_error": frame["diagnostic"]["linf_error"], **result})
    return {
        "schema_version": 1,
        "kind": "outer-band-finite-precursor-diagnostics",
        "source_snapshot_sha256": [sha(path / "manifest.json") for path in paths],
        "source_run_state": [{"status": run["status"], "validated": run["validated"]} for run in runs],
        "source_binary_sha256": runs[0]["binary_sha256"],
        "source_profile_sha256": manifests[0]["profile_sha256"],
        "analyzer_sha256": sha(Path(__file__)),
        "core_half_width": core_width,
        "definitions": {"support_radius": "Volume-equivalent radius of all active cells with speed at least half the global peak; not a connected core radius.", "core_radius": "Volume-equivalent radius of the finest-level half-peak cells inside the prescribed central cube, reported only if the global peak occurs there.", "derivatives": "Centered same-block active-cell interior only; AMR interfaces, patch edges, and outer boundaries excluded."},
        "frames": frames,
        "scope": __doc__.strip(),
        "validated": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output already exists")
    result = measure([path.resolve(strict=True) for path in args.snapshot])
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"frames": len(result["frames"]), "last_time": result["frames"][-1]["time"], "validated": True}))


if __name__ == "__main__":
    main()
