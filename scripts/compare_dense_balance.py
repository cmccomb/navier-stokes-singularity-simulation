"""Compare sampled vorticity-balance fields across fixed-force dense time branches.

Both branches share one from-rest checkpoint and 17 saved event times. This is
local timestep sensitivity of a sampled postprocessed balance, not a solver
residual, continuum error bound, or test of inherited pre-checkpoint error.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from scripts.compare_dense_temporal import compatible
from scripts.compare_mesh_snapshots import active_mask, values
from scripts.compare_mixed_native import load_snapshot
from scripts.export_mesh_snapshot import sha
from scripts.paper_run import write
from scripts.vorticity_balance import balance, eroded_active

TERMS = (
    "residual_five",
    "time_curl_five",
    "advection_curl",
    "viscous_curl",
    "force_curl",
)


def load_branch(paths: list[Path]) -> tuple[list[dict], list[dict], dict]:
    if len(paths) != 5:
        raise ValueError("require five snapshots per time branch")
    loaded = [load_snapshot(path) for path in paths]
    manifests = [manifest for manifest, _ in loaded]
    runs = [run for _, run in loaded]
    probe_hashes = {manifest.get("source_probe_sha256") for manifest in manifests}
    if len(probe_hashes) != 1 or None in probe_hashes:
        raise ValueError("branch mixes dense probe receipts")
    probe_path = paths[0] / "probe-snapshot.json"
    probe = json.loads(probe_path.read_text())
    if sha(probe_path) != next(iter(probe_hashes)):
        raise ValueError("dense probe receipt hash differs")
    times = [manifest["frames"][0]["export"]["time"] for manifest in manifests]
    blocks = manifests[0]["frames"][0]["export"]["blocks"]
    if not probe["centers"] or any(
        abs(time - (times[2] + (index - 2) * probe["spacing"])) > 1e-12
        for index, time in enumerate(times)
    ):
        raise ValueError("branch stencil spacing differs from probe")
    if (
        times != sorted(set(times))
        or abs(
            times[2]
            - probe["centers"][
                min(
                    range(len(probe["centers"])),
                    key=lambda i: abs(probe["centers"][i] - times[2]),
                )
            ]
        )
        > 1e-12
    ):
        raise ValueError("branch lacks one ordered center stencil")
    for manifest, run in zip(manifests, runs, strict=True):
        if (
            len(manifest["frames"]) != 1
            or manifest["frames"][0]["export"]["blocks"] != blocks
            or manifest["parameters"] != manifests[0]["parameters"]
            or manifest["profile_sha256"] != manifests[0]["profile_sha256"]
            or run["binary_sha256"] != runs[0]["binary_sha256"]
        ):
            raise ValueError("branch geometry or source changed")
    return manifests, runs, probe


def measure_pair(
    baseline_paths: list[Path],
    half_paths: list[Path],
    *,
    baseline_time_factor: float = 1,
    allow_equivalent_checkpoint_parent: bool = False,
) -> dict:
    baseline, bruns, bprobe = load_branch(baseline_paths)
    half, hruns, hprobe = load_branch(half_paths)
    control_audit = compatible(
        bprobe,
        hprobe,
        bruns[0],
        baseline_factor=baseline_time_factor,
        allow_equivalent_checkpoint_parent=allow_equivalent_checkpoint_parent,
    )
    times = [manifest["frames"][0]["export"]["time"] for manifest in baseline]
    half_times = [manifest["frames"][0]["export"]["time"] for manifest in half]
    if any(abs(a - b) > 1e-12 for a, b in zip(times, half_times, strict=True)):
        raise ValueError("branch saved times differ")
    blocks = baseline[0]["frames"][0]["export"]["blocks"]
    if blocks != half[0]["frames"][0]["export"]["blocks"]:
        raise ValueError("branch AMR geometry differs")
    if (
        baseline[0]["profile_sha256"] != half[0]["profile_sha256"]
        or baseline[0]["parameters"] != half[0]["parameters"]
        or bruns[0]["binary_sha256"] != hruns[0]["binary_sha256"]
    ):
        raise ValueError("branch force or physical source differs")
    viscosity = float(bruns[0]["profile_manifest"]["parameters"]["viscosity"])
    width = float(baseline[0]["parameters"]["widths"][-1])
    max_level = max(block["level"] for block in blocks)
    raw_a = [
        path / manifest["frames"][0]["path"]
        for path, manifest in zip(baseline_paths, baseline, strict=True)
    ]
    raw_b = [
        path / manifest["frames"][0]["path"]
        for path, manifest in zip(half_paths, half, strict=True)
    ]
    sums: dict[str, dict] = {}
    force_field = {
        "difference_squared_integral": 0.0,
        "reference_squared_integral": 0.0,
        "volume": 0.0,
        "max_component_difference": 0.0,
    }
    velocity_field = {
        "difference_squared_integral": 0.0,
        "reference_squared_integral": 0.0,
        "volume": 0.0,
        "max_component_difference": 0.0,
    }
    for block in blocks:
        h = float(block["spacing"][0])
        active = active_mask(block, blocks)
        force_a = values(raw_a[2], block)[..., 3:][active]
        force_b = values(raw_b[2], block)[..., 3:][active]
        delta = force_a - force_b
        dv = h**3
        force_field["difference_squared_integral"] += float(np.square(delta).sum()) * dv
        force_field["reference_squared_integral"] += (
            float(np.square(force_a).sum()) * dv
        )
        force_field["volume"] += int(active.sum()) * dv
        if delta.size:
            force_field["max_component_difference"] = max(
                force_field["max_component_difference"], float(np.abs(delta).max())
            )
        velocity_a = values(raw_a[2], block)[..., :3][active]
        velocity_b = values(raw_b[2], block)[..., :3][active]
        velocity_delta = velocity_a - velocity_b
        velocity_field["difference_squared_integral"] += (
            float(np.square(velocity_delta).sum()) * dv
        )
        velocity_field["reference_squared_integral"] += (
            float(np.square(velocity_b).sum()) * dv
        )
        velocity_field["volume"] += int(active.sum()) * dv
        if velocity_delta.size:
            velocity_field["max_component_difference"] = max(
                velocity_field["max_component_difference"],
                float(np.abs(velocity_delta).max()),
            )
        mask = eroded_active(active)
        if not mask.any():
            continue
        fields_a = [values(path, block) for path in raw_a]
        fields_b = [values(path, block) for path in raw_b]
        a = balance(fields_a, times, h, viscosity)
        b = balance(fields_b, half_times, h, viscosity)
        core = np.zeros(mask.shape, bool)
        if block["level"] == max_level:
            core = mask.copy()
            for axis in range(3):
                points = (
                    block["origin"][axis]
                    + (np.arange(2, block["shape"][axis] - 2) + 0.5) * h
                )
                shape = [1, 1, 1]
                shape[axis] = len(points)
                core &= (np.abs(points) <= width + 1e-14).reshape(shape)
        for name, selected in (("all_interior", mask), ("finest_core_interior", core)):
            if not selected.any():
                continue
            row = sums.setdefault(
                name,
                {
                    "cells": 0,
                    "volume": 0.0,
                    "squared": {
                        term: {"baseline": 0.0, "half": 0.0, "difference": 0.0}
                        for term in TERMS
                    },
                },
            )
            dv = h**3
            count = int(selected.sum())
            row["cells"] += count
            row["volume"] += count * dv
            for term in TERMS:
                left, right = a[term][selected], b[term][selected]
                row["squared"][term]["baseline"] += float(np.square(left).sum()) * dv
                row["squared"][term]["half"] += float(np.square(right).sum()) * dv
                row["squared"][term]["difference"] += (
                    float(np.square(left - right).sum()) * dv
                )
    for row in sums.values():
        row["rms"] = {
            term: {
                name: math.sqrt(value / row["volume"])
                for name, value in metrics.items()
            }
            for term, metrics in row.pop("squared").items()
        }
        denominator = sum(
            row["rms"][term]["baseline"] for term in TERMS if term != "residual_five"
        )
        row["relative_residual_difference_to_term_sum"] = (
            row["rms"]["residual_five"]["difference"] / denominator
            if denominator
            else None
        )
    if not sums.get("all_interior") or not sums.get("finest_core_interior"):
        raise ValueError("no common interior samples")
    for name, field in (("force", force_field), ("velocity", velocity_field)):
        if not math.isclose(field["volume"], 8, rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"{name} fields lack full active coverage")
        field["reference_rms"] = math.sqrt(
            field["reference_squared_integral"] / field["volume"]
        )
        field["difference_rms"] = math.sqrt(
            field["difference_squared_integral"] / field["volume"]
        )
        field["relative_l2_difference"] = (
            math.sqrt(
                field["difference_squared_integral"]
                / field["reference_squared_integral"]
            )
            if field["reference_squared_integral"]
            else None
        )
    if force_field["difference_rms"] > 1e-10 + 1e-8 * force_field["reference_rms"]:
        raise ValueError("fixed force fields differ materially between time branches")
    return {
        "schema_version": 1,
        "kind": "local-dense-vorticity-balance-time-sensitivity",
        "times": times,
        "center_time": times[2],
        "source_snapshot_sha256": [
            [sha(path / "manifest.json") for path in group]
            for group in (baseline_paths, half_paths)
        ],
        "source_probe_sha256": [
            sha(path / "probe-snapshot.json")
            for path in (baseline_paths[0], half_paths[0])
        ],
        "source_binary_sha256": bruns[0]["binary_sha256"],
        "source_profile_sha256": baseline[0]["profile_sha256"],
        "time_controls": control_audit,
        "velocity_field_sensitivity": velocity_field,
        "force_field_identity": force_field,
        "analyzer_sha256": sha(Path(__file__)),
        "regions": sums,
        "scope": __doc__.strip(),
        "validated": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-snapshot", type=Path, action="append", required=True
    )
    parser.add_argument("--half-snapshot", type=Path, action="append", required=True)
    parser.add_argument("--baseline-time-factor", type=float, default=1)
    parser.add_argument("--allow-equivalent-checkpoint-parent", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise ValueError("output already exists")
    result = measure_pair(
        [path.resolve(strict=True) for path in args.baseline_snapshot],
        [path.resolve(strict=True) for path in args.half_snapshot],
        baseline_time_factor=args.baseline_time_factor,
        allow_equivalent_checkpoint_parent=args.allow_equivalent_checkpoint_parent,
    )
    write(output, result)
    print(
        json.dumps(
            {
                "center_time": result["center_time"],
                "relative_core_difference": result["regions"]["finest_core_interior"][
                    "relative_residual_difference_to_term_sum"
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
