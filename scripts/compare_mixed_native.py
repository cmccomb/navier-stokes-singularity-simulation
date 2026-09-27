"""Conservatively compare native AMR fields when refinement footprints differ.

Every active fine cell contributes its physical volume to its containing active
coarse cell. A coarse cell can therefore contain a mixture of fine AMR levels.
The comparison rejects gaps, overlaps, and source cells coarser than the target.
This is a resolution/mesh-layout sensitivity test, not a continuum error bound.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from scripts.compare_mesh_snapshots import active_mask, validate_blocks, values
from scripts.export_mesh_snapshot import sha
from scripts.paper_run import write
from scripts.revolved_mesh_pilot import block_diagnostics


def restrict_composite(
    raw: Path,
    source: list[dict],
    target: dict,
    required: np.ndarray,
    source_masks: list[np.ndarray] | None = None,
) -> np.ndarray:
    """Volume-average all active source cells into required target cells."""
    shape = tuple(target["shape"][:3])
    if required.shape != shape:
        raise ValueError("target mask shape differs")
    target_lo = np.asarray(target["index_lo"])
    target_dx = float(target["spacing"][0])
    count = int(np.prod(shape))
    coverage = np.zeros(count)
    sums = np.zeros((count, 6))
    if source_masks is None:
        source_masks = [active_mask(block, source) for block in source]
    for block, source_mask in zip(source, source_masks, strict=True):
        source_dx = float(block["spacing"][0])
        ratio_float = target_dx / source_dx
        ratio = round(ratio_float)
        # If an active source cell is coarser than a required target cell,
        # that target cell will remain uncovered and fail the final gate.
        if ratio_float < 1:
            continue
        if not math.isclose(ratio_float, ratio, rel_tol=0, abs_tol=1e-12):
            raise ValueError("source and target cell sizes are not integer aligned")
        source_lo = np.asarray(block["index_lo"])
        source_shape = np.asarray(block["shape"][:3])
        lower = np.maximum(0, target_lo * ratio - source_lo)
        upper = np.minimum(source_shape, (target_lo + shape) * ratio - source_lo)
        if np.any(upper <= lower):
            continue
        selection = tuple(slice(int(a), int(b)) for a, b in zip(lower, upper, strict=True))
        local_active = source_mask[selection]
        if not local_active.any():
            continue
        positions = np.nonzero(local_active)
        mapped = [
            (source_lo[d] + lower[d] + positions[d]) // ratio - target_lo[d]
            for d in range(3)
        ]
        flat = np.ravel_multi_index(mapped, shape)
        keep = required.ravel()[flat]
        if not keep.any():
            continue
        flat = flat[keep]
        weight = (source_dx / target_dx) ** 3
        coverage += np.bincount(flat, weights=np.full(flat.size, weight), minlength=count)
        selected = values(raw, block)[selection][local_active][keep]
        for component in range(6):
            sums[:, component] += np.bincount(
                flat, weights=selected[:, component] * weight, minlength=count
            )
    if not np.allclose(coverage[required.ravel()], 1, rtol=0, atol=1e-12):
        low = float(coverage[required.ravel()].min())
        high = float(coverage[required.ravel()].max())
        raise ValueError(f"source coverage differs from one: {low}, {high}")
    return sums.reshape((*shape, 6))


def compare(coarse_raw: Path, coarse: list[dict], fine_raw: Path, fine: list[dict]) -> dict:
    source_masks = [active_mask(block, fine) for block in fine]
    totals = {
        name: {"difference_squared_integral": 0.0, "reference_squared_integral": 0.0, "active_max_component_difference": 0.0}
        for name in ("velocity", "force")
    }
    volume = 0.0
    for block in coarse:
        active = active_mask(block, coarse)
        if not active.any():
            continue
        reference = restrict_composite(fine_raw, fine, block, active, source_masks)
        actual = values(coarse_raw, block)
        dv = float(np.prod(block["spacing"]))
        volume += int(active.sum()) * dv
        for name, sl in (("velocity", slice(0, 3)), ("force", slice(3, 6))):
            delta = actual[..., sl][active] - reference[..., sl][active]
            basis = reference[..., sl][active]
            item = totals[name]
            item["difference_squared_integral"] += float(np.square(delta).sum()) * dv
            item["reference_squared_integral"] += float(np.square(basis).sum()) * dv
            item["active_max_component_difference"] = max(
                item["active_max_component_difference"], float(np.abs(delta).max())
            )
    if not math.isclose(volume, 8, rel_tol=0, abs_tol=1e-12):
        raise ValueError("coarse active cells do not cover the domain")
    for item in totals.values():
        item["rms_difference"] = math.sqrt(item["difference_squared_integral"] / volume)
        item["relative_l2_difference"] = (
            math.sqrt(item["difference_squared_integral"] / item["reference_squared_integral"])
            if item["reference_squared_integral"]
            else None
        )
    return {"composite_volume": volume, **totals}


def load_snapshot(path: Path) -> tuple[dict, dict]:
    report = json.loads((path / "manifest.json").read_text())
    run = json.loads((path / "run-snapshot.json").read_text())
    if not report["complete"] or sha(path / "run-snapshot.json") != report["source_record_sha256"]:
        raise ValueError("snapshot lineage is incomplete")
    for frame in report["frames"]:
        raw = path / frame["path"]
        if sha(raw) != frame["sha256"]:
            raise ValueError("native export hash differs")
        validate_blocks(frame["export"]["blocks"], raw.stat().st_size)
        audited = block_diagnostics(raw, frame["export"]["blocks"])
        for key in ("volume", "energy", "peak_speed"):
            if not math.isclose(audited[key], frame["diagnostic"][key], rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError(f"native {key} differs from solver")
    return report, run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse", type=Path, required=True)
    parser.add_argument("--fine", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = [p.resolve(strict=True) for p in (args.coarse, args.fine)]
    (a, ar), (b, br) = (load_snapshot(p) for p in paths)
    for key in ("binary_sha256", "adapter_sha256", "inputs_sha256"):
        if ar[key] != br[key]:
            raise ValueError(f"run {key} differs")
    if a["profile_sha256"] != b["profile_sha256"] or a["parameters"]["base_n"] * 2 != b["parameters"]["base_n"]:
        raise ValueError("profile or base-grid ratio differs")
    for key in ("widths", "max_dt", "epsilon_tau_ratio", "frame_dt", "frame_phase_step", "revolved_refinement"):
        if a["parameters"][key] != b["parameters"][key]:
            raise ValueError(f"run {key} differs")
    if len(a["frames"]) != len(b["frames"]):
        raise ValueError("snapshot frame counts differ")
    report = {
        "schema_version": 1,
        "kind": "mixed-geometry-native-sensitivity",
        "source_snapshot_sha256": [sha(p / "manifest.json") for p in paths],
        "source_run_sha256": [a["source_record_sha256"], b["source_record_sha256"]],
        "source_run_state": [
            {"status": run["status"], "validated": run["validated"]}
            for run in (ar, br)
        ],
        "comparator_sha256": sha(Path(__file__)),
        "scope": (
            __doc__.strip()
            + "\nOnly the selected native snapshots are audited; a running source's full trajectory remains unvalidated."
        ),
        "frames": [],
        "validated": False,
    }
    for fa, fb in zip(a["frames"], b["frames"], strict=True):
        if abs(fa["export"]["time"] - fb["export"]["time"]) > 1e-12:
            raise ValueError("snapshot times differ")
        result = compare(
            paths[0] / fa["path"], fa["export"]["blocks"],
            paths[1] / fb["path"], fb["export"]["blocks"],
        )
        report["frames"].append({"time": fa["export"]["time"], "coarse_step": fa["step"], "fine_step": fb["step"], **result})
        print(json.dumps({"time": fa["export"]["time"], "velocity": result["velocity"]["relative_l2_difference"], "force": result["force"]["relative_l2_difference"]}), flush=True)
    args.output.mkdir(parents=True, exist_ok=False)
    report["validated"] = True
    write(args.output / "summary.json", report)


if __name__ == "__main__":
    main()
