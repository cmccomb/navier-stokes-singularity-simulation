"""Compare paired dense checkpoint continuations with one fixed force table.

The two branches share a checkpoint and every saved event. This measures local
timestep sensitivity after that checkpoint; their inherited earlier errors
are common and are not tested here.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

from scripts.paper_run import sha, write


def compatible(baseline: dict, half: dict) -> None:
    if baseline.get("status") != "completed" or half.get("status") != "completed" or not baseline.get("validated") or not half.get("validated"):
        raise ValueError("both dense probes must complete native audits")
    for key in ("parent_record_sha256", "source_binary_sha256", "source_profile_sha256", "checkpoint_step", "checkpoint_time", "checkpoint_files", "centers", "spacing", "planned_times"):
        if baseline[key] != half[key]:
            raise ValueError(f"dense probe {key} differs")
    if (baseline["time_factor"] != 1 or half["time_factor"] != 0.5
            or not math.isclose(half["max_dt"], baseline["max_dt"] / 2, rel_tol=1e-12)
            or not math.isclose(half["integration_phase_step"], baseline["integration_phase_step"] / 2, rel_tol=1e-12)
            or not baseline["force_definition_unchanged"] or not half["force_definition_unchanged"]):
        raise ValueError("require one fixed-force baseline and half-step branch")
    if not (len(baseline["native_frames"]) == len(half["native_frames"]) == len(baseline["planned_times"])):
        raise ValueError("dense frame count differs")


def center_gates(rows: list[dict], centers: list[float], spatial: dict) -> list[dict]:
    result = []
    for center in centers:
        temporal = [row for row in rows if abs(row["time"] - center) <= 1e-12]
        matches = [row for row in spatial["checkpoints"] if abs(row["time"] - center) <= 1e-12 and row["screen"] == "pass"]
        if len(temporal) != 1 or len(matches) != 1:
            raise ValueError("candidate time missing from temporal or spatial record")
        spatial_gap = matches[0]["comparison"]["frames"][0]["velocity"]["relative_l2_difference"]
        limit = spatial_gap / 4
        result.append({"time": center, "spatial_relative_l2": spatial_gap, "temporal_relative_l2": temporal[0]["relative_velocity_l2"], "quarter_spatial_limit": limit, "passed": temporal[0]["relative_velocity_l2"] <= limit})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="completed baseline probe output directory")
    parser.add_argument("--half", type=Path, required=True, help="completed half-step probe output directory")
    parser.add_argument("--checker", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--spatial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-disk-free-gib", type=float, default=30)
    args = parser.parse_args()
    baseline_path, half_path, checker, table, spatial_path = (path.resolve(strict=True) for path in (args.baseline, args.half, args.checker, args.table, args.spatial))
    baseline = json.loads((baseline_path / "probe.json").read_text())
    half = json.loads((half_path / "probe.json").read_text())
    spatial = json.loads(spatial_path.read_text())
    compatible(baseline, half)
    if spatial.get("kind") != "outer-band-prospective-spatial-screen" or spatial["checkpoints"][0]["comparison"]["source_binary_sha256"] != baseline["source_binary_sha256"]:
        raise ValueError("spatial record is unrelated to the dense probe")
    if sha(table) != baseline["source_profile_sha256"]:
        raise ValueError("fixed force table differs")
    force_controls = [item for item in baseline["command"] if item.startswith("ns.epsilon_tau_ratio=")]
    if len(force_controls) != 1 or force_controls != [item for item in half["command"] if item.startswith("ns.epsilon_tau_ratio=")]:
        raise ValueError("force derivative controls differ")
    if args.output.exists():
        raise ValueError("output already exists")
    args.output.mkdir(parents=True)
    report = {
        "schema_version": 1,
        "kind": "local-fixed-force-dense-temporal-sensitivity",
        "source_probe_sha256": [sha(baseline_path / "probe.json"), sha(half_path / "probe.json")],
        "spatial_record_sha256": sha(spatial_path),
        "source_binary_sha256": baseline["source_binary_sha256"],
        "source_profile_sha256": baseline["source_profile_sha256"],
        "checker_sha256": sha(checker),
        "scope": __doc__.strip(),
        "frames": [],
        "center_gates": [],
        "validated": False,
    }
    for index, (left, right, expected) in enumerate(zip(baseline["native_frames"], half["native_frames"], baseline["planned_times"], strict=True)):
        if shutil.disk_usage(args.output).free < args.min_disk_free_gib * 2**30:
            raise RuntimeError("comparison disk reserve reached")
        if abs(left["time"] - expected) > 1e-12 or abs(right["time"] - expected) > 1e-12:
            raise ValueError("paired native time differs")
        argv = [str(checker), f"plot={left['path']}", f"compare={right['path']}", "report_difference=1", "ns.force=paper", f"ns.table_file={table}", force_controls[0]]
        env = {**os.environ, "OMP_NUM_THREADS": "1", "OMP_THREAD_LIMIT": "1", "OMP_DYNAMIC": "FALSE"}
        checked = subprocess.run(argv, capture_output=True, text=True, timeout=900, check=True, env=env)
        (args.output / f"frame-{index:02d}.log").write_text(checked.stdout + checked.stderr)
        rows = [json.loads(line.split(" ", 1)[1]) for line in checked.stdout.splitlines() if line.startswith("NS_ARCHIVE_RESULT ")]
        if len(rows) != 1 or not rows[0]["compared"] or not rows[0]["report_difference"] or rows[0]["reference_resolution_ratio"] != 1 or abs(rows[0]["time"] - expected) > 1e-12 or rows[0]["force_linf_error"] != 0:
            raise ValueError("native comparison failed")
        row = rows[0]
        if row["reference_velocity_l2"] <= 0:
            raise ValueError("zero velocity reference norm")
        report["frames"].append({"time": expected, "baseline_step": left["step"], "half_step": right["step"], "relative_velocity_l2": row["velocity_l2_difference"] / row["reference_velocity_l2"], "relative_force_l2": row["force_l2_difference"] / row["reference_force_l2"] if row["reference_force_l2"] > 0 else None, **row})
    report["center_gates"] = center_gates(report["frames"], baseline["centers"], spatial)
    report["all_local_center_gates_passed"] = all(row["passed"] for row in report["center_gates"])
    report["validated"] = True
    write(args.output / "summary.json", report)
    print(json.dumps({"frames": len(report["frames"]), "centers": len(report["center_gates"]), "local_gate": report["all_local_center_gates_passed"]}))


if __name__ == "__main__":
    main()
