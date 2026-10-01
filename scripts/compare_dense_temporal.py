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

LEGACY_BASELINE_RUNNER_SHA256 = (
    "4d788e2a64e2a58b2deb33d60d847f436ed7407eddec5d985cf9186848f21832"
)


def controls(probe: dict, parent: dict | None = None) -> dict:
    fields = (
        "time_factor",
        "max_dt",
        "integration_phase_step",
        "force_definition_unchanged",
    )
    if all(key in probe for key in fields):
        return {key: probe[key] for key in fields} | {
            "inferred_from_pinned_legacy_runner": False
        }
    if (
        any(key in probe for key in fields)
        or parent is None
        or probe.get("runner_sha256") != LEGACY_BASELINE_RUNNER_SHA256
    ):
        raise ValueError("unrecognized dense probe time-control receipt")
    if (
        parent["binary_sha256"] != probe["source_binary_sha256"]
        or parent["profile_manifest"]["sha256"] != probe["source_profile_sha256"]
        or parent["parameters"].get("integration_phase_step", 0) != 0
    ):
        raise ValueError("legacy dense parent controls differ")
    max_flags = [item for item in probe["command"] if item.startswith("ns.max_dt=")]
    phase_flags = [
        item
        for item in probe["command"]
        if item.startswith("ns.integration_phase_step=")
    ]
    if len(max_flags) != 1 or phase_flags:
        raise ValueError("legacy dense command changes integration controls")
    max_dt = float(max_flags[0].split("=", 1)[1])
    if not math.isclose(max_dt, parent["parameters"]["max_dt"], rel_tol=1e-12):
        raise ValueError("legacy dense max_dt differs from parent")
    return {
        "time_factor": 1,
        "max_dt": max_dt,
        "integration_phase_step": parent["profile_manifest"]["parameters"][
            "forcing_phase_step"
        ],
        "force_definition_unchanged": True,
        "inferred_from_pinned_legacy_runner": True,
    }


def compatible(
    baseline: dict,
    half: dict,
    parent: dict | None = None,
    *,
    baseline_factor: float = 1,
    allow_equivalent_checkpoint_parent: bool = False,
) -> dict:
    if (
        baseline.get("status") != "completed"
        or half.get("status") != "completed"
        or not baseline.get("validated")
        or not half.get("validated")
    ):
        raise ValueError("both dense probes must complete native audits")
    if not math.isfinite(baseline_factor) or baseline_factor <= 0:
        raise ValueError("baseline time factor must be finite and positive")
    parent_records_match = (
        baseline["parent_record_sha256"] == half["parent_record_sha256"]
    )
    if not parent_records_match and not allow_equivalent_checkpoint_parent:
        raise ValueError("dense probe parent_record_sha256 differs")
    for key in (
        "source_binary_sha256",
        "source_profile_sha256",
        "checkpoint_step",
        "checkpoint_time",
        "checkpoint_files",
        "centers",
        "spacing",
        "planned_times",
    ):
        if baseline[key] != half[key]:
            raise ValueError(f"dense probe {key} differs")
    a, b = controls(baseline, parent), controls(half, parent)
    if (
        not math.isclose(a["time_factor"], baseline_factor, rel_tol=1e-12)
        or not math.isclose(b["time_factor"], baseline_factor / 2, rel_tol=1e-12)
        or not math.isclose(b["max_dt"], a["max_dt"] / 2, rel_tol=1e-12)
        or not math.isclose(
            b["integration_phase_step"], a["integration_phase_step"] / 2, rel_tol=1e-12
        )
        or not a["force_definition_unchanged"]
        or not b["force_definition_unchanged"]
    ):
        raise ValueError("require one fixed-force baseline and half-step branch")
    if not (
        len(baseline["native_frames"])
        == len(half["native_frames"])
        == len(baseline["planned_times"])
    ):
        raise ValueError("dense frame count differs")
    return {
        "baseline": a,
        "half": b,
        "checkpoint_parent_identity": (
            "exact_parent_record" if parent_records_match else "exact_checkpoint_state"
        ),
    }


def center_gates(rows: list[dict], centers: list[float], spatial: dict) -> list[dict]:
    result = []
    for center in centers:
        temporal = [row for row in rows if abs(row["time"] - center) <= 1e-12]
        matches = [
            row
            for row in spatial["checkpoints"]
            if abs(row["time"] - center) <= 1e-12 and row["screen"] == "pass"
        ]
        if len(temporal) != 1 or len(matches) != 1:
            raise ValueError("candidate time missing from temporal or spatial record")
        spatial_gap = matches[0]["comparison"]["frames"][0]["velocity"][
            "relative_l2_difference"
        ]
        limit = spatial_gap / 4
        result.append(
            {
                "time": center,
                "spatial_relative_l2": spatial_gap,
                "temporal_relative_l2": temporal[0]["relative_velocity_l2"],
                "quarter_spatial_limit": limit,
                "passed": temporal[0]["relative_velocity_l2"] <= limit,
            }
        )
    return result


def temporal_frame(row: dict, baseline: dict, half: dict, expected: float) -> dict:
    """Keep force readback evidence without inventing a force comparison metric."""
    if (
        not row["compared"]
        or not row["report_difference"]
        or row["reference_resolution_ratio"] != 1
        or abs(row["time"] - expected) > 1e-12
        or any(frame["force_linf_error"] != 0 for frame in (row, baseline, half))
    ):
        raise ValueError("native comparison or force readback failed")
    if row["reference_velocity_l2"] <= 0:
        raise ValueError("zero velocity reference norm")
    return {
        "time": expected,
        "baseline_step": baseline["step"],
        "half_step": half["step"],
        "relative_velocity_l2": row["velocity_l2_difference"]
        / row["reference_velocity_l2"],
        **row,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="completed baseline probe output directory",
    )
    parser.add_argument(
        "--half",
        type=Path,
        required=True,
        help="completed half-step probe output directory",
    )
    parser.add_argument("--checker", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--spatial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-disk-free-gib", type=float, default=30)
    args = parser.parse_args()
    baseline_path, half_path, checker, table, spatial_path = (
        path.resolve(strict=True)
        for path in (args.baseline, args.half, args.checker, args.table, args.spatial)
    )
    baseline = json.loads((baseline_path / "probe.json").read_text())
    half = json.loads((half_path / "probe.json").read_text())
    spatial = json.loads(spatial_path.read_text())
    parent_path = Path(baseline["command"][0]).parent / "run.json"
    if sha(parent_path) != baseline["parent_record_sha256"]:
        raise ValueError("baseline parent record differs")
    parent = json.loads(parent_path.read_text())
    control_audit = compatible(baseline, half, parent)
    if (
        spatial.get("kind") != "outer-band-prospective-spatial-screen"
        or spatial["checkpoints"][0]["comparison"]["source_binary_sha256"]
        != baseline["source_binary_sha256"]
    ):
        raise ValueError("spatial record is unrelated to the dense probe")
    if sha(table) != baseline["source_profile_sha256"]:
        raise ValueError("fixed force table differs")
    force_controls = [
        item for item in baseline["command"] if item.startswith("ns.epsilon_tau_ratio=")
    ]
    if len(force_controls) != 1 or force_controls != [
        item for item in half["command"] if item.startswith("ns.epsilon_tau_ratio=")
    ]:
        raise ValueError("force derivative controls differ")
    if args.output.exists():
        raise ValueError("output already exists")
    args.output.mkdir(parents=True)
    report = {
        "schema_version": 1,
        "kind": "local-fixed-force-dense-temporal-sensitivity",
        "source_probe_sha256": [
            sha(baseline_path / "probe.json"),
            sha(half_path / "probe.json"),
        ],
        "spatial_record_sha256": sha(spatial_path),
        "source_binary_sha256": baseline["source_binary_sha256"],
        "source_profile_sha256": baseline["source_profile_sha256"],
        "checker_sha256": sha(checker),
        "time_controls": control_audit,
        "force_comparison_scope": "Both branches passed independent archived-force readback at every frame; this checker emits velocity difference but no force L2 difference.",
        "scope": __doc__.strip(),
        "frames": [],
        "center_gates": [],
        "validated": False,
    }
    for index, (left, right, expected) in enumerate(
        zip(
            baseline["native_frames"],
            half["native_frames"],
            baseline["planned_times"],
            strict=True,
        )
    ):
        if shutil.disk_usage(args.output).free < args.min_disk_free_gib * 2**30:
            raise RuntimeError("comparison disk reserve reached")
        if (
            abs(left["time"] - expected) > 1e-12
            or abs(right["time"] - expected) > 1e-12
        ):
            raise ValueError("paired native time differs")
        argv = [
            str(checker),
            f"plot={left['path']}",
            f"compare={right['path']}",
            "report_difference=1",
            "ns.force=paper",
            f"ns.table_file={table}",
            force_controls[0],
        ]
        env = {
            **os.environ,
            "OMP_NUM_THREADS": "1",
            "OMP_THREAD_LIMIT": "1",
            "OMP_DYNAMIC": "FALSE",
        }
        checked = subprocess.run(
            argv, capture_output=True, text=True, timeout=900, check=True, env=env
        )
        (args.output / f"frame-{index:02d}.log").write_text(
            checked.stdout + checked.stderr
        )
        rows = [
            json.loads(line.split(" ", 1)[1])
            for line in checked.stdout.splitlines()
            if line.startswith("NS_ARCHIVE_RESULT ")
        ]
        if len(rows) != 1:
            raise ValueError("missing native comparison row")
        report["frames"].append(temporal_frame(rows[0], left, right, expected))
    report["center_gates"] = center_gates(
        report["frames"], baseline["centers"], spatial
    )
    report["all_local_center_gates_passed"] = all(
        row["passed"] for row in report["center_gates"]
    )
    report["validated"] = True
    write(args.output / "summary.json", report)
    print(
        json.dumps(
            {
                "frames": len(report["frames"]),
                "centers": len(report["center_gates"]),
                "local_gate": report["all_local_center_gates_passed"],
            }
        )
    )


if __name__ == "__main__":
    main()
