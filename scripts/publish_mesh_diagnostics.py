"""Build a source-backed documentation record without changing accepted results."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from scripts.export_mesh_snapshot import sha
from scripts.paper_run import write


def checked_report(path):
    report = json.loads(path.read_text())
    if not report["complete"]:
        raise ValueError(f"Incomplete diagnostic: {path}")
    return report


def checked_sensors(path):
    report = checked_report(path)
    for sensor in report["sensors"]:
        if len(sensor["raw_sha256"]) != 4:
            raise ValueError("Four source probes per sensor required")
        for name, digest in sensor["raw_sha256"].items():
            if Path(name).name != name or sha(path.parent / name) != digest:
                raise ValueError("Source probe bytes differ")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--sensors", type=Path, nargs="+", required=True)
    p.add_argument("--ranking", type=Path, required=True)
    p.add_argument("--comparison", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    snapshots = [checked_report(x) for x in (args.baseline, args.candidate)]
    reports = [checked_sensors(x) for x in args.sensors]
    if (
        len(
            {r["profile_sha256"] for r in snapshots}
            | {r["table_sha256"] for r in reports}
        )
        != 1
    ):
        raise ValueError("Diagnostic profile hashes differ")
    ranking = checked_report(args.ranking)
    if ranking["snapshot_manifest_sha256"] != sha(args.candidate):
        raise ValueError("Ranking snapshot differs")
    supplied_reports = {sha(x) for x in args.sensors}
    if not set(ranking["sensor_report_sha256"]) <= supplied_reports:
        raise ValueError("Ranking sensor reports missing")
    for row in ranking["candidates"]:
        folder = args.ranking.parent / row["name"]
        if (
            sha(folder / "mesh.json") != row["mesh_sha256"]
            or sha(folder / "refinement.bands") != row["bands_sha256"]
        ):
            raise ValueError("Candidate geometry changed")
        if row["production_ready"]:
            raise ValueError("Mesh-only record cannot approve production")
    result = {
        "schema_version": 1,
        "kind": "partial-history-mesh-design-diagnostics",
        "recorded_at": datetime.now(UTC).isoformat(),
        "production_ready": False,
        "accepted_best_changed": False,
        "diagnostic_record_complete": True,
        "matched_native_comparison": {
            "status": "pending transfer and independent audit"
        },
        "snapshots": [],
        "source_sensors": reports,
        "candidate_ranking": ranking,
        "limits": [
            "Sparse source sensors are not whole-domain or whole-history error estimates.",
            "Future-time source probes are not evolved Oliver states.",
            "Derivative-window sensitivity is not solver timestep convergence.",
            "Memory estimates are linear screening estimates, not capacity measurements.",
            "No new evolution run, endpoint extension, or replacement best result is authorized by this record.",
        ],
    }
    for path, snapshot in zip((args.baseline, args.candidate), snapshots, strict=True):
        result["snapshots"].append(
            {
                "manifest_sha256": sha(path),
                "source_record_sha256": snapshot["source_record_sha256"],
                "profile_sha256": snapshot["profile_sha256"],
                "exporter_sha256": snapshot["exporter_sha256"],
                "parameters": snapshot["parameters"],
                "frames": [
                    {k: f[k] for k in ("step", "sha256", "native_sha256", "diagnostic")}
                    for f in snapshot["frames"]
                ],
            }
        )
    if args.comparison:
        comparison = checked_report(args.comparison)
        if comparison["snapshot_manifest_sha256"] != [
            sha(x) for x in (args.baseline, args.candidate)
        ]:
            raise ValueError("Native comparison snapshots differ")
        result["matched_native_comparison"] = {
            "status": "complete",
            "sha256": sha(args.comparison),
            "report": comparison,
        }
    # Preserve measured geometry for the leading low-cost ring, including its
    # complete ancestor coverage. This is a candidate, not a new production mesh.
    choices = [
        r
        for r in ranking["candidates"]
        if r["added_cells"] > 0 and r["passes_nominal_11_5_gib_screen"]
    ]
    chosen = max(choices, key=lambda r: r["coverage_score_per_added_million_cells"])
    folder = args.ranking.parent / chosen["name"]
    result["leading_candidate_geometry"] = {
        "name": chosen["name"],
        "selection_rule": "Highest sampled source-sensitivity coverage per added million cells among nominal memory-screen passes",
        "bands_text": (folder / "refinement.bands").read_text(),
        "mesh": json.loads((folder / "mesh.json").read_text()),
    }
    write(args.output, result)


if __name__ == "__main__":
    main()
