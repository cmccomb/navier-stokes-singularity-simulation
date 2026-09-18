"""Publish a compact, explicitly dated activity-design and launch snapshot."""

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from scripts.paper_run import sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--site", type=Path, required=True)
    args = parser.parse_args()
    root, site = args.evidence, args.site

    def read(name):
        return json.loads((root / name).read_text())

    audit = read("action-audit/manifest.json")
    capacity = read("capacity-outer-run.json")
    launched = read("outer-action-launch.json")
    selected = read("design-outer-a075-v05/design.json")
    best = json.loads((site / "data/best.json").read_text())
    if (
        not audit["validated"]
        or not audit["complete_history"]
        or audit["source_record_sha256"] != best["source_record_sha256"]
        or not capacity["validated"]
        or capacity["status"] != "completed"
    ):
        raise ValueError("Complete matching source and capacity evidence required")
    if (
        launched["parameters"]["revolved_refinement"]["sha256"]
        != selected["band_sha256"]
        or launched["profile_manifest"] != capacity["profile_manifest"]
        or launched["binary_sha256"] != capacity["binary_sha256"]
    ):
        raise ValueError("Launch differs from capacity profile/binary/design")
    pilots = []
    for name in ("revolved-pilot", "revolved-pilot-n32"):
        pilot = read(name + "/summary.json")
        pilots.append(
            {
                "summary_sha256": sha(root / name / "summary.json"),
                "passed": pilot["passed"],
                "error_ratio": pilot.get(
                    "error_ratio", pilot.get("error_ratio_16_to_32")
                ),
                "box_decomposition_error_delta": pilot["box_decomposition_error_delta"],
                "cases": [
                    {
                        k: row[k]
                        for k in (
                            "base_n",
                            "box",
                            "final",
                            "archive",
                            "block_diagnostics",
                            "restart",
                        )
                    }
                    for row in pilot["cases"]
                ],
            }
        )
    if not pilots[-1]["passed"]:
        raise ValueError("Finer manufactured pilot must pass")
    plot = read("action-footprint.json")
    if plot["image_sha256"] != sha(root / "action-footprint.png"):
        raise ValueError("Footprint image differs")
    report = {
        "schema_version": 1,
        "kind": "activity-refinement-design-and-launch-snapshot",
        "recorded_at": datetime.now(UTC).isoformat(),
        "source_record_sha256": audit["source_record_sha256"],
        "scope": "Fixed 3D outer-region spatial-sensitivity experiment, not a new best result or a convergence certificate. Launch status is a timestamped snapshot, not a live counter.",
        "audit": {
            "manifest_sha256": sha(root / "action-audit/manifest.json"),
            "saved_states": len(audit["records"]),
            "complete_history": audit["complete_history"],
            "validated": audit["validated"],
            "scope": audit["scope"],
        },
        "full_union_candidates": [
            read(name + "/design.json")
            for name in ("design-a025-v01-v2", "design-a075-v05")
        ],
        "selected_outer_design": selected,
        "manufactured_pilots": pilots,
        "source_sensors": read("outer-source-sensors/summary.json"),
        "capacity": {
            k: capacity[k]
            for k in (
                "status",
                "validated",
                "wall_seconds",
                "binary_sha256",
                "adapter_sha256",
                "parameters",
                "limits",
                "native_frames",
            )
        },
        "capacity_final": capacity["history"][-1],
        "launch": {
            k: launched[k]
            for k in (
                "status",
                "started_at",
                "binary_sha256",
                "adapter_sha256",
                "parameters",
                "execution",
                "limits",
                "planned_frames",
            )
        },
        "archive_budget": {
            "native_fields_gib": capacity["history"][-1]["stored_cells"]
            * 48
            * 280
            / 2**30,
            "storage_preflight_gib": 256,
            "separate_free_reserve_gib": 20,
            "destination_free_gib_before_launch": 452543652 / 2**20,
        },
        "plot": plot,
    }
    shutil.copy2(root / "action-footprint.png", site / "media/action-footprint.png")
    shutil.copy2(
        root / "design-outer-a075-v05/refinement.bands",
        site / "data/outer-refinement.bands",
    )
    write(site / "data/activity-refinement.json", report)


if __name__ == "__main__":
    main()
