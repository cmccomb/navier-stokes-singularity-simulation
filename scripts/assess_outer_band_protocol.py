"""Assess finalized one-frame comparisons against a locked outer-band protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def assess(protocol: dict, reports: list[tuple[str, dict]]) -> dict:
    if protocol.get("kind") != "prospective-outer-band-spatial-checkpoints":
        raise ValueError("unexpected protocol kind")
    checkpoints = protocol["checkpoints"]
    thresholds = protocol["spatial_screen"]
    matched = {}
    for digest, report in reports:
        if report.get("kind") != "mixed-geometry-native-sensitivity" or not report.get("validated"):
            raise ValueError("comparison is not validated")
        for key in ("source_binary_sha256", "source_profile_sha256", "mesh_pair_base_n"):
            if report.get(key) != protocol[key]:
                raise ValueError(f"comparison {key} differs from protocol")
        if len(report["frames"]) != 1 or len(report["source_run_state"]) != 2:
            raise ValueError("expected one frame and both source run states")
        frame = report["frames"][0]
        matches = [i for i, checkpoint in enumerate(checkpoints) if abs(checkpoint["time"] - frame["time"]) < 1e-12]
        if len(matches) != 1 or matches[0] in matched:
            raise ValueError("comparison time missing from protocol or repeated")
        values = [frame[field]["relative_l2_difference"] for field in ("velocity", "force")]
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("invalid comparison difference")
        matched[matches[0]] = (digest, report, values)
    rows = []
    for i, checkpoint in enumerate(checkpoints):
        row = dict(checkpoint)
        if i in matched:
            digest, report, (velocity, force) = matched[i]
            row.update({
                "screen": "pass" if velocity <= thresholds["velocity_relative_l2_max"] and force <= thresholds["force_relative_l2_max"] else "fail",
                "comparison_sha256": digest,
                "comparison": report,
            })
        else:
            row["screen"] = "pending"
        rows.append(row)
    sequences = []
    start = None
    for i in range(len(rows) + 1):
        passed = i < len(rows) and rows[i]["screen"] == "pass"
        if passed and start is None:
            start = i
        if not passed and start is not None:
            if i - start >= 3:
                sequences.append({"first_frame_index": rows[start]["frame_index"], "last_frame_index": rows[i - 1]["frame_index"], "count": i - start})
            start = None
    return {
        "schema_version": 1,
        "kind": "outer-band-prospective-spatial-screen",
        "checkpoints": rows,
        "all_checkpoints_assessed": len(matched) == len(checkpoints),
        "candidate_sequences": sequences,
        "scope": "Only the locked spatial screen is assessed. A candidate sequence still requires completed-run archive, refined fixed-force timestep, and diagnostic gates; it is not a continuum error bound.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.protocol.read_bytes()
    reports = [(hashlib.sha256(path.read_bytes()).hexdigest(), json.loads(path.read_text())) for path in args.comparison]
    result = assess(json.loads(raw), reports)
    result["protocol_sha256"] = hashlib.sha256(raw).hexdigest()
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
