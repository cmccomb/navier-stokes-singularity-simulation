"""Export the final native state from a completed checkpoint continuation.

The one-frame float64 snapshot is for bounded offline diagnostics.  It retains
the hashed parent completion record and continuation receipt and does not turn
the continuation into a new from-rest or spatially converged trajectory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from pathlib import Path

from scripts.archive_path_map import (
    load_path_map,
    path_map_record,
    resolve_archive_path,
)
from scripts.export_mesh_snapshot import bounded, sha, tree_hashes


def final_frame(
    source: Path, path_map: dict[Path, Path] | None = None
) -> tuple[dict, dict, dict]:
    path_map = path_map or {}
    receipt = json.loads((source / "continuation.json").read_text())
    if receipt.get("status") != "completed" or receipt.get("validated") is not True:
        raise ValueError("continuation must be completed and validated")
    stage = receipt.get("stages", {}).get("extension", {})
    if stage.get("status") != "completed" or stage.get("validated") is not True:
        raise ValueError("accepted extension stage is incomplete")
    rows, frames = stage.get("history", []), stage.get("native_frames", [])
    if not rows or not frames:
        raise ValueError("extension lacks final diagnostics or native frames")
    row, frame = rows[-1], frames[-1]
    if (
        not math.isclose(row["time"], receipt["end"], rel_tol=0, abs_tol=2e-14)
        or not math.isclose(frame["time"], receipt["end"], rel_tol=0, abs_tol=1e-12)
        or frame["step"] != row["step"]
        or frame.get("force_linf_error", math.inf) > 1e-12
    ):
        raise ValueError("extension endpoint audit differs from its diagnostics")
    plot = resolve_archive_path(frame["path"], path_map)
    if not plot.is_relative_to(source / "extension") or plot.name != f"plt{row['step']:05d}":
        raise ValueError("extension endpoint path is outside the accepted stage")
    return receipt, row, frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("continuation", "exporter", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--library-dir", type=Path)
    parser.add_argument("--path-map", type=Path)
    args = parser.parse_args()
    source, exporter = (
        args.continuation.resolve(strict=True),
        args.exporter.resolve(strict=True),
    )
    output = args.output.resolve()
    if output.exists():
        raise ValueError("output already exists")
    if args.library_dir:
        os.environ["DYLD_LIBRARY_PATH"] = str(args.library_dir.resolve(strict=True))

    path_map = load_path_map(args.path_map)
    continuation_bytes = (source / "continuation.json").read_bytes()
    receipt, row, frame = final_frame(source, path_map)
    parent = resolve_archive_path(receipt["parent"], path_map)
    parent_bytes = (parent / "run.json").read_bytes()
    run = json.loads(parent_bytes)
    if (
        hashlib.sha256(parent_bytes).hexdigest() != receipt["parent_record_sha256"]
        or run.get("status") != "completed"
        or run.get("validated") is not True
        or run["binary_sha256"] != receipt["bundle_sha256"]["ns_incflo"]
        or run["profile_manifest"]["sha256"]
        != receipt["bundle_sha256"]["profile.tbl"]
    ):
        raise ValueError("continuation parent or pinned solver identity differs")
    if shutil.disk_usage(output.parent).free < 22 * 2**30:
        raise RuntimeError("insufficient export scratch reserve")

    output.mkdir(parents=True)
    (output / "run-snapshot.json").write_bytes(parent_bytes)
    (output / "continuation-snapshot.json").write_bytes(continuation_bytes)
    plot = resolve_archive_path(frame["path"], path_map)
    hashes = tree_hashes(plot)
    raw, log = output / f"{plot.name}.bin", output / f"{plot.name}.log"
    usage = bounded(
        [str(exporter), f"plot={plot}", f"output={raw}", "layout=blocks"],
        output,
        log,
    )
    exported = [
        json.loads(line.split(" ", 1)[1])
        for line in log.read_text().splitlines()
        if line.startswith("NS_VOLUME_RESULT ")
    ]
    if (
        len(exported) != 1
        or exported[0]["schema_version"] != 2
        or exported[0]["time"] != row["time"]
        or exported[0]["bytes"] != raw.stat().st_size
        or exported[0]["dtype"] != "<f8"
        or exported[0]["order"] != "xyz-component"
        or tree_hashes(plot) != hashes
        or (source / "continuation.json").read_bytes() != continuation_bytes
    ):
        raise ValueError("native endpoint changed during export")
    report = {
        "schema_version": 1,
        "complete": True,
        "source_run": str(parent),
        "source_record_sha256": hashlib.sha256(parent_bytes).hexdigest(),
        "source_continuation_sha256": hashlib.sha256(continuation_bytes).hexdigest(),
        "exporter_sha256": sha(exporter),
        "export_script_sha256": sha(Path(__file__)),
        "archive_path_map": path_map_record(path_map),
        "archive_path_map_sha256": sha(args.path_map) if args.path_map else None,
        "profile_sha256": run["profile_manifest"]["sha256"],
        "parameters": run["parameters"],
        "frames": [
            {
                "step": row["step"],
                "path": raw.name,
                "sha256": sha(raw),
                "native_sha256": hashes,
                "diagnostic": row,
                "native_audit": frame,
                "export": exported[0],
                "resource_usage": usage,
            }
        ],
        "scope": "Final audited native frame from a single-mesh continuation; inherited parent and spatial errors remain.",
    }
    (output / "manifest.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({"step": row["step"], "time": row["time"], **usage}))


if __name__ == "__main__":
    main()
