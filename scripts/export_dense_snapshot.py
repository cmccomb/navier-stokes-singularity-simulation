"""Export audited native frames from a completed dense checkpoint continuation.

Each one-frame snapshot retains both the from-rest parent and continuation
receipts. The continuation is not a new from-rest trajectory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from pathlib import Path

from scripts.export_mesh_snapshot import bounded, sha, tree_hashes
from scripts.paper_run import MARKER


def selected_frames(probe: dict, centers: list[float]) -> list[dict]:
    if probe.get("status") != "completed" or not probe.get("validated"):
        raise ValueError("dense probe lacks completed native audit")
    planned = probe["planned_times"]
    frames = probe["native_frames"]
    if len(planned) != len(frames) or len(centers) != len(set(centers)):
        raise ValueError("invalid dense frame selection")
    selected = []
    for center in centers:
        if not any(abs(center - t) <= 1e-12 for t in probe["centers"]):
            raise ValueError("requested center is outside dense probe")
        for offset in (-2, -1, 0, 1, 2):
            expected = center + offset * probe["spacing"]
            matches = [frame for frame in frames if abs(frame["time"] - expected) <= 1e-12]
            if len(matches) != 1:
                raise ValueError("dense stencil frame missing or ambiguous")
            selected.append(matches[0])
    return selected


def diagnostic_rows(log: str) -> dict[int, dict]:
    rows = [json.loads(line[len(MARKER):]) for line in log.splitlines() if line.startswith(MARKER)]
    if len(rows) != len({row["step"] for row in rows}):
        raise ValueError("duplicate dense diagnostic step")
    return {row["step"]: row for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("probe", "parent", "exporter", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--centers", type=float, nargs="+", required=True)
    parser.add_argument("--library-dir", type=Path)
    args = parser.parse_args()
    probe_path, parent, exporter = (path.resolve(strict=True) for path in (args.probe, args.parent, args.exporter))
    output = args.output.resolve()
    if output.exists():
        raise ValueError("output already exists")
    if args.library_dir:
        os.environ["DYLD_LIBRARY_PATH"] = str(args.library_dir.resolve(strict=True))
    probe_bytes = (probe_path / "probe.json").read_bytes()
    probe = json.loads(probe_bytes)
    parent_bytes = (parent / "run.json").read_bytes()
    run = json.loads(parent_bytes)
    if (hashlib.sha256(parent_bytes).hexdigest() != probe["parent_record_sha256"]
            or sha(parent / "run.log") != probe["parent_log_sha256"]
            or sha(parent / "ns_incflo") != probe["source_binary_sha256"]
            or sha(parent / "profile.tbl") != probe["source_profile_sha256"]
            or run["profile_manifest"]["sha256"] != probe["source_profile_sha256"]):
        raise ValueError("dense probe parent lineage differs")
    frames = selected_frames(probe, args.centers)
    diagnostics = diagnostic_rows((probe_path / "dense" / "run.log").read_text())
    if shutil.disk_usage(output.parent).free < 32 * 2**30:
        raise RuntimeError("insufficient export scratch reserve")
    output.mkdir(parents=True)
    for frame in frames:
        step = frame["step"]
        row = diagnostics.get(step)
        plot = Path(frame["path"]).resolve(strict=True)
        if (row is None or not plot.is_relative_to(probe_path / "dense")
                or plot.name != f"plt{step:05d}"
                or abs(row["time"] - frame["time"]) > 1e-12
                or not math.isclose(row["peak_speed"], frame["peak_speed"], rel_tol=1e-9)
                or frame["force_linf_error"] != 0):
            raise ValueError("dense native frame or diagnostic differs")
        folder = output / f"center-{min(range(len(args.centers)), key=lambda i: abs(args.centers[i] - frame['time']))}-{step:05d}"
        folder.mkdir()
        (folder / "run-snapshot.json").write_bytes(parent_bytes)
        (folder / "probe-snapshot.json").write_bytes(probe_bytes)
        hashes = tree_hashes(plot)
        raw = folder / f"{plot.name}.bin"
        log = folder / f"{plot.name}.log"
        usage = bounded([str(exporter), f"plot={plot}", f"output={raw}", "layout=blocks"], folder, log)
        exported = [json.loads(line.split(" ", 1)[1]) for line in log.read_text().splitlines() if line.startswith("NS_VOLUME_RESULT ")]
        if (len(exported) != 1 or exported[0]["schema_version"] != 2
                or exported[0]["time"] != row["time"]
                or exported[0]["bytes"] != raw.stat().st_size
                or exported[0]["dtype"] != "<f8"
                or exported[0]["order"] != "xyz-component"
                or tree_hashes(plot) != hashes):
            raise ValueError("dense native export identity differs")
        report = {
            "schema_version": 1,
            "complete": True,
            "source_run": str(parent),
            "source_record_sha256": hashlib.sha256(parent_bytes).hexdigest(),
            "source_probe_sha256": hashlib.sha256(probe_bytes).hexdigest(),
            "exporter_sha256": sha(exporter),
            "export_script_sha256": sha(Path(__file__)),
            "profile_sha256": probe["source_profile_sha256"],
            "parameters": run["parameters"],
            "frames": [{"step": step, "path": raw.name, "sha256": sha(raw), "native_sha256": hashes,
                        "diagnostic": row, "native_audit": frame, "export": exported[0], "resource_usage": usage}],
            "scope": "One audited native frame from a checkpoint continuation; inherited parent error remains.",
        }
        (folder / "manifest.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"folder": str(folder), "step": step, "time": row["time"]}), flush=True)


if __name__ == "__main__":
    main()
