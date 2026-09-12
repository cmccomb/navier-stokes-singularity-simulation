"""Audit matched native paper-run fields without rewriting either run's status.

Two runs measure sensitivity, not observed order or a continuum error bound.
Near-coincident legacy output events are listed, preserved, and compared; missing
requested events still fail. Aligned finer references are volume-restricted by
the separately verified AMReX archive checker.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

from scripts.paper_run import (
    MARKER,
    SOURCES,
    sha,
    thread_environment,
    validate_history,
    write,
)


def snapshot(path: Path) -> tuple[dict, list[dict]]:
    record = json.loads((path / "run.json").read_text())
    hashes = {name: sha(path / name) for name in SOURCES}
    if (
        hashes != record["adapter_source_hashes"]
        or hashlib.sha256(";".join(hashes.values()).encode()).hexdigest()
        != record["adapter_sha256"]
    ):
        raise ValueError("source snapshot does not match the original run")
    if sha(path / "profile.tbl") != record["profile_manifest"]["sha256"]:
        raise ValueError("profile snapshot hash changed")
    if sha(path / "inputs.paper") != record["inputs_sha256"]:
        raise ValueError("input snapshot hash changed")
    log = (path / "run.log").read_text()
    rows = [
        json.loads(line[len(MARKER) :])
        for line in log.splitlines()
        if line.startswith(MARKER)
    ]
    audit = copy.deepcopy(record)
    # Audit the complete available prefix, not a fictional completed endpoint.
    audit["parameters"]["end"] = rows[-1]["time"]
    validate_history(log, audit)
    return record, rows


def compatible(coarse: dict, fine: dict, kind: str) -> None:
    for key in ("adapter_sha256", "binary_sha256", "inputs_sha256"):
        if coarse[key] != fine[key]:
            raise ValueError(f"comparison changes {key}")
    a, b = copy.deepcopy(coarse["parameters"]), copy.deepcopy(fine["parameters"])
    for params in (a, b):
        params.pop("end")
        params.pop("box")  # the native checker verifies actual physical coverage
    pa = copy.deepcopy(coarse["profile_manifest"]["parameters"])
    pb = copy.deepcopy(fine["profile_manifest"]["parameters"])
    if kind == "temporal":
        if b.pop("max_dt") != a.pop("max_dt") / 2:
            raise ValueError("temporal reference must halve max_dt")
        if pb.pop("forcing_phase_step") != pa.pop("forcing_phase_step") / 2:
            raise ValueError("temporal reference must halve the phase ceiling")
    elif kind == "spatial":
        if b.pop("base_n") != 2 * a.pop("base_n"):
            raise ValueError("spatial reference must double base_n")
    else:
        raise ValueError("unknown comparison kind")
    if a != b or pa != pb:
        raise ValueError("uncontrolled numerical or physical parameter change")


def matched_frames(frames: list[dict], requested: list[float]) -> list[list[dict]]:
    groups = []
    for t in requested:
        candidates = [f for f in frames if abs(f["time"] - t) <= 1e-12]
        if not candidates:
            raise ValueError(f"missing requested native event at t={t}")
        # Prefer closest physical time, retaining every redundant path in the audit.
        groups.append(sorted(candidates, key=lambda f: (abs(f["time"] - t), f["path"])))
    return groups


def frame_headers(path: Path) -> list[dict]:
    frames = []
    for plot in sorted(path.glob("plt[0-9]*")):
        if not plot.is_dir() or not plot.name[3:].isdigit():
            continue
        with (plot / "Header").open() as stream:
            lines = [next(stream).strip() for _ in range(11)]
        if lines[:9] != [
            "HyperCLaw-V1.1",
            "6",
            "velx",
            "vely",
            "velz",
            "forcing_x",
            "forcing_y",
            "forcing_z",
            "3",
        ]:
            raise ValueError("unrecognized native field header")
        t = float(lines[9])
        if not math.isfinite(t):
            raise ValueError("nonfinite native frame time")
        frames.append({"path": plot.name, "time": t})
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--kind", choices=["temporal", "spatial"], required=True)
    parser.add_argument("--times", type=float, nargs="*")
    parser.add_argument(
        "--checker", type=Path, default=Path("build/amrex-omp/ns_archive_check")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    coarse, fine = args.coarse.resolve(strict=True), args.reference.resolve(strict=True)
    checker = args.checker.resolve(strict=True)
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output.resolve()
    a, ah = snapshot(coarse)
    b, bh = snapshot(fine)
    compatible(a, b, args.kind)
    end = min(ah[-1]["time"], bh[-1]["time"])
    times = (
        args.times
        if args.times
        else [
            i * a["parameters"]["frame_dt"]
            for i in range(math.floor(end / a["parameters"]["frame_dt"] + 1e-10) + 1)
        ]
    )
    if any(not math.isfinite(t) or not 0 <= t <= end + 1e-12 for t in times):
        raise ValueError("requested time outside validated history")
    ga, gb = (
        matched_frames(frame_headers(coarse), times),
        matched_frames(frame_headers(fine), times),
    )
    report = {
        "schema_version": 1,
        "kind": args.kind + "-native-sensitivity",
        "scope": __doc__,
        "checker_sha256": sha(checker),
        "runs": [],
        "frames": [],
        "duplicate_events": [],
        "passed": False,
    }
    for path, record, history in ((coarse, a, ah), (fine, b, bh)):
        report["runs"].append(
            {
                "record_sha256": sha(path / "run.json"),
                "original_status": record["status"],
                "original_validated": record["validated"],
                "original_error": record.get("error"),
                "parameters": record["parameters"],
                "adapter_sha256": record["adapter_sha256"],
                "binary_sha256": record["binary_sha256"],
                "profile_sha256": record["profile_manifest"]["sha256"],
                "history_endpoint": history[-1],
            }
        )
    environment = {**os.environ, **thread_environment(1, 1)}

    def check(
        path: Path, record: dict, frame: dict, name: str, reference: Path | None = None
    ) -> dict:
        command = [
            str(checker),
            f"plot={path / frame['path']}",
            "ns.force=paper",
            f"ns.table_file={path / 'profile.tbl'}",
            f"ns.epsilon_tau_ratio={record['parameters']['epsilon_tau_ratio']:.17g}",
        ]
        if reference:
            command += [f"compare={reference}", "report_difference=1"]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env=environment,
            timeout=1800,
            check=False,
        )
        (output / (name + ".log")).write_text(result.stdout + result.stderr)
        if result.returncode:
            raise ValueError(f"native check failed: {name}")
        rows = [
            json.loads(line.removeprefix("NS_ARCHIVE_RESULT "))
            for line in result.stdout.splitlines()
            if line.startswith("NS_ARCHIVE_RESULT ")
        ]
        if len(rows) != 1:
            raise ValueError("missing native check result")
        return rows[0]

    write(output / "summary.json", report)
    for i, (t, ca, fb) in enumerate(zip(times, ga, gb, strict=True)):
        check(fine, b, fb[0], f"{i:03d}-reference")
        row = check(coarse, a, ca[0], f"{i:03d}-difference", fine / fb[0]["path"])
        row.update(
            requested_time=t,
            coarse_frame=ca[0]["path"],
            reference_frame=fb[0]["path"],
            relative_l2_difference=row["velocity_l2_difference"]
            / row["reference_velocity_l2"]
            if row["reference_velocity_l2"]
            else None,
        )
        report["frames"].append(row)
        for label, path, record, group in (
            ("coarse", coarse, a, ca),
            ("reference", fine, b, fb),
        ):
            for j, redundant in enumerate(group[1:]):
                duplicate = check(
                    path,
                    record,
                    redundant,
                    f"{i:03d}-{label}-duplicate-{j}",
                    path / group[0]["path"],
                )
                report["duplicate_events"].append(
                    {
                        "run": label,
                        "requested_time": t,
                        "retained": group[0],
                        "redundant": redundant,
                        "comparison": duplicate,
                    }
                )
        write(output / "summary.json", report)
    report["passed"] = True  # provenance/archive checks, not an accuracy threshold
    write(output / "summary.json", report)
    print(
        json.dumps(
            {
                "kind": report["kind"],
                "matched_events": len(times),
                "duplicate_events": len(report["duplicate_events"]),
                "last": report["frames"][-1],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
