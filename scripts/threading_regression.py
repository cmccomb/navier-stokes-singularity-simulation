"""Check bounded OpenMP correctness; no speed ranking or parameter sweep.

Run the same short from-rest trajectory with one and two threads, verify every
native frame, and compare every velocity component. The standalone box check
also exercises the source at late times and cache reuse/eviction.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from scripts.paper_run import sha, thread_environment, write


def run(command: list[str], directory: Path, log: Path, threads: int) -> str:
    environment = {**os.environ, **thread_environment(threads, threads)}
    with log.open("w") as stream:
        result = subprocess.run(
            command,
            cwd=directory,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=600,
            check=False,
        )
    text = log.read_text()
    if result.returncode:
        raise ValueError(f"correctness check exited {result.returncode}: {log}")
    return text


def compare_records(single: dict, threaded: dict) -> list[tuple[dict, dict]]:
    for record, threads in ((single, 1), (threaded, 2)):
        if record.get("status") != "completed" or not record.get("validated"):
            raise ValueError("both trajectories must pass their clock/archive checks")
        if (
            record["execution"]["threads"] != threads
            or record["execution"]["force_threads"] != threads
        ):
            raise ValueError("thread counts differ from the prescribed check")
    for key in ("binary_sha256", "adapter_sha256", "parameters", "profile_manifest"):
        if single[key] != threaded[key]:
            raise ValueError(f"thread comparison changed {key}")
    a, b = single["history"], threaded["history"]
    if [(r["step"], r["time"], r["dt"]) for r in a] != [
        (r["step"], r["time"], r["dt"]) for r in b
    ]:
        raise ValueError("thread comparison changed the integration clock")
    a, b = single["native_frames"], threaded["native_frames"]
    if not a or len(a) != len(b):
        raise ValueError("native frame sequence is incomplete")
    if [(r["time"], r["levels"], r["stored_cells"]) for r in a] != [
        (r["time"], r["levels"], r["stored_cells"]) for r in b
    ]:
        raise ValueError("native frame times or meshes differ")
    return list(zip(a, b, strict=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=Path("build/amrex-omp"))
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    build, table = args.build.resolve(strict=True), args.table.resolve(strict=True)
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output.resolve()
    report = {
        "schema_version": 1,
        "kind": "openmp-correctness",
        "passed": False,
        "binary_sha256": sha(build / "ns_incflo"),
        "frames": [],
    }
    write(output / "summary.json", report)
    try:
        text = run(
            [
                str(build / "ns_threading_check"),
                "ns.force=paper",
                "ns.force_threads=2",
                "ns.expected_omp_threads=2",
                f"ns.table_file={table}",
            ],
            output,
            output / "source-check.log",
            2,
        )
        checks = [
            json.loads(line.removeprefix("NS_THREADING_RESULT "))
            for line in text.splitlines()
            if line.startswith("NS_THREADING_RESULT ")
        ]
        if (
            len(checks) != 1
            or not checks[0]["passed"]
            or checks[0]["actual_threads"] != 2
        ):
            raise ValueError("missing successful two-thread source check")
        report["source_check"] = checks[0]
        for threads in (1, 2):
            command = [
                sys.executable,
                "-m",
                "scripts.paper_run",
                "--executable",
                str(build / "ns_incflo"),
                "--checker",
                str(build / "ns_archive_check"),
                "--table",
                str(table),
                "--output",
                str(output / f"threads-{threads}"),
                "--threads",
                str(threads),
                "--force-threads",
                str(threads),
                "--base-n",
                "16",
                "--box",
                "8",
                "--widths",
                "0.5",
                "--end",
                "0.56",
                "--max-dt",
                "0.0005",
                "--frame-dt",
                "0.025",
                "--timeout",
                "300",
            ]
            run(command, root, output / f"threads-{threads}.log", threads)
        single = json.loads((output / "threads-1/run.json").read_text())
        threaded = json.loads((output / "threads-2/run.json").read_text())
        for a, b in compare_records(single, threaded):
            log = output / f"compare-{a['path']}.log"
            text = run(
                [
                    str(build / "ns_archive_check"),
                    f"plot={output / 'threads-2' / b['path']}",
                    f"compare={output / 'threads-1' / a['path']}",
                    "ns.force=paper",
                    f"ns.table_file={table}",
                ],
                output,
                log,
                1,
            )
            rows = [
                json.loads(line.removeprefix("NS_ARCHIVE_RESULT "))
                for line in text.splitlines()
                if line.startswith("NS_ARCHIVE_RESULT ")
            ]
            if len(rows) != 1 or not rows[0]["compared"]:
                raise ValueError("missing full native velocity comparison")
            report["frames"].append(rows[0])
            write(output / "summary.json", report)
        report["adapter_sha256"] = single["adapter_sha256"]
        report["velocity_linf_difference"] = max(
            r["velocity_linf_difference"] for r in report["frames"]
        )
        report["passed"] = True
    except (ValueError, subprocess.SubprocessError, OSError) as exc:
        report["error"] = str(exc)
        raise
    finally:
        write(output / "summary.json", report)
    print(
        json.dumps(
            {
                "passed": True,
                "frames": len(report["frames"]),
                "velocity_linf_difference": report["velocity_linf_difference"],
            }
        )
    )


if __name__ == "__main__":
    main()
