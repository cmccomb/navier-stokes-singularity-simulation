"""Verify native coarse/fine comparison against an analytic cell-average result.

This is an averaging-operator correctness check on the initial Taylor field,
not an evolved flow result or a solver-performance benchmark.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

from scripts.paper_run import sha, thread_environment, write


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=Path("build/amrex-omp"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    build = args.build.resolve(strict=True)
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output.resolve()
    for name in ("ns_incflo", "ns_archive_check"):
        shutil.copy2(build / name, output / name)
    shutil.copy2(root / "backends/amrex/inputs.coupled", output / "inputs")
    environment = {**os.environ, **thread_environment(1, 1)}

    def run(
        command: list[str], directory: Path, log: str
    ) -> subprocess.CompletedProcess:
        result = subprocess.run(
            command,
            cwd=directory,
            env=environment,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        (output / log).write_text(result.stdout + result.stderr)
        return result

    for name, n in (("coarse", 16), ("fine", 32)):
        directory = output / name
        directory.mkdir()
        result = run(
            [
                str(output / "ns_incflo"),
                str(output / "inputs"),
                "ns.force=none",
                "incflo.probtype=2",
                "max_step=0",
                "amr.plot_int=1",
                f"amr.n_cell={n} {n} {n}",
                "amr.max_grid_size=16",
            ],
            directory,
            name + ".log",
        )
        if result.returncode:
            raise ValueError("initial-field fixture failed")
    command = [
        str(output / "ns_archive_check"),
        f"plot={output / 'coarse/plt00000'}",
        f"compare={output / 'fine/plt00000'}",
        "ns.force=none",
    ]
    result = run(command + ["report_difference=1"], output, "comparison.log")
    if result.returncode:
        raise ValueError("native restriction failed")
    rows = [
        json.loads(s.removeprefix("NS_ARCHIVE_RESULT "))
        for s in result.stdout.splitlines()
        if s.startswith("NS_ARCHIVE_RESULT ")
    ]
    if len(rows) != 1:
        raise ValueError("missing native restriction result")
    row = rows[0]
    # Averaging the 2x2x2 fine cells multiplies each sine/cosine product by
    # cos(pi/32)^2. The resulting vector RMS and component maximum are known.
    amplitude = 1 - math.cos(math.pi / 32) ** 2
    expected_l2 = amplitude / math.sqrt(2)
    expected_linf = amplitude * math.cos(math.pi / 16) ** 2
    rejected = run(command, output, "restart-rejected.log")
    record = {
        "schema_version": 1,
        "kind": "native-restriction-correctness",
        "scope": __doc__,
        "checker_sha256": sha(output / "ns_archive_check"),
        "expected_l2_difference": expected_l2,
        "expected_linf_difference": expected_linf,
        "measured": row,
        "restart_rejects_mesh_change": bool(
            rejected.returncode
            and "restart mesh changed" in rejected.stdout + rejected.stderr
        ),
    }
    record["passed"] = bool(
        row["reference_resolution_ratio"] == 2
        and row["composite_volume"] == 8
        and abs(row["velocity_l2_difference"] - expected_l2) < 1e-13
        and abs(row["velocity_linf_difference"] - expected_linf) < 1e-12
        and record["restart_rejects_mesh_change"]
    )
    write(output / "summary.json", record)
    print(json.dumps(record), flush=True)
    raise SystemExit(0 if record["passed"] else 2)


if __name__ == "__main__":
    main()
