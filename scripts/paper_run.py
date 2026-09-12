"""Run and audit a versioned finite paper surrogate from exact rest with incflo.

This is not a singularity certificate. The source is mesh-dependent, and a
trajectory requires separate spatial and temporal convergence checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

SOURCES = ("ns_case.H", "incflo_overlay.py", "paper_profile.H", "paper_fields.H")
MARKER = "NS_INCFLO_RESULT "


def thread_environment(threads: int, force_threads: int) -> dict[str, str]:
    """Explicit child-process limits, not mutations to the worker's environment."""
    if threads < 1 or not 1 <= force_threads <= threads:
        raise ValueError("require 1 <= force_threads <= threads")
    return {
        "OMP_NUM_THREADS": str(threads),
        "OMP_THREAD_LIMIT": str(threads),
        "OMP_DYNAMIC": "FALSE",
        "OMP_MAX_ACTIVE_LEVELS": "1",
    }


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, record: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def output_times(end: float, period: float, phase: float, clock: dict) -> list[float]:
    """Predefine output events; the numerical integration clock stays independent."""
    if not 0 < end < clock["t_star"] or period <= 0 or phase < 0:
        raise ValueError("invalid output clock")
    if phase == 0:
        times = [i * period for i in range(math.floor(end / period + 1e-10) + 1)]
        if abs(times[-1] - end) > 1e-13:
            times.append(end)
        return times
    times = [0.0]
    while times[-1] < end:
        t = times[-1]
        dt = min(period, end - t)
        if t < clock["paper_time_cutoff_start"]:
            dt = min(dt, clock["paper_time_cutoff_start"] - t)
        elif clock["forcing_log_rate_bound"] > 0:
            dt = min(
                dt,
                -(clock["t_star"] - t)
                * math.expm1(-phase / clock["forcing_log_rate_bound"]),
            )
        next_time = min(end, t + dt)
        if end - next_time <= 1e-13:
            next_time = end
        if next_time - t <= 1e-11 or len(times) >= 50000:
            raise ValueError("unresolvable or excessive output schedule")
        times.append(next_time)
    return times


def resource_check(
    output: Path, pid: int, max_rss_mib: float, min_disk_free_gib: float
) -> None:
    if (
        min_disk_free_gib
        and shutil.disk_usage(output).free < min_disk_free_gib * 1024**3
    ):
        raise RuntimeError(
            "disk reserve reached; stopping this run without deleting data"
        )
    if max_rss_mib:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "rss="],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if not result.stdout.strip():
            return  # child can finish between the wait and this inspection
        rss = float(result.stdout.strip()) / 1024
        if rss > max_rss_mib:
            raise RuntimeError(
                f"solver RSS {rss:.1f} MiB exceeds {max_rss_mib:.1f} MiB limit"
            )


def wait_bounded(
    process: subprocess.Popen,
    output: Path,
    timeout: float,
    max_rss_mib: float,
    min_disk_free_gib: float,
) -> int:
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(process.args, timeout)
        try:
            return process.wait(timeout=min(5, remaining))
        except subprocess.TimeoutExpired:
            resource_check(output, process.pid, max_rss_mib, min_disk_free_gib)


def validate_history(text: str, record: dict) -> list[dict]:
    rows = [
        json.loads(line[len(MARKER) :])
        for line in text.splitlines()
        if line.startswith(MARKER)
    ]
    if (
        len(rows) < 2
        or rows[0]["time"] != 0
        or rows[0]["peak_speed"] != 0
        or rows[0]["energy"] != 0
    ):
        raise ValueError("trajectory does not begin at exact rest")
    params = record["parameters"]
    clock = record["profile_manifest"]["parameters"]
    for i, row in enumerate(rows):
        if (execution := record.get("execution")) and (
            row.get("openmp_max_threads") != execution["threads"]
            or row.get("force_threads_limit") != execution["force_threads"]
        ):
            raise ValueError("compiled/runtime threading differs from the run record")
        if row["case"] != "paper" or row["adapter_sha256"] != record["adapter_sha256"]:
            raise ValueError("case or compiled adapter does not match the record")
        if row["step"] != i or row["levels"] != len(params["widths"]) + 1:
            raise ValueError("step history or mesh is incomplete")
        for key in (
            "time",
            "dt",
            "volume",
            "energy",
            "peak_speed",
            "l2_error",
            "linf_error",
            "peak_rss_mib",
        ):
            if not math.isfinite(row[key]):
                raise ValueError(f"nonfinite diagnostic: {key}")
        if abs(row["volume"] - 8) > 1e-10:
            raise ValueError("incomplete full-domain diagnostics")
        if row["time"] <= clock["paper_time_cutoff_start"] and (
            row["peak_speed"] != 0 or row["energy"] != 0
        ):
            raise ValueError("quiescent interval is not exactly at rest")
        if i:
            previous = rows[i - 1]["time"]
            if row["dt"] <= 0 or abs(row["time"] - previous - row["dt"]) > 2e-14:
                raise ValueError("inconsistent time increment")
            if previous >= clock["paper_time_cutoff_start"]:
                phase = clock["forcing_log_rate_bound"] * math.log1p(
                    row["dt"] / (clock["t_star"] - row["time"])
                )
                if phase > clock["forcing_phase_step"] * (1 + 1e-9):
                    raise ValueError("forcing phase step exceeded")
                if row["dt"] > params["max_dt"] * (1 + 1e-10):
                    raise ValueError("active timestep ceiling exceeded")
            elif row["time"] > clock["paper_time_cutoff_start"] + 2e-14:
                raise ValueError("quiescent step crossed activation")
    if abs(rows[-1]["time"] - params["end"]) > 2e-14:
        raise ValueError("trajectory did not reach the requested endpoint")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--executable", type=Path, default=Path("build/amrex/ns_incflo")
    )
    parser.add_argument(
        "--checker", type=Path, default=Path("build/amrex/ns_archive_check")
    )
    parser.add_argument(
        "--inputs", type=Path, default=Path("backends/amrex/inputs.paper")
    )
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-n", type=int, default=32)
    parser.add_argument("--widths", type=float, nargs="*", default=[])
    parser.add_argument("--box", type=int, default=32)
    parser.add_argument("--end", type=float, default=0.85)
    parser.add_argument("--max-dt", type=float, default=0.00025)
    parser.add_argument("--epsilon-tau-ratio", type=float, default=0)
    parser.add_argument("--frame-dt", type=float, default=0.025)
    parser.add_argument(
        "--frame-phase-step",
        type=float,
        default=0,
        help="optional forcing-phase output spacing, never an integration timestep",
    )
    parser.add_argument("--timeout", type=float, default=86400)
    parser.add_argument(
        "--max-rss-mib",
        type=float,
        default=0,
        help="solver RSS stop ceiling; zero disables",
    )
    parser.add_argument(
        "--min-disk-free-gib",
        type=float,
        default=0,
        help="stop with this disk reserve; zero disables",
    )
    parser.add_argument("--threads", type=int, default=1, help="solver OpenMP threads")
    parser.add_argument(
        "--force-threads",
        type=int,
        default=1,
        help="concurrent force boxes; at most --threads (extra scratch per worker)",
    )
    args = parser.parse_args()
    try:
        thread_env = thread_environment(args.threads, args.force_threads)
    except ValueError as exc:
        parser.error(str(exc))
    child_env = {**os.environ, **thread_env}
    inputs = args.inputs.resolve(strict=True)
    table = args.table.resolve(strict=True)
    manifest = json.loads(table.with_suffix(table.suffix + ".json").read_text())
    if manifest["sha256"] != sha(table):
        parser.error("table hash differs from its manifest")
    if not (
        0 < args.end < manifest["parameters"]["t_star"]
        and args.max_dt > 0
        and args.frame_dt > 0
        and args.frame_phase_step >= 0
        and args.timeout > 0
        and args.max_rss_mib >= 0
        and args.min_disk_free_gib >= 0
        and all(
            math.isfinite(value)
            for value in (
                args.end,
                args.max_dt,
                args.frame_dt,
                args.frame_phase_step,
                args.timeout,
                args.max_rss_mib,
                args.min_disk_free_gib,
            )
        )
    ):
        parser.error("invalid run clock")
    if args.base_n < 16 or args.base_n % 8 or args.box < 8 or args.box % 8:
        parser.error("mesh dimensions must be multiples of eight")
    planned_frames = output_times(
        args.end, args.frame_dt, args.frame_phase_step, manifest["parameters"]
    )
    source_hashes = {name: sha(inputs.parent / name) for name in SOURCES}
    adapter_hash = hashlib.sha256(";".join(source_hashes.values()).encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output.resolve()
    # Pin immutable run dependencies; rebuilding the checkout cannot change a run.
    for source, name in [
        (args.executable, "ns_incflo"),
        (args.checker, "ns_archive_check"),
        (inputs, "inputs.paper"),
        (table, "profile.tbl"),
    ]:
        shutil.copy2(source.resolve(strict=True), output / name)
    for name in SOURCES:
        shutil.copy2(inputs.parent / name, output / name)
    shutil.copy2(Path(__file__), output / "paper_run.py")
    write(output / "profile.tbl.json", manifest)
    command = [
        str(output / "ns_incflo"),
        str(output / "inputs.paper"),
        f"ns.table_file={output / 'profile.tbl'}",
        f"stop_time={args.end:.17g}",
        f"ns.max_dt={args.max_dt:.17g}",
        f"ns.epsilon_tau_ratio={args.epsilon_tau_ratio:.17g}",
        f"ns.force_threads={args.force_threads}",
        f"ns.expected_omp_threads={args.threads}",
        f"amr.n_cell={args.base_n} {args.base_n} {args.base_n}",
        f"amr.max_level={len(args.widths)}",
        f"amr.max_grid_size={args.box}",
        f"amr.plot_per_exact={args.frame_dt:.17g}",
    ]
    if args.widths:
        command.append(
            "ns.refine_half_width=" + " ".join(f"{w:.17g}" for w in args.widths)
        )
    if args.frame_phase_step:
        command.extend(
            [
                "amr.plot_per_exact=-1",
                "amr.plot_int=1000000000",
                "ns.plot_times=" + " ".join(f"{t:.17g}" for t in planned_frames),
            ]
        )
    record = {
        "schema_version": 1,
        "kind": "finite-paper-surrogate-from-rest-incflo",
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "adapter_sha256": adapter_hash,
        "adapter_source_hashes": source_hashes,
        "binary_sha256": sha(output / "ns_incflo"),
        "checker_sha256": sha(output / "ns_archive_check"),
        "inputs_sha256": sha(output / "inputs.paper"),
        "runner_sha256": sha(output / "paper_run.py"),
        "profile_manifest": manifest,
        "execution": {
            "threads": args.threads,
            "force_threads": args.force_threads,
            "environment": thread_env,
            "library_environment": {
                key: os.environ[key]
                for key in (
                    "DYLD_LIBRARY_PATH",
                    "DYLD_FALLBACK_LIBRARY_PATH",
                    "LD_LIBRARY_PATH",
                )
                if key in os.environ
            },
        },
        "parameters": {
            "base_n": args.base_n,
            "widths": args.widths,
            "box": args.box,
            "end": args.end,
            "max_dt": args.max_dt,
            "epsilon_tau_ratio": args.epsilon_tau_ratio,
            "frame_dt": args.frame_dt,
            "frame_phase_step": args.frame_phase_step,
        },
        "planned_frames": planned_frames,
        "limits": {
            "wall_seconds": args.timeout,
            "max_rss_mib": args.max_rss_mib,
            "min_disk_free_gib": args.min_disk_free_gib,
            "poll_seconds": 5,
        },
        "command": command,
        "native_frames": [],
        "validated": False,
    }
    write(output / "run.json", record)
    start = time.monotonic()
    try:
        resource_check(output, 0, 0, args.min_disk_free_gib)
        with (output / "run.log").open("w") as log:
            process = subprocess.Popen(
                command,
                cwd=output,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=child_env,
            )
            record["pid"] = process.pid
            write(output / "run.json", record)
            try:
                code = wait_bounded(
                    process,
                    output,
                    args.timeout,
                    args.max_rss_mib,
                    args.min_disk_free_gib,
                )
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
        if code:
            raise ValueError(f"solver exited with status {code}")
        history = validate_history((output / "run.log").read_text(), record)
        record["history"] = history
        for plot in sorted(output.glob("plt[0-9]*")):
            checked = subprocess.run(
                [
                    str(output / "ns_archive_check"),
                    f"plot={plot}",
                    "ns.force=paper",
                    f"ns.table_file={output / 'profile.tbl'}",
                    f"ns.epsilon_tau_ratio={args.epsilon_tau_ratio:.17g}",
                ],
                cwd=output,
                capture_output=True,
                text=True,
                timeout=1800,
                check=False,
                env=child_env,
            )
            (output / f"{plot.name}-read.log").write_text(
                checked.stdout + checked.stderr
            )
            if checked.returncode:
                raise ValueError(f"native frame verification failed: {plot.name}")
            rows = [
                json.loads(s.removeprefix("NS_ARCHIVE_RESULT "))
                for s in checked.stdout.splitlines()
                if s.startswith("NS_ARCHIVE_RESULT ")
            ]
            if len(rows) != 1:
                raise ValueError("missing archive verification record")
            record["native_frames"].append({"path": plot.name, **rows[0]})
            write(output / "run.json", record)
        frames = record["native_frames"]
        expected_times = planned_frames
        if len(frames) != len(expected_times) or any(
            abs(f["time"] - t) > 1e-12 for f, t in zip(frames, expected_times)
        ):
            raise ValueError("native archive is missing scheduled frames from rest")
        record.update(status="completed", validated=True)
    except Exception as exc:
        record.update(status="failed", error=str(exc))
        raise
    finally:
        record["wall_seconds"] = time.monotonic() - start
        record["updated_at"] = datetime.now(UTC).isoformat()
        write(output / "run.json", record)
        print(
            json.dumps({k: record[k] for k in ("status", "validated", "wall_seconds")}),
            flush=True,
        )


if __name__ == "__main__":
    main()
