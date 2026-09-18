"""Finite, gated checkpoint continuation; never relabel a restart as fresh rest.

Wait for one parent run, verify its final checkpoint, compare a short tail with
smaller integration steps and unchanged force, then extend the baseline branch.
This is a local sensitivity screen, not spatial/temporal convergence evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from scripts.paper_run import (
    MARKER,
    SOURCES,
    resource_check,
    thread_environment,
    validate_history,
    write,
)


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tree(path: Path) -> dict:
    if path.is_symlink() or not path.is_dir():
        raise ValueError("checkpoint must be a real directory")
    files = {}
    for item in sorted(path.rglob("*")):
        if item.is_symlink() or not (item.is_file() or item.is_dir()):
            raise ValueError("non-regular checkpoint entry")
        if item.is_file():
            files[str(item.relative_to(path))] = {
                "bytes": item.stat().st_size,
                "sha256": sha(item),
            }
    if "Header" not in files or len(files) < 5:
        raise ValueError("incomplete checkpoint tree")
    return files


def checkpoint_clock(path: Path) -> tuple[int, float, int]:
    lines = (path / "Header").read_text().splitlines()
    if lines[0] != "Checkpoint version: 1":
        raise ValueError("unsupported checkpoint format")
    levels, step, t = int(lines[1]) + 1, int(lines[2]), float(lines[3])
    if step < 1 or levels < 1 or not math.isfinite(t):
        raise ValueError("invalid checkpoint clock")
    return step, t, levels


def continuation_times(
    start: float, end: float, period: float, phase: float, clock: dict
) -> list[float]:
    if not all(math.isfinite(v) for v in (start, end, period, phase)) or not (
        clock["paper_time_cutoff_start"] < start < end < clock["t_star"]
        and period > 0
        and phase > 0
        and clock["forcing_log_rate_bound"] > 0
    ):
        raise ValueError("invalid continuation clock")
    times = [start]
    while times[-1] < end:
        t = times[-1]
        dt = min(
            period,
            end - t,
            -(clock["t_star"] - t)
            * math.expm1(-phase / clock["forcing_log_rate_bound"]),
        )
        next_t = min(t + dt, end)
        if end - next_t < 1e-13:
            next_t = end
        if next_t - t <= 1e-11 or len(times) > 50000:
            raise ValueError("unresolvable continuation schedule")
        times.append(next_t)
    return times


def verify_parent(parent: Path) -> tuple[dict, list[dict], Path]:
    record = json.loads((parent / "run.json").read_text())
    if record.get("status") != "completed" or record.get("validated") is not True:
        raise ValueError("parent must have completed its native archive audit")
    if record.get("kind") != "finite-paper-surrogate-from-rest-incflo":
        raise ValueError("this runner requires an original from-rest parent")
    hashes = {name: sha(parent / name) for name in SOURCES}
    if (
        hashes != record["adapter_source_hashes"]
        or hashlib.sha256(";".join(hashes.values()).encode()).hexdigest()
        != record["adapter_sha256"]
    ):
        raise ValueError("parent adapter changed")
    for name, expected in (
        ("ns_incflo", record["binary_sha256"]),
        ("ns_archive_check", record["checker_sha256"]),
        ("inputs.paper", record["inputs_sha256"]),
        ("profile.tbl", record["profile_manifest"]["sha256"]),
    ):
        if sha(parent / name) != expected:
            raise ValueError(f"parent asset changed: {name}")
    refinement = record["parameters"].get("revolved_refinement")
    if refinement is not None and (
        refinement["path"] != "refinement.bands"
        or sha(parent / "refinement.bands") != refinement["sha256"]
    ):
        raise ValueError("parent revolved refinement changed")
    history = validate_history((parent / "run.log").read_text(), record)
    frames = record["native_frames"]
    expected = record["planned_frames"]
    if len(frames) != len(expected) or any(
        abs(f["time"] - t) > 1e-12 for f, t in zip(frames, expected)
    ):
        raise ValueError("parent frame audit is incomplete")
    last = history[-1]
    checkpoint = parent / f"chk{last['step']:05d}"
    if checkpoint_clock(checkpoint) != (last["step"], last["time"], last["levels"]):
        raise ValueError("final checkpoint does not match parent diagnostics")
    return record, history, checkpoint


def command(
    record: dict,
    bundle: Path,
    checkpoint: Path,
    end: float,
    times: list[float],
    max_dt: float,
    *,
    readback: bool = False,
) -> list[str]:
    # Reuse every original runtime setting, replacing only explicit controls.
    args = {}
    for item in record["command"][2:]:
        key, value = item.split("=", 1)
        # paper_run deliberately overrides its initial regular-output period
        # when phase-spaced output is enabled. Preserve that known last value.
        phase_output_override = (
            key == "amr.plot_per_exact"
            and value == "-1"
            and key in args
            and float(args[key]) > 0
        )
        if key in args and not phase_output_override:
            raise ValueError("ambiguous duplicate parent runtime parameter")
        args[key] = value
    args.update(
        {
            "ns.table_file": str(bundle / "profile.tbl"),
            "amr.restart": str(checkpoint),
            "stop_time": f"{end:.17g}",
            "ns.max_dt": f"{max_dt:.17g}",
            "amr.plot_per_exact": "-1",
            "amr.plot_int": "1000000000",
            "amr.plotfile_on_restart": "0",
            "amr.check_int": "-1" if readback else "250",
            "max_step": "0" if readback else "10000000",
            "ns.plot_times": " ".join(f"{t:.17g}" for t in times),
        }
    )
    if "ns.refine_rz_file" in args:
        args["ns.refine_rz_file"] = str(bundle / "refinement.bands")
    return [str(bundle / "ns_incflo"), str(bundle / "inputs.paper")] + [
        key + "=" + value for key, value in args.items()
    ]


def validate_segment(
    log: str, record: dict, initial: dict, end: float, max_dt: float
) -> list[dict]:
    rows = [
        json.loads(x[len(MARKER) :]) for x in log.splitlines() if x.startswith(MARKER)
    ]
    if not rows:
        raise ValueError("continuation produced no diagnostic steps")
    previous = initial
    clock = record["profile_manifest"]["parameters"]
    for row in rows:
        for key in (
            "time",
            "dt",
            "peak_speed",
            "energy",
            "l2_error",
            "linf_error",
            "volume",
            "peak_rss_mib",
        ):
            if not math.isfinite(row[key]):
                raise ValueError("nonfinite continuation diagnostic")
        if (
            row["step"] != previous["step"] + 1
            or row["dt"] <= 0
            or abs(row["time"] - previous["time"] - row["dt"]) > 2e-14
        ):
            raise ValueError("continuation clock/step discontinuity")
        for key in ("adapter_sha256", "case", "levels", "stored_cells", "active_cells"):
            if row[key] != initial[key]:
                raise ValueError(f"continuation changed {key}")
        if abs(row["volume"] - 8) > 1e-10 or row["time"] >= clock["t_star"]:
            raise ValueError("invalid continuation domain/time")
        if row["dt"] > max_dt * (1 + 1e-10):
            raise ValueError("integration ceiling exceeded")
        phase = clock["forcing_log_rate_bound"] * math.log1p(
            row["dt"] / (clock["t_star"] - row["time"])
        )
        if phase > clock["forcing_phase_step"] * (1 + 1e-9):
            raise ValueError("forcing phase ceiling exceeded")
        if row["openmp_max_threads"] != record["execution"]["threads"] or (
            row["force_threads_limit"] != record["execution"]["force_threads"]
        ):
            raise ValueError("thread limits changed")
        previous = row
    if abs(rows[-1]["time"] - end) > 2e-14:
        raise ValueError("continuation did not reach endpoint")
    return rows


def temporal_gate(comparison: dict, l2_limit: float, linf_limit: float) -> dict:
    if not comparison.get("compared") or not comparison.get("report_difference"):
        raise ValueError("missing native temporal comparison")
    if comparison["reference_resolution_ratio"] != 1:
        raise ValueError("temporal comparison changed mesh")
    if not all(
        math.isfinite(x) and x > 0
        for x in (
            comparison["reference_velocity_l2"],
            comparison["peak_speed"],
            l2_limit,
            linf_limit,
        )
    ):
        raise ValueError("invalid temporal normalization or gate")
    l2 = comparison["velocity_l2_difference"] / comparison["reference_velocity_l2"]
    # The checker reports the baseline full-field peak; this is a stated scale,
    # not a pointwise relative error at near-zero velocity.
    linf = comparison["velocity_linf_difference"] / comparison["peak_speed"]
    if not all(math.isfinite(x) and x >= 0 for x in (l2, linf)):
        raise ValueError("invalid temporal sensitivity")
    return {
        "relative_composite_l2": l2,
        "linf_over_baseline_peak": linf,
        "l2_limit": l2_limit,
        "linf_limit": linf_limit,
        "passed": l2 <= l2_limit and linf <= linf_limit,
        "scope": "Short shared-checkpoint sensitivity only; inherited errors remain.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--end", type=float, default=0.9975)
    parser.add_argument("--probe-phase", type=float, default=0.75)
    parser.add_argument("--wait-parent", action="store_true")
    parser.add_argument(
        "--archive-receipt",
        type=Path,
        help="also wait for a verified checkpoint-offload receipt",
    )
    parser.add_argument("--parent-wait-hours", type=float, default=12)
    parser.add_argument("--stage-hours", type=float, default=24)
    parser.add_argument("--relative-l2-limit", type=float, default=0.001)
    parser.add_argument("--relative-linf-limit", type=float, default=0.01)
    args = parser.parse_args()
    if not all(
        math.isfinite(v) and v > 0
        for v in (
            args.end,
            args.probe_phase,
            args.parent_wait_hours,
            args.stage_hours,
            args.relative_l2_limit,
            args.relative_linf_limit,
        )
    ):
        parser.error("limits must be finite and positive")
    parent = args.parent.resolve(strict=True)
    output = args.output.resolve()
    if output == parent or output in parent.parents or parent in output.parents:
        parser.error("continuation must be separate from the original run")
    output.mkdir(parents=True, exist_ok=False)
    runner = output / "runner" / "scripts"
    runner.mkdir(parents=True)
    runner_hashes = {}
    for name in ("__init__.py", "paper_run.py", "paper_continue.py"):
        shutil.copy2(Path(__file__).parent / name, runner / name)
        runner_hashes[name] = sha(runner / name)
    receipt = {
        "schema_version": 1,
        "kind": "checkpoint-continuation-of-from-rest",
        "status": "waiting_parent",
        "parent": str(parent),
        "end": args.end,
        "started_at": datetime.now(UTC).isoformat(),
        "pid": os.getpid(),
        "runner_sha256": sha(Path(__file__)),
        "runner_source_hashes": runner_hashes,
        "stages": {},
        "scope": "Finite grid-dependent surrogate; no singularity/convergence certificate.",
    }

    def save():
        receipt["updated_at"] = datetime.now(UTC).isoformat()
        write(output / "continuation.json", receipt)

    def interrupted(signum, frame):
        raise RuntimeError(f"continuation supervisor received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    save()
    try:
        deadline = time.monotonic() + args.parent_wait_hours * 3600
        while True:
            record = json.loads((parent / "run.json").read_text())
            receipt["parent_status"] = record["status"]
            save()
            archive_ready = args.archive_receipt is None
            if args.archive_receipt is not None and args.archive_receipt.is_file():
                archived = json.loads(args.archive_receipt.read_text())
                if archived.get("source") != str(parent):
                    raise ValueError("archive receipt belongs to another parent")
                archive_ready = archived.get("status") == "completed"
                receipt["archive_status"] = archived.get("status")
                if archive_ready and (
                    not archived.get("checkpoints")
                    or any(
                        not item.get("verified")
                        for item in archived["checkpoints"].values()
                    )
                ):
                    raise ValueError("archive completion lacks verified copies")
            if (
                record["status"] == "completed"
                and record.get("validated") is True
                and archive_ready
            ):
                break
            if record["status"] not in ("running", "validating", "completed"):
                raise ValueError("parent failed or has an unsupported status")
            if not args.wait_parent or time.monotonic() >= deadline:
                raise TimeoutError(
                    "parent validation/archive is not ready; original run unchanged"
                )
            time.sleep(30)

        record, history, checkpoint = verify_parent(parent)
        initial = history[-1]
        clock = record["profile_manifest"]["parameters"]
        times = continuation_times(
            initial["time"],
            args.end,
            record["parameters"]["frame_dt"],
            record["parameters"]["frame_phase_step"],
            clock,
        )
        probe_end = continuation_times(
            initial["time"],
            args.end,
            record["parameters"]["frame_dt"],
            args.probe_phase,
            clock,
        )[1]
        # Keep the production movie schedule intact: the probe ends on its first
        # new saved event. A shorter probe is allowed only for small test fixtures.
        if abs(probe_end - times[1]) > 1e-12:
            times = [times[0], probe_end] + [
                t for t in times[1:] if t > probe_end + 1e-12
            ]
        if probe_end >= args.end:
            raise ValueError("probe must leave a nonempty extension")
        max_dt = record["parameters"]["max_dt"]
        phase_dt_end = -(clock["t_star"] - probe_end) * math.expm1(
            -clock["forcing_phase_step"] / clock["forcing_log_rate_bound"]
        )
        fine_dt = min(max_dt, phase_dt_end) / 2
        limits = record["limits"]
        bundle = output / "bundle"
        bundle.mkdir()
        for name in (
            *SOURCES,
            "ns_incflo",
            "ns_archive_check",
            "inputs.paper",
            "profile.tbl",
        ):
            shutil.copy2(parent / name, bundle / name)
        if record["parameters"].get("revolved_refinement") is not None:
            shutil.copy2(parent / "refinement.bands", bundle / "refinement.bands")
        write(bundle / "parent-run.json", record)
        checkpoint_files = tree(checkpoint)
        pinned_checkpoint = bundle / checkpoint.name
        shutil.copytree(checkpoint, pinned_checkpoint, copy_function=os.link)
        if tree(pinned_checkpoint) != checkpoint_files:
            raise ValueError("pinned checkpoint differs")
        write(bundle / "checkpoint-manifest.json", checkpoint_files)
        assets = {p.name: sha(p) for p in bundle.iterdir() if p.is_file()}
        parent_plot = parent / record["native_frames"][-1]["path"]
        plot_bytes = sum(
            p.stat().st_size for p in parent_plot.rglob("*") if p.is_file()
        )
        checkpoint_bytes = sum(v["bytes"] for v in checkpoint_files.values())
        estimated_steps = math.ceil(
            clock["forcing_log_rate_bound"]
            / clock["forcing_phase_step"]
            * math.log(
                (clock["t_star"] - initial["time"]) / (clock["t_star"] - args.end)
            )
        )
        # Conservative allowance for zero-step readback, probe comparison,
        # periodic/final checkpoints, helper logs and native audit receipts.
        replay_candidates = []
        for candidate in parent.glob("chk[0-9]*"):
            if (
                candidate.name[3:].isdigit()
                and 0 < int(candidate.name[3:]) < initial["step"]
            ):
                step, t, levels = checkpoint_clock(candidate)
                if (
                    initial["step"] - step <= 250
                    and levels == initial["levels"]
                    and t == history[step]["time"]
                ):
                    replay_candidates.append(candidate)
        replay_checkpoint = (
            max(replay_candidates, key=lambda p: int(p.name[3:]))
            if replay_candidates
            else None
        )
        replay_times = []
        if replay_checkpoint is not None:
            replay_step, replay_time, _ = checkpoint_clock(replay_checkpoint)
            replay_times = [replay_time] + [
                t for t in record["planned_frames"] if t > replay_time + 1e-12
            ]
        required = (
            (len(times) + 4 + max(0, len(replay_times) - 1)) * plot_bytes
            + (math.ceil(estimated_steps / 250) + 4 + bool(replay_times))
            * checkpoint_bytes
            + 1024**3
        )
        if (
            shutil.disk_usage(output).free - required
            < limits["min_disk_free_gib"] * 1024**3
        ):
            raise RuntimeError("insufficient storage for all frames plus disk reserve")
        receipt.update(
            status="preflight",
            parent_record_sha256=sha(parent / "run.json"),
            checkpoint=str(checkpoint),
            checkpoint_manifest_sha256=sha(bundle / "checkpoint-manifest.json"),
            bundle_sha256=assets,
            planned_new_frame_times=times[1:],
            coarse_max_dt=max_dt,
            fine_max_dt=fine_dt,
            force_definition_unchanged=True,
            storage_budget_bytes=required,
            limits=limits,
            probe_end=probe_end,
        )
        env = {
            **os.environ,
            **record["execution"].get("library_environment", {}),
            **thread_environment(
                record["execution"]["threads"], record["execution"]["force_threads"]
            ),
        }
        save()

        def run(argv: list[str], directory: Path, log_name: str = "run.log"):
            for name, expected in assets.items():
                if sha(bundle / name) != expected:
                    raise ValueError("pinned continuation asset changed")
            with (directory / log_name).open("x") as log:
                p = subprocess.Popen(
                    argv, cwd=directory, env=env, stdout=log, stderr=subprocess.STDOUT
                )
                receipt["active_child_pid"] = p.pid
                save()
                start = time.monotonic()
                try:
                    while p.poll() is None:
                        if time.monotonic() - start > args.stage_hours * 3600:
                            raise TimeoutError("continuation stage time limit")
                        resource_check(
                            output,
                            p.pid,
                            limits["max_rss_mib"],
                            limits["min_disk_free_gib"],
                        )
                        save()
                        try:
                            p.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            pass
                    if p.returncode:
                        raise RuntimeError(
                            f"child exited {p.returncode}; see {directory / log_name}"
                        )
                finally:
                    if p.poll() is None:
                        p.terminate()
                        try:
                            p.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            p.kill()
                            p.wait(timeout=10)
                    receipt.pop("active_child_pid", None)
                    save()
            return (directory / log_name).read_text()

        def check(
            plot: Path,
            directory: Path,
            name: str,
            compare: Path | None = None,
            difference: bool = False,
        ) -> dict:
            argv = [
                str(bundle / "ns_archive_check"),
                f"plot={plot}",
                "ns.force=paper",
                f"ns.table_file={bundle / 'profile.tbl'}",
                f"ns.epsilon_tau_ratio={record['parameters']['epsilon_tau_ratio']:.17g}",
            ]
            if compare is not None:
                argv.append(f"compare={compare}")
            if difference:
                argv.append("report_difference=1")
            text = run(argv, directory, name)
            rows = [
                json.loads(s[len("NS_ARCHIVE_RESULT ") :])
                for s in text.splitlines()
                if s.startswith("NS_ARCHIVE_RESULT ")
            ]
            if len(rows) != 1 or rows[0]["force_linf_error"] > 1e-12:
                raise ValueError("native force audit failed")
            if (
                rows[0]["levels"] != initial["levels"]
                or rows[0]["stored_cells"] != initial["stored_cells"]
            ):
                raise ValueError("native mesh changed")
            return rows[0]

        readback = output / "restart-readback"
        readback.mkdir()
        receipt["status"] = "checking_restart"
        save()
        run(
            command(
                record,
                bundle,
                pinned_checkpoint,
                initial["time"],
                [initial["time"]],
                max_dt,
                readback=True,
            ),
            readback,
        )
        restored = readback / f"plt{initial['step']:05d}"
        result = check(restored, readback, "compare.log", parent_plot)
        if result["velocity_linf_difference"] > 1e-12 or not result["compared"]:
            raise ValueError("restart did not recover the original full velocity field")
        receipt["restart_check"] = result
        save()

        def segment(
            name: str,
            start_checkpoint: Path,
            start_row: dict,
            frame_times: list[float],
            ceiling: float,
        ):
            directory = output / name
            directory.mkdir()
            argv = command(
                record, bundle, start_checkpoint, frame_times[-1], frame_times, ceiling
            )
            stage = {
                "status": "running",
                "command": argv,
                "start": start_row,
                "max_dt": ceiling,
                "planned_frames": frame_times[1:],
                "native_frames": [],
            }
            receipt["stages"][name] = stage
            receipt["status"] = name
            save()
            rows = validate_segment(
                run(argv, directory), record, start_row, frame_times[-1], ceiling
            )
            stage.update(status="validating", history=rows)
            save()
            plots = sorted(
                p
                for p in directory.glob("plt[0-9]*")
                if p.is_dir() and p.name[3:].isdigit()
            )
            if len(plots) != len(frame_times) - 1:
                raise ValueError("missing or extra continuation movie frames")
            for plot, t in zip(plots, frame_times[1:]):
                checked = check(plot, directory, plot.name + "-read.log")
                if abs(checked["time"] - t) > 1e-12:
                    raise ValueError("continuation frame time mismatch")
                stage["native_frames"].append({"path": str(plot), **checked})
                save()
            end_checkpoint = directory / f"chk{rows[-1]['step']:05d}"
            if checkpoint_clock(end_checkpoint) != (
                rows[-1]["step"],
                rows[-1]["time"],
                rows[-1]["levels"],
            ):
                raise ValueError("continuation final checkpoint mismatch")
            stage.update(
                status="completed", validated=True, final_checkpoint=str(end_checkpoint)
            )
            save()
            return rows[-1], end_checkpoint, plots[-1]

        if replay_checkpoint is not None:
            replay_files = tree(replay_checkpoint)
            replay_pinned = bundle / replay_checkpoint.name
            shutil.copytree(replay_checkpoint, replay_pinned, copy_function=os.link)
            if tree(replay_pinned) != replay_files:
                raise ValueError("pinned replay checkpoint differs")
            write(output / "replay-checkpoint-manifest.json", replay_files)
            _, _, replay_plot = segment(
                "restart-replay",
                replay_pinned,
                history[replay_step],
                replay_times,
                max_dt,
            )
            receipt["restart_replay_check"] = check(
                replay_plot, output, "restart-replay-comparison.log", parent_plot
            )
            save()
        else:
            receipt["restart_replay_check"] = {
                "available": False,
                "reason": "No recent noninitial checkpoint; zero-step full-field recovery was checked.",
            }
            save()
        baseline_last, baseline_checkpoint, baseline_plot = segment(
            "probe-baseline",
            pinned_checkpoint,
            initial,
            [initial["time"], probe_end],
            max_dt,
        )
        fine_last, _, fine_plot = segment(
            "probe-smaller-dt",
            pinned_checkpoint,
            initial,
            [initial["time"], probe_end],
            fine_dt,
        )
        if (
            fine_last["step"] - initial["step"]
            <= baseline_last["step"] - initial["step"]
        ):
            raise ValueError("smaller-dt probe did not actually refine integration")
        comparison = check(
            baseline_plot, output, "temporal-comparison.log", fine_plot, True
        )
        gate = temporal_gate(
            comparison, args.relative_l2_limit, args.relative_linf_limit
        )
        receipt["temporal_comparison"] = {**gate, "native": comparison}
        save()
        if not gate["passed"]:
            raise ValueError(
                "short-tail temporal sensitivity exceeds continuation gates"
            )
        tail_times = [probe_end] + [t for t in times[1:] if t > probe_end + 1e-12]
        segment("extension", baseline_checkpoint, baseline_last, tail_times, max_dt)
        frames = [
            {"path": str(parent / f["path"]), "time": f["time"], "source": "parent"}
            for f in record["native_frames"]
        ]
        for name in ("probe-baseline", "extension"):
            frames.extend(
                {"path": f["path"], "time": f["time"], "source": name}
                for f in receipt["stages"][name]["native_frames"]
            )
        expected = record["planned_frames"] + times[1:]
        if len(frames) != len(expected) or any(
            abs(f["time"] - t) > 1e-12 for f, t in zip(frames, expected)
        ):
            raise ValueError("combined from-rest lineage has a gap")
        write(
            output / "combined-frames.json",
            {
                "kind": "validated-checkpoint-continuation-of-from-rest",
                "frames": frames,
                "starts_at_rest": frames[0]["time"] == 0,
                "through": args.end,
                "excluded_comparison_branch": "probe-smaller-dt",
                "partial": False,
            },
        )
        receipt.update(
            status="completed",
            validated=True,
            combined_frames=len(frames),
            completed_at=datetime.now(UTC).isoformat(),
        )
    except BaseException as exc:
        receipt.update(
            status="failed", validated=False, error=f"{type(exc).__name__}: {exc}"
        )
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
