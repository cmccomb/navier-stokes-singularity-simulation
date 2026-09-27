"""Bounded dense-output probe from a pinned, from-rest native checkpoint.

The staged parent is an immutable copy of a validated running prefix, selected
checkpoint, and reference center plots. This is a checkpoint continuation, not
a second from-rest trajectory. New output events may perturb integration steps;
the center plots are compared against the original for that reason.
"""

from __future__ import annotations

import argparse
import copy
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

from scripts.paper_continue import checkpoint_clock, command, tree, validate_segment
from scripts.paper_run import (
    MARKER,
    SOURCES,
    resource_check,
    sha,
    thread_environment,
    validate_history,
    write,
)


def schedule(start: float, centers: list[float], spacing: float, original: list[float]) -> list[float]:
    if spacing <= 0 or not math.isfinite(spacing) or centers != sorted(set(centers)):
        raise ValueError("invalid dense schedule")
    dense = [center + offset * spacing for center in centers for offset in (-2, -1, 0, 1, 2)]
    end = dense[-1]
    if start >= dense[0] or not all(math.isfinite(t) for t in dense):
        raise ValueError("dense events must follow checkpoint")
    events = sorted(set(dense).union(t for t in original if start < t <= end))
    if any(b - a <= 1e-11 for a, b in zip([start, *events[:-1]], events, strict=True)):
        raise ValueError("unresolvable dense event")
    return [start, *events]


def result_rows(text: str) -> list[dict]:
    return [json.loads(line.split(" ", 1)[1]) for line in text.splitlines() if line.startswith("NS_ARCHIVE_RESULT ")]


def validate_readback(checked: dict, initial: dict) -> None:
    """Verify every state quantity available from the zero-step plot checker."""
    if (checked["step"] != initial["step"]
            or abs(checked["time"] - initial["time"]) > 1e-12
            or checked["levels"] != initial["levels"]
            or checked["stored_cells"] != initial["stored_cells"]
            or abs(checked["composite_volume"] - initial["volume"]) > 1e-10
            or abs(checked["peak_speed"] - initial["peak_speed"]) > 1e-9):
        raise ValueError("restart readback differs from parent diagnostic")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True, help="staged immutable parent prefix and native assets")
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--center-indices", type=int, nargs="+", required=True)
    parser.add_argument("--spacing", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-rss-mib", type=float, required=True)
    parser.add_argument("--min-disk-free-gib", type=float, required=True)
    parser.add_argument("--timeout-hours", type=float, required=True)
    parser.add_argument("--check-only", action="store_true", help="validate staged inputs and storage without starting a child")
    args = parser.parse_args()
    if not all(math.isfinite(x) and x > 0 for x in (args.max_rss_mib, args.min_disk_free_gib, args.timeout_hours)):
        parser.error("resource limits must be finite and positive")
    parent = args.parent.resolve(strict=True)
    output = args.output.resolve()
    if output.exists() or parent == output or parent in output.parents or output in parent.parents:
        parser.error("require a new separate output directory")
    record_bytes = (parent / "run.json").read_bytes()
    log_bytes = (parent / "run.log").read_bytes()
    record = json.loads(record_bytes)
    if record["kind"] != "finite-paper-surrogate-from-rest-incflo" or record["status"] not in ("running", "completed"):
        raise ValueError("require an original from-rest parent prefix")
    for name, digest in record["adapter_source_hashes"].items():
        if name not in SOURCES or sha(parent / name) != digest:
            raise ValueError("adapter source differs")
    for name, digest in (("ns_incflo", record["binary_sha256"]), ("ns_archive_check", record["checker_sha256"]), ("inputs.paper", record["inputs_sha256"]), ("profile.tbl", record["profile_manifest"]["sha256"])):
        if sha(parent / name) != digest:
            raise ValueError(f"parent asset differs: {name}")
    band = record["parameters"].get("revolved_refinement")
    if band and sha(parent / band["path"]) != band["sha256"]:
        raise ValueError("refinement band differs")
    audit = copy.deepcopy(record)
    markers = [json.loads(line[len(MARKER):]) for line in log_bytes.decode().splitlines() if line.startswith(MARKER)]
    audit["parameters"]["end"] = markers[-1]["time"]
    history = validate_history(log_bytes.decode(), audit)
    if not 0 < args.checkpoint_step < len(history):
        raise ValueError("checkpoint missing from staged diagnostic prefix")
    initial = history[args.checkpoint_step]
    checkpoint = parent / f"chk{args.checkpoint_step:05d}"
    if checkpoint_clock(checkpoint) != (initial["step"], initial["time"], initial["levels"]):
        raise ValueError("checkpoint clock differs from from-rest prefix")
    checkpoint_files = tree(checkpoint)
    if args.center_indices != sorted(set(args.center_indices)):
        raise ValueError("center indices must be unique and ordered")
    centers = [record["planned_frames"][index] for index in args.center_indices]
    times = schedule(initial["time"], centers, args.spacing, record["planned_frames"])
    if times[-1] >= record["profile_manifest"]["parameters"]["t_star"]:
        raise ValueError("probe reaches singular time")
    references = {}
    for center in centers:
        matches = [row for row in history if abs(row["time"] - center) <= 1e-12]
        if len(matches) != 1:
            raise ValueError("parent lacks a center diagnostic")
        plot = parent / f"plt{matches[0]['step']:05d}"
        if not (plot / "Header").is_file():
            raise ValueError("parent lacks a center native plot")
        references[center] = plot
    # A source plot gives a conservative storage estimate for each new frame.
    plot_bytes = max(sum(p.stat().st_size for p in plot.rglob("*") if p.is_file()) for plot in references.values())
    required = (len(times) + 2) * plot_bytes + 2 * sum(row["bytes"] for row in checkpoint_files.values())
    if shutil.disk_usage(parent).free - required < args.min_disk_free_gib * 2**30:
        raise RuntimeError("dense probe would violate disk reserve")
    if args.check_only:
        print(json.dumps({"checkpoint_step": args.checkpoint_step, "checkpoint_time": initial["time"], "end": times[-1], "native_frames": len(times) - 1, "storage_budget_bytes": required, "source_binary_sha256": record["binary_sha256"]}))
        return
    output.mkdir(parents=True)
    receipt = {
        "schema_version": 1,
        "kind": "dense-output-checkpoint-probe",
        "status": "prepared",
        "parent_record_sha256": hashlib.sha256(record_bytes).hexdigest(),
        "parent_log_sha256": hashlib.sha256(log_bytes).hexdigest(),
        "source_binary_sha256": record["binary_sha256"],
        "source_profile_sha256": record["profile_manifest"]["sha256"],
        "checkpoint_step": args.checkpoint_step,
        "checkpoint_time": initial["time"],
        "checkpoint_files": checkpoint_files,
        "centers": centers,
        "spacing": args.spacing,
        "planned_times": times[1:],
        "reference_plots": {f"{t:.17g}": str(p) for t, p in references.items()},
        "storage_budget_bytes": required,
        "limits": {"max_rss_mib": args.max_rss_mib, "min_disk_free_gib": args.min_disk_free_gib, "timeout_hours": args.timeout_hours},
        "runner_sha256": sha(Path(__file__)),
        "scope": __doc__.strip(),
        "validated": False,
    }

    def save() -> None:
        receipt["updated_at"] = datetime.now(UTC).isoformat()
        write(output / "probe.json", receipt)

    def interrupted(signum: int, _frame: object) -> None:
        raise RuntimeError(f"dense probe supervisor received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    save()
    env = {**os.environ, **record["execution"].get("library_environment", {}), **thread_environment(record["execution"]["threads"], record["execution"]["force_threads"])}

    def run(argv: list[str], directory: Path, name: str) -> str:
        with (directory / name).open("x") as stream:
            child = subprocess.Popen(argv, cwd=directory, env=env, stdout=stream, stderr=subprocess.STDOUT)
            receipt["active_child_pid"] = child.pid
            save()
            started = time.monotonic()
            try:
                while child.poll() is None:
                    if time.monotonic() - started > args.timeout_hours * 3600:
                        raise TimeoutError("dense probe wall limit reached")
                    resource_check(output, child.pid, args.max_rss_mib, args.min_disk_free_gib)
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
                if child.returncode:
                    raise RuntimeError(f"child exited {child.returncode}; see {directory / name}")
            finally:
                if child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait(timeout=10)
                receipt.pop("active_child_pid", None)
                save()
        return (directory / name).read_text()

    def check(plot: Path, name: str, compare: Path | None = None) -> dict:
        argv = [str(parent / "ns_archive_check"), f"plot={plot}", "ns.force=paper", f"ns.table_file={parent / 'profile.tbl'}", f"ns.epsilon_tau_ratio={record['parameters']['epsilon_tau_ratio']:.17g}"]
        if compare is not None:
            argv += [f"compare={compare}", "report_difference=1"]
        rows = result_rows(run(argv, output, name))
        if len(rows) != 1 or rows[0]["force_linf_error"] != 0:
            raise ValueError("native force audit failed")
        return rows[0]

    try:
        receipt["status"] = "restart_readback"
        save()
        readback = output / "restart-readback"
        readback.mkdir()
        run(command(record, parent, checkpoint, initial["time"], [initial["time"]], record["parameters"]["max_dt"], readback=True), readback, "run.log")
        restored = readback / f"plt{initial['step']:05d}"
        checked = check(restored, "restart-readback-check.log")
        validate_readback(checked, initial)
        receipt["restart_readback"] = checked
        receipt["status"] = "running"
        save()
        dense = output / "dense"
        dense.mkdir()
        argv = command(record, parent, checkpoint, times[-1], times, record["parameters"]["max_dt"])
        receipt["command"] = argv
        save()
        segment = validate_segment(run(argv, dense, "run.log"), record, initial, times[-1], record["parameters"]["max_dt"])
        receipt["segment_steps"] = len(segment)
        receipt["status"] = "auditing"
        save()
        plots = sorted(p for p in dense.glob("plt[0-9]*") if p.is_dir())
        if len(plots) != len(times) - 1:
            raise ValueError("missing or extra dense native frames")
        audited = []
        by_time = {}
        for plot, expected in zip(plots, times[1:], strict=True):
            row = check(plot, plot.name + "-read.log")
            if abs(row["time"] - expected) > 1e-12:
                raise ValueError("dense native frame time differs")
            audited.append({"path": str(plot), **row})
            by_time[expected] = plot
            receipt["native_frames"] = audited
            save()
        comparisons = []
        for center, reference in references.items():
            plot = min(by_time, key=lambda t: abs(t - center))
            if abs(plot - center) > 1e-12:
                raise ValueError("missing center frame")
            row = check(by_time[plot], f"center-{center:.6f}-compare.log", reference)
            if not row["compared"] or not row["report_difference"]:
                raise ValueError("center comparison missing")
            comparisons.append({"time": center, "reference": str(reference), **row})
        receipt.update(status="completed", validated=True, center_comparisons=comparisons, final_step=segment[-1]["step"], final_time=segment[-1]["time"])
        save()
        print(json.dumps({"status": "completed", "frames": len(audited), "center_comparisons": len(comparisons)}), flush=True)
    except Exception as exc:
        receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save()
        raise


if __name__ == "__main__":
    main()
