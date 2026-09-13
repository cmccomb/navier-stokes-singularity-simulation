"""Remove a live Mac paper run's wall deadline without restarting its solver.

Explicit adoption only: preserve the original record, retain positive RAM/disk
guards, and replace the exact parent supervisor after registering kernel exit
status observation. Never signal a process group or an identity-reused PID.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import select
import shutil
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from scripts.paper_run import (
    MARKER,
    SOURCES,
    audit_completed,
    resource_check,
    sha,
    thread_environment,
    validate_history,
    write,
)

# Darwin sys/event.h: permitted for children or targets we may signal. Python
# exposes NOTE_EXIT but not NOTE_EXITSTATUS; EXIT alone does not return status.
NOTE_EXITSTATUS = 0x04000000


def process_info(pid: int) -> dict | None:
    result = subprocess.run(
        ["ps", "-ww", "-p", str(pid), "-o", "pid=,ppid=,uid=,lstart=,state=,command="],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
        env={**os.environ, "LC_ALL": "C"},
    )
    if result.returncode == 1 and not result.stdout.strip():
        return None
    if result.returncode or not result.stdout.strip():
        raise RuntimeError("could not inspect process identity")
    fields = result.stdout.strip().split(None, 9)
    if len(fields) != 10:
        raise RuntimeError("unrecognized ps identity")
    return {
        "pid": int(fields[0]),
        "ppid": int(fields[1]),
        "uid": int(fields[2]),
        "started": " ".join(fields[3:8]),
        "state": fields[8],
        "command": fields[9],
    }


def same_process(current: dict | None, expected: dict) -> bool:
    return bool(current) and all(
        current[key] == expected[key] for key in ("pid", "uid", "started", "command")
    )


def require_process(expected: dict) -> dict:
    current = process_info(expected["pid"])
    if not same_process(current, expected) or current["state"].startswith("Z"):
        raise RuntimeError("process identity changed or process exited")
    return current


class ExitObserver:
    def __init__(self, pid: int):
        if sys.platform != "darwin":
            raise RuntimeError("live adoption requires Darwin kqueue exit status")
        self.pid = pid
        self.queue = select.kqueue()
        try:
            self.queue.control(
                [
                    select.kevent(
                        pid,
                        filter=select.KQ_FILTER_PROC,
                        flags=select.KQ_EV_ADD | select.KQ_EV_ONESHOT,
                        fflags=select.KQ_NOTE_EXIT | NOTE_EXITSTATUS,
                    )
                ],
                0,
                0,
            )
        except BaseException:
            self.queue.close()
            raise

    def poll(self, seconds: float) -> int | None:
        events = self.queue.control(None, 1, seconds)
        if not events:
            return None
        event = events[0]
        if (
            event.ident != self.pid
            or event.flags & select.KQ_EV_ERROR
            or not event.fflags & NOTE_EXITSTATUS
        ):
            raise RuntimeError("missing kernel solver exit status")
        return os.waitstatus_to_exitcode(event.data)

    def close(self) -> None:
        self.queue.close()


def retire_parent(old: dict, solver: dict, receipt: dict, prepared) -> None:
    """Roll back a pre-retirement failure; only the exact supervisor is retired."""
    if require_process(solver)["ppid"] != old["pid"]:
        raise RuntimeError("solver is not a child of the selected supervisor")
    if "T" in require_process(old)["state"]:
        raise RuntimeError("selected supervisor was already stopped")
    stopped = False
    try:
        os.kill(old["pid"], signal.SIGSTOP)
        stopped = True
        deadline = time.monotonic() + 5
        while "T" not in require_process(old)["state"]:
            if time.monotonic() >= deadline:
                raise RuntimeError("supervisor did not stop for handoff")
            time.sleep(0.02)
        require_process(solver)
        prepared()
        require_process(old)
        # A normal exception cleanup in the old runner can terminate its child.
        # Retire only the frozen parent, never the solver or its process group.
        os.kill(old["pid"], signal.SIGKILL)
        receipt["old_supervisor_retired"] = True
    finally:
        if (
            stopped
            and not receipt.get("old_supervisor_retired")
            and same_process(process_info(old["pid"]), old)
        ):
            os.kill(old["pid"], signal.SIGCONT)


def stop_exact_solver(solver: dict, observer: ExitObserver) -> None:
    current = process_info(solver["pid"])
    if not same_process(current, solver) or current["state"].startswith("Z"):
        return
    os.kill(solver["pid"], signal.SIGTERM)
    if observer.poll(15) is None and same_process(process_info(solver["pid"]), solver):
        os.kill(solver["pid"], signal.SIGKILL)
        observer.poll(15)


def verify_assets(output: Path, record: dict) -> None:
    hashes = {name: sha(output / name) for name in SOURCES}
    if (
        hashes != record["adapter_source_hashes"]
        or hashlib.sha256(";".join(hashes.values()).encode()).hexdigest()
        != record["adapter_sha256"]
    ):
        raise ValueError("adapter provenance mismatch")
    for name, expected in (
        ("ns_incflo", record["binary_sha256"]),
        ("ns_archive_check", record["checker_sha256"]),
        ("inputs.paper", record["inputs_sha256"]),
        ("profile.tbl", record["profile_manifest"]["sha256"]),
        ("paper_run.py", record["runner_sha256"]),
    ):
        if sha(output / name) != expected:
            raise ValueError(f"pinned dependency changed: {name}")


def monitor(
    output: Path, solver: dict, observer: ExitObserver, limits: dict, heartbeat
) -> int:
    while True:
        code = observer.poll(5)
        if code is not None:
            return code
        require_process(solver)
        resource_check(
            output, solver["pid"], limits["max_rss_mib"], limits["min_disk_free_gib"]
        )
        heartbeat()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--supervisor-pid", type=int, required=True)
    parser.add_argument(
        "--state",
        type=Path,
        required=True,
        help="fresh persistent handoff record directory",
    )
    parser.add_argument("--remove-wall-limit", action="store_true", required=True)
    parser.add_argument(
        "--reason", required=True, help="operator authorization for this policy change"
    )
    args = parser.parse_args()
    output = args.run.resolve(strict=True)
    original = json.loads((output / "run.json").read_text())
    if original["status"] != "running" or original["validated"]:
        parser.error("only an actively running, unvalidated run can be adopted")
    limits = original["limits"]
    if (
        not limits["wall_seconds"]
        or limits["max_rss_mib"] <= 0
        or limits["min_disk_free_gib"] <= 0
        or limits["poll_seconds"] != 5
    ):
        parser.error(
            "adoption requires an existing deadline and positive resource guards"
        )
    verify_assets(output, original)
    log = (output / "run.log").read_text()
    if not log.endswith("\n"):
        log = log.rsplit("\n", 1)[0] + "\n"
    history = [
        json.loads(line[len(MARKER) :])
        for line in log.splitlines()
        if line.startswith(MARKER)
    ]
    prefix = copy.deepcopy(original)
    prefix["parameters"]["end"] = history[-1]["time"]
    validate_history(log, prefix)
    solver = process_info(original["pid"])
    old = process_info(args.supervisor_pid)
    if (
        not solver
        or not old
        or solver["uid"] != os.getuid()
        or old["uid"] != os.getuid()
        or solver["ppid"] != old["pid"]
        or solver["command"] != " ".join(original["command"])
        or " -m scripts.paper_run " not in old["command"]
    ):
        parser.error("live solver or parent identity does not match the paper run")
    resource_check(
        output, solver["pid"], limits["max_rss_mib"], limits["min_disk_free_gib"]
    )
    observer = ExitObserver(solver["pid"])
    args.state.mkdir(parents=True, exist_ok=False)
    state = args.state.resolve()
    # Single owner across all state-directory names; retain the lock as evidence.
    lock = output / "supervision-owner.json"
    with lock.open("x") as stream:
        json.dump(
            {"supervisor": process_info(os.getpid()), "state": str(state)}, stream
        )
    receipt = {
        "schema_version": 1,
        "status": "preparing",
        "reason": args.reason,
        "started_at": datetime.now(UTC).isoformat(),
        "old_supervisor": old,
        "solver": solver,
        "new_supervisor": process_info(os.getpid()),
        "original_record_sha256": sha(output / "run.json"),
        "original_limits": copy.deepcopy(limits),
        "effective_limits": {**limits, "wall_seconds": None},
        "old_supervisor_retired": False,
        "solver_restarted": False,
        "exit_tracking": "Darwin kqueue NOTE_EXITSTATUS",
        "source_hashes": {
            "paper_supervisor.py": sha(Path(__file__)),
            "paper_run.py": sha(Path(__file__).with_name("paper_run.py")),
        },
        "history_at_handoff": history[-1],
    }
    record = copy.deepcopy(original)

    def heartbeat():
        receipt["updated_at"] = datetime.now(UTC).isoformat()
        write(state / "supervision.json", receipt)

    def prepared():
        if sha(output / "run.json") != receipt["original_record_sha256"]:
            raise RuntimeError(
                "original supervisor changed the record during preparation"
            )
        shutil.copy2(output / "run.json", state / "original-run.json")
        for name in ("paper_supervisor.py", "paper_run.py"):
            shutil.copy2(Path(__file__).with_name(name), state / name)
        receipt["status"] = "ready"
        heartbeat()

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"supervisor received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        retire_parent(old, solver, receipt, prepared)
        require_process(solver)
        record["limits"] = receipt["effective_limits"]
        record["runtime_supervision"] = {
            "state": str(state),
            "supervisor_pid": os.getpid(),
            "original_record_sha256": receipt["original_record_sha256"],
            "original_limits": receipt["original_limits"],
            "reason": args.reason,
            "solver_restarted": False,
            "exit_tracking": receipt["exit_tracking"],
            "source_hashes": receipt["source_hashes"],
        }
        record["updated_at"] = datetime.now(UTC).isoformat()
        write(output / "run.json", record)
        receipt["status"] = "running"
        heartbeat()
        print(
            json.dumps(
                {
                    "status": "adopted",
                    "solver_pid": solver["pid"],
                    "wall_seconds": None,
                    "limits": record["limits"],
                }
            ),
            flush=True,
        )
        code = monitor(output, solver, observer, record["limits"], heartbeat)
        receipt["solver_exit_code"] = code
        record["runtime_supervision"]["solver_exit_code"] = code
        if code:
            raise ValueError(f"solver exited with status {code}")
        receipt["status"] = "validating"
        record["status"] = "validating"
        heartbeat()
        write(output / "run.json", record)
        verify_assets(output, record)
        env = {
            **os.environ,
            **original["execution"].get("library_environment", {}),
            **thread_environment(
                original["execution"]["threads"], original["execution"]["force_threads"]
            ),
        }
        audit_completed(output, record, env)
        receipt["status"] = "completed"
    except BaseException as exc:
        receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        if receipt["old_supervisor_retired"]:
            stop_exact_solver(solver, observer)
            record.update(status="failed", validated=False, error=receipt["error"])
        raise
    finally:
        observer.close()
        if receipt["old_supervisor_retired"]:
            record["wall_seconds"] = (
                datetime.now(UTC) - datetime.fromisoformat(record["started_at"])
            ).total_seconds()
            record["updated_at"] = datetime.now(UTC).isoformat()
            write(output / "run.json", record)
        heartbeat()


if __name__ == "__main__":
    main()
