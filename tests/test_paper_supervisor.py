import os
import subprocess
import sys

import pytest

from scripts.paper_supervisor import (
    ExitObserver,
    monitor,
    process_info,
    retire_parent,
    same_process,
    stop_exact_solver,
)


def test_identity_rejects_reused_pid():
    expected = {"pid": 1, "uid": 2, "started": "first", "command": "fixture"}
    assert same_process({**expected, "ppid": 99, "state": "S"}, expected)
    assert not same_process(None, expected)
    for key in expected:
        assert not same_process({**expected, key: "changed"}, expected)


def test_monitor_has_no_wall_cutoff_but_keeps_resource_limits(monkeypatch, tmp_path):
    class Observer:
        calls = 0

        def poll(self, seconds):
            assert seconds == 5
            self.calls += 1
            return 0 if self.calls == 4 else None

    identity = {"pid": 123}
    guards, beats = [], []
    monkeypatch.setattr("scripts.paper_supervisor.require_process", lambda _: identity)
    monkeypatch.setattr(
        "scripts.paper_supervisor.resource_check", lambda *args: guards.append(args)
    )
    assert (
        monitor(
            tmp_path,
            identity,
            Observer(),
            {"max_rss_mib": 100, "min_disk_free_gib": 20},
            lambda: beats.append(True),
        )
        == 0
    )
    assert guards == [(tmp_path, 123, 100, 20)] * 3 and len(beats) == 3


def test_resource_failure_propagates(monkeypatch, tmp_path):
    class Observer:
        def poll(self, seconds):
            return None

    monkeypatch.setattr("scripts.paper_supervisor.require_process", lambda _: {})

    def fail(*args):
        raise RuntimeError("disk reserve reached")

    monkeypatch.setattr("scripts.paper_supervisor.resource_check", fail)
    with pytest.raises(RuntimeError, match="disk reserve"):
        monitor(
            tmp_path,
            {"pid": 123},
            Observer(),
            {"max_rss_mib": 100, "min_disk_free_gib": 20},
            lambda: None,
        )


def test_stop_does_not_signal_reused_pid(monkeypatch):
    expected = {"pid": 123, "uid": 4, "started": "old", "command": "solver"}
    monkeypatch.setattr(
        "scripts.paper_supervisor.process_info",
        lambda _: {**expected, "started": "new", "state": "R"},
    )
    calls = []
    monkeypatch.setattr("scripts.paper_supervisor.os.kill", lambda *a: calls.append(a))
    stop_exact_solver(expected, None)
    assert calls == []


def parent_and_child(exit_code=7):
    code = (
        "import subprocess,sys; "
        f"p=subprocess.Popen([sys.executable,'-c','import time; "
        f"print(1,flush=True); time.sleep(2); raise SystemExit({exit_code})'], "
        "stdout=subprocess.PIPE,text=True); p.stdout.readline(); "
        "print(p.pid,flush=True); p.wait()"
    )
    parent = subprocess.Popen(
        [sys.executable, "-c", code], stdout=subprocess.PIPE, text=True
    )
    return parent, int(parent.stdout.readline())


@pytest.mark.skipif(sys.platform != "darwin", reason="Darwin live handoff")
def test_live_handoff_preserves_solver_and_observes_nonchild_exit_status():
    parent, pid = parent_and_child()
    observer = ExitObserver(pid)
    solver, old = process_info(pid), process_info(parent.pid)
    receipt, prepared = {}, []
    try:
        retire_parent(old, solver, receipt, lambda: prepared.append(True))
        assert receipt["old_supervisor_retired"] and prepared == [True]
        assert same_process(process_info(pid), solver)
        assert parent.wait(timeout=5) == -9
        assert observer.poll(5) == 7
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait()
        observer.close()


@pytest.mark.skipif(sys.platform != "darwin", reason="Darwin live handoff")
def test_failed_preparation_resumes_original_supervisor():
    parent, pid = parent_and_child(0)
    solver, old = process_info(pid), process_info(parent.pid)
    receipt = {}

    def fail():
        raise RuntimeError("fixture preparation failed")

    try:
        with pytest.raises(RuntimeError, match="preparation failed"):
            retire_parent(old, solver, receipt, fail)
        assert not receipt.get("old_supervisor_retired")
        assert same_process(process_info(pid), solver)
        assert "T" not in process_info(parent.pid)["state"]
        assert parent.wait(timeout=5) == 0
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait()


@pytest.mark.skipif(sys.platform != "darwin", reason="Darwin process identity")
def test_resource_stop_targets_only_the_owned_solver():
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; print(1,flush=True); time.sleep(30)"],
        stdout=subprocess.PIPE,
        text=True,
    )
    process.stdout.readline()
    observer = ExitObserver(process.pid)
    try:
        stop_exact_solver(process_info(process.pid), observer)
        assert process.wait(timeout=5) == -15
        assert process_info(os.getpid()) is not None
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        observer.close()
