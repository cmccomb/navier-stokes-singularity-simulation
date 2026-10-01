"""The final continuation publisher must preserve its scientific boundary."""

import json
from pathlib import Path

import pytest

from scripts.paper_run import sha
from scripts.publish_final_continuation import (
    attach_endpoint_diagnostics,
    audit_continuation,
    publish,
    render,
    updated_results_page,
)


def row(step, time, dt, peak=1.0):
    return {
        "step": step,
        "time": time,
        "dt": dt,
        "levels": 5,
        "stored_cells": 100,
        "active_cells": 90,
        "adapter_sha256": "a" * 64,
        "peak_speed": peak,
        "energy": peak / 10,
        "l2_error": peak / 100,
        "linf_error": peak / 50,
    }


def fixture(tmp_path: Path) -> Path:
    parent = tmp_path / "parent"
    source = tmp_path / "continuation"
    (source / "runner/scripts").mkdir(parents=True)
    (source / "bundle").mkdir()
    parent.mkdir()
    parent_frame = parent / "plt00001"
    parent_frame.mkdir()
    initial = row(1, 0.995, 0.995, 2)
    parent_record = {
        "status": "completed",
        "validated": True,
        "parameters": {"end": 0.995, "base_n": 128, "widths": [0.5, 0.25, 0.125, 0.0625]},
        "planned_frames": [0.0, 0.995],
    }
    (parent / "run.json").write_text(json.dumps(parent_record) + "\n")
    rest = row(0, 0.0, 0.0, 0)
    (parent / "run.log").write_text(
        "NS_INCFLO_RESULT " + json.dumps(rest) + "\n"
        + "NS_INCFLO_RESULT " + json.dumps(initial) + "\n"
    )
    parent_record["history"] = [rest, initial]
    (parent / "run.json").write_text(json.dumps(parent_record) + "\n")
    runner = source / "runner/scripts/paper_continue.py"
    runner.write_text("# pinned\n")
    assets = {}
    for name in ("ns_incflo", "ns_archive_check", "profile.tbl"):
        path = source / "bundle" / name
        path.write_text(name)
        assets[name] = sha(path)
    baseline_row = row(2, 0.996, 0.001, 2.1)
    smaller_rows = [
        row(2, 0.9955, 0.0005, 2.05),
        row(3, 0.996, 0.0005, 2.1),
    ]
    extension_row = row(3, 0.9975, 0.0015, 2.2)
    baseline_frame = source / "probe-baseline/plt00002"
    smaller_frame = source / "probe-smaller-dt/plt00003"
    extension_frame = source / "extension/plt00003"
    baseline_frame.mkdir(parents=True)
    smaller_frame.mkdir(parents=True)
    extension_frame.mkdir(parents=True)
    stages = {
        "restart-replay": {"status": "completed", "validated": True},
        "probe-baseline": {
            "status": "completed",
            "validated": True,
            "start": initial,
            "history": [baseline_row],
            "native_frames": [{"path": str(baseline_frame), "time": 0.996}],
        },
        "probe-smaller-dt": {
            "status": "completed",
            "validated": True,
            "start": initial,
            "history": smaller_rows,
            "native_frames": [{"path": str(smaller_frame), "time": 0.996}],
        },
        "extension": {
            "status": "completed",
            "validated": True,
            "start": baseline_row,
            "history": [extension_row],
            "native_frames": [{"path": str(extension_frame), "time": 0.9975}],
        },
    }
    frames = [
        {"path": str(parent / "plt00000"), "time": 0.0},
        {"path": str(parent_frame), "time": 0.995},
        {"path": str(baseline_frame), "time": 0.996},
        {"path": str(extension_frame), "time": 0.9975},
    ]
    (parent / "plt00000").mkdir()
    combined = {
        "kind": "validated-checkpoint-continuation-of-from-rest",
        "frames": frames,
        "starts_at_rest": True,
        "through": 0.9975,
        "excluded_comparison_branch": "probe-smaller-dt",
        "partial": False,
    }
    (source / "combined-frames.json").write_text(json.dumps(combined) + "\n")
    receipt = {
        "kind": "checkpoint-continuation-of-from-rest",
        "status": "completed",
        "validated": True,
        "completed_at": "2026-10-01T00:00:00+00:00",
        "parent": str(parent),
        "parent_record_sha256": sha(parent / "run.json"),
        "runner_sha256": sha(runner),
        "runner_source_hashes": {"paper_continue.py": sha(runner)},
        "bundle_sha256": assets,
        "force_definition_unchanged": True,
        "probe_end": 0.996,
        "end": 0.9975,
        "planned_new_frame_times": [0.996, 0.9975],
        "combined_frames": 4,
        "stages": stages,
        "restart_check": {
            "compared": True,
            "velocity_linf_difference": 0,
            "force_linf_error": 0,
        },
        "restart_replay_check": {
            "compared": True,
            "velocity_linf_difference": 0,
            "force_linf_error": 0,
        },
        "temporal_comparison": {
            "relative_composite_l2": 0.0001,
            "linf_over_baseline_peak": 0.001,
            "l2_limit": 0.001,
            "linf_limit": 0.01,
            "passed": True,
            "scope": "shared-checkpoint screen",
            "native": {
                "compared": True,
                "report_difference": True,
                "reference_resolution_ratio": 1,
                "force_linf_error": 0,
            },
        },
    }
    (source / "continuation.json").write_text(json.dumps(receipt) + "\n")
    return source


def test_completed_continuation_becomes_bounded_public_record(tmp_path):
    source = fixture(tmp_path)
    result = audit_continuation(source)
    assert result["endpoint"]["time"] == 0.9975
    assert len(result["trajectory"]) == 3
    assert result["temporal_probe"]["passed"] is True
    assert result["claim_boundary"] == {
        "spatially_qualified": False,
        "singularity_or_blowup_demonstrated": False,
        "values_beyond_computed_endpoint": False,
        "interpretation": "Finite continuation of one audited 128^3-base trajectory. The extension is model extrapolation, not a new spatial-convergence certificate or evidence of singularity formation.",
    }
    chart = tmp_path / "chart.svg"
    render(result, chart)
    text = chart.read_text()
    assert "t = 0.9975" in text and "no fit or values beyond the endpoint" in text


def test_endpoint_native_diagnostics_are_hash_and_value_bound(tmp_path):
    source = fixture(tmp_path)
    report = audit_continuation(source)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    manifest = {
        "complete": True,
        "source_continuation_sha256": report["source"][
            "continuation_record_sha256"
        ],
    }
    (snapshot / "manifest.json").write_text(json.dumps(manifest) + "\n")
    endpoint = report["endpoint"]
    frame = {
        "step": endpoint["step"],
        "time": endpoint["time"],
        "peak_speed": endpoint["peak_speed"],
        "kinetic_energy": endpoint["energy"],
        "target_l2_error": endpoint["l2_error"],
        "target_linf_error": endpoint["linf_error"],
        "peak_level": 4,
        "peak_in_finest_core": False,
        "half_peak_support_equivalent_radius": 0.1,
        "finest_core_half_peak_equivalent_radius": None,
        "peak_vorticity": 100,
        "vorticity_rms": 1,
        "force_l2": 2,
        "divergence_rms": 0.01,
        "divergence_linf": 0.02,
    }
    diagnostics = {
        "kind": "outer-band-finite-precursor-diagnostics",
        "validated": True,
        "source_snapshot_sha256": [sha(snapshot / "manifest.json")],
        "source_binary_sha256": report["source"]["binary_sha256"],
        "source_profile_sha256": report["source"]["profile_sha256"],
        "analyzer_sha256": "d" * 64,
        "definitions": {"support_radius": "test"},
        "core_half_width": 0.0625,
        "frames": [frame],
        "scope": "one native endpoint",
    }
    path = tmp_path / "endpoint.json"
    path.write_text(json.dumps(diagnostics) + "\n")
    attach_endpoint_diagnostics(report, path, snapshot)
    assert report["endpoint_native_diagnostics"]["frame"]["peak_vorticity"] == 100
    chart = tmp_path / "native-chart.svg"
    render(report, chart)
    chart_text = chart.read_text()
    assert "peak outside finest core" in chart_text
    assert "no finest-core half-peak radius" in chart_text
    assert "force L² 2" in chart_text
    results = tmp_path / "results.html"
    results.write_text(
        "<article>\n"
        "        <!-- FINAL_CONTINUATION:BEGIN -->\n"
        "        <p>Pending.</p>\n"
        "        <!-- FINAL_CONTINUATION:END -->\n"
        "</article>\n"
    )
    published = updated_results_page(report, results)
    assert 'id="final-continuation"' in published
    assert "peak is outside the finest core" in published
    assert "provisional spatial candidate" in published
    assert "does not add a spatial-convergence certificate" in published
    assert "The analytical paper supplies the blowup result" in published
    site = tmp_path / "site"
    (site / "media").mkdir(parents=True)
    (site / "data").mkdir()
    (site / "results.html").write_text(results.read_text())
    released = publish(source, path, snapshot, site)
    assert released["validated"] is True
    assert (site / "media/final-continuation.svg").is_file()
    saved = json.loads((site / "data/final-continuation.json").read_text())
    assert saved["publication"]["results_page_sha256"] == sha(
        site / "results.html"
    )
    final_page = (site / "results.html").read_text()
    assert "Pending." not in final_page
    assert 'href="data/final-continuation.json"' in final_page
    assert 'src="media/final-continuation.svg"' in final_page
    diagnostics["frames"][0]["peak_speed"] += 1
    path.write_text(json.dumps(diagnostics) + "\n")
    with pytest.raises(ValueError, match="peak_speed differs"):
        attach_endpoint_diagnostics(report, path, snapshot)


@pytest.mark.parametrize("change", ["status", "gate", "lineage", "runner"])
def test_incomplete_or_changed_evidence_is_rejected(tmp_path, change):
    source = fixture(tmp_path)
    receipt_path = source / "continuation.json"
    receipt = json.loads(receipt_path.read_text())
    if change == "status":
        receipt["status"] = "running"
    elif change == "gate":
        receipt["temporal_comparison"]["passed"] = False
    elif change == "lineage":
        combined = json.loads((source / "combined-frames.json").read_text())
        combined["partial"] = True
        (source / "combined-frames.json").write_text(json.dumps(combined))
    else:
        (source / "runner/scripts/paper_continue.py").write_text("# changed\n")
    receipt_path.write_text(json.dumps(receipt) + "\n")
    with pytest.raises(ValueError):
        audit_continuation(source)
