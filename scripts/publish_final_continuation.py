"""Publish an audited diagnostic record for the final finite continuation.

The continuation is an extrapolation of one grid-dependent trajectory.  This
publisher verifies the completed receipt and its saved-frame lineage, then
draws only computed diagnostics.  It never fits or projects values toward
``t*=1`` and never labels the result as a convergence or singularity result.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
from pathlib import Path

import matplotlib

from scripts.paper_run import MARKER, sha, write

matplotlib.use("Agg")
plt = importlib.import_module("matplotlib.pyplot")


REQUIRED_STAGES = (
    "restart-replay",
    "probe-baseline",
    "probe-smaller-dt",
    "extension",
)
DIAGNOSTICS = ("peak_speed", "energy", "l2_error", "linf_error")


def _read(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected an object in {path.name}")
    return value


def _close(left: float, right: float, tolerance: float = 2e-14) -> bool:
    return math.isclose(left, right, rel_tol=0, abs_tol=tolerance)


def _same_state(left: dict, right: dict) -> bool:
    exact = ("step", "levels", "stored_cells", "active_cells", "adapter_sha256")
    return all(left[key] == right[key] for key in exact) and _close(
        left["time"], right["time"]
    )


def _check_history(stage: dict, expected_end: float) -> list[dict]:
    rows = stage.get("history")
    if not isinstance(rows, list) or not rows:
        raise ValueError("completed continuation stage lacks diagnostic history")
    previous = stage.get("start")
    if not isinstance(previous, dict):
        raise TypeError("continuation stage lacks its starting state")
    for row in rows:
        if row["step"] != previous["step"] + 1 or not _close(
            row["time"] - previous["time"], row["dt"]
        ):
            raise ValueError("continuation diagnostic history is discontinuous")
        if not all(math.isfinite(row[key]) for key in ("time", "dt", *DIAGNOSTICS)):
            raise ValueError("continuation diagnostic history is nonfinite")
        previous = row
    if not _close(rows[-1]["time"], expected_end):
        raise ValueError("continuation stage has the wrong endpoint")
    return rows


def _parent_history(parent: Path, record: dict) -> list[dict]:
    rows = [
        json.loads(line.removeprefix(MARKER))
        for line in (parent / "run.log").read_text().splitlines()
        if line.startswith(MARKER)
    ]
    if not rows or rows[-1]["step"] + 1 != len(rows):
        raise ValueError("parent diagnostic history is incomplete")
    if not _close(rows[0]["time"], 0) or rows[0]["peak_speed"] != 0:
        raise ValueError("parent trajectory does not begin at exact rest")
    final = rows[-1]
    if rows != record.get("history"):
        raise ValueError("parent log differs from its hashed completion record")
    if not _close(final["time"], record["parameters"]["end"]):
        raise ValueError("parent diagnostic endpoint differs from its record")
    for key in DIAGNOSTICS:
        if not all(math.isfinite(row[key]) for row in rows):
            raise ValueError("parent diagnostic history is nonfinite")
    return rows


def audit_continuation(source: Path) -> dict:
    source = source.resolve(strict=True)
    receipt_path = source / "continuation.json"
    receipt = _read(receipt_path)
    if (
        receipt.get("kind") != "checkpoint-continuation-of-from-rest"
        or receipt.get("status") != "completed"
        or receipt.get("validated") is not True
    ):
        raise ValueError("only a completed, validated continuation can be published")
    if receipt.get("force_definition_unchanged") is not True:
        raise ValueError("continuation changed or did not audit the force definition")

    runner = source / "runner/scripts/paper_continue.py"
    if (
        sha(runner) != receipt["runner_sha256"]
        or sha(runner) != receipt["runner_source_hashes"]["paper_continue.py"]
    ):
        raise ValueError("pinned continuation runner changed")
    for name, expected in receipt["bundle_sha256"].items():
        if sha(source / "bundle" / name) != expected:
            raise ValueError(f"pinned continuation asset changed: {name}")

    parent = Path(receipt["parent"]).resolve(strict=True)
    parent_record_path = parent / "run.json"
    if sha(parent_record_path) != receipt["parent_record_sha256"]:
        raise ValueError("parent completion record changed")
    parent_record = _read(parent_record_path)
    if parent_record.get("status") != "completed" or parent_record.get("validated") is not True:
        raise ValueError("parent is no longer a completed, validated run")
    parent_rows = _parent_history(parent, parent_record)
    if parent_record["parameters"]["base_n"] != 128 or len(
        parent_record["parameters"]["widths"]
    ) != 4:
        raise ValueError("final publication requires the 128-base five-level model")
    if not _close(receipt["end"], 0.9975):
        raise ValueError("final publication requires the planned t=0.9975 endpoint")

    stages = receipt.get("stages", {})
    if not all(
        stages.get(name, {}).get("status") == "completed"
        and stages[name].get("validated") is True
        for name in REQUIRED_STAGES
    ):
        raise ValueError("all restart, probe, and extension stages must pass")
    initial = parent_rows[-1]
    baseline = stages["probe-baseline"]
    smaller = stages["probe-smaller-dt"]
    extension = stages["extension"]
    if not _same_state(baseline["start"], initial) or not _same_state(
        smaller["start"], initial
    ):
        raise ValueError("temporal probes do not share the parent endpoint")
    baseline_rows = _check_history(baseline, receipt["probe_end"])
    smaller_rows = _check_history(smaller, receipt["probe_end"])
    if len(smaller_rows) <= len(baseline_rows):
        raise ValueError("smaller-step probe did not refine the integration")
    if not _same_state(extension["start"], baseline_rows[-1]):
        raise ValueError("extension does not continue the accepted baseline probe")
    extension_rows = _check_history(extension, receipt["end"])
    expected_stage_times = {
        "probe-baseline": [receipt["probe_end"]],
        "probe-smaller-dt": [receipt["probe_end"]],
        "extension": receipt["planned_new_frame_times"][1:],
    }
    for name, expected in expected_stage_times.items():
        native_frames = stages[name].get("native_frames", [])
        if len(native_frames) != len(expected) or any(
            not _close(frame["time"], time, 1e-12)
            or not Path(frame["path"]).is_dir()
            for frame, time in zip(native_frames, expected, strict=True)
        ):
            raise ValueError(f"{name} native-frame audit is incomplete")

    for name in ("restart_check", "restart_replay_check"):
        check = receipt.get(name, {})
        if (
            check.get("compared") is not True
            or check.get("velocity_linf_difference", math.inf) > 1e-12
            or check.get("force_linf_error", math.inf) > 1e-12
        ):
            raise ValueError(f"{name} did not reproduce the parent field")

    comparison = receipt.get("temporal_comparison", {})
    native = comparison.get("native", {})
    if (
        comparison.get("passed") is not True
        or native.get("compared") is not True
        or native.get("report_difference") is not True
        or native.get("reference_resolution_ratio") != 1
        or native.get("force_linf_error", math.inf) > 1e-12
    ):
        raise ValueError("short-tail fixed-force temporal gate did not pass")

    combined_path = source / "combined-frames.json"
    combined = _read(combined_path)
    frames = combined.get("frames", [])
    expected_times = parent_record["planned_frames"] + receipt["planned_new_frame_times"]
    if (
        combined.get("kind") != "validated-checkpoint-continuation-of-from-rest"
        or combined.get("starts_at_rest") is not True
        or combined.get("partial") is not False
        or not _close(combined.get("through", math.nan), receipt["end"])
        or receipt.get("combined_frames") != len(frames)
        or len(frames) != len(expected_times)
        or any(not _close(frame["time"], time, 1e-12) for frame, time in zip(frames, expected_times, strict=True))
        or any(not Path(frame["path"]).is_dir() for frame in frames)
    ):
        raise ValueError("combined from-rest native-frame lineage is incomplete")

    rows = [initial, *baseline_rows, *extension_rows]
    endpoint = rows[-1]
    trajectory = [
        {"step": row["step"], "time": row["time"], **{key: row[key] for key in DIAGNOSTICS}}
        for row in rows
    ]
    return {
        "schema_version": 1,
        "kind": "audited-final-finite-continuation",
        "status": "completed",
        "validated": True,
        "source_completed_at": receipt["completed_at"],
        "source": {
            "continuation_record_sha256": sha(receipt_path),
            "combined_frames_sha256": sha(combined_path),
            "parent_record_sha256": receipt["parent_record_sha256"],
            "runner_sha256": receipt["runner_sha256"],
            "binary_sha256": receipt["bundle_sha256"]["ns_incflo"],
            "checker_sha256": receipt["bundle_sha256"]["ns_archive_check"],
            "profile_sha256": receipt["bundle_sha256"]["profile.tbl"],
        },
        "mesh": {
            "base_n": parent_record["parameters"]["base_n"],
            "levels": endpoint["levels"],
            "stored_cells": endpoint["stored_cells"],
            "active_cells": endpoint["active_cells"],
            "finest_equivalent_n": parent_record["parameters"]["base_n"]
            * 2 ** (endpoint["levels"] - 1),
        },
        "computed_interval": [initial["time"], endpoint["time"]],
        "from_rest_interval": [0, endpoint["time"]],
        "endpoint": {key: endpoint[key] for key in ("step", "time", *DIAGNOSTICS)},
        "change_over_continuation": {
            key: endpoint[key] - initial[key] for key in DIAGNOSTICS
        },
        "temporal_probe": {
            key: comparison[key]
            for key in (
                "relative_composite_l2",
                "linf_over_baseline_peak",
                "l2_limit",
                "linf_limit",
                "passed",
                "scope",
            )
        },
        "new_native_frames": len(receipt["planned_new_frame_times"]),
        "combined_native_frames": len(frames),
        "new_frame_times": receipt["planned_new_frame_times"],
        "trajectory": trajectory,
        "claim_boundary": {
            "spatially_qualified": False,
            "singularity_or_blowup_demonstrated": False,
            "values_beyond_computed_endpoint": False,
            "interpretation": "Finite continuation of one audited 128^3-base trajectory. The extension is model extrapolation, not a new spatial-convergence certificate or evidence of singularity formation.",
        },
    }


def render(report: dict, output: Path) -> None:
    rows = report["trajectory"]
    times = [row["time"] for row in rows]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "text.color": "#e5eef4",
            "axes.labelcolor": "#bacbd7",
            "xtick.color": "#bacbd7",
            "ytick.color": "#bacbd7",
            "axes.edgecolor": "#40546a",
            "axes.facecolor": "#07111f",
            "figure.facecolor": "#07111f",
            "savefig.facecolor": "#07111f",
            "svg.hashsalt": "final-finite-continuation-v1",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    panels = (
        ("peak_speed", "Peak speed", "model speed"),
        ("energy", "Kinetic energy", "model energy"),
        ("l2_error", "RMS target deviation", "model speed"),
        ("linf_error", "Maximum component deviation", "model speed"),
    )
    for ax, (key, title, ylabel) in zip(axes.flat, panels, strict=True):
        values = [row[key] for row in rows]
        ax.plot(times, values, color="#69d2e7", linewidth=2.2)
        ax.scatter(times[-1], values[-1], color="#ffd88c", s=42, zorder=3)
        ax.set_title(title, fontsize=15, loc="left", color="#e5eef4")
        ax.set_xlabel("Computed time t")
        ax.set_ylabel(ylabel)
        ax.grid(color="#40546a", alpha=0.32, linewidth=0.7)
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)
        ax.margins(x=0.025, y=0.1)
    fig.suptitle(
        "Final 128³-base finite continuation · computed through "
        f"t = {report['endpoint']['time']:.4f}",
        fontsize=20,
        color="#e5eef4",
    )
    fig.text(
        0.02,
        0.012,
        "Gold marks the last computed state. Lines connect solver steps; no fit or values beyond the endpoint. This single-mesh extension is not a singularity or convergence result.",
        color="#aabfce",
        fontsize=11,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="svg", metadata={"Date": None})
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuation", type=Path, required=True)
    parser.add_argument("--site", type=Path, required=True)
    args = parser.parse_args()
    report = audit_continuation(args.continuation)
    chart = args.site / "media/final-continuation.svg"
    render(report, chart)
    report["visualization"] = {
        "path": "media/final-continuation.svg",
        "sha256": sha(chart),
        "scope": "Computed solver-step diagnostics from the accepted baseline continuation branch; no temporal interpolation or fitted projection.",
    }
    report["publisher_sha256"] = sha(Path(__file__))
    write(args.site / "data/final-continuation.json", report)
    print(json.dumps({"endpoint": report["endpoint"], "validated": True}))


if __name__ == "__main__":
    main()
