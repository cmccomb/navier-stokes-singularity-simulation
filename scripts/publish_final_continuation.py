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

from scripts.archive_path_map import (
    load_path_map,
    path_map_record,
    resolve_archive_path,
)
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
RESULTS_BEGIN = "        <!-- FINAL_CONTINUATION:BEGIN -->"
RESULTS_END = "        <!-- FINAL_CONTINUATION:END -->"


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


def audit_continuation(
    source: Path,
    path_map: dict[Path, Path] | None = None,
    path_map_sha256: str | None = None,
) -> dict:
    path_map = path_map or {}
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

    parent = resolve_archive_path(receipt["parent"], path_map)
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
            or not resolve_archive_path(frame["path"], path_map).is_dir()
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
        or any(not resolve_archive_path(frame["path"], path_map).is_dir() for frame in frames)
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
            "archive_path_map": path_map_record(path_map),
            "archive_path_map_sha256": path_map_sha256,
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


def attach_endpoint_diagnostics(
    report: dict, diagnostics_path: Path, snapshot: Path
) -> dict:
    diagnostics_path = diagnostics_path.resolve(strict=True)
    snapshot = snapshot.resolve(strict=True)
    diagnostics = _read(diagnostics_path)
    manifest_path = snapshot / "manifest.json"
    manifest = _read(manifest_path)
    frames = diagnostics.get("frames", [])
    if (
        diagnostics.get("kind") != "outer-band-finite-precursor-diagnostics"
        or diagnostics.get("validated") is not True
        or len(frames) != 1
        or diagnostics.get("source_snapshot_sha256") != [sha(manifest_path)]
        or manifest.get("complete") is not True
        or manifest.get("source_continuation_sha256")
        != report["source"]["continuation_record_sha256"]
        or diagnostics.get("source_binary_sha256")
        != report["source"]["binary_sha256"]
        or diagnostics.get("source_profile_sha256")
        != report["source"]["profile_sha256"]
    ):
        raise ValueError("endpoint native diagnostic lineage is incomplete")
    frame = frames[0]
    endpoint = report["endpoint"]
    for key, diagnostic_key in (
        ("time", "time"),
        ("peak_speed", "peak_speed"),
        ("energy", "kinetic_energy"),
        ("l2_error", "target_l2_error"),
        ("linf_error", "target_linf_error"),
    ):
        if not math.isclose(
            endpoint[key], frame[diagnostic_key], rel_tol=1e-9, abs_tol=1e-12
        ):
            raise ValueError(f"endpoint native {diagnostic_key} differs")
    if endpoint["step"] != frame["step"]:
        raise ValueError("endpoint native step differs")
    report["endpoint_native_diagnostics"] = {
        "source_diagnostics_sha256": sha(diagnostics_path),
        "source_snapshot_manifest_sha256": sha(manifest_path),
        "analyzer_sha256": diagnostics["analyzer_sha256"],
        "definitions": diagnostics["definitions"],
        "core_half_width": diagnostics["core_half_width"],
        "frame": frame,
        "scope": diagnostics["scope"],
    }
    return report


def final_results_section(report: dict) -> str:
    native = report["endpoint_native_diagnostics"]["frame"]
    endpoint = report["endpoint"]
    mesh = report["mesh"]
    probe = report["temporal_probe"]
    boundary = report["claim_boundary"]
    if (
        boundary["spatially_qualified"] is not False
        or boundary["singularity_or_blowup_demonstrated"] is not False
        or boundary["values_beyond_computed_endpoint"] is not False
    ):
        raise ValueError("final results section requires the finite claim boundary")
    core_radius = native["finest_core_half_peak_equivalent_radius"]
    core_text = (
        f"{core_radius:.9g}"
        if core_radius is not None
        else "Not reported; the global peak is outside the finest core"
    )
    return f"""{RESULTS_BEGIN}
        <h2 id="final-continuation">Final finite continuation</h2>
        <p>The accepted 128³-base, five-level branch continues the audited from-rest trajectory through <span class="math">t = {endpoint['time']:.4f}</span>. Its short shared-checkpoint timestep probe passed before extension: relative composite L² {probe['relative_composite_l2']:.6g} against a {probe['l2_limit']:.6g} limit, and L∞ divided by baseline peak {probe['linf_over_baseline_peak']:.6g} against a {probe['linf_limit']:.6g} limit. The <a href="data/final-continuation.json">machine record</a> binds the runner, solver, force profile, complete native-frame lineage, endpoint snapshot, analyzer, and figure.</p>
        <figure class="evidence-figure">
          <div class="plot-viewport" tabindex="0" role="region" aria-label="Final finite continuation figure; scroll horizontally on narrow screens"><img src="media/final-continuation.svg" width="1152" height="864" loading="lazy" alt="Four computed diagnostic histories through t = {endpoint['time']:.4f}, followed by a native endpoint summary for core location, support radius, vorticity, force, and divergence. No values are fitted beyond the endpoint." /></div>
          <p class="plot-scroll-hint">Scroll sideways to inspect all four continuation panels.</p>
          <figcaption><a href="data/final-continuation.json">Audited continuation and endpoint data</a> · computed states only; no fit beyond the endpoint.</figcaption>
        </figure>
        <table>
          <thead><tr><th scope="col">Final computed state</th><th scope="col">Value</th></tr></thead>
          <tbody>
            <tr><th scope="row">Base grid / fixed levels / finest equivalent</th><td>{mesh['base_n']}³ / {mesh['levels']} / {mesh['finest_equivalent_n']}³</td></tr>
            <tr><th scope="row">Endpoint step / time</th><td>{endpoint['step']} / {endpoint['time']:.9g}</td></tr>
            <tr><th scope="row">Peak speed</th><td>{endpoint['peak_speed']:.9g}</td></tr>
            <tr><th scope="row">Kinetic energy</th><td>{endpoint['energy']:.9g}</td></tr>
            <tr><th scope="row">RMS / maximum target deviation</th><td>{endpoint['l2_error']:.9g} / {endpoint['linf_error']:.9g}</td></tr>
            <tr><th scope="row">Half-peak support equivalent radius</th><td>{native['half_peak_support_equivalent_radius']:.9g}</td></tr>
            <tr><th scope="row">Finest-core half-peak equivalent radius</th><td>{core_text}</td></tr>
            <tr><th scope="row">Peak / RMS vorticity</th><td>{native['peak_vorticity']:.9g} / {native['vorticity_rms']:.9g}</td></tr>
            <tr><th scope="row">Manufactured-force L² norm</th><td>{native['force_l2']:.9g}</td></tr>
            <tr><th scope="row">Sampled divergence RMS / maximum</th><td>{native['divergence_rms']:.9g} / {native['divergence_linf']:.9g}</td></tr>
          </tbody>
        </table>
        <p>This extension follows one grid-dependent trajectory beyond the spatially tested interval. It is a finite model extrapolation and does not add a spatial-convergence certificate. The separate refinement evidence identifies 0.615808–0.629071 as a provisional spatial candidate and supports local velocity timestep stability there; mixed residual behavior, material residual timestep sensitivity, and peaks outside the finest core still prevent a resolution-qualified precursor or likely-singularity claim. The analytical paper supplies the blowup result; this computation supplies finite pre-singular diagnostics only.</p>
{RESULTS_END}"""


def updated_results_page(report: dict, page: Path) -> str:
    text = page.read_text()
    if text.count(RESULTS_BEGIN) != 1 or text.count(RESULTS_END) != 1:
        raise ValueError("results page must contain one final-continuation slot")
    start = text.index(RESULTS_BEGIN)
    end = text.index(RESULTS_END, start) + len(RESULTS_END)
    return text[:start] + final_results_section(report) + text[end:]


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
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.subplots_adjust(
        left=0.09,
        right=0.98,
        top=0.87,
        bottom=0.20,
        hspace=0.46,
        wspace=0.28,
    )
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
        ax.set_title(title, fontsize=14, loc="left", color="#e5eef4", pad=10)
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
        y=0.96,
    )
    native = report.get("endpoint_native_diagnostics", {}).get("frame")
    if native is not None:
        core_radius = native["finest_core_half_peak_equivalent_radius"]
        core = (
            f"finest-core half-peak radius {core_radius:.4g}"
            if core_radius is not None
            else "no finest-core half-peak radius"
        )
        peak_location = (
            "peak inside finest core"
            if native["peak_in_finest_core"]
            else "peak outside finest core"
        )
        fig.text(
            0.02,
            0.105,
            "Native endpoint · "
            f"{peak_location}; {core}; support radius "
            f"{native['half_peak_support_equivalent_radius']:.4g}",
            color="#d3e2eb",
            fontsize=10.5,
        )
        fig.text(
            0.02,
            0.077,
            f"|ω|max {native['peak_vorticity']:.4g}; ω RMS "
            f"{native['vorticity_rms']:.4g}; force L² {native['force_l2']:.4g}; "
            f"divergence RMS {native['divergence_rms']:.4g}",
            color="#d3e2eb",
            fontsize=10.5,
        )
    fig.text(
        0.02,
        0.042,
        "Gold marks the last computed state. Lines connect solver steps; no fit or values beyond the endpoint.",
        color="#aabfce",
        fontsize=10.5,
    )
    fig.text(
        0.02,
        0.016,
        "This single-mesh extension is neither a singularity result nor a convergence result.",
        color="#aabfce",
        fontsize=10.5,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output_format = output.suffix.removeprefix(".").lower()
    if output_format == "svg":
        fig.savefig(output, format="svg", metadata={"Date": None})
    elif output_format == "png":
        fig.savefig(output, format="png", dpi=96)
    else:
        raise ValueError("final continuation render must be SVG or PNG")
    plt.close(fig)


def publish(
    continuation: Path,
    endpoint_diagnostics: Path,
    endpoint_snapshot: Path,
    site: Path,
    path_map_path: Path | None = None,
) -> dict:
    path_map = load_path_map(path_map_path)
    report = audit_continuation(
        continuation,
        path_map,
        sha(path_map_path) if path_map_path else None,
    )
    attach_endpoint_diagnostics(
        report, endpoint_diagnostics, endpoint_snapshot
    )
    results_page = site / "results.html"
    results_text = updated_results_page(report, results_page)
    chart = site / "media/final-continuation.svg"
    render(report, chart)
    report["visualization"] = {
        "path": "media/final-continuation.svg",
        "sha256": sha(chart),
        "scope": "Computed solver-step diagnostics from the accepted baseline continuation branch; no temporal interpolation or fitted projection.",
    }
    report["publisher_sha256"] = sha(Path(__file__))
    results_page.write_text(results_text)
    report["publication"] = {
        "results_page": "results.html",
        "results_page_sha256": sha(results_page),
    }
    write(site / "data/final-continuation.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuation", type=Path, required=True)
    parser.add_argument("--endpoint-diagnostics", type=Path, required=True)
    parser.add_argument("--endpoint-snapshot", type=Path, required=True)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--path-map", type=Path)
    args = parser.parse_args()
    report = publish(
        args.continuation,
        args.endpoint_diagnostics,
        args.endpoint_snapshot,
        args.site,
        args.path_map,
    )
    print(json.dumps({"endpoint": report["endpoint"], "validated": True}))


if __name__ == "__main__":
    main()
