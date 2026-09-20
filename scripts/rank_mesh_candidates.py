"""Measure AMReX boxing costs of sparse sensor-informed candidate meshes.

No flow evolution or production-readiness claim. A score is the fraction of
sampled source sensitivity covered by a finer mesh, not predicted error reduction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.export_mesh_snapshot import bounded, sha
from scripts.paper_run import write


def sensor_bands(sensor, base_n):
    q = sensor["request"]
    parent = q["candidate_level"]
    dx = 2 / (base_n * 2**parent)
    radius, z = float(np.hypot(*q["xyz"][:2])), abs(q["xyz"][2])
    result = []
    for ancestor in range(parent + 1):
        # Cover the four-parent-cell patch plus two parent-cell buffer, then
        # supply ancestor clearance. Final AMReX boxes are measured separately.
        pad = 4 * dx + sum(2 * 2 / (base_n * 2**lev) for lev in range(ancestor, parent))
        for center_z in {z, -z}:
            result.append(
                (
                    ancestor,
                    max(0, radius - pad),
                    min(0.999, radius + pad),
                    max(-0.999, center_z - pad),
                    min(0.999, center_z + pad),
                )
            )
    return result


def covers_patch(mesh, request):
    """Exact coverage of the aligned 4h sensor cube by the next AMR level."""
    h = request["dx"]
    fine_level = request["candidate_level"] + 1
    if fine_level >= len(mesh["levels"]):
        return False
    start = np.floor((np.array(request["xyz"]) + 1) / h).astype(int) - 2
    low, high = start * 2, (start + 4) * 2
    cover = np.zeros((8, 8, 8), bool)
    for block in mesh["levels"][fine_level]["boxes"]:
        a = np.maximum(low, block["index_lo"])
        b = np.minimum(high, np.array(block["index_lo"]) + block["shape"])
        if np.all(b > a):
            cover[tuple(slice(x, y) for x, y in zip(a - low, b - low, strict=True))] = (
                True
            )
    return bool(cover.all())


def score(mesh, sensors):
    by_time = {}
    covered = []
    for s in sensors:
        q = s["request"]
        # Normalize each sampled time separately. Patches may overlap, so this
        # is an explicit sensor-weighting heuristic, never a domain integral.
        weight = s["metrics"]["force"]["h_to_half_rms"] ** 2 * (4 * q["dx"]) ** 3
        row = by_time.setdefault(q["time"], [0.0, 0.0])
        row[1] += weight
        if covers_patch(mesh, q):
            row[0] += weight
            covered.append(q["label"])
    fractions = {str(t): a / b if b else 0 for t, (a, b) in by_time.items()}
    return {
        "covered_sensor_count": len(covered),
        "covered_sensor_labels": covered,
        "sampled_sensitivity_coverage_by_time": fractions,
        "mean_sampled_sensitivity_coverage": float(np.mean(list(fractions.values()))),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("executable", "snapshot", "bands", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--sensors", type=Path, nargs="+", required=True)
    args = p.parse_args()
    snapshot = json.loads(args.snapshot.read_text())
    reports = [json.loads(p.read_text()) for p in args.sensors]
    if not snapshot["complete"] or not all(
        r["complete"] and r["table_sha256"] == snapshot["profile_sha256"]
        for r in reports
    ):
        raise ValueError("Incomplete/mismatched source sensors")
    if sha(args.bands) != snapshot["parameters"]["revolved_refinement"]["sha256"]:
        raise ValueError("Original band geometry does not match snapshot")
    sensors = [s for r in reports for s in r["sensors"]]
    if len({s["request"]["label"] for s in sensors}) != len(sensors):
        raise ValueError("Duplicate sensor labels")
    for path, report in zip(args.sensors, reports, strict=True):
        for sensor in report["sensors"]:
            for name, digest in sensor["raw_sha256"].items():
                if sha(path.parent / name) != digest:
                    raise ValueError("Sensor raw data changed")
    n, widths = snapshot["parameters"]["base_n"], snapshot["parameters"]["widths"]
    text = args.bands.read_text().splitlines()
    if text[0] != "NS_RZ_BANDS_V1" or len(text) - 2 != int(text[1]):
        raise ValueError("Invalid original band file")
    original = [
        (int(x[0]), *map(float, x[1:])) for x in (line.split() for line in text[2:])
    ]
    maxima = {
        t: max(
            s["metrics"]["force"]["reference_rms"]
            for s in sensors
            if s["request"]["time"] == t
        )
        for t in {s["request"]["time"] for s in sensors}
    }
    selected = [
        s
        for s in sensors
        if s["metrics"]["force"]["h_to_half_relative"] is not None
        and s["metrics"]["force"]["h_to_half_relative"] >= 0.1
        and s["metrics"]["force"]["reference_rms"]
        >= 0.05 * maxima[s["request"]["time"]]
    ]
    candidates = [
        ("current", widths, []),
        ("deeper-cube-control", widths + [0.03125], []),
        (
            "intermediate-collars",
            widths,
            [s for s in selected if s["request"]["candidate_level"] in (1, 2)],
        ),
        (
            "inner-collars",
            widths,
            [s for s in selected if s["request"]["candidate_level"] == 3],
        ),
        (
            "finest-rings",
            widths + [0.015625],
            [s for s in selected if s["request"]["candidate_level"] == 4],
        ),
    ]
    # Individual buffered rings expose alternatives hidden by expensive unions.
    seen = set()
    for i, s in enumerate(selected):
        geometry = tuple(sensor_bands(s, n))
        if geometry in seen:
            continue
        seen.add(geometry)
        extra_width = [0.015625] if s["request"]["candidate_level"] == 4 else []
        candidates.append((f"local-ring-{i:02d}", widths + extra_width, [s]))
    output = args.output.resolve()
    output.mkdir(exist_ok=False)
    report = {
        "schema_version": 1,
        "complete": False,
        "scope": __doc__,
        "snapshot_manifest_sha256": sha(args.snapshot),
        "sensor_report_sha256": [sha(p) for p in args.sensors],
        "original_bands_sha256": sha(args.bands),
        "mesh_probe_sha256": sha(args.executable),
        "mesh_probe_source_sha256": sha(Path("backends/amrex/mesh_probe.cpp")),
        "tagger_sha256": sha(Path("backends/amrex/ns_case.H")),
        "candidates": [],
        "limits": [
            "Sparse positive-z sensors mirrored for geometric coverage; not complete history/azimuth coverage.",
            "Coverage score is not a measured improvement in evolved solution.",
            "Linear memory extrapolation is a screening estimate; forcing scratch, box count and late-time allocations need a capacity pilot.",
            "No candidate is approved for production or an extended endpoint.",
            "resource_usage.peak_rss_mib is sampled every 0.25 seconds and can miss short-lived peaks; mesh_builder_peak_rss_mib uses the executable's getrusage high-water mark.",
        ],
    }
    baseline_mesh = None
    for name, refined_widths, chosen in candidates:
        folder = output / name
        folder.mkdir()
        bands = sorted(
            set(original + [band for s in chosen for band in sensor_bands(s, n)])
        )
        band_path = folder / "refinement.bands"
        band_path.write_text(
            "NS_RZ_BANDS_V1\n"
            + str(len(bands))
            + "\n"
            + "\n".join(" ".join(str(x) for x in row) for row in bands)
            + "\n"
        )
        command = [
            str(args.executable.resolve()),
            f"amr.n_cell={n} {n} {n}",
            f"amr.max_level={len(refined_widths)}",
            "amr.max_grid_size=32",
            "amr.blocking_factor=8",
            "amr.ref_ratio=2",
            "amr.n_error_buf=0",
            "geometry.prob_lo=-1 -1 -1",
            "geometry.prob_hi=1 1 1",
            "geometry.is_periodic=1 1 1",
            "ns.refine_half_width=" + " ".join(str(v) for v in refined_widths),
            f"ns.refine_rz_file={band_path}",
        ]
        usage = bounded(command, folder, folder / "mesh.log", seconds=120, rss_mib=512)
        rows = [
            json.loads(s.split(" ", 1)[1])
            for s in (folder / "mesh.log").read_text().splitlines()
            if s.startswith("NS_MESH_PROBE ")
        ]
        if len(rows) != 1:
            raise ValueError("Mesh probe record missing")
        mesh = rows[0]
        if baseline_mesh is None:
            actual = snapshot["frames"][0]["export"]["blocks"]
            measured_boxes = {
                (lev["level"], tuple(b["index_lo"]), tuple(b["shape"]))
                for lev in mesh["levels"]
                for b in lev["boxes"]
            }
            actual_boxes = {
                (b["level"], tuple(b["index_lo"]), tuple(b["shape"][:3]))
                for b in actual
            }
            if measured_boxes != actual_boxes:
                raise ValueError("Mesh-only probe does not reproduce production boxes")
            baseline_mesh = mesh
        write(folder / "mesh.json", mesh)
        added = mesh["stored_cells"] - baseline_mesh["stored_cells"]
        estimate = (
            max(f["diagnostic"]["peak_rss_mib"] for f in snapshot["frames"])
            / 1024
            * mesh["stored_cells"]
            / baseline_mesh["stored_cells"]
        )
        row = {
            "name": name,
            "design_sensor_labels": [s["request"]["label"] for s in chosen],
            "widths": refined_widths,
            "bands_sha256": sha(band_path),
            "mesh_sha256": sha(folder / "mesh.json"),
            "stored_cells": mesh["stored_cells"],
            "active_cells": mesh["active_cells"],
            "added_cells": added,
            "linear_peak_rss_estimate_gib": estimate,
            "passes_nominal_11_5_gib_screen": estimate < 11.5,
            "resource_usage": usage,
            "mesh_builder_peak_rss_mib": mesh["peak_rss_mib"],
            **score(mesh, sensors),
            "production_ready": False,
        }
        row["coverage_score_per_added_million_cells"] = (
            row["mean_sampled_sensitivity_coverage"] / (added / 1e6) if added else None
        )
        report["candidates"].append(row)
        print(
            json.dumps({k: v for k, v in row.items() if k != "covered_sensor_labels"}),
            flush=True,
        )
    report["complete"] = True
    write(output / "ranking.json", report)


if __name__ == "__main__":
    main()
