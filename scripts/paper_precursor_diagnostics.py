"""Measure finite precursor diagnostics from audited, fixed-cube native plotfiles.

Centered derivatives use only active cells whose six neighbors are active on
the same AMR level. This deliberately excludes patch and refinement interfaces;
the reported divergence is an interior sampling diagnostic, not a global
discrete Navier--Stokes residual or an error bound.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from scripts.paper_comparison import snapshot
from scripts.paper_run import sha, write


def active_mask(levels: list[dict], index: int) -> np.ndarray:
    level = levels[index]
    active = np.ones(level["shape"][:3], dtype=bool)
    if index + 1 < len(levels):
        fine = levels[index + 1]
        start = np.rint(
            (np.asarray(fine["origin"]) - level["origin"]) / level["spacing"]
        ).astype(int)
        count = np.rint(
            np.asarray(fine["shape"][:3]) * fine["spacing"] / level["spacing"]
        ).astype(int)
        if np.any(start < 0) or np.any(start + count > active.shape):
            raise ValueError("finer cube does not fit inside the coarse cube")
        active[tuple(slice(a, a + n) for a, n in zip(start, count, strict=True))] = False
    return active


def interior_derivatives(velocity: np.ndarray, active: np.ndarray, spacing: list[float]) -> dict:
    """Return curl and divergence on cells with all same-level neighbors active."""
    if velocity.ndim != 4 or velocity.shape[-1] != 3 or active.shape != velocity.shape[:3]:
        raise ValueError("invalid velocity or mask shape")
    if min(active.shape) < 3 or len(spacing) != 3 or any(dx <= 0 for dx in spacing):
        raise ValueError("invalid derivative grid")
    center = (slice(1, -1),) * 3
    valid = active[center].copy()
    for axis in range(3):
        plus = list(center)
        minus = list(center)
        plus[axis] = slice(2, None)
        minus[axis] = slice(None, -2)
        valid &= active[tuple(plus)] & active[tuple(minus)]

    def derivative(component: int, axis: int) -> np.ndarray:
        plus = list(center)
        minus = list(center)
        plus[axis] = slice(2, None)
        minus[axis] = slice(None, -2)
        return (
            velocity[(*plus, component)] - velocity[(*minus, component)]
        ) / (2 * spacing[axis])

    curl_x = derivative(2, 1) - derivative(1, 2)
    curl_y = derivative(0, 2) - derivative(2, 0)
    curl_z = derivative(1, 0) - derivative(0, 1)
    divergence = derivative(0, 0) + derivative(1, 1) + derivative(2, 2)
    vorticity_sq = curl_x**2 + curl_y**2 + curl_z**2
    count = int(valid.sum())
    if not count:
        raise ValueError("no valid interior derivative cells")
    return {
        "cells": count,
        "peak_vorticity": float(np.sqrt(vorticity_sq[valid].max())),
        "vorticity_squared_sum": float(vorticity_sq[valid].sum()),
        "divergence_linf": float(np.abs(divergence[valid]).max()),
        "divergence_squared_sum": float(np.square(divergence[valid]).sum()),
    }


def measure(raw: Path, levels: list[dict], expected: dict) -> dict:
    if any(len(level["shape"]) != 4 or level["shape"][-1] != 6 for level in levels):
        raise ValueError("expected six-component native levels")
    per_level = []
    peak = energy = force_sq = volume = 0.0
    peak_level = None
    for index, level in enumerate(levels):
        field = np.memmap(
            raw,
            dtype="<f8",
            mode="r",
            offset=level["offset_bytes"],
            shape=tuple(level["shape"]),
        )
        if not np.isfinite(field).all():
            raise ValueError("nonfinite native field")
        active = active_mask(levels, index)
        speed_sq = np.sum(np.square(field[..., :3]), axis=-1)
        dv = float(np.prod(level["spacing"]))
        level_peak = float(np.sqrt(speed_sq[active].max()))
        if level_peak > peak:
            peak, peak_level = level_peak, index
        volume += int(active.sum()) * dv
        energy += float(speed_sq[active].sum()) * dv / 2
        force_sq += float(np.square(field[..., 3:][active]).sum()) * dv
        derivatives = interior_derivatives(field[..., :3], active, level["spacing"])
        per_level.append({"level": index, "active_cells": int(active.sum()), "spacing": level["spacing"], "derivatives": derivatives})
        del field, active, speed_sq
    if not math.isclose(volume, 8, rel_tol=0, abs_tol=1e-12):
        raise ValueError("native active volume differs from the full domain")
    if not math.isclose(peak, expected["peak_speed"], rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("native peak differs from the archived audit")
    if not math.isclose(energy, expected["energy"], rel_tol=1e-9, abs_tol=1e-11):
        raise ValueError("native energy differs from the solver diagnostic")

    support_volume = core_volume = 0.0
    if peak:
        for index, level in enumerate(levels):
            field = np.memmap(
                raw,
                dtype="<f8",
                mode="r",
                offset=level["offset_bytes"],
                shape=tuple(level["shape"]),
            )
            speed_sq = np.sum(np.square(field[..., :3]), axis=-1)
            selected = active_mask(levels, index) & (speed_sq >= (peak / 2) ** 2)
            selected_volume = int(selected.sum()) * float(np.prod(level["spacing"]))
            support_volume += selected_volume
            if index == len(levels) - 1:
                core_volume = selected_volume
            del field, speed_sq, selected
    derivative_volume = sum(
        item["derivatives"]["cells"] * float(np.prod(item["spacing"])) for item in per_level
    )
    vorticity_sq = sum(
        item["derivatives"]["vorticity_squared_sum"] * float(np.prod(item["spacing"])) for item in per_level
    )
    divergence_sq = sum(
        item["derivatives"]["divergence_squared_sum"] * float(np.prod(item["spacing"])) for item in per_level
    )
    return {
        "peak_speed": peak,
        "peak_level": peak_level,
        "kinetic_energy": energy,
        "force_l2": math.sqrt(force_sq / volume),
        "half_peak_support_volume": support_volume,
        "half_peak_support_equivalent_radius": (3 * support_volume / (4 * math.pi)) ** (1 / 3),
        "core_half_peak_volume": core_volume,
        "core_half_peak_equivalent_radius": (
            (3 * core_volume / (4 * math.pi)) ** (1 / 3)
            if peak_level == len(levels) - 1 and core_volume
            else None
        ),
        "derivative_interior_volume": derivative_volume,
        "peak_vorticity": max(item["derivatives"]["peak_vorticity"] for item in per_level),
        "vorticity_rms": math.sqrt(vorticity_sq / derivative_volume),
        "divergence_rms": math.sqrt(divergence_sq / derivative_volume),
        "divergence_linf": max(item["derivatives"]["divergence_linf"] for item in per_level),
        "levels": per_level,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--exporter", type=Path, required=True)
    parser.add_argument("--checker", type=Path, required=True)
    parser.add_argument("--times", type=float, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-disk-free-gib", type=float, default=30)
    args = parser.parse_args()
    run, exporter, checker = (p.resolve(strict=True) for p in (args.run, args.exporter, args.checker))
    record, history = snapshot(run)
    if record["status"] != "completed" or record["validated"] is not True:
        raise ValueError("require a completed, validated native run")
    if not all(math.isfinite(t) for t in args.times) or args.times != sorted(set(args.times)):
        raise ValueError("times must be finite, unique and increasing")
    by_step = {row["step"]: row for row in history}
    selected = []
    for time in args.times:
        matches = [f for f in record["native_frames"] if abs(f["time"] - time) <= 1e-12]
        if len(matches) != 1:
            raise ValueError(f"require one native frame at {time}")
        selected.append(matches[0])
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1,
        "kind": "finite-precursor-native-diagnostics",
        "source_record_sha256": sha(run / "run.json"),
        "source_binary_sha256": record["binary_sha256"],
        "source_profile_sha256": record["profile_manifest"]["sha256"],
        "exporter_sha256": sha(exporter),
        "checker_sha256": sha(checker),
        "analyzer_sha256": sha(Path(__file__)),
        "definitions": {
            "half_peak_support_equivalent_radius": "Radius of a sphere with the total active volume where speed is at least half the global peak; it is not a connected-component radius.",
            "core_half_peak_equivalent_radius": "Same volume-equivalent radius using the finest level only, reported only when the global peak lies on that level; it is not an analytic core radius.",
            "vorticity_and_divergence": "Centered level-local derivatives only where the cell and all six neighbors are active on one AMR level; interfaces and outer boundary cells are excluded.",
            "divergence": "Sampled incompressibility diagnostic, not the full discrete momentum residual or a continuum error bound.",
        },
        "frames": [],
        "validated": False,
    }
    environment = dict(os.environ, OMP_NUM_THREADS="1", OMP_THREAD_LIMIT="1")
    with tempfile.TemporaryDirectory(prefix="precursor-native-", dir=args.output) as temporary:
        raw = Path(temporary) / "frame.bin"
        for frame in selected:
            if shutil.disk_usage(args.output).free < args.min_disk_free_gib * 2**30:
                raise RuntimeError("disk reserve reached before native export")
            command = [str(checker), f"plot={run / frame['path']}", "ns.force=paper", f"ns.table_file={run / 'profile.tbl'}", f"ns.epsilon_tau_ratio={record['parameters']['epsilon_tau_ratio']:.17g}"]
            checked = subprocess.run(command, env=environment, capture_output=True, text=True, check=True, timeout=1800)
            rows = [json.loads(x.split(" ", 1)[1]) for x in checked.stdout.splitlines() if x.startswith("NS_ARCHIVE_RESULT ")]
            if len(rows) != 1 or rows[0]["force_linf_error"] != 0 or abs(rows[0]["time"] - frame["time"]) > 1e-12:
                raise ValueError("native checker failed the source frame")
            exported = subprocess.run([str(exporter), f"plot={run / frame['path']}", f"output={raw}"], env=environment, capture_output=True, text=True, check=True, timeout=1800)
            rows = [json.loads(x.split(" ", 1)[1]) for x in exported.stdout.splitlines() if x.startswith("NS_VOLUME_RESULT ")]
            if len(rows) != 1 or rows[0]["schema_version"] != 1 or rows[0]["dtype"] != "<f8" or rows[0]["bytes"] != raw.stat().st_size or abs(rows[0]["time"] - frame["time"]) > 1e-12:
                raise ValueError("native exporter produced an unexpected frame")
            step = frame["step"]
            if step not in by_step or abs(by_step[step]["time"] - frame["time"]) > 1e-12:
                raise ValueError("native frame is missing its solver diagnostic")
            result = measure(raw, rows[0]["levels"], by_step[step])
            report["frames"].append({"time": frame["time"], "step": step, "source_frame": frame["path"], "native_sha256": sha(raw), "target_l2_error": by_step[step]["l2_error"], "target_linf_error": by_step[step]["linf_error"], **result})
            raw.unlink()
    if sha(run / "run.json") != report["source_record_sha256"]:
        raise ValueError("source run changed while measuring")
    report["validated"] = True
    write(args.output / "summary.json", report)
    print(json.dumps({"frames": len(report["frames"]), "last_time": report["frames"][-1]["time"], "validated": True}))


if __name__ == "__main__":
    main()
