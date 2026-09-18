"""Map the completed 3D trajectory into a conservative revolved design footprint.

This is a mesh-design diagnostic, not an axisymmetric solver or an error bound.
Azimuthal maxima retain localized activity; composite integrals exclude covered
coarse cells. Level-local gradients use one-sided differences at patch edges.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from scripts.prepare_native_movies import check_fields, write
from scripts.render_native_3d import sha

METRICS = ("speed", "force", "velocity_gradient", "force_gradient")


def design_edges(finest_dx):
    positive = np.geomspace(finest_dx / 2, np.sqrt(2), 256)
    return np.r_[0, positive], np.r_[-positive[::-1], 0, positive]


def gradient_norm(vectors, spacing):
    result = np.zeros(vectors.shape[:3])
    for c in range(3):
        for d in range(3):
            derivative = np.gradient(vectors[..., c], spacing[d], axis=d, edge_order=2)
            result += derivative * derivative
    return np.sqrt(result)


def active_mask(levels, i):
    level = levels[i]
    mask = np.ones(level["shape"][:3], dtype=bool)
    if i + 1 < len(levels):
        fine = levels[i + 1]
        start = np.rint(
            (np.array(fine["origin"]) - level["origin"]) / level["spacing"]
        ).astype(int)
        size = np.rint(
            np.array(fine["shape"][:3]) * fine["spacing"] / np.array(level["spacing"])
        ).astype(int)
        mask[tuple(slice(a, a + b) for a, b in zip(start, size, strict=True))] = False
    return mask


def frame_maps(fields, levels, r_edges, z_edges):
    shape = (len(r_edges) - 1, len(z_edges) - 1)
    maps = {name: np.zeros(shape) for name in METRICS}
    sums = {name: np.zeros(shape) for name in METRICS}
    volume = np.zeros(shape)
    spacing = np.zeros(shape)
    for i, (values, level) in enumerate(zip(fields, levels, strict=True)):
        mask = active_mask(levels, i)
        axes = [
            lo + (np.arange(n) + 0.5) * dx
            for lo, dx, n in zip(
                level["origin"], level["spacing"], values.shape[:3], strict=True
            )
        ]
        radial = np.hypot(axes[0][:, None], axes[1][None, :])
        rb = np.searchsorted(r_edges, radial, side="right") - 1
        zb = np.searchsorted(z_edges, axes[2], side="right") - 1
        ids = np.broadcast_to(
            rb[:, :, None] * shape[1] + zb[None, None, :], mask.shape
        )[mask]
        if ids.min() < 0 or ids.max() >= np.prod(shape):
            raise ValueError("Design bins do not contain the full native domain")
        dv = float(np.prod(level["spacing"]))
        volume += np.bincount(ids, minlength=volume.size).reshape(shape) * dv
        np.maximum.at(spacing.ravel(), ids, max(level["spacing"]))
        for name, scalar in (
            ("speed", np.linalg.norm(values[..., :3], axis=-1)),
            ("force", np.linalg.norm(values[..., 3:], axis=-1)),
            ("velocity_gradient", gradient_norm(values[..., :3], level["spacing"])),
            ("force_gradient", gradient_norm(values[..., 3:], level["spacing"])),
        ):
            selected = scalar[mask]
            np.maximum.at(maps[name].ravel(), ids, selected)
            sums[name] += np.bincount(
                ids, weights=selected * selected * dv, minlength=volume.size
            ).reshape(shape)
    if abs(volume.sum() - 8) > 1e-12:
        raise ValueError("Composite action-map volume differs")
    return maps, sums, volume, spacing


def audit(args):
    source = json.loads((args.run / "run.json").read_text())
    source_hash = sha(args.run / "run.json")
    prepared = json.loads((args.prepared / "manifest.json").read_text())
    if (
        not source["validated"]
        or source["status"] != "completed"
        or not prepared["complete_history"]
        or not prepared["validated"]
        or prepared["source_record_sha256"] != source_hash
        or prepared["exporter_sha256"] != sha(args.exporter)
    ):
        raise ValueError("Completed source and matching native hash evidence required")
    args.output.mkdir(parents=True, exist_ok=False)
    r_edges, z_edges = design_edges(
        min(prepared["records"][-1]["levels"][-1]["spacing"])
    )
    report = {
        "schema_version": 1,
        "kind": "native-revolved-action-audit",
        "source_record_sha256": source_hash,
        "exporter_sha256": sha(args.exporter),
        "auditor_sha256": sha(__file__),
        "validated": False,
        "complete_history": False,
        "r_edges": r_edges.tolist(),
        "z_edges": z_edges.tolist(),
        "records": [],
        "scope": "All azimuths, active native cells, all saved states. Maxima are revolved for mesh design only; the simulation remains fully 3D. Squared integrals are volume-weighted. Gradients are level-local second-order differences, one-sided at patch boundaries; neither residuals nor convergence errors.",
    }
    union = {name: np.zeros((len(r_edges) - 1, len(z_edges) - 1)) for name in METRICS}
    with tempfile.TemporaryDirectory(
        prefix="action-native-", dir=args.output
    ) as scratch:
        for index, frame in enumerate(source["native_frames"]):
            started = time.monotonic()
            if shutil.disk_usage(args.output).free < 22 * 2**30:
                raise RuntimeError("20 GiB reserve plus bounded scratch guard")
            raw = Path(scratch) / "frame.bin"
            proc = subprocess.run(
                [
                    str(args.exporter),
                    f"plot={args.run / frame['path']}",
                    f"output={raw}",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=300,
                env=dict(os.environ, OMP_NUM_THREADS="1", OMP_THREAD_LIMIT="1"),
            )
            rows = [
                json.loads(line.split(" ", 1)[1])
                for line in proc.stdout.splitlines()
                if line.startswith("NS_VOLUME_RESULT ")
            ]
            if (
                len(rows) != 1
                or rows[0]["time"] != frame["time"]
                or rows[0]["bytes"] != raw.stat().st_size
                or rows[0]["dtype"] != "<f8"
                or rows[0]["order"] != "xyz-component"
            ):
                raise ValueError("Native export identity differs")
            levels = rows[0]["levels"]
            fields = [
                np.memmap(
                    raw,
                    dtype="<f8",
                    mode="r",
                    offset=l["offset_bytes"],
                    shape=tuple(l["shape"]),
                )
                for l in levels
            ]
            hashes = [hashlib.sha256(v.tobytes()).hexdigest() for v in fields]
            evidence = prepared["records"][index]
            if hashes != evidence["level_sha256"] or levels != evidence["levels"]:
                raise ValueError("Native field hashes differ")
            check = check_fields(
                fields,
                levels,
                frame,
                frame["time"]
                <= source["profile_manifest"]["parameters"]["paper_time_cutoff_start"],
            )
            maps, sums, volume, spacing = frame_maps(fields, levels, r_edges, z_edges)
            if not np.isclose(
                sums["speed"].sum() / 2, check["energy"], rtol=1e-12, atol=1e-300
            ):
                raise ValueError(
                    "Action-map energy differs from native composite energy"
                )
            maxima = {name: float(maps[name].max()) for name in METRICS}
            for name in METRICS:
                if maxima[name] > 0:
                    np.maximum(union[name], maps[name] / maxima[name], out=union[name])
            path = args.output / f"frame-{index:04d}.npz"
            np.savez_compressed(
                path,
                **maps,
                **{f"{name}_squared_integral": v for name, v in sums.items()},
                volume=volume,
                native_spacing=spacing,
            )
            report["records"].append(
                {
                    "index": index,
                    "time": frame["time"],
                    "native_level_sha256": hashes,
                    "path": path.name,
                    "sha256": sha(path),
                    "maxima": maxima,
                    "squared_integrals": {k: float(v.sum()) for k, v in sums.items()},
                }
            )
            write(
                args.output / "progress.json",
                {
                    "frames": len(report["records"]),
                    "total": len(source["native_frames"]),
                    "time": frame["time"],
                    "seconds": time.monotonic() - started,
                },
            )
            del fields, maps, sums
            raw.unlink()
            print(
                f"Action {index + 1}/{len(source['native_frames'])} t={frame['time']:.6f} {time.monotonic() - started:.1f}s",
                flush=True,
            )
    if sha(args.run / "run.json") != source_hash:
        raise ValueError("Source changed")
    np.savez_compressed(args.output / "union.npz", **union)
    report.update(
        validated=True,
        complete_history=True,
        union_sha256=sha(args.output / "union.npz"),
    )
    write(args.output / "manifest.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "prepared", "exporter", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    audit(parser.parse_args())
