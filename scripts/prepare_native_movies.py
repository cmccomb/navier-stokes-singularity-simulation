"""Bounded native-plotfile preparation and full-history three-view rendering.

Use one lossless native frame in scratch at a time. No full Zarr copy is needed.
Source plotfiles, solver inputs, and native audits are never modified.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import numpy as np

from scripts.render_native_3d import (
    SCALES,
    THRESHOLDS,
    Scene,
    composite_axes,
    contours,
    encode_frames,
    fit_camera_scales,
    sample_speed,
    sha,
)
from scripts.render_native_triptych import FORCE_THRESHOLDS, Triptych, center_plane


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def read_planes(cache, frame):
    packed = cache / (frame["path"] + ".bin.gz")
    if sha(packed) != frame["compressed_sha256"]:
        raise ValueError("Packed slice hash mismatch")
    with gzip.open(packed, "rb") as stream:
        raw = stream.read()
    if hashlib.sha256(raw).hexdigest() != frame["slice_sha256"]:
        raise ValueError("Native slice hash mismatch")
    value = np.frombuffer(raw, dtype="<f8").reshape(3, 256, 256, 6)
    if not np.isfinite(value).all():
        raise ValueError("Nonfinite native slice")
    return np.array([value[1], value[0]])


def check_fields(fields, levels, expected, rest):
    peak, energy, volume = 0.0, 0.0, 0.0
    for i, (value, level) in enumerate(zip(fields, levels, strict=True)):
        if not np.isfinite(value).all() or (rest and np.any(value)):
            raise ValueError("Nonfinite field or nonzero rest")
        active = np.ones(value.shape[:3], dtype=bool)
        if i + 1 < len(levels):
            fine = levels[i + 1]
            start = np.rint(
                (np.array(fine["origin"]) - level["origin"]) / level["spacing"]
            ).astype(int)
            size = np.rint(
                np.array(fine["shape"][:3])
                * fine["spacing"]
                / np.array(level["spacing"])
            ).astype(int)
            active[tuple(slice(a, a + b) for a, b in zip(start, size))] = False
        speed = np.linalg.norm(value[..., :3], axis=-1)
        peak = max(peak, float(speed[active].max()))
        dv = float(np.prod(level["spacing"]))
        energy += float(np.sum(speed[active] ** 2) * dv / 2)
        volume += int(active.sum()) * dv
    if abs(volume - 8) > 1e-12 or abs(peak - expected["peak_speed"]) > 1e-12:
        raise ValueError("Native composite diagnostics disagree")
    return {"peak_speed": peak, "energy": energy, "volume": volume}


def export_slices(args):
    """Reproduce the compact cache directly from audited native plotfiles."""
    record = json.loads((args.run / "run.json").read_text())
    if record["status"] != "completed" or not record["validated"]:
        raise ValueError("Require completed native audit")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "source_record_sha256": sha(args.run / "run.json"),
        "slice_exporter_sha256": sha(args.exporter),
        "parameters": record["parameters"],
        "profile": record["profile_manifest"]["parameters"],
        "prefix_complete": False,
        "frames": [],
    }
    with tempfile.TemporaryDirectory(
        prefix="native-slices-", dir=args.output
    ) as scratch:
        for frame in record["native_frames"]:
            raw = Path(scratch) / "frame.bin"
            process = subprocess.run(
                [
                    str(args.exporter),
                    f"plot={args.run / frame['path']}",
                    f"output={raw}",
                    "display_n=256",
                ],
                env=dict(os.environ, OMP_NUM_THREADS="1", OMP_THREAD_LIMIT="1"),
                capture_output=True,
                text=True,
                check=True,
                timeout=300,
            )
            rows = [
                json.loads(line.split(" ", 1)[1])
                for line in process.stdout.splitlines()
                if line.startswith("NS_SLICE_RESULT ")
            ]
            if len(rows) != 1 or rows[0]["time"] != frame["time"]:
                raise ValueError("Slice exporter clock mismatch")
            value = np.fromfile(raw, dtype="<f8").reshape(3, 256, 256, 6)
            if not np.isfinite(value).all() or (
                frame["time"] <= manifest["profile"]["paper_time_cutoff_start"]
                and np.any(value)
            ):
                raise ValueError("Invalid slice values")
            packed = args.output / (frame["path"] + ".bin.gz")
            with (
                raw.open("rb") as source,
                gzip.open(packed, "wb", compresslevel=1) as target,
            ):
                shutil.copyfileobj(source, target)
            manifest["frames"].append(
                {**frame, "slice_sha256": sha(raw), "compressed_sha256": sha(packed)}
            )
    if sha(args.run / "run.json") != manifest["source_record_sha256"]:
        raise ValueError("Source record changed")
    manifest["prefix_complete"] = True
    write(args.output / "manifest.json", manifest)


def prepare(args):
    source, cache, output = args.run, args.slices, args.output
    record = json.loads((source / "run.json").read_text())
    cached = json.loads((cache / "manifest.json").read_text())
    native = record["native_frames"]
    if record["status"] != "completed" or not record["validated"]:
        raise ValueError("Require completed native audit")
    if len(native) != len(cached["frames"]) or not cached["prefix_complete"]:
        raise ValueError("Require complete cached slice history")
    for a, b in zip(native, cached["frames"], strict=True):
        if (
            a["path"] != b["path"]
            or a["time"] != b["time"]
            or a["force_linf_error"] != 0
        ):
            raise ValueError("Cached slices do not match audited native history")
    output.mkdir(parents=True, exist_ok=True)
    source_hash, exporter_hash = sha(source / "run.json"), sha(args.exporter)
    indices = list(range(len(native))) if args.indices is None else args.indices
    manifest = {
        "schema_version": 1,
        "kind": "native-plotfile-movie-preparation",
        "source_record_sha256": source_hash,
        "exporter_sha256": exporter_hash,
        "slice_manifest_sha256": sha(cache / "manifest.json"),
        "renderer_sha256": sha(__file__),
        "machine": args.machine,
        "complete_history": False,
        "validated": False,
        "records": [],
        "sampling": "Native scalar magnitudes, trilinear finest-level reconstruction on all composite native centers; no surface smoothing, decimation, or temporal interpolation",
        "scales": {q: dict(SCALES) for q in ("flow", "force")},
        "limits": [0.0, 0.0],
    }
    write(output / "source-run.json", record)
    environment = dict(os.environ, OMP_NUM_THREADS="1", OMP_THREAD_LIMIT="1")
    import pyvista as pv

    with tempfile.TemporaryDirectory(prefix="native-movie-", dir=output) as scratch:
        raw = Path(scratch) / "frame.bin"
        for index in indices:
            started = time.monotonic()
            frame = native[index]
            item_path = output / f"frame-{index:04d}.json"
            if item_path.exists():
                item = json.loads(item_path.read_text())
                if (
                    item["source_record_sha256"] != source_hash
                    or item["exporter_sha256"] != exporter_hash
                ):
                    raise ValueError("Prepared cache has different provenance")
                if item["path"] != frame["path"] or item["time"] != frame["time"]:
                    raise ValueError("Prepared clock differs")
                for q in ("flow", "force"):
                    meshes = []
                    for k, digest in enumerate(item["geometry"][q]):
                        path = output / f"{q}-{index:04d}-{k}.vtp"
                        if digest is not None and sha(path) != digest:
                            raise ValueError("Prepared geometry changed")
                        meshes.append(pv.read(path) if digest else pv.PolyData())
                    fit_camera_scales(meshes, manifest["scales"][q])
            else:
                if shutil.disk_usage(output).free < 21 * 2**30:
                    raise RuntimeError("Movie preparation disk guard")
                process = subprocess.run(
                    [
                        str(args.exporter),
                        f"plot={source / frame['path']}",
                        f"output={raw}",
                    ],
                    env=environment,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                rows = [
                    json.loads(line.split(" ", 1)[1])
                    for line in process.stdout.splitlines()
                    if line.startswith("NS_VOLUME_RESULT ")
                ]
                if (
                    len(rows) != 1
                    or rows[0]["time"] != frame["time"]
                    or rows[0]["bytes"] != raw.stat().st_size
                ):
                    raise ValueError("Native export schema/clock differs")
                levels = rows[0]["levels"]
                fields = [
                    np.memmap(
                        raw,
                        dtype="<f8",
                        mode="r",
                        offset=level["offset_bytes"],
                        shape=tuple(level["shape"]),
                    )
                    for level in levels
                ]
                diagnostics = check_fields(
                    fields,
                    levels,
                    frame,
                    frame["time"]
                    <= record["profile_manifest"]["parameters"][
                        "paper_time_cutoff_start"
                    ],
                )
                expected_energy = record["history"][frame["step"]]["energy"]
                if not np.isclose(
                    diagnostics["energy"], expected_energy, rtol=1e-11, atol=1e-300
                ):
                    raise ValueError("Energy differs from solver history")
                planes = read_planes(cache, cached["frames"][index])
                for a, p in zip(planes, ("xy", "xz"), strict=True):
                    if not np.array_equal(a, center_plane(fields, levels, p)):
                        raise ValueError("Independent native plane exporters disagree")
                axes = composite_axes(levels)
                item = {
                    "index": index,
                    "path": frame["path"],
                    "time": frame["time"],
                    "source_record_sha256": source_hash,
                    "exporter_sha256": exporter_hash,
                    "levels": levels,
                    "display_shape": [len(a) for a in axes],
                    "level_sha256": [
                        hashlib.sha256(f.tobytes()).hexdigest() for f in fields
                    ],
                    "diagnostics": diagnostics,
                    "geometry": {},
                    "triangles": {},
                    "limits": [
                        float(np.linalg.norm(planes[..., k : k + 3], axis=-1).max())
                        for k in (0, 3)
                    ],
                }
                for q, quantity, thresholds in (
                    ("flow", "velocity", THRESHOLDS),
                    ("force", "force", FORCE_THRESHOLDS),
                ):
                    scalar = sample_speed(fields, levels, axes, quantity)
                    meshes = contours(axes, scalar, thresholds)
                    del scalar
                    fit_camera_scales(meshes, manifest["scales"][q])
                    hashes = []
                    for k, mesh in enumerate(meshes):
                        path = output / f"{q}-{index:04d}-{k}.vtp"
                        if mesh.n_points:
                            # Geometry is unchanged. Constant contour scalars and
                            # display normals can be rebuilt by the lit renderer.
                            mesh.clear_data()
                            mesh.save(path, compression="lz4")
                            hashes.append(sha(path))
                        else:
                            hashes.append(None)
                    item["geometry"][q] = hashes
                    item["triangles"][q] = [m.n_cells for m in meshes]
                del fields
                raw.unlink()
                write(item_path, item)
            manifest["records"].append(item)
            manifest["limits"] = list(np.maximum(manifest["limits"], item["limits"]))
            write(
                output / f"progress{args.worker}.json",
                {
                    "prepared": len(manifest["records"]),
                    "total": len(indices),
                    "time": frame["time"],
                },
            )
            print(
                f"Prepared {index + 1}/{len(native)} t={frame['time']:.6f} in {time.monotonic() - started:.1f}s",
                flush=True,
            )
    if sha(source / "run.json") != source_hash:
        raise ValueError("Source completion record changed")
    manifest["complete_history"] = indices == list(range(len(native)))
    manifest["validated"] = True
    write(output / f"manifest{args.worker}.json", manifest)


def render(args):
    import pyvista as pv

    manifest = json.loads((args.prepared / "manifest.json").read_text())
    cached = json.loads((args.slices / "manifest.json").read_text())
    if not manifest["validated"] or not manifest["complete_history"]:
        raise ValueError("A complete audited preparation is required")
    if sha(args.slices / "manifest.json") != manifest["slice_manifest_sha256"]:
        raise ValueError("Wrong slice archive")
    args.output.mkdir(parents=True, exist_ok=False)
    times = [r["time"] for r in manifest["records"]]
    report = {
        "schema_version": 1,
        "kind": "native-three-view",
        "validated": True,
        "complete_history": True,
        "saved_frames": len(times),
        "frame_times": times,
        "view_order": ["xy", "xz", "isometric"],
        "source_record_sha256": manifest["source_record_sha256"],
        "prepared_manifest_sha256": sha(args.prepared / "manifest.json"),
        "renderer_sha256": sha(__file__),
        "surface_sampling": manifest["sampling"],
        "display_shape": manifest["records"][0]["display_shape"],
        "slice_coordinates": {"xy": 0, "xz": 0},
        "surface_thresholds": {"flow": THRESHOLDS, "force": FORCE_THRESHOLDS},
        "slice_color_max": dict(zip(("flow", "force"), manifest["limits"])),
        "isometric_parallel_scale": {
            q: manifest["scales"][q]["isometric"] for q in ("flow", "force")
        },
        "media": {},
    }
    for j, q in enumerate(("flow", "force")):
        folder = args.output / q
        folder.mkdir()
        scene = Scene(800, 600, scales=manifest["scales"][q])
        triptych = Triptych(
            "velocity" if q == "flow" else "force",
            manifest["limits"][j] or 1,
            THRESHOLDS if q == "flow" else FORCE_THRESHOLDS,
            machine=manifest["machine"],
            saved_states=len(times),
        )
        try:
            for index, record in enumerate(manifest["records"]):
                meshes = []
                for k, digest in enumerate(record["geometry"][q]):
                    path = args.prepared / f"{q}-{index:04d}-{k}.vtp"
                    if digest and sha(path) != digest:
                        raise ValueError("Geometry hash mismatch")
                    meshes.append(pv.read(path) if digest else pv.PolyData())
                scene.set_surfaces(meshes)
                planes = read_planes(args.slices, cached["frames"][index])
                triptych.image(
                    planes, scene.image("isometric"), index, times[index], len(times)
                ).save(folder / f"frame-{index:04d}.png")
                print(f"Rendered {q} {index + 1}/{len(times)}", flush=True)
        finally:
            triptych.close()
            scene.close()
        report["media"][q] = encode_frames(
            folder, args.output / f"kay-{q}-views", times, gif_size=(1600, 640)
        )
    write(args.output / "manifest.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    for name in ("run", "exporter", "slices", "output"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--machine", default="Kay")
    p.add_argument("--indices", type=int, nargs="+")
    p.add_argument("--worker", choices=("", "-even", "-odd"), default="")
    r = sub.add_parser("render")
    for name in ("prepared", "slices", "output"):
        r.add_argument("--" + name, required=True, type=Path)
    s = sub.add_parser("slices")
    for name in ("run", "exporter", "output"):
        s.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    {"prepare": prepare, "render": render, "slices": export_slices}[args.command](args)


if __name__ == "__main__":
    main()
