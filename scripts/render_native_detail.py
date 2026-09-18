"""Bounded, source-verified 3D cutaways and native-core circulation movies.

Streamlines are instantaneous, not particle paths. Camera, seeds, transfer
functions and physical core bounds are fixed. Original native fields are read
only; one exported frame occupies temporary scratch per worker.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import pyvista as pv
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageDraw

from scripts.prepare_native_movies import check_fields, write
from scripts.render_native_3d import (
    BACKGROUND,
    COLORS,
    MUTED,
    TEXT,
    THRESHOLDS,
    composite_axes,
    contours,
    encode_frames,
    font,
    sample_speed,
    sha,
)
from scripts.render_native_triptych import FORCE_THRESHOLDS

CORE_HALF = 0.0625
CAMERA = np.array([1.25, -1.65, 0.95])
SCALES = {"flow": 1.45, "force": 1.0, "core": 0.118}
SIZE = (2400, 1440)
PANELS = ((40, 195, 1320, 1080), (1400, 195, 960, 1080))
SEEDS = np.array(
    [
        (r * np.cos(a), r * np.sin(a), z)
        for z in (-0.018, 0.018)
        for r in (0.014, 0.040)
        for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)
    ]
)
DIRECTION_COLORS = ("#edbd78", "#596c83", "#74e3f2")
FORCE_COLORS = (BACKGROUND, "#12364e", "#247f9f", "#57c6d5", "#c0ece7", "#fff0bd")
OPACITY = np.interp(
    np.linspace(0, 5, 256), [0, 1, 2, 3, 4, 5], [0, 0, 0.01, 0.06, 0.22, 0.45]
)


def quarter_cut(mesh):
    """Remove the front x>0, y<0 quadrant without smoothing or new caps."""
    if not mesh.n_points:
        return mesh
    back = mesh.clip(normal=(1, 0, 0), origin=(0, 0, 0), invert=True)
    side = mesh.clip(normal=(1, 0, 0), origin=(0, 0, 0), invert=False)
    side = side.clip(normal=(0, 1, 0), origin=(0, 0, 0), invert=False)
    return back.merge(side, merge_points=True)


def native_core(values, level):
    """Native cell centers become visualization points; no uniform rebinning."""
    dx = np.asarray(level["spacing"])
    grid = pv.ImageData(
        dimensions=values.shape[:3],
        spacing=dx,
        origin=np.asarray(level["origin"]) + dx / 2,
    )
    grid.point_data["velocity"] = np.column_stack(
        [values[..., c].ravel(order="F") for c in range(3)]
    )
    grid.point_data["speed"] = np.linalg.norm(values[..., :3], axis=-1).ravel(order="F")
    grid.point_data["log_force"] = np.log10(
        1 + np.linalg.norm(values[..., 3:], axis=-1)
    ).ravel(order="F")
    return grid


def core_streamlines(grid, max_step=0.5):
    """Fixed seeds, RK45, cell-relative spatial integration; no time advection."""
    if grid["speed"].max() < 1e-10:
        return pv.PolyData()
    result = grid.streamlines_from_source(
        pv.PolyData(SEEDS),
        vectors="velocity",
        integration_direction="both",
        integrator_type=45,
        initial_step_length=0.25,
        min_step_length=0.01,
        max_step_length=max_step,
        step_unit="cl",
        max_steps=3000,
        max_length=0.32,
        terminal_speed=1e-10,
        max_error=1e-8,
        compute_vorticity=False,
    )
    if result.n_points:
        vectors = result["velocity"]
        speed = np.linalg.norm(vectors, axis=1)
        result["axial_fraction"] = np.divide(
            vectors[:, 2], speed, out=np.zeros_like(speed), where=speed > 1e-10
        ).clip(-1, 1)
        if (
            not np.isfinite(result.points).all()
            or np.abs(result.points).max() > CORE_HALF
        ):
            raise ValueError("Invalid streamline geometry")
    return result


class DetailScene:
    def __init__(self, width, height, scale, half):
        self.width, self.height, self.scale, self.half = width, height, scale, half
        self.plotter = pv.Plotter(
            off_screen=True, window_size=(width * 2, height * 2), lighting="none"
        )
        self.plotter.set_background(BACKGROUND)
        self.plotter.ren_win.SetMultiSamples(0)
        if not self.plotter.enable_depth_peeling(number_of_peels=48, occlusion_ratio=0):
            raise RuntimeError("Depth peeling unavailable")
        self.plotter.enable_parallel_projection()
        self.dynamic = []
        for position, intensity, color in (
            ((2, -3, 4), 0.85, "#eaf5ff"),
            ((-3, -1, 1), 0.5, "#a3d8ed"),
            ((0, 3, 2), 1.05, "#fff2d9"),
        ):
            self.plotter.add_light(
                pv.Light(
                    position=position,
                    focal_point=(0, 0, 0),
                    intensity=intensity,
                    color=color,
                )
            )
        self.plotter.add_mesh(
            pv.Box(bounds=(-half, half) * 3).outline(),
            color="#36516b",
            opacity=0.65,
            line_width=1.2,
            lighting=False,
        )
        self.plotter.camera_position = (CAMERA, (0, 0, 0), (0, 0, 1))
        self.plotter.camera.parallel_scale = scale

    def clear(self):
        for actor in self.dynamic:
            self.plotter.remove_actor(actor, render=False)
        self.dynamic.clear()

    def surface(self, mesh, color, opacity):
        if not mesh.n_points:
            return
        actor = self.plotter.add_mesh(
            mesh,
            color=color,
            opacity=opacity,
            smooth_shading=True,
            ambient=0.18,
            diffuse=0.78,
            specular=0.22,
            specular_power=35,
            show_scalar_bar=False,
            reset_camera=False,
            render=False,
        )
        self.dynamic.append(actor)

    def volume(self, core):
        actor = self.plotter.add_volume(
            core,
            scalars="log_force",
            clim=(0, 5),
            # A full-length PyVista transfer table is already in byte units;
            # passing 256 floats in [0, 1] silently rounds opacity to zero.
            opacity=np.rint(OPACITY * 255).astype(np.uint8),
            opacity_unit_distance=0.0125,
            cmap=LinearSegmentedColormap.from_list("force", FORCE_COLORS),
            shade=True,
            ambient=0.35,
            diffuse=0.7,
            specular=0.1,
            show_scalar_bar=False,
            reset_camera=False,
            render=False,
        )
        self.dynamic.append(actor)
        return actor

    def image(self, meshes=()):
        forward = -CAMERA / np.linalg.norm(CAMERA)
        right = np.cross(forward, (0, 0, 1))
        right /= np.linalg.norm(right)
        vertical = np.cross(right, forward)
        for mesh in meshes:
            if not mesh.n_points:
                continue
            x = self.width / 2 + mesh.points @ right * self.height / (2 * self.scale)
            y = self.height / 2 - mesh.points @ vertical * self.height / (
                2 * self.scale
            )
            if (
                x.min() < 20
                or x.max() > self.width - 20
                or y.min() < 20
                or y.max() > self.height - 20
            ):
                raise ValueError("Detail camera clips a visible surface")
        self.plotter.reset_camera_clipping_range()
        # Actor updates are batched with render=False. Screenshot alone can reuse
        # the previous framebuffer after the first frame on VTK/macOS.
        self.plotter.render()
        result = Image.fromarray(self.plotter.screenshot(return_img=True)).resize(
            (self.width, self.height), Image.Resampling.LANCZOS
        )
        draw = ImageDraw.Draw(result)
        origin = np.array((85, self.height - 85))
        for d, label in enumerate("xyz"):
            delta = 48 * np.array([right[d], -vertical[d]])
            end = origin + delta
            draw.line([tuple(origin), tuple(end)], fill=MUTED, width=2)
            draw.text(
                tuple(end + 12 * delta / np.linalg.norm(delta)),
                label,
                fill=TEXT,
                font=font(22),
                anchor="mm",
            )
        return result

    def close(self):
        self.plotter.close()


def compose(images, quantity, native, index, count, stream_count):
    image = Image.new("RGB", SIZE, BACKGROUND)
    draw = ImageDraw.Draw(image)
    for content, (x, y, _, _) in zip(images, PANELS, strict=True):
        image.paste(content, (x, y))

    def text(x, y, value, size=26, color=MUTED, bold=False):
        draw.text((x, y), value, fill=color, font=font(size, bold))

    text(
        52,
        30,
        "FLOW / THREE DIMENSIONS"
        if quantity == "flow"
        else "FORCING / THREE DIMENSIONS",
        22,
        "#74d7e7",
        True,
    )
    text(
        52,
        65,
        "Inside the concentrating flow"
        if quantity == "flow"
        else "Inside the applied force",
        49,
        TEXT,
        True,
    )
    text(1765, 38, f"t = {native['time']:.6f}", 32, TEXT)
    text(1765, 84, f"Saved state {index + 1:03d} / {count}", 24)
    text(52, 155, "01   Cutaway overview", 30, TEXT)
    text(
        1400,
        155,
        "02   Core circulation" if quantity == "flow" else "02   Native core volume",
        30,
        TEXT,
    )
    draw.line((1376, 160, 1376, 1285), fill="#263c51", width=1)
    thresholds = THRESHOLDS if quantity == "flow" else FORCE_THRESHOLDS
    text(
        52,
        1280,
        "Speed surfaces |u|" if quantity == "flow" else "Force surfaces |f|",
        25,
        TEXT,
    )
    for j, (value, color) in enumerate(zip(thresholds, COLORS, strict=True)):
        x = 380 + j * 175
        draw.ellipse((x, 1288, x + 16, 1304), fill=color)
        text(x + 28, 1280, f"{value:g}", 25, TEXT)
    text(52, 1325, "Front quarter removed · reference cube ±0.5", 24)
    if quantity == "flow":
        text(1400, 1277, "Axial fraction u_z / |u|", 24, TEXT)
        for j, (label, color) in enumerate(
            zip(("−1 down", "0 horizontal", "+1 up"), DIRECTION_COLORS, strict=True)
        ):
            text(1400 + j * 285, 1313, label, 23, color)
        text(
            1400,
            1350,
            f"{stream_count} branches · direction, not speed · not particle tracks",
            19,
        )
    else:
        text(1400, 1277, "|f| · fixed log color / opacity mapping", 24, TEXT)
        for j, (label, color) in enumerate(
            zip(
                ("10", "100", "1,000", "10,000", "≥100,000"),
                FORCE_COLORS[1:],
                strict=True,
            )
        ):
            text(1400 + j * 183, 1313, label, 22, color)
        text(1400, 1350, "Opacity reveals structure · not material density", 21)
    text(1400, 200, "Core window ±0.0625 · independent fixed scale", 21)
    draw.line((52, 1384, 2348, 1384), fill="#263c51", width=1)
    text(
        52,
        1403,
        "Kay · 128³ base / 5 levels · model units · all saved states · no temporal interpolation",
        21,
    )
    if native["peak_speed"] == 0:
        text(445, 690, "Initially at rest", 34, TEXT)
        text(1645, 690, "Initially at rest", 30, TEXT)
    return image


def render_frame(fields, levels, scenes, native, index, count, quantities):
    core = native_core(fields[-1], levels[-1])
    lines = core_streamlines(core) if "flow" in quantities else pv.PolyData()
    axes = composite_axes(levels)
    images = {}
    stats = {
        "streamline_branches": lines.n_cells,
        "core_force_max": float(10 ** core["log_force"].max() - 1),
    }
    for quantity in quantities:
        overview, detail = scenes[quantity], scenes["core"]
        overview.clear()
        detail.clear()
        scalar = sample_speed(
            fields, levels, axes, "velocity" if quantity == "flow" else "force"
        )
        full = contours(
            axes, scalar, THRESHOLDS if quantity == "flow" else FORCE_THRESHOLDS
        )
        del scalar
        cut = [quarter_cut(mesh) for mesh in full]
        for mesh, color, opacity in zip(cut, COLORS, (0.30, 0.58, 1.0), strict=True):
            overview.surface(mesh, color, opacity)
        main_image = overview.image(cut)
        if quantity == "flow":
            context = (
                core.contour([1.5, 4.0], scalars="speed")
                if core["speed"].max() > 1.5
                else pv.PolyData()
            )
            context = quarter_cut(context)
            detail.surface(context, "#719bad", 0.13)
            if lines.n_points:
                tube = lines.tube(radius=0.00025, n_sides=10, capping=True)
                actor = detail.plotter.add_mesh(
                    tube,
                    scalars="axial_fraction",
                    clim=(-1, 1),
                    cmap=LinearSegmentedColormap.from_list("axial", DIRECTION_COLORS),
                    smooth_shading=True,
                    ambient=0.3,
                    diffuse=0.7,
                    specular=0.25,
                    show_scalar_bar=False,
                    reset_camera=False,
                    render=False,
                )
                detail.dynamic.append(actor)
            detail_image = detail.image((context, lines))
        else:
            if core["log_force"].max() > 1:
                detail.volume(core)
            detail_image = detail.image()
        images[quantity] = compose(
            (main_image, detail_image), quantity, native, index, count, lines.n_cells
        )
        overview.clear()
        detail.clear()
        del full, cut
    return images, stats


def run(args):
    record = json.loads((args.run / "run.json").read_text())
    source_hash = sha(args.run / "run.json")
    if record["status"] != "completed" or not record["validated"]:
        raise ValueError("A completed native audit is required")
    native = record["native_frames"]
    prepared = json.loads((args.prepared / "manifest.json").read_text())
    if (
        not prepared["validated"]
        or not prepared["complete_history"]
        or prepared["source_record_sha256"] != source_hash
        or prepared["exporter_sha256"] != sha(args.exporter)
    ):
        raise ValueError("Prepared native evidence does not match source")
    args.output.mkdir(parents=True, exist_ok=False)
    if args.indices is not None:
        indices = args.indices
    elif args.worker is not None:
        indices = list(range(args.worker, len(native), 2))
    else:
        indices = list(range(len(native)))
    if (
        not indices
        or sorted(set(indices)) != indices
        or not all(0 <= i < len(native) for i in indices)
    ):
        raise ValueError("Indices must be unique, ordered native state indices")
    for q in ("flow", "force"):
        (args.output / q).mkdir()
    script_hashes = {
        name: sha(Path(__file__).with_name(name))
        for name in (
            "render_native_detail.py",
            "render_native_3d.py",
            "prepare_native_movies.py",
        )
    }
    manifest = {
        "schema_version": 1,
        "kind": "native-detail-3d",
        "source_record_sha256": source_hash,
        "exporter_sha256": sha(args.exporter),
        "renderer_sha256": script_hashes,
        "frame_times": [native[i]["time"] for i in indices],
        "indices": indices,
        "records": [],
        "validated": False,
        "complete_history": False,
        "scales": SCALES,
        "camera": CAMERA.tolist(),
        "core_half_width": CORE_HALF,
        "native_frame_count": len(native),
        "seed_points": SEEDS.tolist(),
        "streamline_policy": "Instantaneous RK45 in finest native patch; fixed 32 seeds; both directions; no temporal advection. Tube radius is a display choice.",
        "cutaway": "Remove x>0 and y<0 quadrant from all overview surfaces; no caps or smoothing.",
        "force_volume": {
            "log_range": [0, 5],
            "opacity": OPACITY.tolist(),
            "opacity_unit_distance": 0.0125,
        },
        "media": {},
    }
    scenes = {
        q: DetailScene(PANELS[0][2], PANELS[0][3], SCALES[q], 0.5)
        for q in ("flow", "force")
    }
    scenes["core"] = DetailScene(PANELS[1][2], PANELS[1][3], SCALES["core"], CORE_HALF)
    try:
        with tempfile.TemporaryDirectory(
            prefix="detail-native-", dir=args.output
        ) as scratch:
            for index in indices:
                started = time.monotonic()
                f = native[index]
                if shutil.disk_usage(args.output).free < 22 * 2**30:
                    raise RuntimeError("20 GiB reserve plus bounded scratch guard")
                raw = Path(scratch) / "frame.bin"
                proc = subprocess.run(
                    [
                        str(args.exporter),
                        f"plot={args.run / f['path']}",
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
                    or rows[0]["time"] != f["time"]
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
                evidence = prepared["records"][index]
                hashes = [hashlib.sha256(v.tobytes()).hexdigest() for v in fields]
                if levels != evidence["levels"] or hashes != evidence["level_sha256"]:
                    raise ValueError("Native field hashes differ")
                diagnostic = check_fields(fields, levels, f, f["time"] <= 0.55)
                if not np.isclose(
                    diagnostic["energy"],
                    record["history"][f["step"]]["energy"],
                    rtol=1e-11,
                    atol=1e-300,
                ):
                    raise ValueError("Native energy differs")
                images, stats = render_frame(
                    fields, levels, scenes, f, index, len(native), ("flow", "force")
                )
                item = dict(
                    index=index,
                    time=f["time"],
                    path=f["path"],
                    native_level_sha256=hashes,
                    diagnostics=diagnostic,
                    **stats,
                    images={},
                )
                for q, image in images.items():
                    path = args.output / q / f"frame-{index:04d}.png"
                    image.save(path)
                    item["images"][q] = {
                        "name": str(path.relative_to(args.output)),
                        "sha256": sha(path),
                    }
                manifest["records"].append(item)
                write(args.output / f"frame-{index:04d}.json", item)
                write(
                    args.output / "progress.json",
                    {
                        "rendered": len(manifest["records"]),
                        "total": len(indices),
                        "index": index,
                        "time": f["time"],
                        "seconds": time.monotonic() - started,
                    },
                )
                del fields, images
                raw.unlink()
                gc.collect()
                print(
                    f"Detail {len(manifest['records'])}/{len(indices)} native={index} t={f['time']:.6f} {time.monotonic() - started:.1f}s",
                    flush=True,
                )
    finally:
        for scene in scenes.values():
            scene.close()
    if sha(args.run / "run.json") != source_hash:
        raise ValueError("Source record changed")
    manifest["validated"] = True
    manifest["complete_history"] = indices == list(range(len(native)))
    write(args.output / "manifest.json", manifest)


def assemble(args):
    """Merge disjoint workers only when every native state and hash agrees."""
    prepared = json.loads((args.prepared / "manifest.json").read_text())
    if not prepared["validated"] or not prepared["complete_history"]:
        raise ValueError("Complete prepared evidence required")
    parts = [json.loads((part / "manifest.json").read_text()) for part in args.parts]
    variable = {"records", "indices", "frame_times", "complete_history", "media"}
    identity = {k: v for k, v in parts[0].items() if k not in variable}
    records = {}
    for folder, part in zip(args.parts, parts, strict=True):
        if (
            not part["validated"]
            or {k: v for k, v in part.items() if k not in variable} != identity
        ):
            raise ValueError("Worker rendering identities differ")
        if (
            part["source_record_sha256"] != prepared["source_record_sha256"]
            or part["exporter_sha256"] != prepared["exporter_sha256"]
            or part["native_frame_count"] != len(prepared["records"])
        ):
            raise ValueError("Worker source differs")
        if part["indices"] != [r["index"] for r in part["records"]] or part[
            "frame_times"
        ] != [r["time"] for r in part["records"]]:
            raise ValueError("Worker clocks differ")
        for item in part["records"]:
            i = item["index"]
            if i in records or i < 0 or i >= len(prepared["records"]):
                raise ValueError("Duplicate or invalid native state")
            expected = prepared["records"][i]
            if (
                item["time"] != expected["time"]
                or item["native_level_sha256"] != expected["level_sha256"]
            ):
                raise ValueError("Native state identity differs")
            for q in ("flow", "force"):
                image = item["images"][q]
                if (
                    image["name"] != f"{q}/frame-{i:04d}.png"
                    or sha(folder / image["name"]) != image["sha256"]
                ):
                    raise ValueError("Rendered frame changed")
            records[i] = (folder, item)
    if sorted(records) != list(range(len(prepared["records"]))):
        raise ValueError("Missing native states")
    args.output.mkdir(parents=True, exist_ok=False)
    for q in ("flow", "force"):
        (args.output / q).mkdir()
    for i in sorted(records):
        folder, item = records[i]
        for image in item["images"].values():
            shutil.copy2(folder / image["name"], args.output / image["name"])
    manifest = dict(parts[0])
    manifest.update(
        records=[records[i][1] for i in sorted(records)],
        indices=sorted(records),
        frame_times=[records[i][1]["time"] for i in sorted(records)],
        complete_history=True,
        media={},
    )
    write(args.output / "manifest.json", manifest)


def encode(args):
    manifest = json.loads((args.output / "manifest.json").read_text())
    if not manifest["validated"] or not manifest["complete_history"]:
        raise ValueError("Cannot encode incomplete history")
    if (
        manifest["indices"] != list(range(manifest["native_frame_count"]))
        or manifest["indices"] != [r["index"] for r in manifest["records"]]
        or manifest["frame_times"] != [r["time"] for r in manifest["records"]]
    ):
        raise ValueError("Movie must include every native state")
    for record in manifest["records"]:
        for image in record["images"].values():
            if sha(args.output / image["name"]) != image["sha256"]:
                raise ValueError("Rendered frame changed")
    for q in ("flow", "force"):
        manifest["media"][q] = encode_frames(
            args.output / q,
            args.output / f"kay-{q}-detail",
            manifest["frame_times"],
            gif_size=(1200, 720),
        )
    write(args.output / "manifest.json", manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("render", "assemble", "encode"))
    parser.add_argument("--output", type=Path, required=True)
    for name in ("run", "exporter", "prepared"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--indices", type=int, nargs="+")
    parser.add_argument("--worker", type=int, choices=(0, 1))
    parser.add_argument("--parts", type=Path, nargs="+")
    args = parser.parse_args()
    if args.command == "render":
        if not all((args.run, args.exporter, args.prepared)):
            parser.error("render requires run, exporter, prepared")
        run(args)
    elif args.command == "assemble":
        if not args.parts or not args.prepared:
            parser.error("assemble requires parts and prepared")
        assemble(args)
    else:
        encode(args)
