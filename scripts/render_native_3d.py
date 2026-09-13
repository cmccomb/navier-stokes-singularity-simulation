"""Lit, fixed-camera speed surfaces from audited native AMReX volumes.

Reconstruct scalar magnitudes linearly within each native level for display,
selecting the finest containing level. This is visualization interpolation,
not additional solver resolution. Never smooth geometry or interpolate time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import GifImagePlugin, Image, ImageColor, ImageDraw, ImageFont

from spaces.native_explorer.reader import NativeDataset

BACKGROUND = "#07111f"
TEXT = "#e9f1f5"
MUTED = "#9fb5c5"
THRESHOLDS = (0.15, 1.5, 4.0)
COLORS = ("#328bb5", "#6cdef0", "#e4fbff")
OPACITIES = (0.20, 0.40, 1.0)
CAMERAS = {
    "isometric": ((1, -1, 1), (0, 0, 1)),
    "side": ((0, -2, 0), (0, 0, 1)),
    "top": ((0, 0, 2), (0, 1, 0)),
    "core": ((1, -1, 1), (0, 0, 1)),
}
SCALES = {"isometric": 1.05, "side": 1.05, "top": 1.05, "core": 0.36}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def composite_axes(levels):
    """Retain native centers outside finer intervals; finest centers in core.

    A tensor-product display mesh, not a replacement simulation mesh. For the
    five-level archive this is 192 samples per axis, with native core spacing.
    """
    axes = []
    for d in range(3):
        axis = np.empty(0)
        for level in levels:
            lo, dx, n = level["origin"][d], level["spacing"][d], level["shape"][d]
            hi = lo + n * dx
            axis = np.r_[
                axis[(axis < lo) | (axis >= hi)], lo + (np.arange(n) + 0.5) * dx
            ]
        axes.append(np.sort(axis))
    return axes


def interpolate_scalar(values, coordinates):
    """Separable trilinear reconstruction, clamped at level cell centers.

    Convex weights cannot exceed the native scalar range. Do not interpolate
    vector magnitudes by taking the magnitude of an interpolated vector.
    """
    result = values
    for d, coordinate in enumerate(coordinates):
        coordinate = np.clip(coordinate, 0, values.shape[d] - 1)
        lo = np.floor(coordinate).astype(int)
        hi = np.minimum(lo + 1, values.shape[d] - 1)
        shape = [1] * 3
        shape[d] = len(lo)
        weight = (coordinate - lo).reshape(shape)
        result = (
            np.take(result, lo, axis=d) * (1 - weight)
            + np.take(result, hi, axis=d) * weight
        )
    return result


def sample_speed(fields, levels, axes, quantity="velocity"):
    """Native scalar reconstruction with fine-over-coarse ownership."""
    if quantity not in {"velocity", "force"}:
        raise ValueError("Unknown vector quantity")
    components = slice(0, 3) if quantity == "velocity" else slice(3, 6)
    output = np.full(tuple(len(x) for x in axes), np.nan, dtype=np.float64)
    for field, level in zip(fields, levels, strict=True):
        native = np.linalg.norm(field[..., components], axis=-1)
        indices, coordinates = [], []
        for d, axis in enumerate(axes):
            lo, dx, n = level["origin"][d], level["spacing"][d], level["shape"][d]
            ids = np.flatnonzero((axis >= lo) & (axis < lo + n * dx))
            indices.append(ids)
            coordinates.append((axis[ids] - lo) / dx - 0.5)
        output[np.ix_(*indices)] = interpolate_scalar(native, coordinates)
    if not np.isfinite(output).all():
        raise ValueError("Display points are not covered by finite native values")
    return output


def audited_fields(dataset, index):
    fields = []
    for array, expected in zip(
        dataset.arrays, dataset.manifest["frames"][index]["level_sha256"], strict=True
    ):
        value = np.asarray(array[index])
        if hashlib.sha256(value.tobytes()).hexdigest() != expected:
            raise ValueError("Native field hash failed: missing or changed data")
        fields.append(value)
    return fields


def contours(axes, scalar, thresholds=THRESHOLDS):
    import pyvista as pv

    grid = pv.RectilinearGrid(*axes)
    grid.point_data["speed"] = scalar.ravel(order="F")
    meshes = []
    for value in thresholds:
        if scalar.min() < value < scalar.max():
            mesh = grid.contour([value], scalars="speed", compute_normals=True)
            if mesh.n_points and not np.isfinite(mesh.points).all():
                raise ValueError("Nonfinite contour geometry")
        else:
            mesh = pv.PolyData()
        meshes.append(mesh)
    return meshes


class Scene:
    def __init__(self, width=1600, height=1200, scales=None):
        import pyvista as pv

        self.pv = pv
        self.width, self.height = width, height
        self.scales = dict(SCALES if scales is None else scales)
        self.plotter = pv.Plotter(
            off_screen=True, window_size=(width * 2, height * 2), lighting="none"
        )
        self.plotter.set_background(BACKGROUND)
        self.plotter.ren_win.SetMultiSamples(0)
        self.depth_peeling = self.plotter.enable_depth_peeling(
            number_of_peels=32, occlusion_ratio=0.0
        )
        if not self.depth_peeling:
            raise RuntimeError("Correct translucent rendering requires depth peeling")
        self.plotter.enable_parallel_projection()
        for position, intensity, color in (
            ((2, -3, 4), 1.0, "#eaf5ff"),
            ((-3, -1, 1), 0.55, "#72c9f5"),
            ((0, 3, 2), 0.9, "#ffffff"),
        ):
            self.plotter.add_light(
                pv.Light(
                    position=position,
                    focal_point=(0, 0, 0),
                    color=color,
                    intensity=intensity,
                )
            )
        # Fixed physical frame: no peak tracking, camera orbit, or auto-zoom.
        self.boxes = [
            self.plotter.add_mesh(
                pv.Box(bounds=(-b, b, -b, b, -b, b)).outline(),
                color="#29445d",
                line_width=1,
                lighting=False,
            )
            for b in (0.5, 0.125)
        ]
        self.actors = []

    def set_surfaces(self, meshes):
        self.meshes = meshes
        for actor, _ in self.actors:
            self.plotter.remove_actor(actor, render=False)
        self.actors = []
        for mesh, color, opacity in zip(meshes, COLORS, OPACITIES, strict=True):
            if mesh.n_points:
                actor = self.plotter.add_mesh(
                    mesh,
                    color=color,
                    opacity=opacity,
                    smooth_shading=True,
                    ambient=0.22,
                    diffuse=0.72,
                    specular=0.35,
                    specular_power=28,
                    show_scalar_bar=False,
                    reset_camera=False,
                    render=False,
                )
                self.actors.append((actor, opacity))

    def image(self, view):
        position, up = CAMERAS[view]
        for i, box in enumerate(self.boxes):
            box.SetVisibility(i == (1 if view == "core" else 0))
        for actor, opacity in self.actors:
            actor.SetVisibility(view != "core" or opacity != OPACITIES[0])
        self.plotter.camera_position = (position, (0, 0, 0), up)
        self.plotter.camera.parallel_scale = self.scales[view]
        # Guard the rendered flow against the caption bands and image edges.
        # Actual vertices are checked, not just a potentially loose bounding box.
        forward = -np.asarray(position, dtype=float)
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        vertical = np.cross(right, forward)
        for i, mesh in enumerate(self.meshes):
            if not mesh.n_points or (view == "core" and i == 0):
                continue
            pixels_x = self.width / 2 + (mesh.points @ right) * self.height / (
                2 * self.scales[view]
            )
            pixels_y = self.height / 2 - (mesh.points @ vertical) * self.height / (
                2 * self.scales[view]
            )
            margin = self.width / 1600
            if (
                pixels_x.min() < 24 * margin
                or pixels_x.max() > self.width - 24 * margin
                or pixels_y.min() < 142 * margin
                or pixels_y.max() > self.height - 132 * margin
            ):
                raise ValueError(f"Flow would be clipped in the fixed {view} camera")
        self.plotter.reset_camera_clipping_range()
        self.plotter.render()
        image = Image.fromarray(self.plotter.screenshot(return_img=True))
        return image.resize((self.width, self.height), Image.Resampling.LANCZOS)

    def close(self):
        self.plotter.close()


def font(size, bold=False):
    import matplotlib

    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(
        str(Path(matplotlib.get_data_path()) / "fonts/ttf" / name), size
    )


def caption(image, view, record, total, index):
    w, h = image.size
    draw = ImageDraw.Draw(image)
    scale = w / 1600

    def text(x, y, value, size=22, fill=MUTED, bold=False):
        draw.text(
            (int(x * scale), int(y * scale)),
            value,
            fill=fill,
            font=font(round(size * scale), bold),
        )

    draw.rectangle((0, 0, w, round(138 * scale)), fill=BACKGROUND)
    text(54, 27, "Velocity in three dimensions", 34, TEXT, True)
    label = {
        "isometric": "Isometric",
        "side": "Side · looking along y",
        "top": "Top · looking along z",
        "core": "Core detail · fixed isometric camera",
    }[view]
    text(54, 77, f"Oliver · from rest · {label}", 24)
    text(1165, 34, f"t = {record['time']:.6f}", 26, TEXT)
    text(1165, 79, f"Saved state {index + 1:03d} / {total}", 21)
    # Legend uses fixed values, not a changing per-frame percentile.
    y = h / scale - 111
    draw.rectangle((0, round((y - 18) * scale), w, h), fill=BACKGROUND)
    text(54, y, "Speed surfaces |u|", 22, TEXT)
    for j, (value, color) in enumerate(zip(THRESHOLDS, COLORS)):
        if view == "core" and j == 0:
            continue
        x = 345 + j * 155
        draw.ellipse(
            (
                round(x * scale),
                round((y + 3) * scale),
                round((x + 17) * scale),
                round((y + 20) * scale),
            ),
            fill=color,
        )
        text(x + 29, y - 1, f"{value:g}", 22, TEXT)
    text(930, y, f"Native peak  {record['composite_peak_speed']:.3f}", 22)
    text(
        54,
        y + 42,
        "Cube ±0.125 · outer 0.15 surface hidden · fixed scale · model units"
        if view == "core"
        else "Fixed physical scale · cube ±0.5 · model units · linear display reconstruction",
        19,
    )
    text(
        54,
        y + 73,
        "All saved states · 5 states/s · nonuniform simulation time · no temporal interpolation",
        18,
    )
    if record["composite_peak_speed"] == 0:
        text(575, 565, "Initially at rest", 28, TEXT)
    elif record["composite_peak_speed"] < THRESHOLDS[1 if view == "core" else 0]:
        text(455, 565, "Flow below the lowest displayed surface", 23, MUTED)
    # A projected coordinate triad, not a velocity-vector overlay.
    forward = -np.asarray(CAMERAS[view][0], dtype=float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, CAMERAS[view][1])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    origin = np.array([1440, 940])
    for d, label in enumerate("xyz"):
        delta = 58 * np.array([right[d], -up[d]])
        if np.linalg.norm(delta) < 1:
            continue  # View-normal coordinate is named in the title.
        end = origin + delta
        draw.line(
            tuple((origin * scale).astype(int)) + tuple((end * scale).astype(int)),
            fill=MUTED,
            width=max(1, round(scale)),
        )
        text(*(end + delta / np.linalg.norm(delta) * 14 - 7), label, 20, TEXT)
    return image


def durations(count):
    if count < 2:
        raise ValueError("A movie needs at least two saved states")
    return [1000] + [200] * (count - 2) + [500]


def fit_camera_scales(meshes, scales):
    """Whole-sequence preflight only; never change scale during animation.

    Caption-safe vertical limits control scale for the 4:3 output. The three
    overview views share the largest required scale, keeping lengths comparable.
    """
    for view, (position, up) in CAMERAS.items():
        forward = -np.asarray(position, dtype=float)
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        vertical = np.cross(right, forward)
        for i, mesh in enumerate(meshes):
            if not mesh.n_points or (view == "core" and i == 0):
                continue
            needed = max(
                float(np.abs(mesh.points @ vertical).max()) / 0.75,
                float(np.abs(mesh.points @ right).max()) / 1.28,
            )
            scales[view] = max(scales[view], needed * 1.04)
    shared = max(scales[v] for v in ("isometric", "side", "top"))
    for view in ("isometric", "side", "top"):
        scales[view] = shared
    return scales


def encode_frames(folder, stem, times, gif_size=(1000, 750)):
    """Bounded GIF writer: all states, fixed global palette, exact holds."""
    paths = sorted(folder.glob("frame-*.png"))
    if len(paths) != len(times) or any(
        path.name != f"frame-{i:04d}.png" for i, path in enumerate(paths)
    ):
        raise ValueError("Missing render frames")
    if times[0] != 0 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Movie times must increase from rest")
    holds = durations(len(paths))
    # One global palette sampled across the entire history, not just endpoint.
    samples = []
    for path in paths:
        with Image.open(path) as im:
            samples.append(im.convert("RGB").resize((80, 60)))
    atlas = Image.new("RGB", (80 * 20, 60 * ((len(samples) + 19) // 20)), BACKGROUND)
    for i, im in enumerate(samples):
        atlas.paste(im, ((i % 20) * 80, (i // 20) * 60))
    palette = atlas.quantize(colors=240, method=Image.Quantize.MEDIANCUT)
    # Small atlas thumbnails dilute white type and bright core highlights.
    # Preserve semantic colors explicitly instead of tinting them cyan.
    reserved = (
        BACKGROUND,
        TEXT,
        MUTED,
        *COLORS,
        "#29445d",
        "#ffffff",
        "#dae5ec",
        "#c8d6df",
        "#b9cbd8",
        "#91a5b7",
        "#718a9f",
        "#4e697f",
        "#334b61",
        "#15263a",
    )
    palette.putpalette(
        palette.getpalette()[: 240 * 3]
        + [value for color in reserved for value in ImageColor.getrgb(color)]
    )
    with stem.with_suffix(".gif").open("wb") as stream:
        for i, path in enumerate(paths):
            with Image.open(path) as im:
                rgb = im.convert("RGB")
                rgb.thumbnail(gif_size, Image.Resampling.LANCZOS)
                frame = rgb.quantize(
                    palette=palette, dither=Image.Dither.FLOYDSTEINBERG
                )
            if i == 0:
                for block in GifImagePlugin._get_global_header(frame, {"loop": 0}):
                    stream.write(block)
            GifImagePlugin._write_frame_data(
                stream, frame, (0, 0), {"duration": holds[i], "disposal": 1}
            )
        stream.write(b";")
    with Image.open(stem.with_suffix(".gif")) as gif:
        if gif.n_frames != len(times):
            raise ValueError("GIF did not retain every saved state")
        decoded_holds = []
        for i in range(gif.n_frames):
            gif.seek(i)
            decoded_holds.append(gif.info["duration"])
        if decoded_holds != holds:
            raise ValueError("GIF timing mismatch")
    # Pipe repeated RGB frames at 10 fps: 1 s rest, .2 s each, .5 s endpoint.
    with Image.open(paths[0]) as im:
        width, height = im.size
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-n",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        "10",
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "17",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(stem.with_suffix(".mp4")),
    ]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
        for path, hold in zip(paths, holds, strict=True):
            with Image.open(path) as im:
                raw = im.convert("RGB").tobytes()
            for _ in range(hold // 100):
                process.stdin.write(raw)
        process.stdin.close()
        if process.wait():
            raise RuntimeError("Video encoding failed")
    probe = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_read_frames,width,height:format=duration",
                "-of",
                "json",
                str(stem.with_suffix(".mp4")),
            ]
        )
    )
    if (
        int(probe["streams"][0]["nb_read_frames"]) != sum(holds) // 100
        or abs(float(probe["format"]["duration"]) - sum(holds) / 1000) > 0.01
    ):
        raise ValueError("MP4 timing or frame coverage mismatch")
    shutil.copyfile(paths[-1], stem.with_suffix(".png"))
    return {
        "saved_states": len(paths),
        "duration_seconds": sum(holds) / 1000,
        "width": width,
        "height": height,
        "files": {
            ext: {
                "name": stem.with_suffix("." + ext).name,
                "sha256": sha(stem.with_suffix("." + ext)),
            }
            for ext in ("gif", "mp4", "png")
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--views", nargs="+", choices=CAMERAS, default=list(CAMERAS))
    parser.add_argument(
        "--stills",
        nargs="+",
        type=int,
        help="Explicit preview indices; no complete-history claim",
    )
    parser.add_argument("--width", type=int, default=1600)
    args = parser.parse_args()
    dataset = NativeDataset(folder=args.dataset)
    if args.stills is None and (
        not dataset.manifest["complete_history"]
        or len(dataset.times) != dataset.manifest["source_frames"]
    ):
        raise ValueError("Movies require the entire audited from-rest sequence")
    if args.width < 800 or args.width % 8:
        raise ValueError("Width must be at least 800 and divisible by 8")
    indices = list(range(len(dataset.times))) if args.stills is None else args.stills
    if any(not 0 <= i < len(dataset.times) for i in indices):
        raise ValueError("Invalid preview index")
    args.output.mkdir(parents=True, exist_ok=False)
    axes = composite_axes(dataset.levels)
    import pyvista as pv

    geometry = args.output / "geometry"
    geometry.mkdir()
    scales = dict(SCALES)
    records = []
    # Precompute once, fitting fixed cameras to the entire saved history. This
    # avoids choosing a flattering endpoint zoom that clips the earlier pulses.
    for index in indices:
        started = time.monotonic()
        fields = audited_fields(dataset, index)
        scalar = sample_speed(fields, dataset.levels, axes)
        meshes = contours(axes, scalar)
        scales = fit_camera_scales(meshes, scales)
        for k, mesh in enumerate(meshes):
            if mesh.n_points:
                mesh.save(geometry / f"frame-{index:04d}-{k}.vtp")
        native = dataset.manifest["frames"][index]
        record = {
            "index": index,
            "time": float(dataset.times[index]),
            "native_hashes": native["level_sha256"],
            "triangles": [m.n_cells for m in meshes],
            "geometry_seconds": time.monotonic() - started,
        }
        records.append(record)
        print(
            f"Prepared {index + 1}/{len(dataset.times)} t={dataset.times[index]:.6f}",
            flush=True,
        )
    print("Fixed camera scales: " + json.dumps(scales), flush=True)
    scene = Scene(args.width, args.width * 3 // 4, scales=scales)
    try:
        for record in records:
            index = record["index"]
            started = time.monotonic()
            meshes = [
                pv.read(geometry / f"frame-{index:04d}-{k}.vtp") if n else pv.PolyData()
                for k, n in enumerate(record["triangles"])
            ]
            scene.set_surfaces(meshes)
            native = dataset.manifest["frames"][index]
            for view in args.views:
                folder = args.output / view
                folder.mkdir(exist_ok=True)
                im = caption(scene.image(view), view, native, len(dataset.times), index)
                im.save(folder / f"frame-{index:04d}.png")
            record["render_seconds"] = time.monotonic() - started
            print(
                f"Rendered {index + 1}/{len(dataset.times)} t={dataset.times[index]:.6f}",
                flush=True,
            )
    finally:
        scene.close()
    media = {}
    if args.stills is None:
        for view in args.views:
            media[view] = encode_frames(
                args.output / view, args.output / f"oliver-3d-{view}", dataset.times
            )
    manifest = {
        "schema_version": 1,
        "kind": "native-speed-surfaces",
        "complete_history": args.stills is None,
        "run_id": dataset.manifest["run_id"],
        "dataset_manifest_sha256": sha(args.dataset / "manifest.json"),
        "renderer_sha256": sha(__file__),
        "source_record_sha256": dataset.manifest["source_record_sha256"],
        "scalar": "velocity magnitude",
        "units": "nondimensional model units",
        "thresholds": THRESHOLDS,
        "colors": COLORS,
        "opacities": OPACITIES,
        "cameras": {v: CAMERAS[v] for v in args.views},
        "orthographic": True,
        "parallel_scale": {v: scales[v] for v in args.views},
        "camera_fit": "Fixed once using all visible surface vertices in the entire saved sequence; overview views share one scale",
        "reference_cube_half_width": {
            v: 0.125 if v == "core" else 0.5 for v in args.views
        },
        "visible_thresholds": {
            v: THRESHOLDS[1:] if v == "core" else THRESHOLDS for v in args.views
        },
        "surface_clipping_checked": True,
        "source_domain": [-1, 1],
        "display_axis_samples": list(map(len, axes)),
        "sampling": "Finest containing native level; trilinear interpolation of native scalar magnitudes, clamped at level centers. Contours linearly reconstructed on a nonuniform display grid. No spatial filter, geometry smoothing, or temporal interpolation.",
        "antialiasing": "2x linear supersampling and Lanczos downsampling",
        "depth_peeling": bool(scene.depth_peeling),
        "frames": records,
        "media": media,
        "scope": dataset.manifest["scope"],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
