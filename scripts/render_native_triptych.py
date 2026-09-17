"""Synchronized XY slice, XZ slice, isometric movies from all native states.

Fixed center slices use native-cell values, never smoothing. The isometric
panel uses the separately documented linear scalar reconstruction. Camera and
color scales stay fixed throughout each complete movie. No source writes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm
from PIL import Image

from scripts.render_native_3d import (
    BACKGROUND,
    COLORS,
    MUTED,
    SCALES,
    TEXT,
    THRESHOLDS,
    Scene,
    audited_fields,
    composite_axes,
    contours,
    encode_frames,
    fit_camera_scales,
    sample_speed,
    sha,
)
from spaces.native_explorer.reader import PLANES, NativeDataset

ORDER = ("xy", "xz", "isometric")
FORCE_THRESHOLDS = (10.0, 100.0, 1000.0)


def center_plane(fields, levels, plane, n=256):
    """Nearest native cell at the fixed zero plane, fine over coarse.

    Zero is a cell face; the positive-side cell owns that boundary, as in the
    native exporter and interactive reader. Array axes are horizontal, vertical.
    """
    h, v, f = PLANES[plane]
    axis = -1 + (np.arange(n) + 0.5) * 2 / n
    result = np.full((n, n, 6), np.nan)
    for values, level in zip(fields, levels, strict=True):
        origin, dx, shape = map(
            np.asarray, (level["origin"], level["spacing"], level["shape"][:3])
        )
        fixed = int(np.floor(-origin[f] / dx[f]))
        if not 0 <= fixed < shape[f]:
            continue
        ih = np.floor((axis - origin[h]) / dx[h]).astype(int)
        iv = np.floor((axis - origin[v]) / dx[v]).astype(int)
        ph = np.flatnonzero((ih >= 0) & (ih < shape[h]))
        pv = np.flatnonzero((iv >= 0) & (iv < shape[v]))
        slab = np.take(values, fixed, axis=f)
        result[np.ix_(ph, pv)] = slab[np.ix_(ih[ph], iv[pv])]
    if not np.isfinite(result).all():
        raise ValueError("Incomplete native center plane")
    return result


class Triptych:
    def __init__(self, quantity, limit, thresholds, machine="Oliver", saved_states=280):
        self.quantity = quantity
        self.fig = plt.figure(figsize=(15, 6), dpi=160, facecolor=BACKGROUND)
        cmap = LinearSegmentedColormap.from_list(
            "native", [BACKGROUND, "#177eab", "#69d2e7", "#fff2c0"]
        )
        norm = SymLogNorm(linthresh=0.02 * limit, vmin=0, vmax=limit)
        self.images = []
        for left, plane in zip((0.045, 0.365), ORDER[:2], strict=True):
            ax = self.fig.add_axes((left, 0.21, 0.255, 0.6375), facecolor=BACKGROUND)
            im = ax.imshow(
                np.zeros((256, 256)),
                origin="lower",
                extent=(-1, 1, -1, 1),
                cmap=cmap,
                norm=norm,
                interpolation="nearest",
            )
            self.images.append(im)
            ax.set(xlabel="x", ylabel=plane[1], xticks=(-1, 0, 1), yticks=(-1, 0, 1))
            ax.tick_params(colors=MUTED, labelsize=11)
            ax.xaxis.label.set(color=MUTED, size=12)
            ax.yaxis.label.set(color=MUTED, size=12)
            for spine in ax.spines.values():
                spine.set_color("#29445d")
            ax.set_title(
                f"x–{plane[1]} slice · {'z' if plane == 'xy' else 'y'} = 0",
                color=TEXT,
                fontsize=14,
                pad=14,
            )
        ax = self.fig.add_axes((0.665, 0.19, 0.33, 0.66))
        self.iso = ax.imshow(np.zeros((600, 800, 3), dtype=np.uint8))
        ax.set_axis_off()
        self.fig.text(
            0.83, 0.872, "Isometric · 3D surfaces", ha="center", color=TEXT, size=14
        )
        self.fig.text(
            0.045,
            0.947,
            "Flow" if quantity == "velocity" else "Applied force",
            color=TEXT,
            size=20,
            weight="bold",
        )
        self.time = self.fig.text(0.61, 0.947, "", color=TEXT, size=14)
        symbol = "|u|" if quantity == "velocity" else "|f|"
        bar = self.fig.colorbar(
            self.images[0],
            cax=self.fig.add_axes((0.045, 0.085, 0.575, 0.017)),
            orientation="horizontal",
        )
        bar.ax.tick_params(colors=MUTED, labelsize=10)
        bar.ax.xaxis.set_label_position("top")
        bar.set_label(
            f"{symbol} · fixed slice scale · linear/log transition: {0.02 * limit:.3g}",
            color=MUTED,
            size=10,
        )
        self.fig.text(
            0.68, 0.16, f"{symbol} surfaces · fixed thresholds", color=TEXT, size=11
        )
        for i, (threshold, color) in enumerate(zip(thresholds, COLORS, strict=True)):
            self.fig.text(
                0.68 + i * 0.10, 0.12, f"● {threshold:g}", color=color, size=13
            )
        self.fig.text(
            0.68, 0.074, "Reference cube ±0.5 · fixed camera", color=MUTED, size=10
        )
        self.fig.text(
            0.045,
            0.018,
            f"{machine} · from rest · all {saved_states} saved states · model units · 5 states/s · nonuniform simulation time · no temporal interpolation",
            color=MUTED,
            size=10,
        )

    def image(self, planes, isometric, index, time, total):
        component = slice(0, 3) if self.quantity == "velocity" else slice(3, 6)
        for im, plane in zip(self.images, planes, strict=True):
            im.set_data(np.linalg.norm(plane[..., component], axis=-1).T)
        self.iso.set_data(isometric)
        self.time.set_text(
            f"t = {time:.6f}     ·     saved state {index + 1:03d} / {total}"
        )
        self.fig.canvas.draw()
        return Image.fromarray(np.asarray(self.fig.canvas.buffer_rgba())[..., :3])

    def close(self):
        plt.close(self.fig)


def main():
    renderer_sha256 = sha(__file__)
    surface_renderer_sha256 = sha(Path(__file__).with_name("render_native_3d.py"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--velocity-geometry", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--prepared",
        type=Path,
        help="Reuse a hash-verified complete preflight for presentation-only changes",
    )
    args = parser.parse_args()
    dataset = NativeDataset(folder=args.dataset)
    velocity = json.loads((args.velocity_geometry / "manifest.json").read_text())
    if (
        not dataset.manifest["complete_history"]
        or len(dataset.times) != dataset.manifest["source_frames"]
    ):
        raise ValueError("Complete from-rest dataset required")
    if (
        velocity["dataset_manifest_sha256"] != sha(args.dataset / "manifest.json")
        or not velocity["complete_history"]
        or [f["time"] for f in velocity["frames"]] != list(dataset.times)
        or velocity["thresholds"] != list(THRESHOLDS)
    ):
        raise ValueError("Velocity geometry does not match this archive")
    args.output.mkdir(parents=True, exist_ok=False)
    geometry = args.output / "geometry"
    geometry.mkdir()
    import pyvista as pv

    axes = composite_axes(dataset.levels)
    scales = dict(SCALES)
    records, limits = [], np.zeros(2)
    cache = args.output
    preparation = None
    if args.prepared:
        cache = args.prepared
        preparation = json.loads((cache / "preflight.json").read_text())
        if preparation["dataset_manifest_sha256"] != sha(
            args.dataset / "manifest.json"
        ) or [r["time"] for r in preparation["records"]] != list(dataset.times):
            raise ValueError("Prepared cache does not match the complete archive")
        records, scales, limits = (
            preparation["records"],
            preparation["scales"],
            preparation["limits"],
        )
        geometry = cache / "geometry"
        for index, record in enumerate(records):
            if (
                record["index"] != index
                or sha(cache / f"planes-{index:04d}.npy") != record["planes_sha256"]
            ):
                raise ValueError("Prepared plane cache changed")
            for k, digest in enumerate(record["geometry_sha256"]):
                path = geometry / f"frame-{index:04d}-{k}.vtp"
                if (
                    digest is not None and (not path.exists() or sha(path) != digest)
                ) or (digest is None and path.exists()):
                    raise ValueError("Prepared geometry cache changed")
        print("Verified complete preflight cache", flush=True)
    for index, time in [] if preparation else enumerate(dataset.times):
        fields = audited_fields(dataset, index)
        planes = np.array([center_plane(fields, dataset.levels, p) for p in ORDER[:2]])
        np.save(args.output / f"planes-{index:04d}.npy", planes)
        limits = np.maximum(
            limits,
            [np.linalg.norm(planes[..., k : k + 3], axis=-1).max() for k in (0, 3)],
        )
        scalar = sample_speed(fields, dataset.levels, axes, "force")
        meshes = contours(axes, scalar, FORCE_THRESHOLDS)
        fit_camera_scales(meshes, scales)
        hashes = []
        for k, mesh in enumerate(meshes):
            if mesh.n_points:
                path = geometry / f"frame-{index:04d}-{k}.vtp"
                mesh.save(path)
                hashes.append(sha(path))
            else:
                hashes.append(None)
        records.append(
            {
                "index": index,
                "time": float(time),
                "triangles": [m.n_cells for m in meshes],
                "geometry_sha256": hashes,
                "planes_sha256": sha(args.output / f"planes-{index:04d}.npy"),
            }
        )
        print(f"Prepared force + slices {index + 1}/{len(dataset.times)}", flush=True)
    (args.output / "preflight.json").write_text(
        json.dumps(
            {
                "dataset_manifest_sha256": sha(args.dataset / "manifest.json"),
                "records": records,
                "scales": scales,
                "limits": list(limits),
                "renderer_sha256": renderer_sha256,
            },
            indent=2,
        )
        + "\n"
    )
    media = {}
    for j, quantity in enumerate(("velocity", "force")):
        is_velocity = quantity == "velocity"
        source = args.velocity_geometry / "geometry" if is_velocity else geometry
        source_records = velocity["frames"] if is_velocity else records
        scale = velocity["parallel_scale"] if is_velocity else scales
        thresholds = THRESHOLDS if is_velocity else FORCE_THRESHOLDS
        name = "flow" if is_velocity else "force"
        folder = args.output / name
        folder.mkdir()
        scene = Scene(800, 600, scales=scale)
        triptych = Triptych(quantity, float(limits[j] or 1), thresholds)
        try:
            for index, time in enumerate(dataset.times):
                meshes = [
                    pv.read(source / f"frame-{index:04d}-{k}.vtp")
                    if n
                    else pv.PolyData()
                    for k, n in enumerate(source_records[index]["triangles"])
                ]
                scene.set_surfaces(meshes)
                im = triptych.image(
                    np.load(cache / f"planes-{index:04d}.npy"),
                    scene.image("isometric"),
                    index,
                    time,
                    len(dataset.times),
                )
                im.save(folder / f"frame-{index:04d}.png")
                print(f"Rendered {name} {index + 1}/{len(dataset.times)}", flush=True)
        finally:
            triptych.close()
            scene.close()
        media[name] = encode_frames(
            folder,
            args.output / f"oliver-{name}-views",
            dataset.times,
            gif_size=(1600, 640),
        )
    manifest = {
        "schema_version": 1,
        "kind": "native-three-view",
        "validated": True,
        "complete_history": True,
        "saved_frames": len(dataset.times),
        "frame_times": list(dataset.times),
        "view_order": ORDER,
        "slice_coordinates": {"xy": 0, "xz": 0},
        "slice_sampling": "Finest native cell, positive-side zero-plane ownership; no smoothing",
        "surface_sampling": velocity["sampling"],
        "source_record_sha256": dataset.manifest["source_record_sha256"],
        "dataset_manifest_sha256": sha(args.dataset / "manifest.json"),
        "renderer_sha256": renderer_sha256,
        "surface_renderer_sha256": surface_renderer_sha256,
        "velocity_geometry_manifest_sha256": sha(
            args.velocity_geometry / "manifest.json"
        ),
        "slice_color_max": dict(zip(("flow", "force"), limits)),
        "surface_thresholds": {"flow": THRESHOLDS, "force": FORCE_THRESHOLDS},
        "isometric_parallel_scale": {
            "flow": velocity["parallel_scale"]["isometric"],
            "force": scales["isometric"],
        },
        "force_geometry": records,
        "prepared_manifest_sha256": sha(cache / "preflight.json"),
        "media": media,
        "scope": dataset.manifest["scope"],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
