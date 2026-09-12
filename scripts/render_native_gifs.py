"""Render every validated native event in perpendicular views, with fixed scales.

Display pixels use the finest containing cell, without spatial smoothing or
invented time frames. Zero-plane ties use the positive-side native cell. This
is a diagnostic comparison-mesh preview, not a high-resolution or blowup claim.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from scripts.paper_comparison import frame_headers, matched_frames, snapshot
from scripts.paper_run import output_times, sha, thread_environment, write


def prefix_frames(
    record: dict, history: list[dict], headers: list[dict], through: float
):
    """Select a complete saved prefix, retaining every legacy duplicate file."""
    if not math.isfinite(through) or not 0 < through <= history[-1]["time"] + 1e-12:
        raise ValueError("preview endpoint is outside the available history")
    params = record["parameters"]
    schedule = record.get("planned_frames") or output_times(
        params["end"],
        params["frame_dt"],
        params.get("frame_phase_step", 0),
        record["profile_manifest"]["parameters"],
    )
    expected = [t for t in schedule if t <= through + 1e-12]
    if not expected or abs(expected[-1] - through) > 1e-12:
        raise ValueError("preview must end at a scheduled saved event")
    selected = [f for f in headers if f["time"] <= through + 1e-12]
    groups = matched_frames(selected, expected)  # Missing even a zero state fails.
    if sum(map(len, groups)) != len(selected):
        raise ValueError("native prefix contains an unscheduled event")
    for frame in selected:
        step = int(frame["path"][3:])
        if step >= len(history) or abs(history[step]["time"] - frame["time"]) > 1e-12:
            raise ValueError("native frame does not match its diagnostic step")
    return selected, expected


def check_native(
    source: Path,
    frame: dict,
    record: dict,
    checker: Path,
    output: Path,
    environment: dict,
) -> dict:
    result = subprocess.run(
        [
            str(checker),
            f"plot={source / frame['path']}",
            "ns.force=paper",
            f"ns.table_file={source / 'profile.tbl'}",
            f"ns.epsilon_tau_ratio={record['parameters']['epsilon_tau_ratio']:.17g}",
        ],
        capture_output=True,
        text=True,
        env=environment,
        timeout=1800,
        check=False,
    )
    (output / f"{frame['path']}-read.log").write_text(result.stdout + result.stderr)
    rows = [
        json.loads(line.removeprefix("NS_ARCHIVE_RESULT "))
        for line in result.stdout.splitlines()
        if line.startswith("NS_ARCHIVE_RESULT ")
    ]
    if result.returncode or len(rows) != 1:
        raise ValueError(f"native prefix check failed: {frame['path']}")
    row = rows[0]
    if abs(row["time"] - frame["time"]) > 1e-12 or row["step"] != int(
        frame["path"][3:]
    ):
        raise ValueError("native readback changed the selected event")
    return {"path": frame["path"], **row}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--executable", type=Path, default=Path("build/amrex-omp/ns_slice_export")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--through",
        type=float,
        help="explicit partial-preview endpoint; audit every saved file from rest",
    )
    parser.add_argument(
        "--checker", type=Path, default=Path("build/amrex-omp/ns_archive_check")
    )
    args = parser.parse_args()
    source, executable = (
        args.run.resolve(strict=True),
        args.executable.resolve(strict=True),
    )
    record, history = snapshot(source)
    if args.through is None and (
        not record["validated"] or record["status"] != "completed"
    ):
        raise ValueError("this renderer requires a validated complete native sequence")
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output.resolve()
    environment = {**os.environ, **thread_environment(1, 1)}
    native_frames = record["native_frames"]
    expected = [f["time"] for f in native_frames]
    checker = None
    if args.through is not None:
        checker = args.checker.resolve(strict=True)
        selected, expected = prefix_frames(
            record, history, frame_headers(source), args.through
        )
        native_frames = []
        for frame in selected:
            checked = check_native(source, frame, record, checker, output, environment)
            diagnostic = history[checked["step"]]
            if (
                checked["levels"] != diagnostic["levels"]
                or checked["stored_cells"] != diagnostic["stored_cells"]
                or abs(checked["composite_volume"] - 8) > 1e-10
            ):
                raise ValueError(
                    "native preview mesh differs from the diagnostic history"
                )
            native_frames.append(checked)
            print(f"Checked {frame['path']} at t={frame['time']:.6f}", flush=True)
    n = 256
    slices = []
    provenance = {
        "scope": __doc__,
        "source_record_sha256": sha(source / "run.json"),
        "source_adapter_sha256": record["adapter_sha256"],
        "source_binary_sha256": record["binary_sha256"],
        "exporter_sha256": sha(executable),
        "checker_sha256": sha(checker) if checker else record["checker_sha256"],
        "original_run_status": record["status"],
        "original_run_validated": record["validated"],
        "render_scope": "audited partial preview"
        if args.through is not None
        else "completed sequence",
        "through": args.through,
        "scheduled_events": expected,
        "duplicate_files_retained": len(native_frames) - len(expected),
        "native_checks": native_frames,
        "frames": [],
        "gifs": {},
    }
    for frame in native_frames:
        path = output / (frame["path"] + ".bin")
        result = subprocess.run(
            [
                str(executable),
                f"plot={source / frame['path']}",
                f"output={path}",
                f"display_n={n}",
            ],
            capture_output=True,
            text=True,
            env=environment,
            timeout=120,
            check=False,
        )
        path.with_suffix(".log").write_text(result.stdout + result.stderr)
        if result.returncode:
            raise ValueError("native slice export failed")
        data = np.fromfile(path, dtype="<f8").reshape(3, n, n, 6)
        if (
            not np.isfinite(data).all()
            or np.max(np.linalg.norm(data[..., :3], axis=-1))
            > frame["peak_speed"] + 1e-12
        ):
            raise ValueError(
                "invalid slice or slice speed exceeds native full-field peak"
            )
        if frame["time"] <= record["profile_manifest"]["parameters"][
            "paper_time_cutoff_start"
        ] and np.any(data):
            raise ValueError("quiescent native slice is not zero")
        slices.append(data)
        provenance["frames"].append(
            {
                "native_path": frame["path"],
                "time": frame["time"],
                "slice_sha256": sha(path),
            }
        )
    maxima = [
        max(float(np.max(np.linalg.norm(f[..., k : k + 3], axis=-1))) for f in slices)
        for k in (0, 3)
    ]
    # An entirely quiescent prefix must still map zero to the bottom of the scale.
    maxima = [value if value > 0 else 1.0 for value in maxima]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "text.color": "#e5edf5",
            "axes.labelcolor": "#e5edf5",
            "xtick.color": "#c4d1de",
            "ytick.color": "#c4d1de",
            "axes.edgecolor": "#8295a7",
            "figure.facecolor": "#0a1521",
            "axes.facecolor": "#0a1521",
            "savefig.facecolor": "#0a1521",
        }
    )
    coords = -1 + 2 * (np.arange(n) + 0.5) / n
    stride = 16
    xx, yy = np.meshgrid(coords[::stride], coords[::stride], indexing="xy")
    for plane, (name, horizontal, vertical, components) in enumerate(
        (("vertical-xz", "x", "z", (0, 2)), ("equatorial-xy", "x", "y", (0, 1)))
    ):
        for field, k, limit in (("velocity", 0, maxima[0]), ("forcing", 3, maxima[1])):
            frames = []
            for i, (data, native) in enumerate(zip(slices, native_frames, strict=True)):
                vector = data[plane, ..., k : k + 3]
                magnitude = np.linalg.norm(vector, axis=-1)
                fig, ax = plt.subplots(figsize=(8.4, 8.2), dpi=110)
                fig.subplots_adjust(left=0.11, right=0.88, bottom=0.18, top=0.80)
                title = "Velocity" if field == "velocity" else "Applied body force"
                fig.text(
                    0.11,
                    0.945,
                    f"{title} · {'vertical XZ' if plane == 0 else 'equatorial XY'}",
                    fontsize=20,
                    weight="bold",
                )
                fig.text(
                    0.11,
                    0.900,
                    f"AMReX / incflo · {'partial from rest' if args.through is not None else 'from rest'} · t = {native['time']:.3f}",
                    fontsize=13,
                )
                base = record["parameters"]["base_n"]
                levels = len(record["parameters"]["widths"]) + 1
                fig.text(
                    0.11,
                    0.859,
                    f"Comparison mesh: {base}³ base + {levels - 1} refined levels · core {base * 2 ** (levels - 1)}³-equivalent",
                    fontsize=11,
                )
                im = ax.imshow(
                    magnitude.T,
                    origin="lower",
                    extent=(-1, 1, -1, 1),
                    vmin=0,
                    vmax=limit,
                    cmap="magma",
                    interpolation="nearest",
                    aspect="equal",
                )
                u, v = (vector[::stride, ::stride, c].T for c in components)
                norm = np.hypot(u, v)
                valid = norm > 0.025 * limit
                u = np.ma.array(
                    np.divide(u, norm, out=np.zeros_like(u), where=norm > 0),
                    mask=~valid,
                )
                v = np.ma.array(
                    np.divide(v, norm, out=np.zeros_like(v), where=norm > 0),
                    mask=~valid,
                )
                ax.quiver(
                    xx,
                    yy,
                    u,
                    v,
                    color="#eaf2f7",
                    angles="xy",
                    scale_units="xy",
                    scale=16,
                    width=0.003,
                    alpha=0.82,
                )
                ax.set(
                    xlabel=horizontal,
                    ylabel=vertical,
                    xticks=[-1, -0.5, 0, 0.5, 1],
                    yticks=[-1, -0.5, 0, 0.5, 1],
                )
                colorbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.035)
                colorbar.set_label(
                    "speed · model units"
                    if field == "velocity"
                    else "force magnitude · model units"
                )
                fig.text(
                    0.11,
                    0.110,
                    "Color scale fixed across all frames and both views.",
                    fontsize=11,
                )
                fig.text(
                    0.11,
                    0.078,
                    "Arrows: in-plane direction above 2.5% of scale. Native cells; no smoothing.",
                    fontsize=10,
                )
                fig.text(
                    0.11,
                    0.046,
                    f"Saved file {i + 1}/{len(slices)} · full domain · {provenance['duplicate_files_retained']} duplicate files retained",
                    fontsize=10,
                    color="#a8bacb",
                )
                fig.canvas.draw()
                image = Image.fromarray(
                    np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
                )
                if i in (0, len(slices) - 1):
                    image.save(
                        output / f"{field}-{name}-{'first' if i == 0 else 'last'}.png"
                    )
                frames.append(image)
                plt.close(fig)
            path = output / f"{field}-{name}.gif"
            frames[0].save(
                path,
                save_all=True,
                append_images=frames[1:],
                duration=[110] * (len(frames) - 1) + [1100],
                loop=0,
                disposal=2,
                optimize=False,
            )
            with Image.open(path) as gif:
                if gif.n_frames != len(slices):
                    raise ValueError("GIF dropped source events")
            provenance["gifs"][path.name] = {
                "sha256": sha(path),
                "frames": len(frames),
                "color_max": limit,
            }
            write(output / "render.json", provenance)
            print(path, flush=True)
    write(output / "render.json", provenance)


if __name__ == "__main__":
    main()
