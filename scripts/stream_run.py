"""Bounded per-run publisher: atomic solver snapshots -> compact media -> main.

Run on the designated controller in a dedicated clean clone. No scheduler,
service installation, credentials, or source-machine paths are published.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import shlex
import subprocess
import tempfile
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm
from PIL import GifImagePlugin, Image

from scripts.volume_history import render_3d, validate_descriptor

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

OUTPUTS = [
    "site/data/stream.json",
    "site/media/stream-flow.gif",
    "site/media/stream-flow.mp4",
    "site/media/stream-force.gif",
    "site/media/stream-force.mp4",
    "site/media/stream-flow-3d.html",
    "site/media/stream-force-3d.html",
]
RENDER_REVISION = 12


def package_outputs(run: dict) -> list[str]:
    """Explicit compact files only; never transfer native archives or broad dirs."""
    if run["clip_times_3d"] != run["clip_times"]:
        raise ValueError("3D must cover every saved movie time from rest")
    paths = list(OUTPUTS)
    for field in ("velocity", "force"):
        paths.extend(
            validate_descriptor(
                run["render"][field]["volume_history"], field, run["clip_times"]
            )
        )
    return list(dict.fromkeys(paths))


class CachedFrames(Sequence):
    """Compact disk-backed history; keep only one decoded snapshot in memory."""

    def __init__(self, paths):
        self.paths = paths
        self._last_index = None
        self._last_frame = None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        if index != self._last_index:
            with np.load(self.paths[index], allow_pickle=False) as values:
                self._last_frame = {key: values[key] for key in values.files}
            self._last_index = index
        return self._last_frame


def build_playback(times: list[float], slow_motion_after: float | None = None) -> dict:
    """Show every saved state at 5 snapshots/s, not a uniform simulation clock.

    Capture uses a nonuniform phase clock. Equal visible holds keep every
    snapshot inspectable, without interpolating states or accelerating an
    ever-longer history into a fixed-duration clip. The old activation argument
    remains accepted for callers; there is no late-window speed change.
    """
    values = np.asarray(times, dtype=float)
    if (
        values.ndim != 1
        or not len(values)
        or not np.isfinite(values).all()
        or not np.all(np.diff(values) > 0)
    ):
        raise ValueError("playback needs finite, strictly increasing saved times")
    fps = 50
    span = float(values[-1] - values[0])
    repeats = [10] * (len(values) - 1) + [25 if span else 200]
    # Make the real initial rest state visible without fabricating more states.
    if span and values[0] == 0:
        repeats[0] = 50
    traversal_ticks = sum(repeats[:-1])
    end_ticks = repeats[-1]
    return {
        "mode": "all saved frames, equal snapshot holds; nonuniform simulation time",
        "applies_to": ["flow_gif", "flow_mp4", "force_gif", "force_mp4"],
        "simulation_time_span": span,
        "saved_frames_per_second": 5,
        "simulation_time_proportional": False,
        "initial_rest_hold_seconds": 1 if span and values[0] == 0 else 0,
        "traversal_seconds": traversal_ticks / fps,
        "terminal_hold_seconds": end_ticks / fps,
        "duration_seconds": (traversal_ticks + end_ticks) / fps,
        "mp4_fps": fps,
        "mp4_frame_repeats": repeats,
        "gif_timing_quantum_ms": 20,
        "source_frame_duration_ms": [int(n * 20) for n in repeats],
        "terminal_hold_policy": "editorial pause at the final saved time, not additional simulated time",
        "quantization_policy": "exact 200 ms snapshot holds, 1 s initial rest and 0.5 s final hold",
    }


def read_frames(folder: Path, window: int | None = None, cache: Path | None = None):
    """Derive native planes and browser volumes from each finalized archive.

    Keep at most one native vector field in memory, independent of clip length.
    The archival file is never rewritten, moved, or reduced.
    """
    if window is not None and window < 1:
        raise ValueError("display window must be positive")
    paths = sorted(folder.glob("frame-*.npz"))
    paths = [p for p in paths if not p.name.endswith(".tmp.npz")]
    count = len(paths)
    if window is None and any(
        p.name != f"frame-{i:06d}.npz" for i, p in enumerate(paths)
    ):
        raise ValueError(
            "full history requires contiguous saved frame indices from zero"
        )
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
    frames = []
    cached_paths = []
    for path in paths if window is None else paths[-window:]:
        cached = None
        if cache is not None:
            stat = path.stat()
            key = hashlib.sha256(
                f"native-planes-v1:{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode()
            ).hexdigest()
            cached = cache / f"{key}.npz"
            cached_paths.append(cached)
            if cached.exists():
                continue
        with np.load(path, allow_pickle=False) as values:
            frame = {key: values[key].copy() for key in ("time", "axis")}
            if not all(np.isfinite(v).all() for v in frame.values()):
                raise ValueError(f"nonfinite snapshot: {path.name}")
            n = len(frame["axis"])
            if n < 2 or not np.all(np.diff(frame["axis"]) > 0):
                raise ValueError("coordinates must increase")
            index = int(np.argmin(np.abs(frame["axis"])))
            stride = max(1, int(np.ceil(n / 32)))
            frame["volume_axis"] = frame["axis"][::stride].copy()
            for field in ("velocity", "force"):
                vectors = values[field]
                if vectors.shape != (n, n, n, 3):
                    raise ValueError("invalid vector shape")
                if not np.isfinite(vectors).all():
                    raise ValueError(f"nonfinite snapshot: {path.name}")
                frame[f"{field}_planes"] = np.stack(
                    (vectors[:, index, :, :], vectors[:, :, index, :])
                ).astype(np.float32)
                frame[field] = np.array(
                    vectors[::stride, ::stride, ::stride], dtype=np.float32, copy=True
                )
                frame[f"{field}_archive_dtype"] = str(vectors.dtype)
                del vectors
        if cached is None:
            frames.append(frame)
        else:
            pending = cached.with_suffix(".tmp.npz")
            np.savez(pending, **frame)
            pending.replace(cached)
    if cache is not None:
        frames = CachedFrames(cached_paths)
    if frames:
        axis = frames[0]["axis"].copy()
        if any(not np.array_equal(f["axis"], axis) for f in frames):
            raise ValueError("archive coordinates changed within the clip")
    if frames and not np.all(np.diff([float(f["time"]) for f in frames]) > 0):
        raise ValueError("snapshot times must increase")
    return frames, count


def plane_vectors(frame: dict, field: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Nearest saved planes to zero, with their actual coordinate retained."""
    index = int(np.argmin(np.abs(frame["axis"])))
    if f"{field}_planes" in frame:
        planes = frame[f"{field}_planes"]
        return planes[0], planes[1], float(frame["axis"][index])
    vectors = frame[field]
    return vectors[:, index, :, :], vectors[:, :, index, :], float(frame["axis"][index])


def frame_magnitudes(frame: dict, field: str):
    return tuple(
        np.linalg.norm(p.astype(np.float64), axis=-1)
        for p in plane_vectors(frame, field)[:2]
    )


def verify_gif(path: Path, expected_durations: list[int]) -> dict:
    """Decode encoded frame delays; manifest counts alone are not verification."""
    with Image.open(path) as gif:
        if gif.n_frames != len(expected_durations):
            raise ValueError("GIF does not contain every saved frame")
        for index, duration in enumerate(expected_durations):
            gif.seek(index)
            if gif.info.get("duration") != duration:
                raise ValueError("GIF frame delays differ from the playback record")
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"frames": len(expected_durations), "sha256": digest}


def render_pair(
    frames: list[dict],
    field: str,
    stem: Path,
    resolution: int,
    half_domain: float = 1.0,
    slow_motion_after: float | None = None,
    *,
    mesh_label: str | None = None,
    color_max: float | None = None,
    save_poster: bool = False,
) -> dict:
    playback = build_playback(
        [float(frame["time"]) for frame in frames], slow_motion_after
    )
    peak = max(
        float(np.max(p)) for frame in frames for p in frame_magnitudes(frame, field)
    )
    vmax = color_max if color_max is not None else (peak if peak > 0 else 1.0)
    if not np.isfinite(vmax) or vmax <= 0 or peak > vmax + 1e-12:
        raise ValueError("color scale must cover the sampled magnitude")
    norm = SymLogNorm(linthresh=0.02 * vmax, vmin=0, vmax=vmax)
    cmap = LinearSegmentedColormap.from_list(
        "stream", ["#07111f", "#177eab", "#69d2e7", "#fff2c0"]
    )
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.4), dpi=180, facecolor="#07111f")
    fig.subplots_adjust(left=0.11, right=0.86, bottom=0.23, top=0.80, wspace=0.38)
    coord = frames[0]["axis"]
    spacing = float(coord[1] - coord[0])
    extent = (coord[0] - spacing / 2, coord[-1] + spacing / 2) * 2
    images = []
    initial_magnitudes = frame_magnitudes(frames[0], field)
    coordinate = frames[0].get(
        "plane_coordinates", [plane_vectors(frames[0], field)[2]]
    )[0]
    for j, (axis, label) in enumerate(zip(axes, ("x–z", "x–y"))):
        axis.set_facecolor("#07111f")
        images.append(
            axis.imshow(
                initial_magnitudes[j].T,
                origin="lower",
                extent=extent,
                norm=norm,
                cmap=cmap,
                interpolation="nearest",
            )
        )
        held = "y" if j == 0 else "z"
        axis.set_title(
            f"{label} · {held}={coordinate:.3f}", color="#e9f1f5", fontsize=12.8
        )
        axis.set_xlabel("x", color="#9fb3c2", fontsize=11.2)
        axis.set_ylabel("z" if j == 0 else "y", color="#9fb3c2", fontsize=11.2)
        axis.tick_params(colors="#9fb3c2", labelsize=11.2)
        axis.set_xlim(-half_domain, half_domain)
        axis.set_ylim(-half_domain, half_domain)
    color_axis = fig.add_axes((0.89, 0.24, 0.02, 0.47))
    bar = fig.colorbar(images[0], cax=color_axis)
    bar.ax.tick_params(colors="#9fb3c2", labelsize=11.2)
    bar.set_label(
        "|u|" if field == "velocity" else "|f|", color="#e9f1f5", fontsize=11.2
    )
    fig.text(
        0.11,
        0.92,
        "Velocity" if field == "velocity" else "Applied force",
        color="#e9f1f5",
        fontsize=16,
        ha="left",
    )
    time_label = fig.text(
        0.50,
        0.92,
        "",
        color="#e9f1f5",
        fontsize=12.8,
        family="DejaVu Sans Mono",
        ha="left",
    )
    status_label = fig.text(0.88, 0.92, "", color="#9fb3c2", fontsize=11.2, ha="right")
    fig.text(
        0.5,
        0.035,
        f"{mesh_label or f'{resolution}³ solver'} · {len(coord)}² planes · all {len(frames)} frames · shared scale",
        ha="center",
        color="#9fb3c2",
        fontsize=11.2,
    )

    replay_label = fig.text(
        0.11,
        0.09,
        "",
        ha="left",
        color="#9fb3c2",
        fontsize=11.2,
    )

    def update(index: int):
        magnitudes = frame_magnitudes(frames[index], field)
        for j, image in enumerate(images):
            image.set_data(magnitudes[j].T)
            coordinates = frames[index].get(
                "plane_coordinates", [coordinate, coordinate]
            )
            moving = frames[index].get("moving_xy", False) and j == 1
            axes[j].set_title(
                f"{'x–z · y' if j == 0 else 'x–y · z'}={coordinates[j]:.5f}"
                + (" · peak" if moving else ""),
                color="#e9f1f5",
                fontsize=12.8,
            )
        time_label.set_text(f"t = {float(frames[index]['time']):.6f}")
        status_label.set_text(
            "zero field" if all(not np.any(p) for p in magnitudes) else ""
        )
        replay_label.set_text(
            f"Frame {index + 1}/{len(frames)} · 5 saved frames/s · simulation-time spacing varies"
            if len(frames) > 1
            else "Single saved state · no simulated-time advance"
        )

    stem.parent.mkdir(parents=True, exist_ok=True)
    video = stem.with_suffix(".tmp.mp4")
    gif = stem.with_suffix(".tmp.gif")
    # Encode each GIF frame immediately with a shared palette: no image-history
    # buffer and no optimizer allowed to merge identical fluid states.
    background = np.asarray([7, 17, 31])
    palette_colors = (
        np.concatenate(
            [
                cmap(np.linspace(0, 1, 128))[:, :3] * 255,
                np.linspace(background, [233, 241, 245], 64),
                np.linspace(background, [159, 179, 194], 64),
            ]
        )
        .round()
        .astype(np.uint8)
    )
    palette = Image.new("P", (1, 1))
    palette.putpalette(palette_colors.ravel().tolist())
    try:
        width, height = fig.canvas.get_width_height()
        # Render each source state once; repetition affects only encoded timing.
        with tempfile.TemporaryFile() as errors, gif.open("wb") as gif_output:
            process = subprocess.Popen(
                [
                    str(matplotlib.rcParams["animation.ffmpeg_path"]),
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgba",
                    "-s",
                    f"{width}x{height}",
                    "-r",
                    str(playback["mp4_fps"]),
                    "-i",
                    "pipe:0",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-crf",
                    "18",
                    "-movflags",
                    "+faststart",
                    str(video),
                ],
                stdin=subprocess.PIPE,
                stderr=errors,
            )
            try:
                for i, repeats in enumerate(playback["mp4_frame_repeats"]):
                    update(i)
                    fig.canvas.draw()
                    pixels = bytes(fig.canvas.buffer_rgba())
                    with (
                        Image.frombytes("RGBA", (width, height), pixels).convert(
                            "RGB"
                        ) as still,
                        still.quantize(
                            palette=palette, dither=Image.Dither.NONE
                        ) as indexed,
                    ):
                        if i == 0:
                            header, _ = GifImagePlugin.getheader(
                                indexed, info={"loop": 0, "optimize": False}
                            )
                            for block in header:
                                gif_output.write(block)
                        for block in GifImagePlugin.getdata(
                            indexed,
                            duration=playback["source_frame_duration_ms"][i],
                            disposal=2,
                        ):
                            gif_output.write(block)
                    for _ in range(repeats):
                        process.stdin.write(pixels)
                process.stdin.close()
                if process.wait(timeout=120):
                    errors.seek(0)
                    raise RuntimeError(errors.read().decode(errors="replace"))
                gif_output.write(b";")
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=10)
        verification = verify_gif(gif, playback["source_frame_duration_ms"])
        if save_poster:
            fig.savefig(stem.with_suffix(".png"), facecolor="#07111f")
        video.replace(stem.with_suffix(".mp4"))
        gif.replace(stem.with_suffix(".gif"))
    finally:
        plt.close(fig)
        palette.close()
    return {
        "color_max": vmax,
        "plane_coordinate": coordinate,
        "gif_verification": verification,
        "scale_policy": "shared across both planes and all frames in this clip; recomputed on publication",
    }


def build_manifest(remote: dict, frames: list[dict], count: int, render: dict) -> dict:
    times = [float(f["time"]) for f in frames]
    revision = hashlib.sha256(
        json.dumps(
            [
                remote.get("id"),
                remote["source_commit"],
                remote["config"],
                times,
                count,
                f"renderer-v{RENDER_REVISION}",
            ]
        ).encode()
    ).hexdigest()[:12]
    return {
        "schema_version": 1,
        "id": remote.get("id", "best-guess-n192-rest-t0985"),
        "label": "Current best · streaming from rest",
        "status": remote["status"],
        "observed_at": remote["observed_at"],
        "started_at": remote["started_at"],
        "source_commit": remote["source_commit"],
        "config": remote["config"],
        "revision": revision,
        "render_revision": RENDER_REVISION,
        "latest_t": times[-1],
        "captured_frames": count,
        "clip_times": times,
        "history_policy": "every finalized saved snapshot from rest; no frame thinning",
        "clip_times_3d": times,
        "history_policy_3d": "every saved snapshot from rest; on-demand volumes with a three-frame browser cache",
        "playback": build_playback(
            times, remote["config"].get("paper_time_cutoff_start")
        ),
        "display_resolution": len(frames[-1]["axis"]),
        "display_resolution_3d": len(frames[-1].get("volume_axis", frames[-1]["axis"])),
        "archive": {
            "resolution": len(frames[-1]["axis"]),
            "fields": ["velocity", "force"],
            "components_per_field": 3,
            "dtype": str(frames[-1].get("velocity_archive_dtype", "unknown")),
            "policy": "Immutable saved fields; native planes and reduced browser volumes derived after capture",
        },
        "diagnostics": remote.get("diagnostics"),
        "progress": remote.get("progress"),
        "media": {
            "flow_gif": "media/stream-flow.gif",
            "flow_mp4": "media/stream-flow.mp4",
            "force_gif": "media/stream-force.gif",
            "force_mp4": "media/stream-force.mp4",
            "flow_3d": "media/stream-flow-3d.html",
            "force_3d": "media/stream-force-3d.html",
        },
        "render": render,
        "scope_warning": "Finite manufactured-solution surrogate; exploratory and spatially unvalidated. Movies and browser 3D include every saved snapshot from rest, not every solver step. Saved-frame playback is not uniform simulation time. Browser 3D loads reduced spatial samples on demand; native archives remain intact.",
    }


def command(argv: list[str], **kwargs) -> str:
    return subprocess.check_output(
        argv, text=True, timeout=kwargs.pop("timeout", 120), **kwargs
    ).rstrip("\n")


def read_remote(ssh: list[str], host: str, run: str) -> dict:
    script = f"""import csv, json, subprocess
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
p=Path({run!r})
launch=json.loads((p/'launch.json').read_text())
cfg=json.loads((p/'preview-volumes/manifest.json').read_text())['config']
result=subprocess.run(['ps','-p',str(launch['pid']),'-o','command='],capture_output=True,text=True)
alive=result.returncode==0 and str(p) in result.stdout and 'navier_stokes_sim.cli' in result.stdout
progress=json.loads((p/'progress.json').read_text()) if (p/'progress.json').exists() else None
diagnostics=None
if (p/'partial-checkpoint.npz').exists():
 with np.load(p/'partial-checkpoint.npz',allow_pickle=False) as f:
  history=json.loads(str(f['metadata'].item()))['diagnostics']
  diagnostics=history[-1] if history else None
complete=(p/'run.json').exists() and (p/'final-state.npz').exists()
if complete:
 with (p/'diagnostics.csv').open() as handle:
  rows=list(csv.DictReader(handle))
 if rows:
  diagnostics={{k: (v.lower()=='true' if v.lower() in ('true','false') else float(v)) for k,v in rows[-1].items()}}
paths=sorted(f for f in (p/'preview-volumes').glob('frame-*.npz') if not f.name.endswith('.tmp.npz'))
latest=None
if paths:
 with np.load(paths[-1],allow_pickle=False) as f: latest=float(f['time'])
print(json.dumps(dict(id=p.name,status='complete' if complete else ('running' if alive else 'stopped'),observed_at=datetime.now(timezone.utc).isoformat(),started_at=launch['started_at'],source_commit=launch['source_commit'],config=cfg,progress=progress,diagnostics=diagnostics,captured_frames=len(paths),latest_t=latest)))
"""
    # The simulation interpreter, not an ambient system Python lacking NumPy.
    python = str(Path(run).parents[1] / ".venv/bin/python")
    if host == "local":
        return json.loads(command([python, "-c", script]))
    return json.loads(command([*ssh, host, shlex.join([python, "-c", script])]))


def publish(repo: Path, message: str) -> str:
    outputs = package_outputs(json.loads((repo / "site/data/stream.json").read_text()))
    dirty = command(["git", "status", "--porcelain", "--untracked-files=all"], cwd=repo)
    if any(line[3:] not in outputs for line in dirty.splitlines()):
        raise RuntimeError("publisher clone has changes outside its allowlist")
    command(["git", "add", "--", *outputs], cwd=repo)
    if command(["git", "diff", "--cached", "--name-only"], cwd=repo):
        command(["git", "commit", "-m", message], cwd=repo)
    for attempt in range(3):
        try:
            command(["git", "pull", "--rebase", "origin", "main"], cwd=repo)
            command(["git", "push", "origin", "HEAD:main"], cwd=repo, timeout=600)
            return command(["git", "rev-parse", "HEAD"], cwd=repo)
        except subprocess.CalledProcessError:
            if attempt == 2:
                raise
            time.sleep(3)
    raise RuntimeError("unreachable publish retry state")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="Source SSH host, or local")
    parser.add_argument("--run", required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--identity", type=Path)
    parser.add_argument(
        "--deadline",
        required=True,
        help="ISO time with timezone; controller stops by this bound",
    )
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--no-push", action="store_true")
    args = parser.parse_args()
    if args.host != "local" and args.identity is None:
        parser.error("an explicit SSH identity is required for a remote source")
    if args.host == "local" and not args.once:
        parser.error("local publication is a bounded --once operation")
    deadline = datetime.fromisoformat(args.deadline)
    if deadline.tzinfo is None:
        parser.error("deadline needs a timezone")
    args.cache.mkdir(parents=True, exist_ok=True)
    lock = (args.cache / "publisher.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    ssh = [
        "ssh",
        "-i",
        str(args.identity),
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
    ]
    last_key = None
    last_render = None
    failures = 0
    while datetime.now(UTC) < deadline:
        try:
            remote = read_remote(ssh, args.host, args.run)
            if args.host == "local":
                folder = Path(args.run) / "preview-volumes"
            else:
                # Dense archives stay on their owning machine. Render there
                # with --host local --once, then transfer only compact outputs.
                if remote["config"]["preview_resolution"] > 32:
                    raise RuntimeError(
                        "render native archives on their source with --host local --once"
                    )
                folder = args.cache / "preview-volumes"
                folder.mkdir(exist_ok=True)
                command(
                    [
                        "rsync",
                        "-az",
                        "--exclude",
                        "*.tmp*",
                        "--exclude",
                        ".*",
                        "-e",
                        shlex.join(ssh),
                        f"{args.host}:{args.run}/preview-volumes/",
                        str(folder) + "/",
                    ]
                )
            frames, count = read_frames(folder, cache=args.cache / "derived-frames")
            if not frames:
                raise RuntimeError("no finalized preview snapshot yet")
            if float(frames[0]["time"]) != remote["config"]["t_start"]:
                raise ValueError(
                    "full-history movie must start at the initial saved time"
                )
            if remote["config"]["t_start"] == 0 and any(
                np.any(frames[0][field]) for field in ("velocity", "force")
            ):
                raise ValueError(
                    "from-rest movie requires an actual zero initial field"
                )
            frame_key = (count, float(frames[-1]["time"]))
            key = (*frame_key, remote["status"])
            if key != last_key:
                if frame_key != last_render:
                    rendering = {
                        field: render_pair(
                            frames,
                            field,
                            args.repo / f"site/media/stream-{name}",
                            remote["config"]["resolution"],
                            remote["config"]["half_domain"],
                            remote["config"]["paper_time_cutoff_start"],
                        )
                        for field, name in (("velocity", "flow"), ("force", "force"))
                    }
                    for field, name in (("velocity", "flow"), ("force", "force")):
                        rendering[field]["volume_history"] = render_3d(
                            frames,
                            field,
                            args.repo / f"site/media/stream-{name}-3d.html",
                            remote["config"],
                        )
                    last_render = frame_key
                manifest = build_manifest(remote, frames, count, rendering)
                destination = args.repo / "site/data/stream.json"
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(json.dumps(manifest, indent=2) + "\n")
                commit = (
                    None
                    if args.no_push
                    else publish(
                        args.repo,
                        f"Stream 192-cubed frame {count}: t={manifest['latest_t']:.6f}",
                    )
                )
                receipt = {
                    "published_at": datetime.now(UTC).isoformat(),
                    "commit": commit,
                    "frames": count,
                    "latest_t": manifest["latest_t"],
                    "status": remote["status"],
                }
                (args.cache / "receipt.json").write_text(
                    json.dumps(receipt, indent=2) + "\n"
                )
                print(json.dumps(receipt), flush=True)
                last_key = key
            failures = 0
            if args.once or remote["status"] in ("complete", "stopped"):
                return
        except Exception as exc:
            failures += 1
            failure = {
                "failed_at": datetime.now(UTC).isoformat(),
                "consecutive_failures": failures,
                "error": str(exc),
            }
            (args.cache / "failure.json").write_text(
                json.dumps(failure, indent=2) + "\n"
            )
            print(json.dumps(failure), flush=True)
            if args.once or failures >= 5:
                raise
        time.sleep(max(5, args.poll_seconds))
    print(
        json.dumps({"status": "deadline-reached", "at": datetime.now(UTC).isoformat()}),
        flush=True,
    )


if __name__ == "__main__":
    main()
