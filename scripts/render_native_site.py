"""Prepare completed native results in the existing site's paired-image style.

No publication occurs. Native vectors remain on the archive host; only compact
movies, posters and a source-backed completion record are prepared for review.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from scripts.paper_comparison import snapshot
from scripts.paper_run import sha, thread_environment, write
from scripts.stream_run import build_playback, render_pair


class NativePlanes(Sequence):
    def __init__(self, folder, frames, moving=False):
        self.folder, self.frames, self.moving = folder, frames, moving

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        frame = self.frames[index]
        data = np.memmap(
            self.folder / (frame["path"] + ".bin"),
            mode="r",
            dtype="<f8",
            shape=(4, 256, 256, 6),
        )
        selected = (0, 3 if self.moving else 1)
        return {
            "time": frame["time"],
            "axis": -1 + (np.arange(256) + 0.5) / 128,
            "velocity_planes": [data[p, ..., :3] for p in selected],
            "force_planes": [data[p, ..., 3:] for p in selected],
            "plane_coordinates": [0, frame["peak_position"][2] if self.moving else 0],
            "moving_xy": self.moving,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, executable = (
        args.run.resolve(strict=True),
        args.executable.resolve(strict=True),
    )
    record, history = snapshot(source)
    if record["status"] != "completed" or not record["validated"]:
        raise ValueError("Only a validated complete native run can be featured")
    native = record["native_frames"]
    # Native times can differ in their last bit from the prescribed clock.
    if len(native) != len(record["planned_frames"]) or not np.allclose(
        [f["time"] for f in native], record["planned_frames"], rtol=0, atol=1e-12
    ):
        raise ValueError("The complete saved schedule is required")
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output.resolve()
    slices = output / "slices"
    slices.mkdir()
    frames, maxima = [], np.zeros(2)
    environment = {**os.environ, **thread_environment(1, 1)}
    for frame in native:
        path = slices / (frame["path"] + ".bin")
        result = subprocess.run(
            [
                str(executable),
                f"plot={source / frame['path']}",
                f"output={path}",
                "display_n=256",
                "peak_xy=1",
            ],
            capture_output=True,
            text=True,
            timeout=180,
            env=environment,
            check=True,
        )
        path.with_suffix(".log").write_text(result.stdout + result.stderr)
        rows = [
            json.loads(line.removeprefix("NS_SLICE_RESULT "))
            for line in result.stdout.splitlines()
            if line.startswith("NS_SLICE_RESULT ")
        ]
        if len(rows) != 1 or not rows[0]["peak_xy"]:
            raise ValueError("Missing native maximum-location evidence")
        row = rows[0]
        data = np.memmap(path, dtype="<f8", mode="r", shape=(4, 256, 256, 6))
        if abs(row["time"] - frame["time"]) > 1e-12 or not np.isfinite(data).all():
            raise ValueError("Invalid native frame")
        if row["peak_speed"] > frame["peak_speed"] + 1e-12:
            raise ValueError("Composite peak exceeds the archived all-cell maximum")
        if (
            frame["time"]
            <= record["profile_manifest"]["parameters"]["paper_time_cutoff_start"]
        ) and (np.any(data) or row["peak_position"] != [0, 0, 0]):
            raise ValueError("The initial rest interval must remain exactly zero")
        maxima = np.maximum(
            maxima,
            [np.linalg.norm(data[..., k : k + 3], axis=-1).max() for k in (0, 3)],
        )
        frames.append({"path": frame["path"], "slice_sha256": sha(path), **row})
        print(
            f"Exported {len(frames)}/{len(native)} t={row['time']:.6f} z_peak={row['peak_position'][2]:.6g}",
            flush=True,
        )
    times = [f["time"] for f in frames]
    media, renders = {}, {}
    for moving in (False, True):
        view = "peak" if moving else "midplane"
        sequence = NativePlanes(slices, frames, moving)
        for field, label, limit in (
            ("velocity", "flow", maxima[0]),
            ("force", "force", maxima[1]),
        ):
            stem = output / "site/media" / f"oliver-{label}-{view}"
            rendered = render_pair(
                sequence,
                field,
                stem,
                record["parameters"]["base_n"],
                mesh_label="64³ base · 5 fixed levels",
                color_max=float(limit or 1),
                save_poster=True,
            )
            renders[f"{label}_{view}"] = rendered
            media[f"{label}_{view}"] = {
                ext: {
                    "path": f"media/{stem.name}.{ext}",
                    "sha256": sha(stem.with_suffix("." + ext)),
                }
                for ext in ("gif", "mp4", "png")
            }
            print(f"Rendered {stem.name}", flush=True)
    final = history[-1]
    public = {
        "schema_version": 1,
        "kind": "completed-native-best",
        "machine": "Oliver",
        "id": "reference-n64-l5-rest-t0995-loader-fixed",
        "status": "completed",
        "validated": True,
        "production_accuracy_certified": False,
        "source_record_sha256": sha(source / "run.json"),
        "source_log_sha256": sha(source / "run.log"),
        "source_adapter_sha256": record["adapter_sha256"],
        "source_binary_sha256": record["binary_sha256"],
        "exporter_sha256": sha(executable),
        "renderer_sha256": sha(Path(__file__)),
        "completed_at": record.get("completed_at"),
        "elapsed_seconds": record.get("elapsed_seconds", record.get("wall_seconds")),
        "parameters": record["parameters"],
        "profile": record["profile_manifest"]["parameters"],
        "stored_cells": final["stored_cells"],
        "active_cells": final["active_cells"],
        "finest_equivalent_n": 1024,
        "finest_spacing": 2 / 1024,
        "diagnostics": final,
        "saved_frames": len(frames),
        "frame_times": times,
        "native_checks": native,
        "frames": frames,
        "display_n": 256,
        "playback": build_playback(times),
        "media": media,
        "render": renders,
        "vector_arrows": False,
        "peak_slice_policy": "Horizontal slice at the active composite-grid maximum of velocity magnitude in each saved frame. Exact ties prefer smallest absolute z, then positive z, then y/x. Rest uses z=0. No smoothing of slice height.",
        "scope": "Finite localized three-pulse manufactured-force surrogate; archive validation is not convergence certification or evidence of a singularity. 1024³-equivalent spacing applies only to the innermost refined cube, not the full domain.",
    }
    write(output / "site/data/best.json", public)
    print(output / "site/data/best.json", flush=True)


if __name__ == "__main__":
    main()
