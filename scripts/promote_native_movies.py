"""Stage one complete native movie release; never push or overwrite history."""

import argparse
import json
from pathlib import Path
import shutil

from scripts.render_native_3d import durations, sha
from scripts.render_grid_views import render, render_panel
from scripts.render_native_mesh import (
    hierarchy,
    plane_lines,
    transition_cells,
    render_svg,
)


def stage(source, slices, movies, site):
    record = json.loads(source.read_text())
    cached = json.loads((slices / "manifest.json").read_text())
    views = json.loads((movies / "manifest.json").read_text())
    native = record["native_frames"]
    times = [f["time"] for f in native]
    if not (
        record["status"] == "completed"
        and record["validated"]
        and views["validated"]
        and views["complete_history"]
        and views["source_record_sha256"] == sha(source)
        and times == views["frame_times"]
        and times[0] == 0
        and times[-1] == record["parameters"]["end"]
        and len(times) == len(cached["frames"]) == len(record["planned_frames"])
    ):
        raise ValueError("Native source and complete movies disagree")
    if any(
        a["path"] != b["path"] or a["time"] != b["time"] or a["force_linf_error"] != 0
        for a, b in zip(native, cached["frames"], strict=True)
    ):
        raise ValueError("Native slice or force audit mismatch")
    if any(a >= b for a, b in zip(times, times[1:])) or any(
        abs(a - b) > 1e-12 for a, b in zip(times, record["planned_frames"], strict=True)
    ):
        raise ValueError("Saved history differs from the prescribed schedule")
    media = {}
    for quantity in ("flow", "force"):
        clip = views["media"][quantity]
        if clip["saved_states"] != len(times):
            raise ValueError("Movie drops native states")
        for ext, asset in clip["files"].items():
            path = movies / asset["name"]
            if path.stat().st_size >= 100 * 1024**2 or sha(path) != asset["sha256"]:
                raise ValueError("Invalid movie asset")
            destination = site / "media" / path.name
            shutil.copy2(path, destination)
            asset["path"] = "media/" + path.name
        media[quantity + "_views"] = clip["files"]
    final = record["history"][-1]
    finest = record["parameters"]["base_n"] * 2 ** (final["levels"] - 1)
    holds = durations(len(times))
    best = {
        "schema_version": 1,
        "kind": "completed-native-best",
        "machine": "Kay",
        "id": "refined-n128-l5-rest-t0995",
        "status": "completed",
        "validated": True,
        "production_accuracy_certified": False,
        "vector_arrows": False,
        "source_record_sha256": sha(source),
        "source_adapter_sha256": record["adapter_sha256"],
        "source_binary_sha256": record["binary_sha256"],
        "source_checker_sha256": record["checker_sha256"],
        "completed_at": record.get("completed_at", record.get("updated_at")),
        "elapsed_seconds": record.get("wall_seconds"),
        "parameters": record["parameters"],
        "profile": record["profile_manifest"]["parameters"],
        "stored_cells": final["stored_cells"],
        "active_cells": final["active_cells"],
        "finest_equivalent_n": finest,
        "finest_spacing": 2 / finest,
        "diagnostics": final,
        "saved_frames": len(times),
        "frame_times": times,
        "native_checks": native,
        "display_n": 256,
        "frames": [
            {**n, "slice_sha256": c["slice_sha256"]}
            for n, c in zip(native, cached["frames"], strict=True)
        ],
        "slice_policy": "Fixed x-y at z=0 and x-z at y=0; finest native cell, positive-side ownership. No peak-following slice or smoothing.",
        "playback": {
            "source_frame_duration_ms": holds,
            "mp4_frame_repeats": [h // 100 for h in holds],
            "duration_seconds": sum(holds) / 1000,
        },
        "media": media,
        "render": views,
        "scope": "Finite localized three-pulse manufactured-force surrogate; archive validation is not convergence certification or evidence of a singularity. 2048³-equivalent spacing applies only to the innermost refined cube, not the full domain.",
    }

    def write(path, obj):
        path.write_text(json.dumps(obj, indent=2) + "\n")

    write(site / "data/best.json", best)
    write(site / "data/three-view.json", views)
    levels = hierarchy(best)
    lines = plane_lines(levels)
    cells, bounds = transition_cells(levels)
    write(
        site / "data/native-mesh.json",
        {
            "run_id": best["id"],
            "source_record_sha256": best["source_record_sha256"],
            "levels": levels,
            "planes": lines,
            "cells": cells,
            "detail_bounds": bounds,
            "stored_cells": best["stored_cells"],
            "active_cells": best["active_cells"],
        },
    )
    (site / "media/native-mesh.svg").write_text(render_svg(levels, lines))
    (site / "media/kay-grid-views.svg").write_text(render(best))
    for view in ("xy", "isometric"):
        (site / f"media/mesh-{view}.svg").write_text(render_panel(best, view))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "slices", "movies", "site"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    stage(args.source, args.slices, args.movies, args.site)
