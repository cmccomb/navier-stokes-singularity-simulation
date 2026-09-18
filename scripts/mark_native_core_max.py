"""Annotate the native-core velocity maximum without retracing rendered fields."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from scripts.prepare_native_movies import write
from scripts.render_native_3d import TEXT, font, sha
from scripts.render_native_detail import CAMERA, CORE_HALF, PANELS, SCALES


def core_maximum(values, level):
    """Exact discrete argmax; first C-order cell wins an exact tie."""
    speed = np.linalg.norm(values[..., :3], axis=-1)
    if not np.isfinite(speed).all():
        raise ValueError("Nonfinite native core")
    index = np.unravel_index(int(np.argmax(speed)), speed.shape)
    peak = float(speed[index])
    if peak == 0:
        return {"value": 0.0, "cell_index": None, "xyz": None}
    point = np.asarray(level["origin"]) + (np.asarray(index) + 0.5) * level["spacing"]
    if np.abs(point).max() >= CORE_HALF:
        raise ValueError("Maximum outside the native core")
    return {"value": peak, "cell_index": [int(i) for i in index], "xyz": point.tolist()}


def project(point):
    x, y, width, height = PANELS[1]
    forward = -CAMERA / np.linalg.norm(CAMERA)
    right = np.cross(forward, (0, 0, 1))
    right /= np.linalg.norm(right)
    vertical = np.cross(right, forward)
    factor = height / (2 * SCALES["core"])
    return np.array(
        [
            x + width / 2 + np.dot(point, right) * factor,
            y + height / 2 - np.dot(point, vertical) * factor,
        ]
    )


def annotate(image, maximum):
    if maximum["xyz"] is None:
        return image.copy()
    px, py = project(maximum["xyz"])
    # Supersample only the annotation layer; unmarked data pixels are unchanged.
    layer = Image.new("RGBA", (image.width * 2, image.height * 2))
    draw = ImageDraw.Draw(layer)
    label = f"core max |u| = {maximum['value']:.4g}"
    face = font(44, True)
    box = draw.textbbox((0, 0), label, font=face)
    width, height = (box[2] - box[0]) / 2 + 26, 39
    tx = px + 28
    if tx + width > 2340:
        tx = px - width - 28
    ty = max(265, min(py - 60, 1180))
    corner = (tx if tx > px else tx + width, ty + height / 2)
    draw.line(
        [(px * 2, py * 2), (corner[0] * 2, corner[1] * 2)], fill="#dcebf1", width=2
    )
    draw.rounded_rectangle(
        (tx * 2, ty * 2, (tx + width) * 2, (ty + height) * 2),
        radius=9,
        fill=(7, 17, 31, 238),
        outline="#506b7b",
        width=2,
    )
    draw.text(((tx + 13) * 2, (ty + 7) * 2 - box[1]), label, font=face, fill=TEXT)
    for radius, color, stroke in ((11, "#07111f", 8), (8, "#ffffff", 3)):
        draw.ellipse(
            (
                (px - radius) * 2,
                (py - radius) * 2,
                (px + radius) * 2,
                (py + radius) * 2,
            ),
            outline=color,
            width=stroke * 2,
        )
    layer = layer.resize(image.size, Image.Resampling.LANCZOS)
    return Image.alpha_composite(image.convert("RGBA"), layer).convert("RGB")


def extract(args):
    record = json.loads((args.run / "run.json").read_text())
    prepared = json.loads((args.prepared / "manifest.json").read_text())
    source_hash = sha(args.run / "run.json")
    if (
        not record["validated"]
        or record["status"] != "completed"
        or not prepared["complete_history"]
        or not prepared["validated"]
        or prepared["source_record_sha256"] != source_hash
        or prepared["exporter_sha256"] != sha(args.exporter)
    ):
        raise ValueError("Completed, source-matched native evidence required")
    args.output.mkdir(parents=True, exist_ok=False)
    native = record["native_frames"]
    indices = list(range(len(native))) if args.indices is None else args.indices
    result = {
        "source_record_sha256": source_hash,
        "exporter_sha256": sha(args.exporter),
        "extractor_sha256": sha(Path(__file__)),
        "records": [],
        "validated": False,
        "complete_history": indices == list(range(len(native))),
        "policy": "Maximum velocity magnitude within the finest native patch; exact native cell center, C-order first exact tie, no smoothing, no unique marker at rest.",
    }
    with tempfile.TemporaryDirectory(
        prefix="core-max-native-", dir=args.output
    ) as scratch:
        for i in indices:
            if shutil.disk_usage(args.output).free < 22 * 2**30:
                raise RuntimeError("20 GiB reserve plus bounded scratch guard")
            frame = native[i]
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
                json.loads(s.split(" ", 1)[1])
                for s in proc.stdout.splitlines()
                if s.startswith("NS_VOLUME_RESULT ")
            ]
            if (
                len(rows) != 1
                or rows[0]["time"] != frame["time"]
                or rows[0]["levels"] != prepared["records"][i]["levels"]
                or rows[0]["bytes"] != raw.stat().st_size
                or rows[0]["dtype"] != "<f8"
                or rows[0]["order"] != "xyz-component"
            ):
                raise ValueError("Native export identity differs")
            level = rows[0]["levels"][-1]
            core = np.memmap(
                raw,
                dtype="<f8",
                mode="r",
                offset=level["offset_bytes"],
                shape=tuple(level["shape"]),
            )
            native_hash = hashlib.sha256(core.tobytes()).hexdigest()
            if native_hash != prepared["records"][i]["level_sha256"][-1]:
                raise ValueError("Native core hash differs")
            peak = core_maximum(core, level)
            if peak["value"] > frame["peak_speed"] + 1e-12 or (
                frame["time"] <= 0.55 and peak["value"] != 0
            ):
                raise ValueError("Core maximum contradicts full-field diagnostic")
            item = {
                "index": i,
                "time": frame["time"],
                "core_sha256": native_hash,
                **peak,
            }
            result["records"].append(item)
            write(args.output / f"frame-{i:04d}.json", item)
            del core
            raw.unlink()
            print(
                f"Core maximum {len(result['records'])}/{len(indices)} native={i} value={peak['value']:.5g}",
                flush=True,
            )
    if sha(args.run / "run.json") != source_hash:
        raise ValueError("Native source changed")
    result["validated"] = True
    write(args.output / "manifest.json", result)


def mark(args):
    original = json.loads((args.rendered / "manifest.json").read_text())
    maxima = json.loads((args.maxima / "manifest.json").read_text())
    if (
        not all(r["validated"] and r["complete_history"] for r in (original, maxima))
        or original["source_record_sha256"] != maxima["source_record_sha256"]
        or original["frame_times"] != [r["time"] for r in maxima["records"]]
        or original["indices"] != [r["index"] for r in maxima["records"]]
    ):
        raise ValueError("Marker history must match every source state")
    for frame, maximum in zip(original["records"], maxima["records"], strict=True):
        if frame["native_level_sha256"][-1] != maximum["core_sha256"]:
            raise ValueError("Marker core source differs")
        for asset in frame["images"].values():
            if sha(args.rendered / asset["name"]) != asset["sha256"]:
                raise ValueError("Source render changed")
    args.output.mkdir(parents=True, exist_ok=False)
    for q in ("flow", "force"):
        (args.output / q).mkdir()
    for frame, maximum in zip(original["records"], maxima["records"], strict=True):
        flow = frame["images"]["flow"]
        frame["unmarked_flow_sha256"] = flow["sha256"]
        frame["core_velocity_maximum"] = maximum
        with Image.open(args.rendered / flow["name"]) as image:
            annotate(image, maximum).save(args.output / flow["name"])
        flow["sha256"] = sha(args.output / flow["name"])
        force = frame["images"]["force"]
        shutil.copy2(args.rendered / force["name"], args.output / force["name"])
    original["velocity_marker"] = {
        "policy": maxima["policy"],
        "annotation_sha256": sha(Path(__file__)),
        "maxima_manifest_sha256": sha(args.maxima / "manifest.json"),
        "unmarked_manifest_sha256": sha(args.rendered / "manifest.json"),
        "projection": "Orthographic overlay on core panel, visible through rendered geometry; no force marker.",
    }
    original["media"] = {}
    write(args.output / "manifest.json", original)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("extract", "mark"))
    parser.add_argument("--output", type=Path, required=True)
    for name in ("run", "prepared", "exporter", "rendered", "maxima"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--indices", nargs="+", type=int)
    args = parser.parse_args()
    if args.command == "extract":
        if not all((args.run, args.prepared, args.exporter)):
            parser.error("extract requires run, prepared, exporter")
        extract(args)
    else:
        if not args.rendered or not args.maxima:
            parser.error("mark requires rendered and maxima")
        mark(args)
