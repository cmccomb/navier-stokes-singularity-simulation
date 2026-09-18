"""Publish verified complete-history 3D media beside the current best result."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from PIL import Image


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate(manifest, best, views):
    count = best["saved_frames"]
    if (
        manifest["kind"] != "native-detail-3d"
        or not manifest["validated"]
        or not manifest["complete_history"]
        or manifest["native_frame_count"] != count
        or manifest["indices"] != list(range(count))
        or [r["index"] for r in manifest["records"]] != list(range(count))
        or manifest["source_record_sha256"] != best["source_record_sha256"]
        or manifest["frame_times"] != best["frame_times"]
        or [r["time"] for r in manifest["records"]] != best["frame_times"]
    ):
        raise ValueError("Detail rendering must match the entire best-run history")
    for q in ("flow", "force"):
        media = manifest["media"][q]
        if (
            media["saved_states"] != count
            or media["duration_seconds"] != views["media"][q]["duration_seconds"]
            or (media["width"], media["height"]) != (2400, 1440)
        ):
            raise ValueError("Media clock or dimensions differ")


def publish(rendered, site):
    manifest_path = rendered / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    best = json.loads((site / "data/best.json").read_text())
    views = json.loads((site / "data/three-view.json").read_text())
    validate(manifest, best, views)
    count = best["saved_frames"]
    holds = [1000] + [200] * (count - 2) + [500]
    # Validate every native-state render before copying any public media.
    for i, item in enumerate(manifest["records"]):
        for q in ("flow", "force"):
            image = item["images"][q]
            if (
                image["name"] != f"{q}/frame-{i:04d}.png"
                or digest(rendered / image["name"]) != image["sha256"]
            ):
                raise ValueError("Native-state image changed")
    replacements = []
    for q in ("flow", "force"):
        for ext in ("gif", "mp4", "png"):
            record = manifest["media"][q]["files"][ext]
            name = f"kay-{q}-detail.{ext}"
            path = rendered / name
            if record["name"] != name or digest(path) != record["sha256"]:
                raise ValueError("Encoded media hash differs")
            if path.stat().st_size >= 100_000_000:
                raise ValueError("Media exceeds conservative Git file limit")
            replacements.append((path, site / "media" / name))
            record.update(path=f"media/{name}", bytes=path.stat().st_size)
        with Image.open(rendered / f"kay-{q}-detail.gif") as gif:
            if gif.n_frames != count or gif.size != (1200, 720):
                raise ValueError("GIF states or dimensions differ")
            for i, hold in enumerate(holds):
                gif.seek(i)
                if gif.info["duration"] != hold:
                    raise ValueError("GIF clock differs")
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
                    "stream=width,height,nb_read_frames:format=duration",
                    "-of",
                    "json",
                    str(rendered / f"kay-{q}-detail.mp4"),
                ]
            )
        )
        video = probe["streams"][0]
        if (
            (video["width"], video["height"]) != (2400, 1440)
            or int(video["nb_read_frames"]) != sum(holds) // 100
            or abs(float(probe["format"]["duration"]) - sum(holds) / 1000) > 0.01
        ):
            raise ValueError("MP4 readback differs")
    total = sum(p.stat().st_size for p in site.rglob("*") if p.is_file())
    total += sum(
        a.stat().st_size - (b.stat().st_size if b.exists() else 0)
        for a, b in replacements
    )
    # Leave room for the provenance record and future small text updates.
    if total > 990_000_000:
        raise ValueError("Publication would exceed the Pages size budget")
    additional = sum(a.stat().st_size for a, _ in replacements)
    if shutil.disk_usage(site).free - additional < 20 * 2**30:
        raise ValueError("Publication would consume the 20 GiB local disk reserve")
    manifest.update(
        saved_frames=count,
        source_render_manifest_sha256=digest(manifest_path),
        playback={"gif_hold_ms": holds, "duration_seconds": sum(holds) / 1000},
        reconstruction=(
            f"Overview scalar magnitudes use fixed {manifest['overview_volume']['display_resolution']}-cubed display resampling with finest-containing-level ownership. "
            if "overview_volume" in manifest
            else "Overview magnitude surfaces are display reconstructions. "
        )
        + "The core retains native samples. Instantaneous streamlines and volume ray interpolation are display reconstructions, not added solver resolution; original native fields are unchanged.",
    )
    for source, destination in replacements:
        shutil.copy2(source, destination)
        if digest(source) != digest(destination):
            raise ValueError("Public media copy differs")
    (site / "data/detail-view.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"Published {count} states per quantity; site approximately {total / 2**20:.1f} MiB"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rendered", type=Path, required=True)
    parser.add_argument("--site", type=Path, required=True)
    args = parser.parse_args()
    publish(args.rendered, args.site)
