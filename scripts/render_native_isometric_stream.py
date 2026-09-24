"""Render completed preparation records with fixed cameras and bounded scratch.

Only verified, regenerable surface files are removed when explicitly requested.
Native plotfiles and checkpoints are never opened for writing or removed.
"""

import argparse
import json
import time
from pathlib import Path

import pyvista as pv

from scripts.render_native_3d import SCALES, Scene, sha

# Chosen physical scales, not per-frame fits. Every actual mesh is checked by
# Scene.image; any clipping aborts the release rather than changing the camera.
ISOMETRIC_SCALES = {"flow": 1.65, "force": 1.10}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--remove-verified-geometry", action="store_true")
    parser.add_argument("--wait-seconds", type=float, default=7200)
    parser.add_argument("--single-worker", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = json.loads((args.prepared / "source-run.json").read_text())
    source_hash = sha(args.prepared / "source-run.json")
    scenes = {
        q: Scene(800, 600, scales={**SCALES, "isometric": scale})
        for q, scale in ISOMETRIC_SCALES.items()
    }
    deadline = time.monotonic() + args.wait_seconds
    records = []
    try:
        for index, native in enumerate(source["native_frames"]):
            item_path = args.prepared / f"frame-{index:04d}.json"
            # The two independent preparation workers publish only completed
            # records. Their progress establishes that an early benchmark frame
            # is no longer awaiting producer-side cache verification.
            while True:
                worker = "even" if index % 2 == 0 else "odd"
                progress_path = args.prepared / (
                    "progress.json" if args.single_worker else f"progress-{worker}.json"
                )
                try:
                    progress = json.loads(progress_path.read_text())
                    ready = progress["prepared"] > (
                        index if args.single_worker else index // 2
                    )
                    item = json.loads(item_path.read_text()) if ready else None
                except (FileNotFoundError, json.JSONDecodeError):
                    item = None
                if item is not None:
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("Native preparation did not finish")
                time.sleep(2)
            if (
                item["source_record_sha256"] != source_hash
                or item["path"] != native["path"]
                or item["time"] != native["time"]
            ):
                raise ValueError("Native preparation identity changed")
            receipt_path = args.output / f"frame-{index:04d}.json"
            receipt = {
                "index": index,
                "time": native["time"],
                "source_record_sha256": source_hash,
                "preparation_sha256": sha(item_path),
                "scales": ISOMETRIC_SCALES,
                "images": {},
            }
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                if (
                    receipt["preparation_sha256"] != sha(item_path)
                    or receipt["scales"] != ISOMETRIC_SCALES
                ):
                    raise ValueError("Rendered cache changed")
                for image in receipt["images"].values():
                    if sha(args.output / image["name"]) != image["sha256"]:
                        raise ValueError("Rendered image changed")
            else:
                for q, scene in scenes.items():
                    meshes = []
                    for k, digest in enumerate(item["geometry"][q]):
                        path = args.prepared / f"{q}-{index:04d}-{k}.vtp"
                        if digest is not None and sha(path) != digest:
                            raise ValueError("Prepared geometry hash mismatch")
                        meshes.append(pv.read(path) if digest else pv.PolyData())
                    scene.set_surfaces(meshes)
                    image_path = args.output / f"{q}-{index:04d}.png"
                    scene.image("isometric").save(image_path)
                    receipt["images"][q] = {
                        "name": image_path.name,
                        "sha256": sha(image_path),
                    }
                receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
            if args.remove_verified_geometry:
                for q in ("flow", "force"):
                    for k, digest in enumerate(item["geometry"][q]):
                        path = args.prepared / f"{q}-{index:04d}-{k}.vtp"
                        if digest is not None and path.exists():
                            if sha(path) != digest:
                                raise ValueError(
                                    "Geometry changed before scratch cleanup"
                                )
                            path.unlink()
            records.append(receipt)
            (args.output / "progress.json").write_text(
                json.dumps(
                    {
                        "rendered": len(records),
                        "total": len(source["native_frames"]),
                        "time": native["time"],
                    }
                )
                + "\n"
            )
            print(
                f"Isometric pair {index + 1}/{len(source['native_frames'])}", flush=True
            )
    finally:
        for scene in scenes.values():
            scene.close()
    manifest = {
        "schema_version": 1,
        "complete_history": True,
        "validated": True,
        "source_record_sha256": source_hash,
        "frames": records,
        "renderer_sha256": sha(__file__),
        "surface_renderer_sha256": sha(Path(__file__).with_name("render_native_3d.py")),
        "isometric_parallel_scale": ISOMETRIC_SCALES,
        "camera_policy": "Fixed physical scales selected before image generation; every native surface checked for clipping; no per-frame zoom",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
