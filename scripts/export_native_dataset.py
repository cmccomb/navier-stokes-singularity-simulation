"""Package complete or explicit prototype native frames as lossless sharded Zarr.

No publication. One native frame in scratch at a time; original plotfiles are
read-only. The completion manifest is written only after every selected frame
round-trips exactly. Prototype subsets are clearly marked, never called complete.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import zarr


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def export(source, best_path, exporter, output, indices=None, min_free_gib=20):
    best = json.loads(best_path.read_text())
    if (
        not best["validated"]
        or best["status"] != "completed"
        or best["frame_times"][0] != 0
    ):
        raise ValueError("Require a validated completed native source")
    indices = list(range(best["saved_frames"])) if indices is None else indices
    if (
        not indices
        or indices[0] != 0
        or indices != sorted(set(indices))
        or indices[-1] >= best["saved_frames"]
    ):
        raise ValueError("Frame subset must start at rest and be unique and increasing")
    output.mkdir(parents=True, exist_ok=False)
    root = zarr.open_group(str(output / "fields.zarr"), mode="w")
    records, levels, arrays, maxima = [], None, [], [0.0, 0.0]
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "OMP_THREAD_LIMIT": "1"}
    with tempfile.TemporaryDirectory(prefix="native-export-", dir=output) as temporary:
        for out_index, source_index in enumerate(indices):
            if shutil.disk_usage(output).free < min_free_gib * 2**30:
                raise RuntimeError(
                    "Export stopped at the free-disk guard; source preserved"
                )
            frame = best["frames"][source_index]
            name = frame["path"]
            if not name.startswith("plt") or not name[3:].isdigit():
                raise ValueError("Invalid source frame path")
            raw = Path(temporary) / "frame.bin"
            process = subprocess.run(
                [str(exporter), f"plot={source / name}", f"output={raw}"],
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            rows = [
                json.loads(line.split(" ", 1)[1])
                for line in process.stdout.splitlines()
                if line.startswith("NS_VOLUME_RESULT ")
            ]
            if len(rows) != 1 or abs(rows[0]["time"] - frame["time"]) > 1e-12:
                raise ValueError("Native export timestamp mismatch")
            result = rows[0]
            if (
                result["dtype"] != "<f8"
                or result["order"] != "xyz-component"
                or raw.stat().st_size != result["bytes"]
            ):
                raise ValueError("Native binary schema mismatch")
            if levels is None:
                levels = result["levels"]
                for lev in levels:
                    shape = tuple(lev["shape"])
                    arrays.append(
                        root.create_array(
                            f"level_{lev['level']}",
                            shape=(len(indices), *shape),
                            chunks=(1, 8, 8, 8, 3),
                            shards=(1, *shape),
                            dtype="<f8",
                            fill_value=0,
                            compressors=[
                                zarr.codecs.BloscCodec(
                                    cname="zstd", clevel=5, shuffle="bitshuffle"
                                )
                            ],
                            dimension_names=("t", "x", "y", "z", "component"),
                        )
                    )
            elif levels != result["levels"]:
                raise ValueError("Source is not a fixed hierarchy")
            hashes = []
            composite_peak, energy, volume = 0.0, 0.0, 0.0
            for i, lev in enumerate(levels):
                field = np.memmap(
                    raw,
                    dtype="<f8",
                    mode="r",
                    offset=lev["offset_bytes"],
                    shape=tuple(lev["shape"]),
                )
                if not np.isfinite(field).all() or (
                    frame["time"] <= best["profile"]["paper_time_cutoff_start"]
                    and np.any(field)
                ):
                    raise ValueError("Nonfinite field or nonzero rest interval")
                arrays[i][out_index] = field
                returned = np.asarray(arrays[i][out_index])
                if not np.array_equal(returned, field):
                    raise ValueError("Lossless voxel round-trip failed")
                hashes.append(hashlib.sha256(returned.tobytes()).hexdigest())
                active = np.ones(field.shape[:3], dtype=bool)
                if i + 1 < len(levels):
                    finer = levels[i + 1]
                    start = np.rint(
                        (np.array(finer["origin"]) - lev["origin"]) / lev["spacing"]
                    ).astype(int)
                    count = np.rint(
                        np.array(finer["shape"][:3])
                        * finer["spacing"]
                        / np.array(lev["spacing"])
                    ).astype(int)
                    active[tuple(slice(a, a + b) for a, b in zip(start, count))] = False
                speed = np.linalg.norm(field[..., :3], axis=-1)
                force = np.linalg.norm(field[..., 3:], axis=-1)
                composite_peak = max(composite_peak, float(speed[active].max()))
                maxima[0] = max(maxima[0], float(speed[active].max()))
                maxima[1] = max(maxima[1], float(force[active].max()))
                dv = np.prod(lev["spacing"])
                energy += float(np.sum(speed[active] ** 2) * dv / 2)
                volume += int(active.sum()) * dv
                del field, returned
            if (
                abs(volume - 8) > 1e-12
                or abs(composite_peak - frame["peak_speed"]) > 1e-12
            ):
                raise ValueError("Composite-grid native diagnostic mismatch")
            if source_index == best["saved_frames"] - 1 and not np.isclose(
                energy, best["diagnostics"]["energy"], rtol=1e-12
            ):
                raise ValueError("Endpoint energy mismatch")
            records.append(
                {
                    "source_index": source_index,
                    "source_frame": name,
                    "time": frame["time"],
                    "level_sha256": hashes,
                    "composite_peak_speed": composite_peak,
                    "energy": energy,
                }
            )
            print(
                f"Verified {out_index + 1}/{len(indices)} t={frame['time']:.8f}",
                flush=True,
            )
    manifest = {
        "schema_version": 1,
        "kind": "native-fixed-level-voxels",
        "validated": True,
        "complete_history": len(indices) == best["saved_frames"],
        "source_frames": best["saved_frames"],
        "run_id": best["id"],
        "source_record_sha256": best["source_record_sha256"],
        "best_sha256": sha(best_path),
        "exporter_sha256": sha(exporter),
        "levels": levels,
        "times": [r["time"] for r in records],
        "frames": records,
        "components": ["u_x", "u_y", "u_z", "f_x", "f_y", "f_z"],
        "color_max": {"velocity": maxima[0], "force": maxima[1]},
        "domain": [[-1, 1]] * 3,
        "dtype": "float64",
        "units": "nondimensional model units",
        "sampling": "finest containing native cell; positive side at cell boundaries; no spatial or temporal interpolation",
        "scope": "Finite localized manufactured-force surrogate, not a singularity or spatial-convergence certificate",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    shutil.copyfile(best_path, output / "source-best.json")
    print(
        json.dumps(
            {
                "output": str(output),
                "frames": len(records),
                "complete": manifest["complete_history"],
            }
        ),
        flush=True,
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--best", type=Path, required=True)
    parser.add_argument("--exporter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--indices", type=int, nargs="+")
    parser.add_argument("--min-free-gib", type=float, default=20)
    args = parser.parse_args()
    export(
        args.source.resolve(),
        args.best.resolve(),
        args.exporter.resolve(),
        args.output.resolve(),
        args.indices,
        args.min_free_gib,
    )
