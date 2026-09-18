"""Bounded manufactured-flow, native block export, and restart tests for r-z bands."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from scripts.coupled_pilot import parse_history
from scripts.paper_run import sha, thread_environment, write


def block_diagnostics(raw, blocks):
    """Independent active-cell integration, including ragged coverage and holes."""
    volume = energy = peak = 0.0
    for block in blocks:
        shape = tuple(block["shape"])
        values = np.memmap(
            raw, mode="r", dtype="<f8", offset=block["offset_bytes"], shape=shape
        )
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite exported native values")
        active = np.ones(shape[:3], dtype=bool)
        lo = np.array(block["index_lo"])
        hi = lo + shape[:3]
        for fine in blocks:
            if fine["level"] != block["level"] + 1:
                continue
            flo = np.array(fine["index_lo"])
            fshape = np.array(fine["shape"][:3])
            if np.any(flo % 2) or np.any(fshape % 2):
                raise ValueError("Native fine blocks are not coarse-cell aligned")
            a = np.maximum(lo, flo // 2)
            b = np.minimum(hi, (flo + fshape) // 2)
            if np.all(b > a):
                active[
                    tuple(slice(x, y) for x, y in zip(a - lo, b - lo, strict=True))
                ] = False
        speed2 = np.sum(values[..., :3] ** 2, axis=-1)[active]
        dv = np.prod(block["spacing"])
        volume += active.sum() * dv
        energy += speed2.sum() * dv / 2
        if speed2.size:
            peak = max(peak, np.sqrt(speed2.max()))
    return {"volume": float(volume), "energy": float(energy), "peak_speed": float(peak)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--bands", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-n", type=int, choices=(16, 32, 64), default=32)
    args = parser.parse_args()
    build, bands = args.build.resolve(), args.bands.resolve()
    inputs = Path("backends/amrex/inputs.coupled").resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output.resolve()
    environment = {**os.environ, **thread_environment(2, 1)}
    report = {
        "band_sha256": sha(bands),
        "binary_sha256": sha(build / "ns_incflo"),
        "exporter_sha256": sha(build / "ns_volume_export"),
        "cases": [],
        "passed": False,
    }

    def run(command, folder, log):
        result = subprocess.run(
            [str(x) for x in command],
            cwd=folder,
            env=environment,
            text=True,
            capture_output=True,
            timeout=300,
            check=False,
        )
        (folder / log).write_text(result.stdout + result.stderr)
        if result.returncode:
            raise ValueError(f"Pilot failed: {folder / log}")
        return result.stdout

    def record(stdout, marker):
        rows = [
            json.loads(s[len(marker) :])
            for s in stdout.splitlines()
            if s.startswith(marker)
        ]
        if len(rows) != 1:
            raise ValueError("Missing or duplicate native receipt")
        return rows[0]

    coarse_n = args.base_n
    for n, box in ((coarse_n, 32), (coarse_n * 2, 32), (coarse_n * 2, 16)):
        folder = output / f"n{n}-b{box}"
        folder.mkdir()
        command = [
            build / "ns_incflo",
            inputs,
            "ns.force=shear",
            "stop_time=0.025",
            "ns.max_dt=0.003125",
            f"amr.n_cell={n} {n} {n}",
            f"amr.max_grid_size={box}",
            "amr.max_level=2",
            "ns.refine_half_width=0.5 0.25",
            f"ns.refine_rz_file={bands}",
            "amr.plot_int=4",
            "amr.check_int=4",
        ]
        history = parse_history(
            run(command, folder, "run.log"),
            mode="shear",
            levels=3,
            end=0.025,
            max_dt=0.003125,
        )
        plot = folder / "plt00008"
        archive = record(
            run(
                [build / "ns_archive_check", f"plot={plot}", "ns.force=shear"],
                folder,
                "archive.log",
            ),
            "NS_ARCHIVE_RESULT ",
        )
        raw = folder / "blocks.bin"
        exported = record(
            run(
                [
                    build / "ns_volume_export",
                    f"plot={plot}",
                    f"output={raw}",
                    "layout=blocks",
                ],
                folder,
                "export.log",
            ),
            "NS_VOLUME_RESULT ",
        )
        if (
            exported["schema_version"] != 2
            or exported["bytes"] != raw.stat().st_size
            or exported["time"] != history[-1]["time"]
        ):
            raise ValueError("Block export identity differs")
        diagnostic = block_diagnostics(raw, exported["blocks"])
        for key in ("volume", "energy", "peak_speed"):
            if not np.isclose(
                diagnostic[key], history[-1][key], rtol=1e-11, atol=1e-14
            ):
                raise ValueError(f"Block export diagnostic differs: {key}")
        restart = folder / "restart"
        restart.mkdir()
        run(command + [f"amr.restart={folder / 'chk00004'}"], restart, "run.log")
        comparison = record(
            run(
                [
                    build / "ns_archive_check",
                    f"plot={restart / 'plt00008'}",
                    f"compare={plot}",
                    "ns.force=shear",
                ],
                restart,
                "compare.log",
            ),
            "NS_ARCHIVE_RESULT ",
        )
        if not comparison["compared"]:
            raise ValueError("Missing full-field restart comparison")
        report["cases"].append(
            {
                "base_n": n,
                "box": box,
                "final": history[-1],
                "archive": archive,
                "block_export": exported,
                "block_diagnostics": diagnostic,
                "restart": comparison,
            }
        )
        write(output / "summary.json", report)
        print(
            json.dumps({"n": n, "box": box, "l2_error": history[-1]["l2_error"]}),
            flush=True,
        )
    errors = [r["final"]["l2_error"] for r in report["cases"]]
    report["resolution_pair"] = [coarse_n, coarse_n * 2]
    report["error_ratio"] = errors[0] / errors[1]
    report["box_decomposition_error_delta"] = abs(errors[1] - errors[2])
    report["passed"] = (
        2.5 < report["error_ratio"] < 6
        and report["box_decomposition_error_delta"] < 1e-10
    )
    write(output / "summary.json", report)
    if not report["passed"]:
        raise ValueError("Manufactured convergence or decomposition gate failed")


if __name__ == "__main__":
    main()
