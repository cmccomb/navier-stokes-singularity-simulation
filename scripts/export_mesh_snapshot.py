"""Freeze selected native states for bounded, offline mesh comparisons.

Works with an immutable completed prefix of a running run. Does not change the
run, imply a complete archive audit, or invoke the solver. Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_hashes(path):
    return {
        str(p.relative_to(path)): sha(p) for p in sorted(path.rglob("*")) if p.is_file()
    }


def bounded(command, folder, log, seconds=300, rss_mib=1536):
    env = dict(
        os.environ, OMP_NUM_THREADS="1", OMP_THREAD_LIMIT="1", OMP_DYNAMIC="FALSE"
    )
    started, peak = time.monotonic(), 0
    with log.open("x") as out:
        process = subprocess.Popen(
            command, stdout=out, stderr=subprocess.STDOUT, env=env
        )
        try:
            while process.poll() is None:
                if shutil.disk_usage(folder).free < 20 * 2**30:
                    raise RuntimeError("20 GiB reserve reached")
                rss = subprocess.run(
                    ["ps", "-o", "rss=", "-p", str(process.pid)],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                ).stdout.strip()
                peak = max(peak, int(rss or 0))
                if peak > rss_mib * 1024 or time.monotonic() - started > seconds:
                    raise RuntimeError(
                        "Diagnostic worker exceeded its memory/time guard"
                    )
                time.sleep(0.25)
            if process.returncode:
                raise RuntimeError(f"Diagnostic failed: {log}")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    return {"seconds": time.monotonic() - started, "peak_rss_mib": peak / 1024}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("run", "exporter", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    parser.add_argument(
        "--library-dir",
        type=Path,
        help="Explicit child-only macOS runtime library path",
    )
    args = parser.parse_args()
    if args.library_dir:
        os.environ["DYLD_LIBRARY_PATH"] = str(args.library_dir.resolve(strict=True))
    run, exporter, output = (
        args.run.resolve(),
        args.exporter.resolve(),
        args.output.resolve(),
    )
    record_bytes = (run / "run.json").read_bytes()
    record = json.loads(record_bytes)
    for name, digest in record["adapter_source_hashes"].items():
        if sha(run / name) != digest:
            raise ValueError("Adapter source differs")
    if sha(run / "profile.tbl") != record["profile_manifest"]["sha256"]:
        raise ValueError("Profile differs")
    if sha(run / "inputs.paper") != record["inputs_sha256"]:
        raise ValueError("Input template differs")
    band = record["parameters"].get("revolved_refinement")
    if band and sha(run / band["path"]) != band["sha256"]:
        raise ValueError("Refinement geometry differs")
    rows = {
        r["step"]: r
        for r in (
            json.loads(s.split(" ", 1)[1])
            for s in (run / "run.log").read_text().splitlines()
            if s.startswith("NS_INCFLO_RESULT ")
        )
    }
    if len(set(args.steps)) != len(args.steps) or not set(args.steps) <= rows.keys():
        raise ValueError("Require distinct completed steps")
    if shutil.disk_usage(output.parent).free < 22 * 2**30:
        raise RuntimeError("Insufficient export scratch reserve")
    output.mkdir(exist_ok=False)
    (output / "run-snapshot.json").write_bytes(record_bytes)
    report = {
        "schema_version": 1,
        "complete": False,
        "source_run": str(run),
        "source_record_sha256": hashlib.sha256(record_bytes).hexdigest(),
        "exporter_sha256": sha(exporter),
        "export_script_sha256": sha(__file__),
        "profile_sha256": record["profile_manifest"]["sha256"],
        "parameters": record["parameters"],
        "frames": [],
        "scope": "Selected immutable native states; not the full archive audit.",
    }
    for step in args.steps:
        plot = run / f"plt{step:05d}"
        if not (plot / "Header").is_file():
            raise ValueError("Requested step has no saved native state")
        hashes = tree_hashes(plot)
        raw, log = output / f"{plot.name}.bin", output / f"{plot.name}.log"
        usage = bounded(
            [str(exporter), f"plot={plot}", f"output={raw}", "layout=blocks"],
            output,
            log,
        )
        exported = [
            json.loads(s.split(" ", 1)[1])
            for s in log.read_text().splitlines()
            if s.startswith("NS_VOLUME_RESULT ")
        ]
        if (
            len(exported) != 1
            or exported[0]["schema_version"] != 2
            or exported[0]["time"] != rows[step]["time"]
            or exported[0]["bytes"] != raw.stat().st_size
            or exported[0]["dtype"] != "<f8"
            or exported[0]["order"] != "xyz-component"
            or tree_hashes(plot) != hashes
        ):
            raise ValueError("Native state changed or export identity differs")
        report["frames"].append(
            {
                "step": step,
                "path": raw.name,
                "sha256": sha(raw),
                "native_sha256": hashes,
                "diagnostic": rows[step],
                "export": exported[0],
                "resource_usage": usage,
            }
        )
        print(json.dumps({"step": step, **usage}), flush=True)
    report["complete"] = True
    (output / "manifest.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
