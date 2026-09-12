"""Validate the coupled incflo adapter; no paper-surrogate trajectory is claimed."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import shutil
import subprocess
import time
from itertools import pairwise
from pathlib import Path

from scripts.coupled_archive import native_roundtrip
from scripts.diffusion_pilot import orders
from scripts.refinement_pilot import AMREX_COMMIT, write_json

INCFLO_COMMIT = "7491eea4d69bbfde0581a5fe7f95d803841a8a09"
MARKER = "NS_INCFLO_RESULT "


def parse_history(
    stdout: str,
    *,
    mode: str,
    levels: int,
    end: float,
    max_dt: float,
    expected_adapter: str | None = None,
) -> list[dict]:
    rows = [
        json.loads(s[len(MARKER) :])
        for s in stdout.splitlines()
        if s.startswith(MARKER)
    ]
    if len(rows) < 2 or rows[0].get("time") != 0 or rows[0].get("step") != 0:
        raise ValueError("missing from-zero history")
    if "WARNING: fixed_dt" in stdout:
        raise ValueError("unchecked fixed time step")
    for index, row in enumerate(rows):
        if (
            row.get("schema_version") != 1
            or row.get("step") != index
            or row.get("case") != mode
            or row.get("levels") != levels
        ):
            raise ValueError("wrong case, level count, or incomplete step history")
        if not re.fullmatch(r"[0-9a-f]{64}", str(row.get("adapter_sha256", ""))):
            raise ValueError("missing compiled source identity")
        if expected_adapter is not None and row["adapter_sha256"] != expected_adapter:
            raise ValueError("executable does not match adapter source")
        for key in (
            "time",
            "l2_error",
            "linf_error",
            "peak_speed",
            "energy",
            "volume",
            "peak_rss_mib",
        ):
            if (
                type(row.get(key)) not in (int, float)
                or not math.isfinite(row[key])
                or row[key] < 0
            ):
                raise ValueError(f"invalid {key}")
        if (
            abs(row["volume"] - 8) > 1e-10
            or not 0 < row["active_cells"] <= row["stored_cells"]
        ):
            raise ValueError("invalid composite coverage")
        if len(row.get("velocity_integral", [])) != 3 or not all(
            math.isfinite(v) for v in row["velocity_integral"]
        ):
            raise ValueError("invalid integral")
        if index and not (
            0 < row["dt"] <= max_dt * (1 + 1e-12)
            and math.isclose(
                row["time"] - rows[index - 1]["time"],
                row["dt"],
                rel_tol=1e-10,
                abs_tol=1e-15,
            )
        ):
            raise ValueError("invalid clock or timestep ceiling")
    if not math.isclose(rows[-1]["time"], end, rel_tol=1e-11, abs_tol=1e-14):
        raise ValueError("run did not reach its endpoint")
    if mode != "none" and rows[0]["peak_speed"] != 0:
        raise ValueError("manufactured test did not start at exact rest")
    if mode == "rest" and any(r["peak_speed"] != 0 for r in rows):
        raise ValueError("unforced rest changed")
    if mode == "none" and any(
        b["energy"] > a["energy"] + 1e-8 for a, b in pairwise(rows)
    ):
        raise ValueError("unforced Taylor vortex increased energy")
    return rows


def run_case(
    executable: Path,
    inputs: Path,
    output: Path,
    mode: str,
    n: int,
    levels: int,
    steps: int,
    box: int,
    end: float = 0.2,
    scheme: str = "MOL",
    expected_adapter: str | None = None,
) -> dict:
    output.mkdir()
    dt = end / steps
    args = [
        str(executable),
        str(inputs),
        f"ns.force={mode}",
        f"amr.n_cell={n} {n} {n}",
        f"amr.max_level={levels - 1}",
        f"amr.max_grid_size={box}",
        f"ns.max_dt={dt:.17g}",
        f"stop_time={end:.17g}",
        f"incflo.advection_type={scheme}",
    ]
    if levels > 1:
        args.append(
            "ns.refine_half_width="
            + " ".join(str(0.5**lev) for lev in range(1, levels))
        )
    if mode == "none":
        args.append("incflo.probtype=2")
    start = time.monotonic()
    # Preserve partial diagnostics on timeout, and make long checks inspectable.
    with (output / "run.log").open("w") as log:
        process = subprocess.run(
            args,
            cwd=output,
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=1800,
            check=False,
        )
    if process.returncode:
        raise ValueError(f"exit {process.returncode}; see {output.name}/run.log")
    rows = parse_history(
        (output / "run.log").read_text(),
        mode=mode,
        levels=levels,
        end=end,
        max_dt=dt,
        expected_adapter=expected_adapter,
    )
    record = {
        "mode": mode,
        "base_n": n,
        "levels": levels,
        "steps_requested": steps,
        "steps": rows[-1]["step"],
        "max_grid_size": box,
        "scheme": scheme,
        "max_dt": dt,
        "elapsed_seconds": time.monotonic() - start,
        "initial": rows[0],
        "final": rows[-1],
        "l2_error": rows[-1]["l2_error"],
    }
    write_json(output / "result.json", record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--executable", type=Path, default=Path("build/amrex/ns_incflo")
    )
    parser.add_argument(
        "--inputs", type=Path, default=Path("backends/amrex/inputs.coupled")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--archive-checker", type=Path, default=Path("build/amrex/ns_archive_check")
    )
    args = parser.parse_args()
    executable, inputs = (
        args.executable.resolve(strict=True),
        args.inputs.resolve(strict=True),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        parser.error("preserve previous results: output directory must be empty")
    # A concurrent rebuild must not change the executable between cases.
    pinned_executable = (args.output / "ns_incflo").resolve()
    shutil.copy2(executable, pinned_executable)
    executable = pinned_executable
    source_hashes = {
        name: hashlib.sha256((inputs.parent / name).read_bytes()).hexdigest()
        for name in (
            "ns_case.H",
            "incflo_overlay.py",
            "paper_profile.H",
            "paper_fields.H",
        )
    }
    expected_adapter = hashlib.sha256(
        ";".join(source_hashes.values()).encode()
    ).hexdigest()
    pinned_inputs = (args.output / "inputs.coupled").resolve()
    shutil.copy2(inputs, pinned_inputs)
    inputs = pinned_inputs
    checker = (args.output / "ns_archive_check").resolve()
    shutil.copy2(args.archive_checker.resolve(strict=True), checker)
    report = {
        "schema_version": 1,
        "scope": __doc__,
        "platform": platform.platform(),
        "amrex_commit": AMREX_COMMIT,
        "incflo_commit": INCFLO_COMMIT,
        "binary_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        "inputs_sha256": hashlib.sha256(inputs.read_bytes()).hexdigest(),
        "adapter_sha256": expected_adapter,
        "adapter_source_hashes": source_hashes,
        "archive_checker_sha256": hashlib.sha256(checker.read_bytes()).hexdigest(),
        "cases": [],
        "convergence": {},
        "failures": [],
        "passed": False,
    }
    cases = [("rest", 16, 3, 16, 32, "MOL")]
    cases += [
        ("oscillation", 16, 3, s, 32, scheme)
        for scheme in ("MOL", "Godunov")
        for s in (16, 32, 64)
    ]
    cases += [("shear", n, 3, 64, 32, "MOL") for n in (16, 32, 64)]
    cases += [("none", n, 1, 64, 32, "MOL") for n in (16, 32, 64)]
    cases += [("shear", 32, 3, 64, 16, "MOL")]
    for mode, n, levels, steps, box, scheme in cases:
        name = f"{mode}-n{n}-l{levels}-s{steps}-b{box}-{scheme.lower()}"
        try:
            row = run_case(
                executable,
                inputs,
                args.output / name,
                mode,
                n,
                levels,
                steps,
                box,
                scheme=scheme,
                expected_adapter=expected_adapter,
            )
            report["cases"].append(row)
            print(name, f"error={row['l2_error']:.6g}", flush=True)
        except (ValueError, subprocess.TimeoutExpired) as exc:
            report["failures"].append({"case": name, "error": str(exc)})
            print(name, str(exc), flush=True)
        write_json(args.output / "summary.json", report)
    for mode, scheme, param in [
        ("oscillation", "MOL", "steps"),
        ("oscillation", "Godunov", "steps"),
        ("shear", "MOL", "base_n"),
        ("none", "MOL", "base_n"),
    ]:
        rows = [
            r
            for r in report["cases"]
            if r["mode"] == mode and r["scheme"] == scheme and r["max_grid_size"] == 32
        ]
        key = f"{mode}-{scheme}"
        if len(rows) != 3:
            report["failures"].append(
                {"gate": key, "error": "missing convergence case"}
            )
            continue
        try:
            report["convergence"][key] = orders(rows, param)
            if not all(r["passed"] for r in report["convergence"][key]):
                report["failures"].append(
                    {"gate": key, "error": "second-order convergence failed"}
                )
        except ValueError as exc:
            report["failures"].append({"gate": key, "error": str(exc)})
    decomposed = [
        r for r in report["cases"] if r["mode"] == "shear" and r["base_n"] == 32
    ]
    if len(decomposed) == 2:
        delta = abs(decomposed[0]["l2_error"] - decomposed[1]["l2_error"])
        report["decomposition_error_delta"] = delta
        if delta > 1e-9:
            report["failures"].append(
                {"gate": "decomposition", "error": "error norm changed"}
            )
    else:
        report["failures"].append({"gate": "decomposition", "error": "missing case"})
    try:
        report["native_archive"] = native_roundtrip(
            executable, checker, inputs, args.output / "native"
        )
    except (ValueError, subprocess.TimeoutExpired) as exc:
        report["failures"].append({"gate": "native_archive", "error": str(exc)})
    report["passed"] = not report["failures"] and len(report["cases"]) == len(cases)
    write_json(args.output / "summary.json", report)
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
