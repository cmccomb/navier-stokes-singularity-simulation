"""Optional built-backend test of all slice axes and finest-cell selection."""

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from scripts.paper_run import thread_environment


def test_native_slices_match_known_refined_initial_field(tmp_path):
    root = Path(__file__).resolve().parents[1]
    build = next(
        (
            path
            for path in (root / "build/amrex-omp", root / "build/amrex")
            if (path / "ns_slice_export").exists()
        ),
        None,
    )
    if build is None:
        pytest.skip("optional AMReX slice-export binary is not built")
    environment = {**os.environ, **thread_environment(1, 1)}
    subprocess.run(
        [
            str(build / "ns_incflo"),
            str(root / "backends/amrex/inputs.coupled"),
            "ns.force=none",
            "incflo.probtype=2",
            "max_step=0",
            "incflo.do_initial_proj=0",
            "incflo.initial_iterations=0",
            "amr.plot_int=1",
            "amr.n_cell=16 16 16",
            "amr.max_level=2",
            "ns.refine_half_width=0.5 0.25",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        timeout=60,
        check=True,
    )
    output = tmp_path / "slices.bin"
    subprocess.run(
        [
            str(build / "ns_slice_export"),
            f"plot={tmp_path / 'plt00000'}",
            f"output={output}",
            "display_n=64",
        ],
        env=environment,
        capture_output=True,
        timeout=60,
        check=True,
    )
    actual = np.fromfile(output, dtype="<f8").reshape(3, 64, 64, 6)
    axis = -1 + (np.arange(64) + 0.5) * 2 / 64
    a, b = np.meshgrid(axis, axis, indexing="ij")
    for plane, (h, v, f) in enumerate(((0, 2, 1), (0, 1, 2), (1, 2, 0))):
        xyz = np.zeros((64, 64, 3))
        xyz[..., h], xyz[..., v] = a, b
        level = (np.max(abs(xyz), axis=-1) < 0.5).astype(int)
        level[np.max(abs(xyz), axis=-1) < 0.25] = 2
        dx = 2 / (16 * 2**level)
        sample = -1 + (np.floor((xyz + 1) / dx[..., None]) + 0.5) * dx[..., None]
        expected = np.zeros((64, 64, 6))
        x, y = sample[..., 0], sample[..., 1]
        expected[..., 0] = 1 - np.cos(np.pi * x) * np.sin(np.pi * y)
        expected[..., 1] = 1 + np.sin(np.pi * x) * np.cos(np.pi * y)
        np.testing.assert_allclose(actual[plane], expected, rtol=0, atol=1e-13)

    peak_output = tmp_path / "peak-slices.bin"
    process = subprocess.run(
        [
            str(build / "ns_slice_export"),
            f"plot={tmp_path / 'plt00000'}",
            f"output={peak_output}",
            "display_n=64",
            "peak_xy=1",
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    report = next(
        json.loads(line.removeprefix("NS_SLICE_RESULT "))
        for line in process.stdout.splitlines()
        if line.startswith("NS_SLICE_RESULT ")
    )
    peak = np.fromfile(peak_output, dtype="<f8").reshape(4, 64, 64, 6)
    np.testing.assert_array_equal(peak[:3], actual)
    assert report["peak_xy"] and report["planes"][-1] == "xy_peak"
    # Independently enumerate the active composite cells, including all levels.
    maximum = 0
    for level in range(3):
        count = 16 * 2**level
        coords = -1 + (np.arange(count) + 0.5) * 2 / count
        x, y, z = np.meshgrid(coords, coords, coords, indexing="ij")
        radius = np.maximum.reduce((abs(x), abs(y), abs(z)))
        active = radius < 2**-level
        if level < 2:
            active &= radius >= 2 ** -(level + 1)
        speed = np.hypot(
            1 - np.cos(np.pi * x) * np.sin(np.pi * y),
            1 + np.sin(np.pi * x) * np.cos(np.pi * y),
        )
        maximum = max(maximum, speed[active].max())
    assert report["peak_speed"] == pytest.approx(maximum, abs=1e-13)
    xyz = np.zeros((64, 64, 3))
    xyz[..., 0], xyz[..., 1] = a, b
    xyz[..., 2] = report["peak_position"][2]
    levels = (np.max(abs(xyz), axis=-1) < 0.5).astype(int)
    levels[np.max(abs(xyz), axis=-1) < 0.25] = 2
    dx = 2 / (16 * 2**levels)
    sample = -1 + (np.floor((xyz + 1) / dx[..., None]) + 0.5) * dx[..., None]
    expected = np.zeros((64, 64, 6))
    x, y = sample[..., 0], sample[..., 1]
    expected[..., 0] = 1 - np.cos(np.pi * x) * np.sin(np.pi * y)
    expected[..., 1] = 1 + np.sin(np.pi * x) * np.cos(np.pi * y)
    np.testing.assert_allclose(peak[3], expected, rtol=0, atol=1e-13)
