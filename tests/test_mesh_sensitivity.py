import json
import sys

import numpy as np
import pytest

from scripts.compare_mesh_snapshots import (
    active_mask,
    compare,
    interface_mask,
    restrict_to,
    validate_blocks,
)
from scripts.export_mesh_snapshot import sha
from scripts.mesh_source_sensors import directional_variation, metrics
from scripts.publish_mesh_diagnostics import checked_sensors
from scripts.rank_mesh_candidates import covers_patch, sensor_bands
from scripts.revolved_mesh_pilot import block_diagnostics


def archive(tmp_path, name, specifications):
    blocks, offset = [], 0
    path = tmp_path / name
    with path.open("wb") as stream:
        for level, start, shape, value in specifications:
            dx = 2 / (8 * 2**level)
            field = np.full((*shape, 6), value, dtype="<f8")
            block = {
                "level": level,
                "index_lo": list(start),
                "shape": list(field.shape),
                "spacing": [dx] * 3,
                "origin": [-1 + x * dx for x in start],
                "offset_bytes": offset,
            }
            stream.write(field.tobytes())
            offset += field.nbytes
            blocks.append(block)
    return path, blocks


def test_conservative_composite_refinement(tmp_path):
    base, a = archive(tmp_path, "base", [(0, (0, 0, 0), (8, 8, 8), 1)])
    fine, b = archive(
        tmp_path, "fine", [(0, (0, 0, 0), (8, 8, 8), 1), (1, (4, 4, 4), (8, 8, 8), 2)]
    )
    result = compare(base, a, fine, b, [])
    assert result["regions"]["all/velocity"]["volume"] == 8
    assert result["regions"]["all/velocity"]["delta_squared_integral"] == 3
    assert result["regions"]["newly_refined/velocity"]["volume"] == 1
    assert (
        result["regions"]["unchanged_spacing/velocity"]["delta_squared_integral"] == 0
    )


def test_native_core_is_not_coarsened_or_double_counted(tmp_path):
    base, a = archive(
        tmp_path, "base", [(0, (0, 0, 0), (8, 8, 8), 9), (1, (4, 4, 4), (8, 8, 8), 2)]
    )
    other, b = archive(
        tmp_path, "other", [(0, (0, 0, 0), (8, 8, 8), 9), (1, (4, 4, 4), (8, 8, 8), 3)]
    )
    result = compare(base, a, other, b, [0.5])["regions"]
    assert result["all/velocity"]["volume"] == 8
    assert result["level_0/velocity"]["delta_squared_integral"] == 0
    assert result["level_1/velocity"]["delta_squared_integral"] == 3
    assert active_mask(a[0], a).sum() == 448


def test_restriction_averages_subcells(tmp_path):
    _, target = archive(tmp_path, "base", [(0, (0, 0, 0), (8, 8, 8), 0)])
    path, fine = archive(tmp_path, "fine", [(1, (0, 0, 0), (16, 16, 16), 0)])
    data = np.memmap(path, mode="r+", dtype="<f8", shape=(16, 16, 16, 6))
    data[::2] = 2
    data.flush()
    result, _ = restrict_to(path, fine, target[0], np.ones((8, 8, 8), bool))
    np.testing.assert_array_equal(result, 1)


def test_missing_fine_coverage_is_rejected(tmp_path):
    _, target = archive(tmp_path, "base", [(1, (4, 4, 4), (8, 8, 8), 0)])
    path, coarse = archive(tmp_path, "coarse", [(0, (0, 0, 0), (8, 8, 8), 0)])
    with pytest.raises(ValueError, match="at least as refined"):
        restrict_to(path, coarse, target[0], np.ones((8, 8, 8), bool))


def test_interface_is_cube_boundary_not_coordinate_plane():
    xyz = [np.array([0.49, 0.49, 0.8]), np.array([0.0, 0.8, 0.0]), np.zeros(3)]
    np.testing.assert_array_equal(
        interface_mask(xyz, 0.01, [0.5]), [True, False, False]
    )


def test_directional_probe_recovers_axial_gradient():
    f = np.zeros((8, 8, 8, 6))
    f[..., 0] = (np.arange(8) + 0.5)[None, None, :]
    result = directional_variation(f, [1, 1, 0], 1, 2)["velocity"]
    assert result["radial"]["gradient_rms"] == 0
    assert result["azimuthal"]["gradient_rms"] == 0
    assert result["axial"]["gradient_rms"] == 1


def test_sensor_metrics_separate_window_sensitivity():
    a = np.ones((4, 4, 4, 6))
    result = metrics(a * 3, a * 1.5, a, a)["force"]
    assert result["difference_ratio"] == pytest.approx(3)
    assert result["epsilon_half_relative"] == 0


def test_geometry_overlap_and_offset_guard(tmp_path):
    path, blocks = archive(tmp_path, "a", [(0, (0, 0, 0), (8, 8, 8), 1)])
    validate_blocks(blocks, path.stat().st_size)
    blocks[0]["origin"][0] += 0.01
    with pytest.raises(ValueError, match="geometry"):
        validate_blocks(blocks, path.stat().st_size)
    _, blocks = archive(
        tmp_path, "b", [(0, (0, 0, 0), (8, 8, 8), 1), (0, (0, 0, 0), (8, 8, 8), 1)]
    )
    with pytest.raises(ValueError, match="Overlapping"):
        validate_blocks(blocks, 8**3 * 6 * 8 * 2)


def test_candidate_score_requires_patch_not_just_center_coverage():
    request = {"dx": 0.25, "xyz": [0, 0, 0], "candidate_level": 0}
    partial = {"levels": [{}, {"boxes": [{"index_lo": [7, 7, 7], "shape": [2, 2, 2]}]}]}
    assert not covers_patch(partial, request)
    full = {"levels": [{}, {"boxes": [{"index_lo": [4, 4, 4], "shape": [8, 8, 8]}]}]}
    assert covers_patch(full, request)


def test_sensor_bands_buffer_patch_and_ancestors():
    s = {"request": {"candidate_level": 3, "xyz": [0.1, 0, 0.04]}}
    bands = sensor_bands(s, 128)
    assert {b[0] for b in bands} == {0, 1, 2, 3}
    assert any(b[3] < -0.04 < b[4] for b in bands)
    assert any(b[3] < 0.04 < b[4] for b in bands)
    for level, lo, hi, _, _ in bands:
        assert lo < 0.1 < hi


def test_source_publication_rejects_tampered_raw_bytes(tmp_path):
    path = tmp_path / "sensors.json"
    hashes = {}
    for name in ("h", "half", "quarter", "epsilon_half"):
        raw = tmp_path / f"{name}.bin"
        raw.write_bytes(b"native-source-sample")
        hashes[raw.name] = sha(raw)
    path.write_text(json.dumps({"complete": True, "sensors": [{"raw_sha256": hashes}]}))
    assert checked_sensors(path)["complete"]
    (tmp_path / "quarter.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="bytes differ"):
        checked_sensors(path)


def test_native_comparison_cli_checks_hashes_and_independent_audits(
    tmp_path, monkeypatch
):
    from scripts.compare_mesh_snapshots import main

    folders = []
    for name, value in (("baseline", 1), ("candidate", 2)):
        folder = tmp_path / name
        folder.mkdir()
        raw, blocks = archive(folder, "state.bin", [(0, (0, 0, 0), (8, 8, 8), value)])
        source = folder / "run-snapshot.json"
        source.write_text(
            json.dumps(
                {
                    "adapter_source_hashes": {
                        n: "pinned"
                        for n in (
                            "paper_fields.H",
                            "paper_profile.H",
                            "incflo_overlay.py",
                        )
                    }
                }
            )
        )
        (folder / "manifest.json").write_text(
            json.dumps(
                {
                    "complete": True,
                    "profile_sha256": "same-profile",
                    "source_record_sha256": sha(source),
                    "parameters": {
                        "base_n": 8,
                        "widths": [],
                        "max_dt": 0.001,
                        "epsilon_tau_ratio": 0,
                    },
                    "frames": [
                        {
                            "step": 10,
                            "path": raw.name,
                            "sha256": sha(raw),
                            "export": {
                                "time": 0.5,
                                "blocks": blocks,
                                "bytes": raw.stat().st_size,
                            },
                            "diagnostic": block_diagnostics(raw, blocks),
                        }
                    ],
                }
            )
        )
        folders.append(folder)
    output = tmp_path / "comparison"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare",
            "--baseline",
            str(folders[0]),
            "--candidate",
            str(folders[1]),
            "--output",
            str(output),
        ],
    )
    main()
    result = json.loads((output / "comparison.json").read_text())
    assert result["complete"]
    assert all(
        a["volume"] == 8 and a["all_six_components_finite"]
        for a in result["frames"][0]["native_audits"]
    )
    assert result["frames"][0]["regions"]["all/velocity"]["relative_l2_difference"] == 1
    assert result["frames"][0]["regions"]["all/force"]["relative_l2_difference"] == 1


def test_sensor_score_normalizes_each_time_and_requires_complete_patch():
    from scripts.rank_mesh_candidates import score

    mesh = {"levels": [{}, {"boxes": [{"index_lo": [4, 4, 4], "shape": [8, 8, 8]}]}]}
    sensors = [
        {
            "request": {
                "dx": 0.25,
                "xyz": [x, 0, 0],
                "candidate_level": 0,
                "time": t,
                "label": str(t),
            },
            "metrics": {"force": {"h_to_half_rms": weight}},
        }
        for x, t, weight in ((0, 0.5, 1), (0.5, 0.9, 100))
    ]
    result = score(mesh, sensors)
    assert result["covered_sensor_count"] == 1
    assert result["mean_sampled_sensitivity_coverage"] == 0.5
