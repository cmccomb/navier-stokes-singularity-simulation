"""Raw native movie preparation must preserve scalar and history evidence."""

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("zarr")
from scripts.prepare_native_movies import check_fields, read_planes, rendered_image
from scripts.render_native_3d import composite_axes, sha
from scripts.render_native_triptych import Triptych


def test_full_kay_display_retains_all_native_core_centers():
    levels = [
        {
            "origin": [-1 / 2**i] * 3,
            "spacing": [2 / (128 * 2**i)] * 3,
            "shape": [128, 128, 128, 6],
        }
        for i in range(5)
    ]
    for axis in composite_axes(levels):
        assert len(axis) == 384
        np.testing.assert_allclose(axis, -axis[::-1])
        core = axis[np.abs(axis) < 0.0625]
        assert len(core) == 128
        np.testing.assert_array_equal(np.diff(core), np.full(127, 2 / 2048))


def test_native_composite_excludes_covered_coarse_cells():
    levels = [
        {"origin": [-1] * 3, "spacing": [0.5] * 3, "shape": [4, 4, 4, 6]},
        {"origin": [-0.5] * 3, "spacing": [0.25] * 3, "shape": [4, 4, 4, 6]},
    ]
    fields = [np.zeros((4, 4, 4, 6)), np.zeros((4, 4, 4, 6))]
    fields[0][..., 0] = 2
    fields[0][1:3, 1:3, 1:3, 0] = 100
    fields[1][..., 0] = 3
    report = check_fields(fields, levels, {"peak_speed": 3}, False)
    assert report == {"peak_speed": 3, "energy": 18.5, "volume": 8}
    with pytest.raises(ValueError, match="diagnostics"):
        check_fields(fields, levels, {"peak_speed": 4}, False)
    with pytest.raises(ValueError, match="nonzero rest"):
        check_fields(fields, levels, {"peak_speed": 3}, True)


def test_cached_slices_have_fixed_xy_xz_order_and_hash_guards(tmp_path):
    data = np.zeros((3, 256, 256, 6), dtype="<f8")
    data[0], data[1], data[2] = 1, 2, 3
    raw = data.tobytes()
    path = tmp_path / "plt00001.bin.gz"
    with gzip.open(path, "wb") as stream:
        stream.write(raw)
    frame = {
        "path": "plt00001",
        "compressed_sha256": sha(path),
        "slice_sha256": hashlib.sha256(raw).hexdigest(),
    }
    result = read_planes(tmp_path, frame)
    assert np.all(result[0] == 2) and np.all(result[1] == 1)
    with pytest.raises(ValueError, match="slice hash"):
        read_planes(tmp_path, {**frame, "slice_sha256": "wrong"})


def test_triptych_caption_uses_selected_run():
    fig = Triptych("velocity", 7.5, (0.15, 1.5, 4), machine="Kay", saved_states=280)
    try:
        assert fig.fig.texts[-1].get_text().startswith("Kay · from rest · all 280")
    finally:
        fig.close()


def test_isometric_image_must_match_frame_source_and_hash(tmp_path):
    from PIL import Image

    path = tmp_path / "flow-0000.png"
    Image.new("RGB", (800, 600), "black").save(path)
    preparation = {"time": 0, "source_record_sha256": "source"}
    receipt = {
        "index": 0,
        **preparation,
        "images": {"flow": {"name": path.name, "sha256": sha(path)}},
    }
    assert rendered_image(tmp_path, receipt, preparation, 0, "flow").shape == (
        600,
        800,
        3,
    )
    with pytest.raises(ValueError, match="lineage"):
        rendered_image(tmp_path, {**receipt, "time": 0.5}, preparation, 0, "flow")
    with pytest.raises(ValueError, match="lineage"):
        rendered_image(
            tmp_path,
            {**receipt, "source_record_sha256": "other"},
            preparation,
            0,
            "flow",
        )
    receipt["images"]["flow"]["sha256"] = "changed"
    with pytest.raises(ValueError, match="hash"):
        rendered_image(tmp_path, receipt, preparation, 0, "flow")


def test_storage_recheck_preserves_budget_and_does_not_claim_restart():
    site = Path(__file__).parents[1] / "site"
    check = json.loads((site / "data/continuation-storage-check.json").read_text())
    required = (
        check["budgeted_plotfiles"] * check["plotfile_bytes"]
        + check["budgeted_checkpoints"] * check["checkpoint_bytes"]
        + check["extra_margin_bytes"]
    )
    assert required == check["required_bytes"]
    assert check["reserve_bytes"] == 20 * 2**30
    assert check["passed"] == (check["free_bytes"] >= required + check["reserve_bytes"])
    assert check["shortfall_bytes"] == max(
        0, required + check["reserve_bytes"] - check["free_bytes"]
    )
    assert check["planned_new_frames"] == 40 and not check["solver_started"]


def test_current_field_documentation_matches_current_run():
    site = Path(__file__).parents[1] / "site"
    run = json.loads((site / "data/best.json").read_text())
    page = (site / "voxels.html").read_text()
    assert run["machine"] == "Oliver" and run["parameters"]["base_n"] == 128
    assert f'{run["stored_cells"]:,}' in page
    assert f'{run["active_cells"]:,}' in page
    assert "huggingface.co/spaces" not in page
