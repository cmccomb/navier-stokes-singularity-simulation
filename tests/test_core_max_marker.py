"""Velocity-only maximum markers use actual native cells, not visual peaks."""

import json
from argparse import Namespace

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("zarr")
pytest.importorskip("pyvista")
from scripts.mark_native_core_max import annotate, core_maximum, mark, project, sha


def field():
    return np.zeros((8, 8, 8, 6)), {"origin": [-0.0625] * 3, "spacing": [0.015625] * 3}


def test_maximum_uses_velocity_not_force_and_native_centers():
    values, level = field()
    values[2, 3, 5, :3] = (3, 4, 0)
    values[6, 1, 2, 3:] = 1000000
    result = core_maximum(values, level)
    assert result["value"] == 5
    assert result["cell_index"] == [2, 3, 5]
    np.testing.assert_allclose(
        result["xyz"],
        np.array(level["origin"]) + (np.array([2, 3, 5]) + 0.5) * level["spacing"],
    )


def test_rest_has_no_unique_marker_and_changes_no_pixels():
    values, level = field()
    maximum = core_maximum(values, level)
    assert maximum["xyz"] is None and maximum["cell_index"] is None
    image = Image.new("RGB", (2400, 1440), "#07111f")
    np.testing.assert_array_equal(annotate(image, maximum), image)


def test_exact_ties_are_deterministic_without_tracking():
    values, level = field()
    values[2, 3, 5, 1] = 5
    values[4, 1, 2, 2] = 5
    assert core_maximum(values, level)["cell_index"] == [2, 3, 5]
    values[4, 1, 2, 2] += 1e-10
    assert core_maximum(values, level)["cell_index"] == [4, 1, 2]


def test_projection_is_core_panel_center_and_marker_stays_in_core_panel():
    np.testing.assert_allclose(project([0, 0, 0]), [1880, 735])
    image = Image.new("RGB", (2400, 1440), "#07111f")
    for point in ([0, 0, 0], [0.062, -0.062, 0.062], [-0.062, 0.062, -0.062]):
        marked = np.asarray(annotate(image, {"xyz": point, "value": 7.415}))
        changed = np.any(marked != np.asarray(image), axis=-1)
        y, x = np.where(changed)
        assert len(x) > 20
        assert x.min() >= 1400 and x.max() < 2400
        assert y.min() > 230 and y.max() < 1275


@pytest.mark.parametrize("damage", [None, "source", "core", "clock", "incomplete"])
def test_annotation_is_velocity_only_and_history_bound(tmp_path, damage):
    rendered, maxima = tmp_path / "rendered", tmp_path / "maxima"
    rendered.mkdir()
    maxima.mkdir()
    original = {
        "validated": True,
        "complete_history": True,
        "source_record_sha256": "native",
        "frame_times": [0, 0.995],
        "indices": [0, 1],
        "records": [],
        "media": {"stale": "must reencode"},
    }
    peaks = {
        "validated": True,
        "complete_history": True,
        "source_record_sha256": "native",
        "policy": "native cells",
        "records": [],
    }
    for q in ("flow", "force"):
        (rendered / q).mkdir()
    for i, t in enumerate(original["frame_times"]):
        frame = {"index": i, "time": t, "native_level_sha256": [str(i)], "images": {}}
        for q in ("flow", "force"):
            path = rendered / q / f"frame-{i:04d}.png"
            Image.new("RGB", (2400, 1440), "#07111f").save(path)
            frame["images"][q] = {
                "name": str(path.relative_to(rendered)),
                "sha256": sha(path),
            }
        original["records"].append(frame)
        peaks["records"].append(
            {
                "index": i,
                "time": t,
                "core_sha256": str(i),
                "xyz": [0, 0, 0] if i else None,
                "value": float(i),
            }
        )
    if damage == "source":
        peaks["source_record_sha256"] = "other"
    elif damage == "core":
        peaks["records"][1]["core_sha256"] = "different"
    elif damage == "clock":
        peaks["records"][1]["time"] = 0.99
    elif damage == "incomplete":
        peaks["complete_history"] = False
    (rendered / "manifest.json").write_text(json.dumps(original))
    (maxima / "manifest.json").write_text(json.dumps(peaks))
    args = Namespace(rendered=rendered, maxima=maxima, output=tmp_path / "marked")
    if damage:
        with pytest.raises(ValueError):
            mark(args)
        assert not args.output.exists()
        return
    mark(args)
    result = json.loads((args.output / "manifest.json").read_text())
    assert not result["media"]
    for i in range(2):
        assert (
            result["records"][i]["images"]["force"]
            == original["records"][i]["images"]["force"]
        )
        for q in ("flow", "force"):
            assert (
                sha(rendered / original["records"][i]["images"][q]["name"])
                == original["records"][i]["images"][q]["sha256"]
            )
        force = f"force/frame-{i:04d}.png"
        assert sha(args.output / force) == sha(rendered / force)
    assert (
        result["records"][1]["images"]["flow"]["sha256"]
        != original["records"][1]["images"]["flow"]["sha256"]
    )
