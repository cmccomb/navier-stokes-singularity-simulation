"""Check the drawn mesh has the same active geometry as the saved hierarchy."""

import itertools
import json
from pathlib import Path

import pytest

from scripts.render_native_mesh import hierarchy, plane_lines, transition_cells

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def levels():
    record = json.loads((ROOT / "site/data/best.json").read_text())
    return hierarchy({**record, "stored_cells": 10485760, "active_cells": 9437184})


def test_plane_edges_are_on_cell_faces_and_exclude_covered_interiors(levels):
    for i, x0, y0, x1, y1 in plane_lines(levels):
        level = levels[i]
        assert x0 == x1 or y0 == y1
        for value in (x0, y0, x1, y1):
            index = (value + level["half"]) / level["dx"]
            assert index == pytest.approx(round(index))
            assert -level["half"] <= value <= level["half"]
        if i + 1 < len(levels):
            h = levels[i + 1]["half"]
            assert not (abs((x0 + x1) / 2) < h and abs((y0 + y1) / 2) < h)


def test_transition_block_fills_space_once_with_native_cells(levels):
    cells, bounds = transition_cells(levels)
    volume = sum(cell[-1] ** 3 for cell in cells)
    expected = 1
    for lo, hi in bounds:
        expected *= hi - lo
    assert volume == pytest.approx(expected)
    assert {c[0] for c in cells} == {3, 4}
    for level, *values in cells:
        low, dx = values[:3], values[3]
        assert dx == levels[level]["dx"]
        assert all(
            lo <= v and v + dx <= hi for v, (lo, hi) in zip(low, bounds, strict=True)
        )
    for a, b in itertools.combinations(cells, 2):
        assert any(a[k] + a[4] <= b[k] or b[k] + b[4] <= a[k] for k in (1, 2, 3))


def test_generated_mesh_matches_current_record(levels):
    data = json.loads((ROOT / "site/data/native-mesh.json").read_text())
    run = json.loads((ROOT / "site/data/best.json").read_text())
    assert data["schema_version"] == 2
    assert data["source_record_sha256"] == run["source_record_sha256"]
    assert data["levels"] == levels
    cells, bounds = transition_cells(levels)
    assert data["cells"] == cells and data["detail_bounds"] == bounds
    assert data["active_cells"] == run["active_cells"] == 10810304
    assert data["stored_cells"] == run["stored_cells"] == 12055040
    assert data["composite_volume"] == 8
    assert data["planes3d"] == [
        edge for plane in data["planes_by_fixed_axis"].values() for edge in plane
    ]
    for fixed, edges in data["planes_by_fixed_axis"].items():
        assert edges
        for level, *coordinates in edges:
            a, b = coordinates[:3], coordinates[3:]
            assert a[int(fixed)] == b[int(fixed)] == 0
            assert sum(x != y for x, y in zip(a, b, strict=True)) == 1
            for value in coordinates:
                index = (value + 1) / levels[level]["dx"]
                assert index == pytest.approx(round(index))
                assert -1 <= value <= 1
