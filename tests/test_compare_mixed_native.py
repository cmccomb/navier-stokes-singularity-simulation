"""Conservation and coverage checks for changing native AMR footprints."""

import numpy as np
import pytest

from scripts.compare_mixed_native import restrict_composite


def block(level, shape, spacing, offset, index_lo=(0, 0, 0)):
    return {
        "level": level,
        "shape": [*shape, 6],
        "spacing": [spacing] * 3,
        "index_lo": list(index_lo),
        "origin": [-1 + i * spacing for i in index_lo],
        "offset_bytes": offset,
    }


def test_mixed_levels_conserve_volume_inside_one_target_cell(tmp_path):
    target = block(0, (2, 2, 2), 1, 0)
    coarse = block(0, (4, 4, 4), 0.5, 0)
    fine = block(1, (2, 2, 2), 0.25, 4**3 * 6 * 8)
    source = [coarse, fine]
    raw = tmp_path / "mixed.bin"
    with raw.open("wb") as stream:
        stream.write(np.ones((4, 4, 4, 6), dtype="<f8").tobytes())
        stream.write(np.full((2, 2, 2, 6), 3, dtype="<f8").tobytes())
    result = restrict_composite(raw, source, target, np.ones((2, 2, 2), bool))
    assert result[0, 0, 0] == pytest.approx([1.25] * 6)
    assert np.all(result[1:, :, :] == 1)
    assert np.all(result[:, 1:, :] == 1)
    assert np.all(result[:, :, 1:] == 1)


def test_missing_source_cells_fail_coverage_gate(tmp_path):
    target = block(0, (2, 2, 2), 1, 0)
    partial = block(0, (2, 2, 2), 0.5, 0)
    raw = tmp_path / "partial.bin"
    raw.write_bytes(np.ones((2, 2, 2, 6), dtype="<f8").tobytes())
    with pytest.raises(ValueError, match="coverage differs from one"):
        restrict_composite(raw, [partial], target, np.ones((2, 2, 2), bool))
