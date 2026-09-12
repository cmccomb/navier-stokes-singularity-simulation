import numpy as np

from scripts.force_resolution_scan import average


def test_sensor_cell_averaging_keeps_components_and_constants():
    field = np.arange(8**3 * 6, dtype=float).reshape(8, 8, 8, 6)
    coarse = average(field, 4)
    assert coarse.shape == (2, 2, 2, 6)
    np.testing.assert_array_equal(
        coarse[0, 0, 0], field[:4, :4, :4].mean(axis=(0, 1, 2))
    )
    np.testing.assert_array_equal(
        average(np.ones_like(field), 4), np.ones((2, 2, 2, 6))
    )
