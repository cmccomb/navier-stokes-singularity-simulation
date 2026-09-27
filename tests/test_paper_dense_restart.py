"""Check that a dense probe preserves intervening original output events."""

import pytest

from scripts.paper_dense_restart import (
    controlled_command,
    schedule,
    time_controls,
    validate_readback,
)


def test_dense_schedule_keeps_original_events_and_center_stencils():
    start = 0.6005534368496662
    centers = [0.6158075087303938, 0.6224977197922241, 0.6290714295556976]
    original = [0.6020692874341146, 0.6089987316985057, *centers]
    times = schedule(start, centers, 0.0005, original)
    assert times[0] == start
    assert times[-1] == centers[-1] + 0.001
    assert all(time in times for time in original)
    assert len(times) == 18  # checkpoint plus 15 dense events and two earlier originals
    assert times == sorted(set(times))


def test_dense_schedule_rejects_event_before_checkpoint():
    with pytest.raises(ValueError, match="follow checkpoint"):
        schedule(0.61, [0.6105], 0.0005, [])


def test_zero_step_checker_can_validate_without_energy_field():
    initial = {"step": 250, "time": 0.6, "levels": 5, "stored_cells": 100, "volume": 8.0, "peak_speed": 0.03}
    checked = {"step": 250, "time": 0.6, "levels": 5, "stored_cells": 100, "composite_volume": 8.0, "peak_speed": 0.03}
    validate_readback(checked, initial)
    with pytest.raises(ValueError, match="restart readback"):
        validate_readback({**checked, "stored_cells": 99}, initial)


def test_fixed_force_time_controls_halve_only_integration_limits(tmp_path):
    record = {
        "parameters": {"max_dt": 0.00025, "integration_phase_step": 0},
        "profile_manifest": {"parameters": {"forcing_phase_step": 0.0375}},
        "command": ["solver", "inputs", "stop_time=0.995", "ns.table_file=table", "ns.max_dt=0.00025"],
    }
    assert time_controls(record, 0.5) == (0.000125, 0.01875)
    argv = controlled_command(record, tmp_path, tmp_path / "chk00250", [0.6005, 0.63], 0.5)
    assert "ns.max_dt=0.000125" in argv
    assert "ns.integration_phase_step=0.018749999999999999" in argv
    assert "ns.table_file=" + str(tmp_path / "profile.tbl") in argv
    with pytest.raises(ValueError, match="time factor"):
        time_controls(record, 1.5)
