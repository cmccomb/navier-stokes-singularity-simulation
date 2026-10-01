"""The summary figure must preserve the published qualification boundary."""

from pathlib import Path

import pytest

from scripts.plot_refinement_summary import collect, render


def test_summary_uses_all_audited_sources(tmp_path):
    report = collect(Path("site/data").resolve())
    candidate = report["candidate"]
    assert candidate["interval"] == pytest.approx(
        [0.6158075087303938, 0.6290714295556976]
    )
    assert len(report["spatial"]["times"]) == 8
    assert max(candidate["from_rest_half_step_velocity_relative_l2"]) < min(
        candidate["quarter_spatial_velocity_gap"]
    )
    assert [
        n128 < n64
        for n64, n128 in zip(
            candidate["finest_core_balance_rms_n64"],
            candidate["finest_core_balance_rms_n128"],
            strict=True,
        )
    ] == [False, True, True]
    assert all(
        0.35 < ratio < 0.5
        for ratio in candidate["half_to_quarter_balance_gap_over_quarter_residual"]
    )
    assert report["conclusion"]["locked_protocol_fully_qualified"] is False
    output = tmp_path / "summary.svg"
    render(report, output)
    text = output.read_text()
    assert "Where the refinement evidence holds" in text
    assert "prevent a qualified precursor claim" in text
