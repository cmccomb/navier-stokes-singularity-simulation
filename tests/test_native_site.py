"""The HF viewer must not relabel a prototype as a full history or replace GIFs."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"


def test_explorer_is_opt_in_and_keeps_complete_movie_fallbacks():
    html = (SITE / "index.html").read_text()
    assert '<details class="native-explorer"' in html
    assert 'data-src="https://ccm-navier-stokes-singularity-simulation.hf.space"' in html
    assert 'id="flow-video"' in html and 'id="force-video"' in html
    assert "the GIFs above already contain all 280" in html
    assert 'href="voxels.html"' in (SITE / "documentation.html").read_text()
    script = (SITE / "native-explorer.js").read_text()
    assert 'panel.open && !frame.getAttribute("src")' in script
    assert "data.complete_history && data.saved_frames === 280" in script


def test_site_and_space_pin_the_same_release():
    record = json.loads((SITE / "data/native-explorer.json").read_text())
    config = json.loads((ROOT / "spaces/native_explorer/dataset.json").read_text())
    assert record["dataset"] == config
    assert record["validated"] and record["start_time"] == 0
    assert len(config["revision"]) == 40 and config["prefix"].startswith("releases/")
    assert record["complete_history"] == (record["saved_frames"] == record["source_frames"])
