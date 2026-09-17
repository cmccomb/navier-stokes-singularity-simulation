"""The HF viewer must not relabel a prototype as a full history or replace GIFs."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"


def test_native_embed_loads_on_approach_and_accepts_only_its_own_resize_messages():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is needed for embed controller checks")
    subprocess.run(
        [node, str(ROOT / "tests/native_site_ui.cjs")], check=True, timeout=30
    )


def test_homepage_keeps_mesh_interaction_without_the_data_explorer():
    html = (SITE / "index.html").read_text()
    assert 'id="native-explorer"' not in html and 'id="native-frame"' not in html
    assert 'src="native-explorer.js"' not in html
    assert 'src="mesh-explorer.html?embed=whole&amp;v=kay128"' in html
    assert (
        html.index('src="media/mesh-xy.svg?v=kay128"')
        < html.index('src="media/mesh-isometric.svg?v=kay128"')
        < html.index('src="mesh-explorer.html?embed=whole&amp;v=kay128"')
    )
    assert 'id="flow-video"' in html and 'id="force-video"' in html
    assert "All 280 saved states are included." in html
    assert 'href="voxels.html"' in (SITE / "documentation.html").read_text()
    script = (SITE / "native-explorer.js").read_text()
    assert "IntersectionObserver" in script and "observer.disconnect()" in script
    assert "Prototype:" not in html and 'id="native-coverage"' not in html


def test_site_and_space_pin_the_same_release():
    record = json.loads((SITE / "data/native-explorer.json").read_text())
    config = json.loads((ROOT / "spaces/native_explorer/dataset.json").read_text())
    assert record["dataset"] == config
    assert record["validated"] and record["start_time"] == 0
    assert len(config["revision"]) == 40 and config["prefix"].startswith("releases/")
    assert record["complete_history"] == (
        record["saved_frames"] == record["source_frames"]
    )
