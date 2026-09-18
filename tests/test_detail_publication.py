"""Detail publication must preserve the source, native clock and site budget."""

import copy
import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.publish_native_detail import digest, validate

SITE = Path(__file__).parents[1] / "site"


def fixture():
    best = {
        "saved_frames": 2,
        "source_record_sha256": "source",
        "frame_times": [0, 0.995],
    }
    media = {
        q: {"saved_states": 2, "duration_seconds": 1.5, "width": 2400, "height": 1440}
        for q in ("flow", "force")
    }
    manifest = {
        "kind": "native-detail-3d",
        "validated": True,
        "complete_history": True,
        "native_frame_count": 2,
        "indices": [0, 1],
        "source_record_sha256": "source",
        "frame_times": best["frame_times"],
        "records": [{"index": i, "time": t} for i, t in enumerate(best["frame_times"])],
        "media": copy.deepcopy(media),
    }
    return manifest, best, {"media": media}


@pytest.mark.parametrize(
    "damage",
    [None, "source", "missing", "time", "record", "duration", "count", "shape"],
)
def test_publication_matches_best_native_clock(damage):
    manifest, best, views = fixture()
    if damage == "source":
        manifest["source_record_sha256"] = "another-run"
    elif damage == "missing":
        manifest["indices"] = [1]
    elif damage == "time":
        manifest["frame_times"] = [0, 0.99]
    elif damage == "record":
        manifest["records"][1]["time"] = 0.99
    elif damage == "duration":
        manifest["media"]["flow"]["duration_seconds"] = 1.2
    elif damage == "count":
        manifest["media"]["force"]["saved_states"] = 1
    elif damage == "shape":
        manifest["media"]["flow"]["width"] = 1200
    if damage:
        with pytest.raises(ValueError):
            validate(manifest, best, views)
    else:
        validate(manifest, best, views)


def test_published_detail_record_and_all_gif_states():
    path = SITE / "data/detail-view.json"
    if not path.exists():
        pytest.skip("Full render has not been published yet")
    manifest = json.loads(path.read_text())
    best = json.loads((SITE / "data/best.json").read_text())
    views = json.loads((SITE / "data/three-view.json").read_text())
    validate(manifest, best, views)
    assert manifest["saved_frames"] == 280
    assert manifest["core_half_width"] == 0.0625
    assert len(manifest["seed_points"]) == 32
    assert manifest["records"][0]["streamline_branches"] == 0
    assert manifest["records"][-1]["streamline_branches"] > 0
    assert (
        manifest["records"][-1]["diagnostics"]["peak_speed"]
        == best["diagnostics"]["peak_speed"]
    )
    for q in ("flow", "force"):
        for asset in manifest["media"][q]["files"].values():
            file = SITE / asset["path"]
            assert digest(file) == asset["sha256"]
            assert file.stat().st_size == asset["bytes"] < 100_000_000
        with Image.open(SITE / manifest["media"][q]["files"]["gif"]["path"]) as gif:
            assert gif.size == (1200, 720)
            assert gif.n_frames == 280
            for i, hold in enumerate(manifest["playback"]["gif_hold_ms"]):
                gif.seek(i)
                assert gif.info["duration"] == hold
    assert sum(p.stat().st_size for p in SITE.rglob("*") if p.is_file()) < 1_000_000_000
