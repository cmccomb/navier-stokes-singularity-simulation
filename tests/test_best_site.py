"""The staged best run and its media must share one complete native history."""

import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).parents[1]
SITE = ROOT / "site"


def test_best_is_complete_native_history_with_traceable_peak_planes():
    run = json.loads((SITE / "data/best.json").read_text())
    assert not run.get("preview_record")
    assert run["status"] == "completed" and run["validated"]
    assert not run["production_accuracy_certified"] and not run["vector_arrows"]
    assert (
        run["source_record_sha256"]
        == "036814dd3dcb1f15758c8061532c070dadef80b0df704778f61eeb9ee2e5295a"
    )
    assert len(run["frames"]) == len(run["native_checks"]) == run["saved_frames"] == 280
    assert run["frame_times"] == [r["time"] for r in run["frames"]]
    assert run["frame_times"][0] == 0 and run["frame_times"][-1] == 0.995
    assert all(a < b for a, b in zip(run["frame_times"], run["frame_times"][1:]))
    assert run["stored_cells"] == 1310720 and run["active_cells"] == 1179648
    assert run["diagnostics"]["step"] == 5510
    for frame, native in zip(run["frames"], run["native_checks"], strict=True):
        assert frame["path"] == native["path"] and frame["time"] == native["time"]
        assert native["force_linf_error"] == 0
        assert frame["peak_xy"] and frame["peak_speed"] <= native["peak_speed"] + 1e-12
        assert all(math.isfinite(v) and -1 <= v < 1 for v in frame["peak_position"])
        if frame["time"] <= 0.55:
            assert frame["peak_position"] == [0, 0, 0] and frame["peak_level"] == -1
        else:
            spacing = 2 / (64 * 2 ** frame["peak_level"])
            for value in frame["peak_position"]:
                index = (value + 1) / spacing - 0.5
                assert index == pytest.approx(round(index), abs=1e-12)
    assert run["frames"][-1]["peak_position"][2] == 0.0380859375


def test_best_movies_have_all_frames_matching_durations_and_fixed_scales():
    run = json.loads((SITE / "data/best.json").read_text())
    for key, assets in run["media"].items():
        for asset in assets.values():
            path = SITE / asset["path"]
            assert 0 < path.stat().st_size < 100 * 1024**2
            assert hashlib.sha256(path.read_bytes()).hexdigest() == asset["sha256"]
        with Image.open(SITE / assets["gif"]["path"]) as gif:
            assert gif.n_frames == 280 and gif.size == (1440, 792)
            for i, duration in enumerate(run["playback"]["source_frame_duration_ms"]):
                gif.seek(i)
                assert gif.info["duration"] == duration
        field = key.split("_")[0]
        assert (
            run["render"][key]["color_max"]
            == run["render"][field + "_midplane"]["color_max"]
        )
    assert run["playback"]["source_frame_duration_ms"] == [1000] + [200] * 278 + [500]


def test_best_videos_preserve_the_gif_clock():
    probe = shutil.which("ffprobe")
    if probe is None:
        pytest.skip("ffprobe is optional for local media verification")
    run = json.loads((SITE / "data/best.json").read_text())
    for assets in run["media"].values():
        report = json.loads(
            subprocess.check_output(
                [
                    probe,
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=nb_frames,width,height,duration",
                    "-of",
                    "json",
                    str(SITE / assets["mp4"]["path"]),
                ],
                text=True,
            )
        )
        video = report["streams"][0]
        assert (video["width"], video["height"]) == (1440, 792)
        assert int(video["nb_frames"]) == sum(run["playback"]["mp4_frame_repeats"])
        assert float(video["duration"]) == pytest.approx(
            run["playback"]["duration_seconds"], abs=0.02
        )


def test_homepage_uses_only_fixed_midplane_media():
    html = (SITE / "index.html").read_text()
    script = (SITE / "best.js").read_text()
    assert 'src="best.js"' in html and 'src="app.js"' not in html
    assert 'fetch("data/best.json"' in script and "data/stream.json" not in script
    assert "At peak-speed height" not in html and "moving slice" not in script
    assert 'data-view="peak"' not in html and 'role="tab' not in html
    assert 'run.media[`${kind}_midplane`]' in script
    assert html.count("Fixed planes: y = 0 and z = 0.") == 2
    assert "Active solver cells" in html and "1024³-equivalent core only" in html
    assert "media/stream-flow-3d.html" not in html
    assert "media/stream-flow-3d.html" in (SITE / "results.html").read_text()


def test_best_page_controller_with_fixed_planes_and_reduced_motion():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is optional for controller unit tests")
    subprocess.run([node, str(ROOT / "tests/best_site_ui.cjs")], check=True, timeout=30)
