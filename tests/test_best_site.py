"""The staged best run and its media must share one complete native history."""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).parents[1]
SITE = ROOT / "site"


def test_best_is_complete_native_history_with_fixed_planes():
    run = json.loads((SITE / "data/best.json").read_text())
    assert not run.get("preview_record")
    assert run["status"] == "completed" and run["validated"]
    assert not run["production_accuracy_certified"] and not run["vector_arrows"]
    assert (
        run["source_record_sha256"]
        == "afe8898110109ff25600c3cb58e3a9911b5baf6f6f8f7f14ff3dbbd826ded97c"
    )
    assert len(run["frames"]) == len(run["native_checks"]) == run["saved_frames"] == 280
    assert run["frame_times"] == [r["time"] for r in run["frames"]]
    assert run["frame_times"][0] == 0 and run["frame_times"][-1] == 0.995
    assert all(a < b for a, b in zip(run["frame_times"], run["frame_times"][1:]))
    assert run["stored_cells"] == 12055040 and run["active_cells"] == 10810304
    assert run["machine"] == "Oliver" and run["parameters"]["base_n"] == 128
    assert run["finest_equivalent_n"] == 2048
    assert run["diagnostics"]["step"] == 5510
    for frame, native in zip(run["frames"], run["native_checks"], strict=True):
        assert frame["path"] == native["path"] and frame["time"] == native["time"]
        assert native["force_linf_error"] == 0
        assert frame["peak_speed"] == native["peak_speed"]
        assert len(frame["slice_sha256"]) == 64
        if frame["time"] <= 0.55:
            assert frame["peak_speed"] == 0
    assert run["render"]["slice_coordinates"] == {"xy": 0, "xz": 0}
    assert run["diagnostics"]["peak_speed"] == pytest.approx(7.415301682371514)


def test_best_movies_have_all_frames_matching_durations_and_fixed_scales():
    run = json.loads((SITE / "data/best.json").read_text())
    for key, assets in run["media"].items():
        for asset in assets.values():
            path = SITE / asset["path"]
            assert 0 < path.stat().st_size < 100 * 1024**2
            assert hashlib.sha256(path.read_bytes()).hexdigest() == asset["sha256"]
        with Image.open(SITE / assets["gif"]["path"]) as gif:
            assert gif.n_frames == 280 and gif.size == (1600, 640)
            for i, duration in enumerate(run["playback"]["source_frame_duration_ms"]):
                gif.seek(i)
                assert gif.info["duration"] == duration
        field = key.split("_")[0]
        assert run["render"]["slice_color_max"][field] > 0
        assert run["render"]["source_record_sha256"] == run["source_record_sha256"]
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
        assert (video["width"], video["height"]) == (2400, 960)
        assert int(video["nb_frames"]) == sum(run["playback"]["mp4_frame_repeats"])
        assert float(video["duration"]) == pytest.approx(
            run["playback"]["duration_seconds"], abs=0.02
        )


def test_homepage_uses_only_fixed_midplane_media():
    html = (SITE / "index.html").read_text()
    script = (SITE / "best.js").read_text()
    assert 'src="best.js?v=outer-complete-2"' in html and 'src="app.js"' not in html
    assert '"data/best.json"' in script and "data/stream.json" not in script
    assert "At peak-speed height" not in html and "moving slice" not in script
    assert 'data-view="peak"' not in html and 'role="tab' not in html
    assert "views.media[kind].files" in script
    assert '["xy","xz","isometric"]' in script
    assert html.count("Fixed planes: y = 0 and z = 0.") == 2
    assert "Active solver cells" in html and "2048³-equivalent core only" in html
    assert "media/stream-flow-3d.html" not in html
    assert "media/stream-flow-3d.html" not in (SITE / "results.html").read_text()


def test_best_page_controller_with_fixed_planes_and_reduced_motion():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is optional for controller unit tests")
    subprocess.run([node, str(ROOT / "tests/best_site_ui.cjs")], check=True, timeout=30)
