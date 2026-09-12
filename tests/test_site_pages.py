import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

SITE = Path(__file__).parents[1] / "site"


class _References(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.paths: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        attributes = dict(attrs)
        for name in ("href", "src"):
            value = attributes.get(name)
            if value:
                self.paths.append(value)


def test_documentation_is_a_native_multipage_site() -> None:
    index = (SITE / "index.html").read_text(encoding="utf-8")
    assert 'href="documentation.html"' in index
    for name in (
        "documentation.html",
        "method.html",
        "derivation.html",
        "running.html",
        "results.html",
        "reproduction.html",
        "review.html",
        "accuracy.html",
        "refinement.html",
    ):
        assert (SITE / name).is_file()

    derivation = (SITE / "derivation.html").read_text(encoding="utf-8")
    assert "U<sup>*</sup>(η) = 4η + j₀" in derivation
    assert "spacing gain" in derivation
    assert "1.47" in derivation


def test_site_pages_have_no_broken_local_links() -> None:
    missing: list[str] = []
    for page in SITE.glob("*.html"):
        parser = _References()
        parser.feed(page.read_text(encoding="utf-8"))
        for reference in parser.paths:
            parsed = urlsplit(reference)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            target = page.parent / parsed.path
            if not target.exists():
                missing.append(f"{page.name}: {reference}")
    assert not missing, f"broken site links: {missing}"


def test_site_keeps_peak_speed_without_illustrative_comparisons() -> None:
    index = (SITE / "index.html").read_text(encoding="utf-8")
    app = (SITE / "app.js").read_text(encoding="utf-8")
    faq = (SITE / "faq.html").read_text(encoding="utf-8")
    styles = (SITE / "styles.css").read_text(encoding="utf-8")
    assert 'id="peak-speed"' in index
    assert 'id="speed-time"' in index
    assert "How is peak speed measured?" in faq
    content = f"{index}\n{app}\n{faq}\n{styles}"
    for removed in (
        "speed-comparison", "comparison-scale", "slowReferences",
        "fingernail", "Arctic coast", "Statue speed", "emoji comparison",
    ):
        assert removed not in content


def test_refinement_pilot_is_documentation_not_a_replacement_flow_run() -> None:
    docs = (SITE / "documentation.html").read_text()
    page = (SITE / "refinement.html").read_text()
    assert 'href="refinement.html"' in docs
    assert 'href="data/refinement-pilot.json"' in page
    assert "not a new best movie or a singularity demonstration" in page
    assert "manufactured tests, not validation of the paper-surrogate forcing" in page
    report = json.loads((SITE / "data/refinement-pilot.json").read_text())
    assert report["passed"] and len(report["cases"]) == 9
    row = next(r for r in report["cases"] if r["base_n"] == 128 and r["levels"] == 4)
    assert row["finest_effective_n"] == 1024
    assert row["stored_cells"] == 8388608
    assert row["active_cells"] == 7602176
    assert row["coarse_fine_flux_mismatch"] == 0
    assert row["divergence_after_linf"] < 1e-8


def test_tenfold_design_stays_distinct_from_measured_operator_results() -> None:
    page = (SITE / "refinement.html").read_text()
    assert "extrapolations, not new simulated frames" in page
    design = json.loads((SITE / "data/tenfold-design.json").read_text())
    assert [r["speed_multiplier"] for r in design["forecasts"]] == [1,10,100]
    assert not any(r["production_ready"] for r in design["tenfold_mesh_candidates"])
    diffusion = json.loads((SITE / "data/diffusion-pilot.json").read_text())
    assert diffusion["passed"] and len(diffusion["cases"]) == 11
    assert all(r["passed"] for r in diffusion["convergence"]["temporal"])


def test_coupled_validation_remains_distinct_from_the_project_force():
    page = (SITE / "refinement.html").read_text()
    assert 'href="data/coupled-pilot.json"' in page
    assert "not sharing one simulation's domain" in page
    report = json.loads((SITE / "data/coupled-pilot.json").read_text())
    assert report["passed"] and len(report["cases"]) == 14
    assert len(report["native_archive"]["frames"]) == 5
    assert report["native_archive"]["restart"]["velocity_linf_difference"] == 0
    assert report["native_archive"]["frames"][0]["peak_speed"] == 0
    assert all(r["initial"]["adapter_sha256"] == report["adapter_sha256"] for r in report["cases"])
    capacity = report["fleet_checks"]["nonbinary_capacity"]
    assert capacity["passed"] is False  # Keep the original failed gate visible.
    assert capacity["resolution"]["status"] == "fixed"
    assert all(r["volume"] == 8 for r in capacity["resolution"]["regression"]["rows"])


def test_actual_forcing_port_keeps_version_and_accuracy_boundaries():
    page = (SITE / "refinement.html").read_text()
    assert 'href="data/paper-port.json"' in page
    assert "Existing homepage movies remain v1" in page
    assert "convergence probes, not accepted production meshes" in page
    report = json.loads((SITE / "data/paper-port.json").read_text())
    assert report["localization"]["default_changed"] is False
    for revision in ("legacy", "localized"):
        assert report["profile_points"][revision]["passed"]
        assert len(report["profile_points"][revision]["cases"]) == 40
        assert report["fields"][revision]["passed"]
        assert len(report["fields"][revision]["cases"]) == 14
    smoke = report["smoke"]
    assert smoke["validated"] and smoke["status"] == "completed"
    assert smoke["final"]["time"] == 0.62
    assert len(smoke["native_frames"]) == 26
    assert smoke["native_frames"][0]["peak_speed"] == 0
    assert all(f["force_linf_error"] == 0 for f in smoke["native_frames"])
    assert report["earlier_output_failure"]["validated"] is False
    assert report["fine_force"]["passed"]
    assert len(report["fine_force"]["cases"]) == 18
    assert {r["n"] for r in report["fine_force"]["cases"]} == {1024, 16384}


def test_threading_record_is_correctness_not_a_speedup_claim():
    page = (SITE / "refinement.html").read_text()
    assert "correctness checks, not measured speedups" in page
    record = json.loads((SITE / "data/threading-validation.json").read_text())
    assert record["limits"]["mpi_enabled"] is False
    for key in ("same_binary_one_two", "same_binary_one_four"):
        assert record[key]["passed"]
        assert len(record[key]["frames"]) == 26
        assert record[key]["velocity_linf_difference"] < 2e-16
        assert record[key]["frames"][0]["peak_speed"] == 0
    assert record["native_restriction"]["passed"]
    assert record["native_restriction"]["restart_rejects_mesh_change"]


def test_fleet_sensitivity_keeps_failed_audit_and_uncertainty_visible():
    page = (SITE / "refinement.html").read_text()
    record = json.loads((SITE / "data/fleet-sensitivity.json").read_text())
    assert 'href="data/fleet-sensitivity.json"' in page
    assert "not an accepted fully resolved forcing or a blowup claim" in page
    assert record["temporal"]["passed"]
    assert len(record["temporal"]["frames"]) == 35
    assert len(record["temporal"]["duplicate_events"]) == 6
    assert record["temporal"]["runs"][0]["original_status"] == "failed"
    assert len(record["spatial_prefix"]["frames"]) == 5
    assert record["source_sensors"]["completed"]
    assert not record["candidate"]["production_accuracy_certified"]
    assert record["candidate"]["planned_frames"] == 280
