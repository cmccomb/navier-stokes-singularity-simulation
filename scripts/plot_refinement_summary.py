"""Plot the audited spatial, temporal, and residual qualification evidence."""

from __future__ import annotations

import argparse
import importlib
import json
import math
from pathlib import Path

import matplotlib

from scripts.paper_run import sha, write

matplotlib.use("Agg")
plt = importlib.import_module("matplotlib.pyplot")


CENTERS = (
    ("t06158", 0.6158075087303938),
    ("t06225", 0.6224977197922241),
    ("t06291", 0.6290714295556976),
)


def read(path: Path) -> dict:
    result = json.loads(path.read_text())
    if not isinstance(result, dict):
        raise TypeError(f"expected object in {path.name}")
    return result


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0, abs_tol=1e-12)


def collect(data: Path) -> dict:
    protocol_path = data / "outer-band-protocol-20260927.json"
    spatial_path = data / "outer-band-prospective-prefix-20260927.json"
    temporal_path = data / "outer-from-rest-n64-temporal.json"
    protocol, spatial, temporal = map(read, (protocol_path, spatial_path, temporal_path))
    if (
        spatial.get("kind") != "outer-band-prospective-spatial-screen"
        or spatial["protocol_sha256"] != sha(protocol_path)
        or temporal.get("kind") != "temporal-fixed-force-native-sensitivity"
        or temporal.get("passed") is not True
    ):
        raise ValueError("spatial or temporal source audit is incomplete")
    limits = protocol["spatial_screen"]
    checkpoints = [item for item in spatial["checkpoints"] if "comparison" in item]
    if not checkpoints or any(
        item["comparison"].get("validated") is not True for item in checkpoints
    ):
        raise ValueError("spatial comparison sequence is incomplete")
    candidates = spatial.get("candidate_sequences", [])
    if len(candidates) != 1 or candidates[0]["count"] != 3:
        raise ValueError("expected one three-checkpoint candidate sequence")
    first, last = candidates[0]["first_frame_index"], candidates[0]["last_frame_index"]
    candidate_rows = [item for item in checkpoints if first <= item["frame_index"] <= last]
    if len(candidate_rows) != 3 or any(item["screen"] != "pass" for item in candidate_rows):
        raise ValueError("candidate interval does not contain three passes")
    candidate_times = [item["time"] for item in candidate_rows]
    temporal_frames = temporal["frames"]
    if len(temporal_frames) != 3 or any(
        not close(frame["time"], time)
        for frame, time in zip(temporal_frames, candidate_times, strict=True)
    ):
        raise ValueError("from-rest temporal controls do not match the candidate interval")

    sources = [protocol_path, spatial_path, temporal_path]
    residual_n64, residual_n128, residual_ratio, quarter_velocity = [], [], [], []
    for tag, center in CENTERS:
        if not close(center, candidate_times[len(residual_n64)]):
            raise ValueError("configured residual center differs from spatial candidate")
        n64_path = data / f"vorticity-balance-dense-outer-n64-{tag}.json"
        n128_path = data / f"vorticity-balance-dense-outer-n128-{tag}.json"
        gap_path = data / f"vorticity-balance-dense-quarter-gap-outer-n64-{tag}.json"
        n64, n128, gap = map(read, (n64_path, n128_path, gap_path))
        sources.extend((n64_path, n128_path, gap_path))
        if not all(item.get("validated") is True for item in (n64, n128, gap)):
            raise ValueError("residual source audit is incomplete")
        if not all(close(item["center_time"], center) for item in (n64, n128, gap)):
            raise ValueError("residual center times differ")
        core64 = n64["regions"]["finest_core_interior"]["rms"]["residual_five"]
        core128 = n128["regions"]["finest_core_interior"]["rms"]["residual_five"]
        residual = gap["regions"]["finest_core_interior"]["rms"]["residual_five"]
        if residual["half"] <= 0:
            raise ValueError("invalid quarter-step residual normalization")
        residual_n64.append(core64)
        residual_n128.append(core128)
        residual_ratio.append(residual["difference"] / residual["half"])
        quarter_velocity.append(
            gap["velocity_field_sensitivity"]["relative_l2_difference"]
        )

    spatial_times = [item["time"] for item in checkpoints]
    velocity = [
        item["comparison"]["frames"][0]["velocity"]["relative_l2_difference"]
        for item in checkpoints
    ]
    force = [
        item["comparison"]["frames"][0]["force"]["relative_l2_difference"]
        for item in checkpoints
    ]
    quarter_spatial = [
        item["comparison"]["frames"][0]["velocity"]["relative_l2_difference"] / 4
        for item in candidate_rows
    ]
    result = {
        "schema_version": 1,
        "kind": "refinement-qualification-summary-figure",
        "source_sha256": {path.name: sha(path) for path in sources},
        "spatial": {
            "times": spatial_times,
            "velocity_relative_l2": velocity,
            "force_relative_l2": force,
            "velocity_limit": limits["velocity_relative_l2_max"],
            "force_limit": limits["force_relative_l2_max"],
        },
        "candidate": {
            "times": candidate_times,
            "interval": [candidate_times[0], candidate_times[-1]],
            "from_rest_half_step_velocity_relative_l2": [
                frame["relative_l2_difference"] for frame in temporal_frames
            ],
            "local_half_to_quarter_velocity_relative_l2": quarter_velocity,
            "quarter_spatial_velocity_gap": quarter_spatial,
            "finest_core_balance_rms_n64": residual_n64,
            "finest_core_balance_rms_n128": residual_n128,
            "half_to_quarter_balance_gap_over_quarter_residual": residual_ratio,
        },
        "conclusion": {
            "spatial_screen_candidate": True,
            "velocity_timestep_stable": True,
            "residual_spatially_converged": False,
            "residual_timestep_stable": False,
            "locked_protocol_fully_qualified": False,
            "reason": "The three spatial-screen passes and independent velocity timestep control support a candidate interval. Mixed 64-to-128 residual behavior, material residual timestep sensitivity, peaks outside the finest core, and the incomplete 64-base parent trajectory prevent a resolution-qualified precursor claim.",
        },
        "scope": "Audited finite-surrogate comparisons. Thresholds are presentation gates, not continuum error bounds or singularity criteria.",
    }
    return result


def render(report: dict, output: Path) -> None:
    spatial, candidate = report["spatial"], report["candidate"]
    candidate_start, candidate_end = candidate["interval"]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "text.color": "#e5eef4",
            "axes.labelcolor": "#bacbd7",
            "xtick.color": "#bacbd7",
            "ytick.color": "#bacbd7",
            "axes.edgecolor": "#40546a",
            "axes.facecolor": "#07111f",
            "figure.facecolor": "#07111f",
            "savefig.facecolor": "#07111f",
            "svg.hashsalt": "refinement-summary-v1",
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    cyan, amber, magenta, green = "#69d2e7", "#ffb35c", "#ef7fc8", "#8cdda3"

    ax = axes[0, 0]
    times = spatial["times"]
    ax.plot(times, [100 * x for x in spatial["velocity_relative_l2"]], "o-", color=cyan, label="Velocity")
    ax.plot(times, [100 * x for x in spatial["force_relative_l2"]], "s-", color=amber, label="Force")
    ax.axhline(100 * spatial["velocity_limit"], color=cyan, linestyle="--", linewidth=1, label="Velocity gate")
    ax.axhline(100 * spatial["force_limit"], color=amber, linestyle="--", linewidth=1, label="Force gate")
    ax.axvspan(candidate_start, candidate_end, color=green, alpha=0.13)
    ax.set_title("Spatial velocity and forcing", loc="left", fontsize=15, pad=12)
    ax.set_ylabel("Relative L² difference (%)")
    ax.legend(fontsize=9, frameon=False, loc="upper left")

    ax = axes[0, 1]
    ct = candidate["times"]
    ax.plot(ct, [math.log10(100 * x) for x in candidate["from_rest_half_step_velocity_relative_l2"]], "o-", color=cyan, label="From-rest half step")
    ax.plot(ct, [math.log10(100 * x) for x in candidate["local_half_to_quarter_velocity_relative_l2"]], "s-", color=green, label="Half → quarter step")
    ax.plot(ct, [math.log10(100 * x) for x in candidate["quarter_spatial_velocity_gap"]], "--", color=amber, label="¼ spatial gap")
    ax.set_title("Velocity timestep checks", loc="left", fontsize=15, pad=12)
    ax.set_ylabel("Relative L² difference (%) · log scale")
    ax.set_ylim(math.log10(0.0025), math.log10(0.7))
    ax.set_yticks([math.log10(x) for x in (0.005, 0.02, 0.1, 0.5)])
    ax.set_yticklabels(["0.005", "0.02", "0.1", "0.5"])
    ax.legend(fontsize=9, frameon=False, loc="upper center")

    ax = axes[1, 0]
    ax.plot(ct, [math.log10(x) for x in candidate["finest_core_balance_rms_n64"]], "o-", color=cyan, label="64³ base")
    ax.plot(ct, [math.log10(x) for x in candidate["finest_core_balance_rms_n128"]], "s-", color=magenta, label="128³ base")
    ax.set_title("Finest-core balance RMS", loc="left", fontsize=15, pad=12)
    ax.set_ylabel("Vorticity-balance RMS · log scale")
    ax.set_ylim(math.log10(0.0025), math.log10(0.016))
    ax.set_yticks([math.log10(x) for x in (0.003, 0.005, 0.01, 0.015)])
    ax.set_yticklabels(["0.003", "0.005", "0.01", "0.015"])
    ax.legend(fontsize=9, frameon=False, loc="upper left")

    ax = axes[1, 1]
    ratios = [100 * x for x in candidate["half_to_quarter_balance_gap_over_quarter_residual"]]
    ax.plot(ct, ratios, "o-", color=magenta)
    for x, y in zip(ct, ratios, strict=True):
        ax.annotate(f"{y:.0f}%", (x, y), xytext=(0, 8), textcoords="offset points", ha="center", color="#e5eef4")
    ax.set_title("Residual timestep sensitivity", loc="left", fontsize=15, pad=12)
    ax.set_ylabel("Gap / quarter residual (%)")
    ax.set_ylim(0, max(ratios) * 1.3)

    for ax in axes.flat:
        ax.set_xlabel("Computed time, t")
        ax.grid(color="#40546a", alpha=0.32, linewidth=0.7)
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.87, bottom=0.17, hspace=0.43, wspace=0.26)
    fig.suptitle(
        "Where the refinement evidence holds — and where it stops",
        fontsize=20,
        y=0.96,
    )
    fig.text(
        0.02,
        0.065,
        "Green band: three consecutive spatial-screen passes. Velocity timestep checks pass.",
        color="#aabfce",
        fontsize=11,
    )
    fig.text(
        0.02,
        0.035,
        "Mixed spatial residuals and 38–48% residual sensitivity prevent a qualified precursor claim.",
        color="#aabfce",
        fontsize=11,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="svg", metadata={"Date": None})
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("site/data"))
    parser.add_argument("--output", type=Path, default=Path("site/media/refinement-summary.svg"))
    parser.add_argument("--manifest", type=Path, default=Path("site/data/refinement-summary-figure.json"))
    args = parser.parse_args()
    report = collect(args.data.resolve(strict=True))
    render(report, args.output)
    report["plotter_sha256"] = sha(Path(__file__))
    report["image_sha256"] = sha(args.output)
    write(args.manifest, report)
    print(json.dumps({"candidate_interval": report["candidate"]["interval"], "image": str(args.output)}))


if __name__ == "__main__":
    main()
