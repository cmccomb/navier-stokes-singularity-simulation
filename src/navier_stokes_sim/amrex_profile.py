"""Export the recorded finite surrogate, not the paper's infinite construction."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .config import SimulationConfig
from .profile import _paper_interpolants
from .time_stepping import forcing_log_rate_bound

PARAMETERS = (
    "h",
    "t_star",
    "base_radius",
    "base_height",
    "velocity_scale",
    "swirl_ratio",
    "half_domain",
    "paper_annulus_xa",
    "paper_annulus_xb",
    "paper_eta_support",
    "paper_eta_taper",
    "paper_axial_slope",
    "paper_axis_offset",
    "paper_swirl_bias",
    "paper_time_cutoff_start",
    "paper_time_cutoff_end",
    "localization_inner",
    "localization_outer",
    "pulses_enabled",
    "pulse_rtheta_strength",
    "pulse_rz_strength",
    "pulse_azimuthal_mode",
    "pulse_radial_frequency",
    "pulse_axial_frequency",
    "pulse_log_frequency",
    "pulse_time_width",
    "pulse_hierarchy_levels",
    "pulse_mode_stride",
    "pulse_scale_ratio",
    "pulse_correction_strength",
    "pulse_correction_passes",
    "derivative_epsilon",
    "forcing_phase_step",
    "viscosity",
)


def export_profile(cfg: SimulationConfig, output: Path) -> dict:
    cfg.validate()
    if (
        cfg.profile_model != "paper-surrogate"
        or cfg.profile_interpolation != "cubic"
        or cfg.paper_profile_revision
        not in {"appendix-b-axis-v1", "appendix-b-axis-v2-localized"}
    ):
        raise ValueError(
            "this adapter requires the recorded cubic Appendix-B finite surrogate"
        )
    if cfg.forcing_phase_step is None or cfg.forcing_end is not None:
        raise ValueError("export requires a phase ceiling and uninterrupted forcing")
    if cfg.pulse_correction_passes > 4:
        raise ValueError("at most four discrete correction passes are supported")
    params = {name: float(getattr(cfg, name)) for name in PARAMETERS}
    params["forcing_log_rate_bound"] = forcing_log_rate_bound(cfg)
    params["fully_localized"] = float(
        cfg.paper_profile_revision == "appendix-b-axis-v2-localized"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation keeps previously exported physics immutable.
    with output.open("x") as stream:
        stream.write(f"NS_PAPER_TABLES_V1\n{len(params)}\n")
        for name, value in params.items():
            stream.write(f"{name} {value:.17g}\n")
        tables = _paper_interpolants(cfg)
        stream.write(f"{len(tables)}\n")
        for name, spline in tables.items():
            stream.write(f"{name} {len(spline.x)}\n")
            np.savetxt(stream, spline.x, fmt="%.17g")
            np.savetxt(stream, spline.c.T, fmt="%.17g")
    manifest = {
        "schema_version": 1,
        "kind": "finite-paper-surrogate-cubic-tables",
        "config": asdict(cfg),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "profile_source_sha256": hashlib.sha256(
            (Path(__file__).parent / "profile.py").read_bytes()
        ).hexdigest(),
        "force_definition": "centered time derivative plus centered velocity advection minus seven-point viscosity, at each level's own spacing",
        "parameters": params,
        "tables": {name: len(spline.x) for name, spline in tables.items()},
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, help="a config JSON or a run record containing config"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--localized",
        action="store_true",
        help="explicitly select the corrected v2 core cutoff",
    )
    args = parser.parse_args()
    values = json.loads(args.config.read_text()) if args.config else {}
    cfg = SimulationConfig(**values.get("config", values))
    if args.localized:
        cfg = replace(cfg, paper_profile_revision="appendix-b-axis-v2-localized")
    print(json.dumps(export_profile(cfg, args.output), indent=2))


if __name__ == "__main__":
    main()
