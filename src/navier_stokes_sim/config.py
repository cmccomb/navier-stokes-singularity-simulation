"""Configuration for the proof-informed numerical experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isclose, isfinite

MESH_PRESETS: dict[str, dict[str, float]] = {
    "full-domain": {
        "half_domain": 1.0,
        "localization_inner": 0.72,
        "localization_outer": 0.94,
    },
    "core-refined": {
        "half_domain": 0.68,
        "localization_inner": 0.61,
        "localization_outer": 0.66,
    },
}


def mesh_preset_parameters(name: str) -> dict[str, float]:
    """Return an independent copy of a validated static-grid preset."""

    try:
        return dict(MESH_PRESETS[name])
    except KeyError as error:
        raise ValueError(f"unknown mesh preset {name!r}") from error


@dataclass(frozen=True)
class SimulationConfig:
    """Numerical and similarity-profile parameters.

    The singular time is normalized to ``t_star=1`` as in the paper.  The
    default exponent ``h=0.008`` satisfies the paper's ``0 < h < 1/100``.
    """

    resolution: int = 32
    mesh_preset: str = "full-domain"
    half_domain: float = 1.0
    viscosity: float = 0.01
    h: float = 0.008
    base_radius: float = 0.52
    base_height: float = 0.52
    velocity_scale: float = 0.25
    swirl_ratio: float = 0.85
    profile_model: str = "paper-surrogate"
    profile_interpolation: str = "cubic"
    paper_profile_revision: str = "appendix-b-axis-v1"
    paper_annulus_xa: float = 0.60
    paper_annulus_xb: float = 1.45
    paper_eta_support: float = 0.86
    paper_eta_taper: float = 0.98
    paper_axial_slope: float = 4.0
    paper_axis_offset: float = 0.03
    # Retained only so v0.6/v0.7 checkpoints with the old hand-shaped profile
    # remain loadable.  The Appendix-B profile does not read this parameter.
    paper_axial_bias: float = 0.12
    paper_swirl_bias: float = 0.08
    paper_axis_lambda: float = 2.0
    paper_axis_sigma: float = 0.25
    pulses_enabled: bool = True
    pulse_rtheta_strength: float = 0.075
    pulse_rz_strength: float = 0.060
    pulse_azimuthal_mode: int = 4
    pulse_radial_frequency: float = 4.0
    pulse_axial_frequency: float = 2.5
    pulse_log_frequency: float = 1.0
    pulse_time_width: float = 0.46
    pulse_hierarchy_levels: int = 3
    pulse_mode_stride: int = 1
    pulse_scale_ratio: float = 0.68
    pulse_correction_strength: float = 0.04
    pulse_correction_passes: int = 1
    localization_inner: float = 0.72
    localization_outer: float = 0.94
    paper_time_cutoff_start: float = 0.55
    paper_time_cutoff_end: float = 0.775
    ramp_time: float = 0.15
    t_start: float = 0.0
    t_end: float = 0.94
    t_star: float = 1.0
    frames: int = 25
    frame_spacing: str = "linear"
    cfl: float = 0.32
    max_dt: float = 0.02
    pressure_rel_tol: float = 1e-7
    pressure_abs_tol: float = 1e-9
    pressure_max_iterations: int = 2000
    min_cells_per_scale: float = 4.0
    integrator: str = "rk2"
    pressure_projection: str = "auto"
    derivative_epsilon: float = 2e-4
    forcing_phase_step: float | None = 0.15
    forcing_end: float | None = None
    capture_volumes: bool = True
    capture_force_volumes: bool = False
    volume_frames: int = 6
    stream_volumes: bool = False
    preview_phase_step: float | None = None
    preview_resolution: int = 32

    def validate(self) -> None:
        if self.profile_interpolation not in {"linear", "cubic"}:
            raise ValueError("profile_interpolation must be linear or cubic")
        if self.preview_phase_step is not None and (
            not isfinite(self.preview_phase_step)
            or not 0 < self.preview_phase_step <= 1
        ):
            raise ValueError("preview_phase_step must lie in (0, 1] or be None")
        if self.preview_resolution < 8:
            raise ValueError("preview_resolution must be at least 8")
        if self.forcing_phase_step is not None and (
            not isfinite(self.forcing_phase_step)
            or not 0 < self.forcing_phase_step <= 1
        ):
            raise ValueError("forcing_phase_step must lie in (0, 1] or be None")
        if self.resolution < 8:
            raise ValueError("resolution must be at least 8")
        if self.resolution % 2:
            raise ValueError(
                "resolution must be even so the singular axis lies between cells"
            )
        if self.mesh_preset not in MESH_PRESETS:
            raise ValueError(f"mesh_preset must be one of {sorted(MESH_PRESETS)}")
        expected_mesh = MESH_PRESETS[self.mesh_preset]
        for field_name, expected in expected_mesh.items():
            actual = float(getattr(self, field_name))
            if not isclose(actual, expected, rel_tol=0, abs_tol=1e-12):
                raise ValueError(
                    f"{self.mesh_preset!r} mesh requires {field_name}={expected:g}; "
                    f"got {actual:g}"
                )
        if not (0 < self.h < 0.01):
            raise ValueError("h must satisfy the paper's 0 < h < 1/100")
        if not (0 <= self.t_start < self.t_end < self.t_star):
            raise ValueError("require 0 <= t_start < t_end < t_star")
        if self.frames < 2:
            raise ValueError("frames must be at least 2")
        if self.volume_frames < 2:
            raise ValueError("volume_frames must be at least 2")
        if self.frame_spacing not in {"linear", "similarity"}:
            raise ValueError("frame_spacing must be 'linear' or 'similarity'")
        if self.integrator not in {"euler", "rk2"}:
            raise ValueError("integrator must be 'euler' or 'rk2'")
        if self.pressure_projection not in {"auto", "sparse", "matrix-free", "fft"}:
            raise ValueError(
                "pressure_projection must be 'auto', 'sparse', 'matrix-free', or 'fft'"
            )
        if self.profile_model not in {"separable", "paper-surrogate"}:
            raise ValueError("profile_model must be 'separable' or 'paper-surrogate'")
        if self.paper_profile_revision not in {
            "legacy-hand-shaped",
            "appendix-b-axis-v1",
            "appendix-b-axis-v2-localized",
        }:
            raise ValueError(
                "paper_profile_revision must be 'legacy-hand-shaped', "
                "'appendix-b-axis-v1', or 'appendix-b-axis-v2-localized'"
            )
        if not (0 < self.paper_annulus_xa < self.paper_annulus_xb):
            raise ValueError("paper annulus must satisfy 0 < Xa < Xb")
        if not (0 < self.paper_eta_support < 1):
            raise ValueError("paper_eta_support must lie in (0, 1)")
        if not (self.paper_eta_support < self.paper_eta_taper < 1):
            raise ValueError("paper_eta_taper must lie between paper_eta_support and 1")
        if not (0 < self.paper_axial_slope <= 8):
            raise ValueError("paper_axial_slope must lie in (0, 8]")
        if not (0 < self.paper_axis_offset <= 0.05):
            raise ValueError("paper_axis_offset must lie in (0, 0.05]")
        if self.pulse_azimuthal_mode < 1:
            raise ValueError("pulse_azimuthal_mode must be positive")
        if self.pulse_hierarchy_levels < 1:
            raise ValueError("pulse_hierarchy_levels must be positive")
        if self.pulse_mode_stride < 1:
            raise ValueError("pulse_mode_stride must be positive")
        if not (0 < self.pulse_scale_ratio <= 1):
            raise ValueError("pulse_scale_ratio must lie in (0, 1]")
        if not (0 <= self.pulse_correction_strength <= 0.25):
            raise ValueError("pulse_correction_strength must lie in [0, 0.25]")
        if not (0 <= self.pulse_correction_passes <= 4):
            raise ValueError("pulse_correction_passes must lie in [0, 4]")
        if not (0 < self.pulse_time_width < 0.5):
            raise ValueError("pulse_time_width must lie in (0, 0.5)")
        if min(self.pulse_rtheta_strength, self.pulse_rz_strength) < 0:
            raise ValueError("pulse strengths cannot be negative")
        if not (
            0 < self.localization_inner < self.localization_outer < self.half_domain
        ):
            raise ValueError(
                "localization radii must satisfy 0 < inner < outer < half_domain"
            )
        if not (
            0 <= self.paper_time_cutoff_start < self.paper_time_cutoff_end < self.t_star
        ):
            raise ValueError(
                "paper temporal cutoff must satisfy 0 <= start < end < t_star"
            )
        if self.forcing_end is not None and not (
            self.t_start < self.forcing_end < self.t_end
        ):
            raise ValueError(
                "forcing_end must lie strictly inside the simulated interval"
            )
        if (
            min(
                self.half_domain,
                self.viscosity,
                self.base_radius,
                self.base_height,
                self.velocity_scale,
                self.cfl,
                self.max_dt,
                self.derivative_epsilon,
                self.pulse_radial_frequency,
                self.pulse_axial_frequency,
                self.paper_axis_lambda,
                self.paper_axis_sigma,
            )
            <= 0
        ):
            raise ValueError("positive physical and time-step parameters are required")
        if self.base_radius >= self.half_domain or self.base_height >= self.half_domain:
            raise ValueError("the compact profile must fit inside the periodic domain")

    @property
    def dx(self) -> float:
        return 2 * self.half_domain / self.resolution

    @property
    def similarity_fit_start(self) -> float:
        """First time at which the selected profile is fully activated."""

        if self.profile_model == "paper-surrogate":
            return self.paper_time_cutoff_end
        return self.ramp_time

    def to_dict(self) -> dict[str, float | int | str | bool | None]:
        return asdict(self)
