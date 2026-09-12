# Fixed-refinement incompressible backend

This backend validates the pressure-projection operator needed for full-domain,
fixed local refinement. It uses the established AMReX-Hydro MAC projector and
AMReX multilevel multigrid, with double precision and immutable upstream source
pins in `CMakeLists.txt`. The coupled `incflo` integrator advances
incompressible Navier–Stokes with both verification sources and the project's
finite paper-surrogate force. The latter has cross-language field checks and
a small from-rest trajectory test, not yet a spatial-convergence certificate.

The [first-class documentation page](https://cmccomb.com/navier-stokes-singularity-simulation/refinement.html)
contains the mesh derivation, measured results, limitations, and next gates.
The [machine-readable record](../../site/data/refinement-pilot.json) retains all
nine cases, source/binary hashes, errors, timings, and peak process RSS.

## Build and reproduce

Requires Python 3.11+, a C++20 compiler, and network access for the three pinned
source archives. The pilot runner uses only Python's standard library. No
PhiFlow, MPI, GPU, system-wide installation, or fleet service is required.

```sh
python3 -m venv .venv-amrex
.venv-amrex/bin/python -m pip install -r backends/amrex/requirements-build.txt
.venv-amrex/bin/cmake -S backends/amrex -B build/amrex -DCMAKE_BUILD_TYPE=Release
.venv-amrex/bin/cmake --build build/amrex --parallel 2
.venv-amrex/bin/ctest --test-dir build/amrex --output-on-failure
.venv-amrex/bin/python -m scripts.refinement_pilot \
  --levels 4 --resolutions 32 64 128 --output outputs/projection-32-128
```

Use a fresh output directory. The full nine-case suite checks exact rest,
preservation of an analytic solenoidal field, uniform and nested-grid spatial
convergence, and box-decomposition invariance. Each process is capped at ten
minutes. Output and source identities are checked before computing convergence;
any missing case or failed gate produces exit code 2. Partial summaries and raw
process logs remain available after failures. The output directory must be empty
to avoid overwriting a previous experiment.

The smaller `--resolutions 16 32 64` suite is the default. `--levels` can be 2, 3,
or 4; each suite includes a one-level control. For one operator test:

```sh
build/amrex/ns_projection_pilot n_cell=128 levels=4 test_case=mixed
```

The largest recorded case has local 1024³-equivalent spacing only within
`[-0.125,0.125]³`, with a 128³ base grid across the full periodic `[-1,1]³`
domain. The 1.55 GiB measured peak RSS is for this operator pilot, including
two projections and diagnostics, **not** a full NS memory budget. It must not
be compared directly with a macOS physical-footprint measurement.

## Launch boundary

`ns_incflo` builds the established
[AMReX-Fluids/incflo](https://github.com/AMReX-Fluids/incflo) integrator at
`7491eea4d69bbfde0581a5fe7f95d803841a8a09` against the same AMReX libraries.
`incflo_overlay.py` generates a separate source tree with exact-count edits;
it leaves the downloaded upstream unchanged and refuses source drift.
`ns_case.H` supplies verification or finite-surrogate forces, fixed-region tagging,
and volume-weighted composite diagnostics. Covered coarse cells are excluded.

```sh
python -m scripts.coupled_pilot --output outputs/coupled-check
```

Fourteen evolution cases check exact rest, oscillatory time convergence for
MOL and Godunov, spatial convergence of a fully 3D manufactured shear flow on
three levels, the upstream translating/decaying Taylor vortex on uniform grids,
and decomposition invariance. The spatial force is continuous and independent
of grid spacing. Five native 3D frames are read back with `ns_archive_check`;
all velocity and forcing components must exist, forcing must equal the external
acceleration at each frame's time, and a mid-run checkpoint must reproduce the
uninterrupted final full velocity field. The runner pins a private executable
copy, checks the compiled adapter hash, preserves logs, and refuses nonempty
output directories.

The integration edits matter for an oscillatory case from rest:

- MOL predicts with `f(t)` and corrects with `[f(t)+f(t+dt)]/2. Godunov traces
  with the old-time force and updates with the midpoint force. Pressure retains
  the upstream treatment; it is not averaged with the external force.
- `ns.max_dt` is a ceiling in addition to the CFL and output/end-time limits.
  It avoids the upstream zero-CFL timestep-halving fallback during quiescent
  intervals. The unsafe `incflo.fixed_dt` override is refused with this ceiling.
- Plotfile forcing excludes the solver's pressure gradient. Native archives
  contain `velx`, `vely`, `velz`, `forcing_x`, `forcing_y`, and `forcing_z` on
  every stored level.

These smooth manufactured problems do not establish accuracy for the actual
oscillatory pulse hierarchy. The current build is serial CPU, double precision,
and constant unit density; the fleet runs independent checks, not distributed
subdomains of one trajectory. Set `NS_BUILD_INCFLO=OFF` for operator-only builds.

The archived fourteen-case suite identifies source commit `ec253643b31d`.
A subsequent 96³-base capacity check exposed accumulated roundoff in diagnostic
volume sums. The current source computes volume from integer active-cell counts
per level and includes a non-binary 96³-base/four-level exact-rest CTest. The
original failed check and corrected regression are both retained in the record;
the volume tolerance was not widened.

The companion `ns_diffusion_pilot` now advances three-component diffusion with
Crank–Nicolson on the fixed periodic hierarchy. Run
`python -m scripts.diffusion_pilot --output outputs/diffusion-check` after the
same build. Eleven cases verify exact rest, constants, decay, forcing-integral
balance, separate space/time convergence, decomposition invariance, and an
eight-level evolution smoke test. The temporal manufactured force is built from
the semidiscrete Laplacian independently of dt; the spatial test uses a separate
continuous analytic source. Neither applies the actual project's force.

The [10× design forecast](../../site/data/tenfold-design.json) anchors the leading
similarity law to the last 192³ peak. It is not a new run or a verified peak law.
The sampled 3D phase screen rejects all tested simple nested-cube candidates,
including an eight-level 16,384-equivalent core; coarse forcing regions remain
underresolved by that conservative criterion. The new
`navier_stokes_sim.refinement_design` module reproduces this screen. It does not
generate a production mesh or certify the full force, weak tails, or errors.

Before a refined from-rest trajectory can replace the existing 192³ result,
verify its actual-force convergence and the multilevel movie/export path.
The upstream coupled update, interpolation, synchronization, and native archive
have manufactured-case checks, not a project-force convergence certificate.
Audit the entire active
support and all relevant phase gradients, then separate spatial, temporal, and
force-difference convergence. The pilot's convenient nested cubes are not yet
an accepted forcing-informed production mesh.

CI builds the actual C++ backend and runs the four-level convergence suite when
backend-related files change. The Python unit tests alone do not validate the
numerical operator.

## Finite paper forcing, versioned and cross-checked

`amrex_profile` exports the existing SciPy cubic coefficients and physical
parameters. `paper_profile.H` evaluates those same potentials in C++;
`paper_fields.H` applies the original centered curl, finite correction passes,
and force stencil at each level's spacing. This preserves a **grid-dependent
finite surrogate**, not the proof's infinite pulse/correction construction.
Force is prescribed independently of the evolving numerical state:
`f_h = centered_dt(u_h) + u_h · centered_grad(u_h) - nu * laplace_h(u_h)`.

The port exposed missing fixed localization on the core swirl potential.
`appendix-b-axis-v2-localized` applies that cutoff to the core `A_z` term;
the already-localized exterior term is unchanged. The old v1 remains selectable
and remains the default for reproducibility. Existing movies are v1, not v2.
The new revision agrees exactly on the interior plateau and remains a discrete
curl. This is a versioned modeling change, not a relabeling of old results.

```sh
python -m navier_stokes_sim.amrex_profile --localized --output outputs/profile.tbl
python -m scripts.profile_bridge --table outputs/profile.tbl --output outputs/profile-check.json
python -m scripts.force_bridge --table outputs/profile.tbl --output outputs/force-check
python -m scripts.paper_run --table outputs/profile.tbl --output outputs/paper-smoke \
  --base-n 16 --widths 0.5 --end 0.62 --max-dt 0.0005 --frame-dt 0.025
```

Export and Python/PhiFlow comparisons require the project's scientific Python
dependencies. `paper_run` itself uses only the standard library and built C++
binaries. Each run pins its binary, input, table, manifest, and adapter sources;
it records every solver step and reads back all native velocity/force frames.
It refuses existing output directories and has a bounded runtime. `run.json`
starts as `running`; a `completed` status also requires clock and archive checks.

The force clock retains the exported log-phase ceiling in addition to CFL and
`ns.max_dt`. An optional `ns.epsilon_tau_ratio` caps the force derivative window
relative to `T-t`; zero preserves the original rule. Only an exactly zero CFL
before activation permits larger quiescent steps. Those steps stop at the next
output time and at activation, so no scheduled frame is skipped. Two cached
force times per level avoid repeated evaluations within MOL stages; this adds
six double-precision values per stored cell. Native forcing excludes pressure.

The first two-level v2 trajectory passes exact rest, all 26 scheduled frames
through `t=0.62`, and zero measured full-field force readback error. Its coarse
mesh does not establish physical accuracy. Separate fleet runs test spatial and
temporal sensitivity. See the [forcing-port record](../../site/data/paper-port.json)
and [documentation](https://cmccomb.com/navier-stokes-singularity-simulation/refinement.html#paper-force).
