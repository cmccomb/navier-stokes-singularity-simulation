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
oscillatory pulse hierarchy. The default build is serial CPU, double precision,
and constant unit density; opt-in OpenMP shares work within one Mac. The fleet
runs independent checks, not distributed
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
python -m navier_stokes_sim.amrex_profile --config backends/amrex/paper-reference.json --output outputs/profile.tbl
python -m scripts.profile_bridge --table outputs/profile.tbl --output outputs/profile-check.json
python -m scripts.force_bridge --table outputs/profile.tbl --output outputs/force-check
python -m scripts.fine_force_bridge --table outputs/profile.tbl --output outputs/fine-force-check
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

The fine-force cross-check compares 18 bounded patches, each 12³ cells, at
1024³- and 16,384³-equivalent spacing through `t=0.99983872`. Padded Python
arrays preserve the original full-domain sample coordinates; only interior
cells, clear of artificial periodic patch edges, enter the comparison. Both
velocity and force agree within the recorded gates. This is a local stencil
agreement test, not a simulation at that effective full-domain resolution.

For matched-grid temporal sensitivity, `ns_archive_check` accepts
`compare=other_plot report_difference=1`. It retains instantaneous-force
verification but reports the velocity difference instead of enforcing the
restart-equality gate. Composite L² excludes covered coarse cells; L∞ includes
all stored cells. Domain, variable names, refinement regions, and output times
must match (within 1e-12 for the report mode). A finer reference can be volume
averaged to the comparison grid at an aligned integer refinement ratio;
uncovered/mismatched regions are rejected. The default still enforces restart
equality without changing meshes. `scripts.archive_restriction_check` verifies
the averaging against an analytic initial Taylor field, not an evolved result.
At `t=0.625`, the first reference/finer-time pair differs by 0.0414% relative
composite L². This is not a late-time bound or a measured convergence order.

The longer pinned probes exposed roundoff-scale duplicate output events.
The next-launch clipper uses the same 1e-12 acceptance tolerance as upstream
`writeNow`, and never enlarges an existing CFL-limited step. A 0.00025-ceiling
rerun has all 26 scheduled frames without duplicate events; a final tiny
endpoint cleanup step remains. The running pinned probes were not patched.

## Bounded OpenMP

Configure with `-DNS_ENABLE_OPENMP=ON` to enable AMReX/Hydro CPU threading.
On macOS, the verified build uses an existing Homebrew LLVM/OpenMP toolchain:

```sh
cmake -S backends/amrex -B build/amrex-omp -DCMAKE_BUILD_TYPE=Release \
  -DNS_ENABLE_OPENMP=ON \
  -DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm/bin/clang \
  -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++
cmake --build build/amrex-omp --parallel 2
python -m scripts.threading_regression --build build/amrex-omp \
  --table outputs/profile.tbl --output outputs/threading-check --end 0.62
python -m scripts.paper_run --executable build/amrex-omp/ns_incflo \
  --checker build/amrex-omp/ns_archive_check --threads 2 --force-threads 2 \
  --table outputs/profile.tbl --output outputs/threaded-paper-check \
  --base-n 16 --widths 0.5 --end 0.62 --max-dt 0.00025
```

The default remains one thread. The runner explicitly caps `OMP_NUM_THREADS`,
`OMP_THREAD_LIMIT`, nested teams, and dynamic sizing, and records those limits.
`--force-threads` separately bounds per-box source scratch memory. Cached force
lookup/insertion and diagnostic summation stay outside parallel regions.
The same binary agrees within 1.4e-16 for one/two threads and 1.7e-16 for
one/four threads across all 26 from-rest native frames through 0.62. Late-time
per-box source comparisons are exactly equal. See the
[threading record](../../site/data/threading-validation.json). These checks do
not measure a speedup or establish late-time accuracy. Verify the destination's
OpenMP runtime before deploying a dynamically linked binary. MPI stays off.

## Native comparisons, phase-spaced output, and run limits

`scripts.paper_comparison` checks pinned source/input/profile identities, the
complete available diagnostic prefix from rest, all requested native fields,
and controlled spatial or temporal parameter changes. It does not rewrite a
run's original status. Legacy duplicate output events are listed and compared;
missing requested events still fail. The 32³/64³-base pair differs by 6.12%
relative composite velocity L² at 0.65, versus 0.0439% for halved time limits.
The temporal difference at 0.85 is 0.0672%. Two runs do not establish order.

```sh
python -m scripts.paper_comparison --coarse outputs/reference \
  --reference outputs/time-fine --kind temporal --output outputs/time-comparison
python -m scripts.paper_comparison --coarse outputs/reference \
  --reference outputs/space-fine --kind spatial --times 0 0.625 0.65 \
  --output outputs/space-comparison
```

`--frame-phase-step` optionally predefines increasingly dense native output
times using the exported forcing-rate bound. `--frame-dt` remains the maximum
physical-time output gap. These output parameters are distinct from numerical
`--max-dt` and the table's forcing-phase ceiling. Events can only shorten the
integration step; all expected events, including rest and the endpoint, are
retained in `run.json` and checked against native archives. The default zero
frame-phase step preserves uniform output scheduling.

`--max-rss-mib` and `--min-disk-free-gib` optionally stop the owned solver if a
resource threshold is crossed. They are polled every five seconds, not OS hard
limits. `--timeout` limits solver wall time. A stop records failure and leaves
existing native frames and checkpoints intact. The runner itself is copied
and hashed along with the solver inputs; per-process loader overrides are
recorded. No fleet machine's global library configuration is changed.

The current large candidate is 128³ base with half-widths
`0.5 0.25 0.125 0.0625`, from rest to 0.995. At output phase 0.75 and maximum gap
0.025 it needs 280 full-native frames, about 131.25 GiB before checkpoints.
Deployment requires a full-size activation/memory check first. Local
`scripts.force_resolution_scan` sensors show appreciable source discretization
sensitivity even when velocity sensitivity is below 1%; they are overlapping
local samples, not a global norm or production accuracy certificate. See the
[first-class refinement documentation](https://cmccomb.com/navier-stokes-singularity-simulation/refinement.html#next-refined-run)
and [machine-readable evidence](../../site/data/fleet-sensitivity.json).

`ns_slice_export` reads native AMReX plotfiles and samples three orthogonal
zero-plane views using the finest containing cell. Ties at a zero-plane face
use the positive-side native cell. No spatial smoothing is applied. An analytic
initial-field fixture with three refinement levels checks all orientations and
level selection independently of the rendered pictures. Disable initialization
projection only for this fixture so it remains exactly the known field; normal
scientific runs retain their existing projection settings.

```sh
python -m scripts.render_native_gifs --run outputs/validated-paper-run \
  --output outputs/native-gifs
```

The GIF renderer requires a validated complete event sequence, preserves every
saved state, fixes color scales across time and orientations, and records field
and GIF hashes. It produces velocity and body-force GIFs in XZ and XY views.
Display pixel count is not a solver resolution. The first full-size resource
preflight remains recorded as failed: an upstream tiny-step fallback enlarged
a roundoff-limited activation step and overshot the requested endpoint. The
controlled-run overlay now suppresses that fallback, recognizes activation
roundoff, and uses the archive contract's 2e-14 endpoint tolerance. A dedicated
0.55-spaced output regression exercises the trigger; full-size repetition is
required before the long run.
