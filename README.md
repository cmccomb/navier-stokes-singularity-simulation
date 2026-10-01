# Navier–Stokes Singularity Simulation

[![CI](https://github.com/cmccomb/navier-stokes-singularity-simulation/actions/workflows/ci.yml/badge.svg)](https://github.com/cmccomb/navier-stokes-singularity-simulation/actions/workflows/ci.yml)
[![Results site](https://github.com/cmccomb/navier-stokes-singularity-simulation/actions/workflows/pages.yml/badge.svg)](https://cmccomb.com/navier-stokes-singularity-simulation/)
[![License: MIT](https://img.shields.io/badge/License-MIT-69d2e7.svg)](LICENSE)

An open, reproducible 3D incompressible-flow experiment informed by OpenAI's
[*Finite Time Blowup for Navier–Stokes*](https://cdn.openai.com/pdf/32d9f210-8b73-45e0-91bc-82a30aef8a9a/navier-stokes.pdf).
The featured completed run uses [AMReX / incflo](backends/amrex/README.md)
with fixed nested grids and multilevel projection. The earlier PhiFlow 3.4
backend, projected RK2 and periodic FFT projection remain available.

The experiment starts from rest, smoothly activates a compactly supported
manufactured force, and follows an inward-spiraling, axially stretching vortex
toward the normalized singular time `t*=1`.

> [!IMPORTANT]
> This is a finite surrogate, not an independent proof and not yet the paper's
> exact infinite pulse hierarchy or all-order correction cycle. Numerical instability, overflow, or a
> grid-dependent peak is not evidence of blow-up.

## Current completed simulation

The [results site](https://cmccomb.com/navier-stokes-singularity-simulation/)
shows **Oliver's completed outer-refinement run: all 280 audited states from
exact rest through t = 0.995**. The velocity and forcing movies, numerical
record, and interactive mesh use this same September 23 completion.

| Run configuration | Value |
|---|---:|
| Grid / domain | `128^3` base, 5 fixed levels plus 27 outer bands, full `[-1,1]^3` box |
| Stored / active cells | `12,055,040` / `10,810,304` |
| Saved time interval | `0` to `0.995`, all 280 states |
| Exact rest interval | `0` to `0.55` |
| Endpoint peak speed | `7.415301682` model units |
| Reported endpoint RMS deviation | `0.001613819` |
| Profile | `appendix-b-axis-v2-localized`, finite three-pulse surrogate |
| Maximum timestep / phase advance | `0.00025` / `0.0375` radians |
| Force-difference half-window | `2e-7` |

The [completion record](site/data/best.json) binds each saved state to the
current GIFs and MP4s. The fixed x–y and x–z planes and isometric surfaces use
one saved-time clock, fixed scales, and no temporal interpolation. Mesh views
show the actual active cell edges, including the irregular outer bands.
2048³-equivalent spacing applies **only to the innermost core**.

The native float64 archive remains on Oliver; older published movies have
been removed from the site. The [field documentation](site/voxels.html)
describes the current archive and its readback checks.

Archive validation establishes completion and data integrity. It does not
certify spatial convergence or singularity formation. The discrete forcing
and diagnostic quadrature depend on the mesh.
[Run details and limitations](site/results.html) ·
[Backend validation](site/refinement.html).

The fixed-cube 32³/64³/128³ runs provide a controlled spatial-sensitivity
sequence. All three completed from rest through `t=0.995` with the same binary,
force profile, physical refinement cubes, and output clock. At the endpoint,
the relative velocity L² gaps decrease from **10.29%** for 32³→64³ to
**4.54%** for 64³→128³; forcing-field gaps decrease from **26.87%** to
**18.19%**. The finest-pair active-cell maximum velocity-component difference
is **0.521 model units**. See the [32³→64³](site/data/three-grid-n32-n64-20260927.json)
and [64³→128³](site/data/paired-refinement-20260927.json) native comparisons.
Nearly identical peak speeds do not establish field convergence, and the
manufactured force itself changes with grid spacing.

An exploratory [earlier active-interval search](site/results.html#paired-refinement)
compares eight more matched states from `t≈0.566` to `0.700`. The finest-pair
velocity gap remains 2.19–2.47%; forcing is below 5% only through `t≈0.629`.
No sampled active state meets both working spatial targets. This selection was
made after the late-time comparison and does not define a prospective pass.
The [32³→64³](site/data/onset-n32-n64-20260927.json) and
[64³→128³](site/data/onset-n64-n128-20260927.json) audit records are retained.

The outer-band 64³ companion has supplied two finalized active frames while
its full run continues. Conservative comparison with the completed outer-band
128³ run gives velocity gaps of 2.24% and 2.13% at `t≈0.566` and `0.588`;
forcing gaps are 2.34% and 2.61%. The
[native prefix record](site/data/outer-band-prefix-20260927.json) verifies
complete active-volume coverage across different AMR footprints. Neither
velocity gap passes the 2% working target. These selected frames are checked,
but the 64³ trajectory is not yet complete or fully audited.
The [prospective outer-band checkpoint protocol](site/data/outer-band-protocol-20260927.json)
locks thirteen subsequent scheduled states and the spatial screen before
those states are produced.
Its first eight [audited comparisons](site/data/outer-band-prospective-prefix-20260927.json)
show two velocity misses followed by three consecutive passes at
`t=0.615808–0.629071` under the 2% velocity / 5% forcing working screen.
This is a provisional spatial candidate at sampled times. The 64³ full-run
audit, refined fixed-force timestep control, and diagnostic gates remain
outstanding. The next three locked checkpoints through `t=0.699573` fail the
forcing screen, reaching 10.601% even as the velocity gap falls to 1.672%.
Five later checkpoints are pending.

The [outer-band candidate diagnostics](site/results.html#precursor-diagnostics)
show rising peak speed and sampled vorticity with shrinking broad half-peak
support, but all three global peaks lie on level 1, outside the finest core.
The 128³ sampled vorticity maximum is about 9–10% higher than the 64³ value.
The spatial-screen passes therefore do not yet support a localized singularity
precursor. The machine-readable [64³](site/data/outer-precursor-n64-candidate.json)
and [128³](site/data/outer-precursor-n128-candidate.json) records retain hashes,
energy, forcing, divergence, sampled volumes, and exact definitions.

A [five-frame pressure-free balance diagnostic](site/results.html#precursor-diagnostics)
covers all three candidate checkpoints on both outer-band meshes. It
samples the curl of the momentum equation only where every needed same-block
neighbor is active. Within the prescribed finest core, switching from a
three-frame to a five-frame time derivative changes the measured balance by
more than the five-frame residual at every checkpoint. This output clock does
not support a residual-convergence claim. A completed
[dense 64³ checkpoint continuation](site/data/outer-dense-n64-probe.json)
reduces the finest-core three-versus-five-frame derivative difference from
roughly 0.04 to 0.00025–0.00028. Its measured residual rises from 0.00293
to 0.01359 over the candidate centers and differs from the sparse estimate.
The [three dense balance records](site/results.html#precursor-diagnostics)
do not establish residual convergence. A completed
[local fixed-force half-step comparison](site/data/outer-dense-n64-local-time.json)
passes the candidate velocity gate at all three centers (0.0340%, 0.0224%,
0.00835% relative L²). An audited
[quarter-step continuation](site/data/outer-dense-n64-quarter-probe.json) reduces
the half-to-quarter velocity gaps to 0.00519%, 0.00342%, and 0.00905%. Its
fieldwise balance gaps are 0.00252, 0.00521, and 0.00372, smaller than the
baseline-to-half gaps at all three centers but still 38–48% of the quarter-step
residual norms. A separate [from-rest half-step run](site/data/outer-from-rest-n64-temporal.json)
gives 0.0192%, 0.0166%, and 0.0152% velocity gaps with exact force readback.

The matched [dense 128³ continuation](site/data/outer-dense-n128-probe.json)
also completed. Its finest-core residuals are 0.00677, 0.00478, and 0.01121;
the finer value is lower at the latter two centers but more than twice the 64³
value at the first. The dense derivative stencil is stable, but the spatial
residual sequence is mixed. These results support local velocity timestep
stability through the three candidate times; they do not establish residual
convergence or a resolution-qualified singularity precursor.

An [audited native diagnostic trace](site/results.html#precursor-diagnostics)
at the same six times shows increasing peak speed and sampled vorticity and a
shrinking half-peak support volume. At `t=0.995`, the vorticity maximum changes
by 16.2% between 64³ and 128³, and the local finest-level half-peak radius
does not contract from the preceding checkpoint. Centered divergence sampled
away from AMR interfaces is reported separately from the target deviation;
neither is a full momentum residual. The machine-readable [64³](site/data/precursor-n64-fixed.json)
and [128³](site/data/precursor-n128-fixed.json) records retain all six
checkpoint measurements, hashes, and exact diagnostic definitions.

The working completion goal is a **resolution-qualified precursor movie**:
identify the latest interval where matched full-field, source and timestep
refinements show decreasing differences; report peak growth, core contraction,
vorticity, energy and numerical residuals there; and label any extension toward
`t*=1` as a model extrapolation. A practical visualization gate is at most 2%
finest-pair velocity L² difference and 5% forcing L² difference at predefined
matched checkpoints, with the halved-timestep difference below one-quarter of
the spatial gap and peak/core/vorticity trends stable under further refinement.
These thresholds set presentation fidelity, not a mathematical error bound.
A completed [fixed-force timestep control](site/data/temporal-fixed-force-late-control.json)
on a 16³-base/two-level mesh reached `t=0.995` with 5,146 versus 10,269
integration steps. At six matched checkpoints its velocity difference stayed
below 0.21% relative composite L², reaching 0.083% at the endpoint; saved
endpoint forcing fields were identical. This verifies the late-time comparison
method on a coarse mesh, not the timestep accuracy of the 64³/128³ runs.
The 32³-base control on Mali and the 96³-base intermediate run on Kay both
completed their native audits. The 64³-base outer-band companion reached about
`t=0.902` before its wall-clock guard stopped the process, leaving its native
prefix and checkpoint intact. The completed, audited 128³ outer-band model is
therefore the final production mesh. A bounded continuation from its `t=0.995`
checkpoint toward `t=0.9975` is an explicitly extrapolative final run, not a
new spatial-convergence certificate.

## Earlier PhiFlow publication workflow

With noninteractive GitHub authentication configured, the publisher detects
finalized phase-clock snapshots every 30 seconds and commits the latest
manifest plus full-history GIF/MP4 movies to `main`.
GitHub Pages adds build and cache latency. Both velocity and forcing movies
include **every finalized saved snapshot, from the actual zero-velocity start
through the latest saved time**, in order. There is no rolling window or frame
thinning in the movies. This means every saved output, not every internal solver
timestep; unsaved states cannot be recovered from the movies. Raw volumes stay
outside Git. Both interactive 3D explorers cover the same complete history,
recorded in `clip_times_3d`. Each selected 32³, three-component float32 volume
loads on demand; a three-frame browser cache bounds memory without dropping
saved times. Magnitude and signed x/y/z views retain fixed scales and camera
position. Playback waits for each frame and does not interpolate simulation time.

Movies play at **5 saved snapshots per second** (200 ms per state), with a
1-second initial-rest hold and a 0.5-second final hold. A single-state movie
holds for 4 seconds. Duration grows with the saved history; there is no fixed
12-second cap or late-window speed change. Capture uses nonuniform simulation
times, so this is explicitly labeled saved-frame playback, not uniform physical
time. No fluid states are interpolated. GIF and MP4 use the same holds; MP4
repeats states at 50 fps. Exports remain 1440×792 pixels.

The exporter checks contiguous frame indices and the actual initial rest state.
Before publication, it decodes each GIF to verify that its frame count equals
`captured_frames` and that every frame delay matches the manifest. GIF hashes
are recorded in `render`, and publication rejects incomplete histories. Existing
movie playback position is retained when a new full-history revision arrives.

### Saving the full field

Use `--preview-resolution 192` with `--resolution 192` and
`--preview-phase-step 0.3` to save every cell of both three-component fields
at the existing phase-clock events. The historical directory name is
`preview-volumes/`, but these files contain native 192³ data in float32.
This setting changes only the stored copies: solver arithmetic, restart
checkpoints, and the sparse `full-volumes/` snapshots remain float64.
Use a fresh output directory when changing capture settings; configuration
checks deliberately prevent mixing different capture histories.

For the current interval, 487 dense snapshots require about **82.7 GB
(77.0 GiB)** before compression: `487 × 192³ × 6 × 4` bytes. Budget another
4.1 GB for 12 sparse float64 snapshots, checkpoint/temporary-file space,
and a disk reserve. Compression can reduce usage, but capacity planning
should not depend on it. Keep one full-resolution production run active
at a time and retain its archive outside Git. Extract native 192² planes
and reduce browser 3D samples only after the full snapshot is finalized.

To refresh compact media once on the archive's machine, run
`python -m scripts.stream_run --host local --once --no-push --run <run-directory>
--repo <checkout> --cache <receipt-directory> --deadline <ISO-time-with-timezone>`.
The exporter reads one native field at a time and caches derived native 2D
planes and 32³ browser copies outside Git. The complete 2D history is read
lazily from this cache, and GIF frames are encoded individually instead of
buffering a growing image history in RAM. Subsequent updates reuse the compact
cache; the native archive is never rewritten. Browser 3D exports every cached
32³ vector field as a losslessly gzip-compressed, content-addressed chunk. The
HTML bundles Plotly and a small time index, not the entire volume history.
Serve the HTML and its `stream-volumes/` directory together over HTTPS (or
localhost); these explorers are no longer standalone single-file downloads.
Sandboxed embedding requires CORS access to the chunks, as GitHub Pages provides.
Publish the compact outputs from the authorized publishing checkout. Remote
publication rejects dense-archive transfers; raw histories are not rsynced to
another machine or committed to Git.

### Live native-field publication

`python -m scripts.live_native` owns one bounded publication loop on the
designated controller. Supply `--host`, `--run`, `--source-repo` (a dedicated
render checkout on the archive machine), `--repo` (a clean publishing checkout
on the controller), `--cache`, `--identity`, `--site-url`, and a timezone-aware
`--deadline`. Both checkouts need the project environment installed. Use the
controller's authenticated login session; an SSH session may not have access
to the same macOS Keychain credentials. Never copy credentials between hosts.

The controller polls finalized frame headers every 30 seconds, also watching
the diagnostic clock and completion status. On change, it invokes a serial
`--host local --once --no-push` render on the source, collects the seven fixed
website files plus explicitly indexed 32³ frame chunks, verifies run identity,
finite ordered times, complete 2D/3D coverage, and each chunk's SHA-256 and shape,
and commits/pushes them to `main`. It checks the public Pages manifest before
recording deployment as live. Raw fields stay on the source machine.

The source and controller reject missing, corrupt, nonfinite, or out-of-order
3D frames. Frame paths are restricted to content-addressed browser assets;
native volume paths cannot enter the transfer allowlist. Published chunks are
immutable and retained so an already-open explorer can keep loading its history
after a new update. Each explorer needs a current browser with WebGL, gzip
decompression, and Web Crypto. Failed loads pause playback and keep the last
successfully displayed frame labeled until the reader retries.

The cache contains an exclusive controller lock, atomic `status.json`,
`receipt.json`, and a failure record. Transient failures retry with bounded
backoff up to five minutes; they do not silently stop after five attempts.
The loop exits after the final complete/stopped record reaches Pages, or at
its explicit deadline. Render/build/transfer time adds to the polling cadence;
the website itself checks for published updates every minute. A run-specific
launcher and its deadline belong with the controller's runtime records, not
in the numerical archive or in a permanent service on the solver machine.


## Quick start

Requirements: Python 3.11–3.13 and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --no-editable --extra dev
uv run ns-blowup --resolution 20 --t-end 0.85 --frames 16 --output outputs/smoke
uv run pytest -q
```

Results are written under `outputs/`, which is intentionally excluded from Git.
See the [experiment runbook](https://cmccomb.com/navier-stokes-singularity-simulation/running.html) for start-from-rest, refinement,
control, validation, and media commands.

## Documentation

- [Documentation index](https://cmccomb.com/navier-stokes-singularity-simulation/documentation.html)
- [Model and numerical method](https://cmccomb.com/navier-stokes-singularity-simulation/method.html)
- [Profile derivation and mesh calculation](https://cmccomb.com/navier-stokes-singularity-simulation/derivation.html)
- [Experiment runbook and outputs](https://cmccomb.com/navier-stokes-singularity-simulation/running.html)
- [Finite-truncation reproduction target](https://cmccomb.com/navier-stokes-singularity-simulation/reproduction.html)
- [Results and validation record](https://cmccomb.com/navier-stokes-singularity-simulation/results.html)
- [Contribution guide](CONTRIBUTING.md)

## Contributing

Contributions are welcome in the mathematical model, discretization,
validation, visualization, and documentation. Start with
[CONTRIBUTING.md](CONTRIBUTING.md) and propose experiments with explicit
resolution, convergence, and falsification criteria.

Large checkpoints and raw fleet output stay outside Git. Compact featured media
and result metadata are published automatically through GitHub Pages.

## Primary sources

- OpenAI, [announcement](https://openai.com/index/navier-stokes-solution/),
  September 8, 2026.
- OpenAI, [*Finite Time Blowup for Navier–Stokes*](https://cdn.openai.com/pdf/32d9f210-8b73-45e0-91bc-82a30aef8a9a/navier-stokes.pdf).
- Holl and Thuerey, [PhiFlow](https://github.com/tum-pbs/PhiFlow), ICML 2024.
