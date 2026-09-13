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

## Current best

The [results site](https://cmccomb.com/navier-stokes-singularity-simulation/)
features **Oliver's completed AMReX run: 280 saved states from exact rest through t = 0.995**.
Its 64³ base and four fixed refined levels give 1024³-equivalent spacing
**only in the core**, not throughout the domain. The
[completion record](site/data/best.json) preserves native checks and media
provenance. Archive validation does not establish convergence or a singularity.

| Completed run configuration | Value |
|---|---:|
| Grid / domain | `64^3` base, 5 fixed levels, full `[-1,1]^3` box |
| Stored / active cells | `1,310,720` / `1,179,648` |
| Saved time interval | `0` to `0.995`, all 280 states |
| Exact rest interval | `0` to `0.55` |
| Profile | `appendix-b-axis-v2-localized`, finite three-pulse surrogate |
| Maximum timestep / phase advance | `0.00025` / `0.0375` radians |
| Force-difference half-window | `2e-7` |

Arrow-free GIFs and MP4s use the established site styling and fixed midplanes.
The homepage movies retain every saved state. [Run details and limitations](site/results.html#oliver-complete).

The [native-voxel explorer](https://huggingface.co/spaces/ccm/navier-stokes-singularity-simulation)
adds x/y/z slice controls, velocity/forcing selection, and saved-time playback.
Prototype subsets are explicitly labeled. The
[dataset documentation](https://cmccomb.com/navier-stokes-singularity-simulation/voxels.html)
covers the lossless float64 format, native-grid sampling, validation, and publication.

The earlier PhiFlow from-rest 32³ cubic-profile temporal pilot recovered a successive-difference
ratio of 3.989, consistent with second-order time convergence at that coarse
resolution. Spatial convergence is not established. This remains a best-guess
finite-surrogate experiment, not a singularity demonstration.

The [refined-backend documentation](https://cmccomb.com/navier-stokes-singularity-simulation/refinement.html)
separates operator validation, completed flow histories, memory measurements
and the remaining accuracy gates. The previous uniform-grid movies and 3D
explorers remain in the [numerical record](site/results.html#legacy-media).

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

The separate `288^3` late-window calculation remains the resolution maximum and
is documented in the [full numerical record](https://cmccomb.com/navier-stokes-singularity-simulation/results.html).

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
