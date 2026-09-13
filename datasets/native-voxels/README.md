---
license: mit
language:
  - en
tags:
  - physics
  - computational-fluid-dynamics
  - navier-stokes
  - amrex
  - zarr
pretty_name: Native Navier–Stokes simulation voxels
---

# Native Navier–Stokes simulation voxels

Lossless exports of velocity and manufactured forcing from Oliver's completed
AMReX / incflo experiment. The source has **280 saved states from exact rest
through t = 0.995**, including an exactly quiescent interval through t = 0.55.
An explicitly labeled prototype may contain fewer states; each release's
`manifest.json` is authoritative about its coverage.

## Layout and access

Each content-addressed `releases/<manifest-hash-prefix>/` directory contains
`manifest.json`, `source-best.json`, and `fields.zarr`. The manifest lists actual
times, source plotfile names, component order, level geometry, and per-frame,
per-level SHA-256 hashes of uncompressed values. Publication never overwrites
another release. Pin both the dataset commit and release prefix when citing,
downloading, or analyzing the fields.

`fields.zarr/level_0` through `level_4` have axes `(t, x, y, z, component)`.
Every level occupies a 64 × 64 × 64 rectangular native patch. Components are
`u_x, u_y, u_z, f_x, f_y, f_z`, stored as little-endian float64 in nondimensional
model units. Origins are cell-boundary coordinates; cell centers are
`origin + (index + 0.5) * spacing`.

The full periodic domain is [−1,1]³. The base spacing is 2/64; four fixed inner
cubes have half-widths 0.5, 0.25, 0.125, and 0.0625, with spacing halved at each
level. Thus 1024³-equivalent spacing applies **only to the innermost core**.
There are 1,310,720 stored cells and 1,179,648 active composite cells per state.
Covered coarse cells are retained for fidelity to the native files. Exclude
them from composite integrals and select the finest containing cell for slices.

Zarr v3 uses lossless Blosc Zstd/bitshuffle compression, 8³-cell chunks split
into velocity and force triplets, and one shard per level per saved state.
The entire uncompressed field payload is 16.40625 GiB. No dense 1024³ regridding,
float32 conversion, spatial smoothing, or temporal interpolation is performed.
Pressure, ghost cells, and solver checkpoints are not included.

Use Zarr 3.3 and `huggingface_hub` / fsspec to open
`hf://datasets/ccm/navier-stokes-singularity-simulation@<commit>/releases/<hash>/fields.zarr`.
The [interactive Space](https://huggingface.co/spaces/ccm/navier-stokes-singularity-simulation)
provides a three-plane viewer without requiring a whole-history download.

## Validation and limitations

Each exported level is read back and compared exactly with its native float64
values. Checks also require finite fields, exact rest, source timestamps,
active-grid peak speed, full-domain volume, and endpoint kinetic energy.
Independent C++ slice comparisons test all three axes, and synthetic nested-grid
tests check axis order, boundary selection, and refinement precedence.

These are archive and visualization integrity checks, not accuracy bounds.
The experiment uses a finite, localized, three-pulse manufactured-force
surrogate inspired by the analytical construction. It does not implement that
construction's exact infinite hierarchy or certify a Navier–Stokes singularity.
The coarser outer grid can remain under-resolved; resampling cannot repair it.

## Provenance and reuse

Created by Chris McComb's independent numerical project, not affiliated with
OpenAI. Run settings, upstream solver revisions, archive hashes, and diagnostics
are retained in each release's `source-best.json`.

- [Simulation source](https://github.com/cmccomb/navier-stokes-singularity-simulation)
- [Numerical record](https://cmccomb.com/navier-stokes-singularity-simulation/results.html#oliver-complete)
- [Method and limitations](https://cmccomb.com/navier-stokes-singularity-simulation/method.html)

Released under the project's MIT license. Cite this dataset's repository,
immutable commit, release prefix, and source run ID so the exact fields remain
identifiable.
