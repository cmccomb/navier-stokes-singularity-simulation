---
title: Native Navier–Stokes Explorer
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.27.0
python_version: '3.12'
app_file: app.py
pinned: false
license: mit
short_description: Native velocity and forcing slices through saved time
---

# Native Navier–Stokes explorer

Move x, y, and z to inspect three perpendicular planes. Select velocity or
forcing, then use the saved-frame playbar, playback, or single-frame buttons.
The displayed time is the actual solver output time; it is not uniformly spaced.

The app reads a commit-pinned native dataset, preserves float64 values, and
selects the finest containing cell at each location on a fixed 256² display
plane. The time control shows the current saved frame and actual simulation
time. Playback targets five saved states per second. A three-frame startup
buffer and bounded read-ahead prepare frames while the browser displays the
previous frame; revisits reuse a 12-frame image cache. Static plot axes and
color scales are retained, so only image pixels are redrawn. Playback still
waits when data is late and never skips or invents saved states. This is an
interactive slice explorer, not a volume renderer or a solver restart archive.

On the Space, a bounded background download caches the selected archive on
ephemeral server disk when it fits (18 GiB raw limit, 5 GiB free-space reserve).
Until the complete snapshot is ready, slices are read remotely. A partial Zarr
download is never exposed as a valid zero-filled field. Browser clients receive
only the rendered planes, not the voxel history.

[Results and GIFs](https://cmccomb.com/navier-stokes-singularity-simulation/)
remain available if the Space is sleeping or busy.
[Source code](https://github.com/cmccomb/navier-stokes-singularity-simulation/tree/main/spaces/native_explorer)
is maintained with the simulation repository.

This finite manufactured-force experiment is not the analytical proof's exact
infinite construction. Archive integrity does not certify convergence or a
singularity. No arrows or automatic peak-height tracking are used.

## Run locally

Install `requirements.txt` in a separate Python 3.12 environment. Run `python
app.py` with the published `dataset.json`, or set `NS_DATASET_LOCAL` to an
exported dataset directory. Remote overrides are `NS_DATASET_REPO`,
`NS_DATASET_REVISION` (a 40-character commit), and `NS_DATASET_PREFIX`.
No Hugging Face token is needed to view the public dataset.
