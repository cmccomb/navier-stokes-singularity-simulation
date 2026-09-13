"""Read fixed native refinement levels, loading only requested native planes."""

import json
import re
import shutil
from functools import lru_cache
from pathlib import Path
from threading import Thread

import numpy as np
import zarr

PLANES = {"yz": (1, 2, 0), "xz": (0, 2, 1), "xy": (0, 1, 2)}


class NativeDataset:
    def __init__(self, folder=None, repo_id=None, revision=None, prefix=""):
        self.cache_status = (
            "Local archive" if folder is not None else "Remote native slices"
        )
        self._local_arrays = None
        self.remote_config = None if folder is not None else (repo_id, revision, prefix)
        storage_options = None
        if folder is not None:
            folder = Path(folder)
            self.manifest = json.loads((folder / "manifest.json").read_text())
            store = str(folder / "fields.zarr")
        else:
            if not repo_id or not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
                raise ValueError("Remote data requires a dataset ID and pinned commit")
            if prefix and not re.fullmatch(r"releases/[0-9a-f]{16}", prefix):
                raise ValueError("Invalid immutable release prefix")
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(
                repo_id,
                f"{prefix}/manifest.json" if prefix else "manifest.json",
                repo_type="dataset",
                revision=revision,
                token=False,
            )
            self.manifest = json.loads(Path(path).read_text())
            store = f"hf://datasets/{repo_id}@{revision}/{prefix + '/' if prefix else ''}fields.zarr"
            storage_options = {"token": False}
        m = self.manifest
        if (
            m.get("schema_version") != 1
            or m.get("kind") != "native-fixed-level-voxels"
            or not m.get("validated")
        ):
            raise ValueError("An audited native dataset is required")
        self.times = np.asarray(m["times"], dtype=float)
        if (
            not len(self.times)
            or self.times[0] != 0
            or not np.isfinite(self.times).all()
            or np.any(np.diff(self.times) <= 0)
        ):
            raise ValueError("Times must start at rest and increase strictly")
        self.group = zarr.open_group(store, mode="r", storage_options=storage_options)
        self.levels = m["levels"]
        self.arrays = [self.group[f"level_{i}"] for i in range(len(self.levels))]
        for a, lev in zip(self.arrays, self.levels):
            if a.shape != (len(self.times), *lev["shape"]) or a.dtype != np.dtype(
                "<f8"
            ):
                raise ValueError("Native array shape or precision changed")
        self.native_plane = lru_cache(maxsize=48)(self._native_plane)

    def _native_plane(self, frame, level, plane, fixed, quantity):
        """Cache small native slabs, not the entire time series or full box."""
        f = PLANES[plane][2]
        component = slice(0, 3) if quantity == "velocity" else slice(3, 6)
        select = [frame, slice(None), slice(None), slice(None), component]
        select[f + 1] = fixed
        arrays = self._local_arrays if self._local_arrays is not None else self.arrays
        return np.asarray(arrays[level][tuple(select)])

    def warm_archive_cache(self):
        """Optional CPU-Space disk cache; stay usable remotely while it fills."""
        if self.remote_config is None:
            return
        from httpx import HTTPError
        from huggingface_hub import constants, snapshot_download

        repo_id, revision, prefix = self.remote_config
        # The current archive fits CPU Basic's ephemeral disk. Larger future
        # archives continue to use remote slabs instead of exhausting storage.
        raw_bytes = sum(a.nbytes for a in self.arrays)
        cache = Path(constants.HF_HUB_CACHE)
        cache.mkdir(parents=True, exist_ok=True)
        if (
            raw_bytes > 18 * 2**30
            or shutil.disk_usage(cache).free < raw_bytes + 5 * 2**30
        ):
            self.cache_status = (
                "Remote native slices · archive exceeds disk-cache budget"
            )
            return
        self.cache_status = (
            "Warming server-side archive cache · cold frames may be slow"
        )

        def warm():
            try:
                snapshot = snapshot_download(
                    repo_id,
                    repo_type="dataset",
                    revision=revision,
                    allow_patterns=[f"{prefix}/fields.zarr/**"],
                    token=False,
                    max_workers=4,
                )
                group = zarr.open_group(
                    str(Path(snapshot) / prefix / "fields.zarr"), mode="r"
                )
                arrays = [group[f"level_{i}"] for i in range(len(self.levels))]
                if any(
                    a.shape != b.shape or a.dtype != b.dtype
                    for a, b in zip(arrays, self.arrays, strict=True)
                ):
                    raise ValueError("Cached layout differs from pinned native archive")
                # Never read a half-downloaded store: absent chunks mean zeros.
                # Publish the local arrays only after the entire snapshot is ready.
                self._local_arrays = arrays
                self.cache_status = (
                    "Native archive cached on server · no browser volume download"
                )
                print("Native archive disk cache ready", flush=True)
            except (OSError, ValueError, RuntimeError, KeyError, HTTPError) as error:
                self.cache_status = "Remote native slices · disk cache unavailable"
                print(f"Archive cache unavailable: {type(error).__name__}", flush=True)

        Thread(target=warm, name="native-archive-cache", daemon=True).start()

    def sample_plane(self, frame, plane, coordinate, quantity="velocity", n=256):
        if (
            not isinstance(frame, (int, float, np.integer, np.floating))
            or not np.isfinite(frame)
            or isinstance(frame, bool)
            or int(frame) != frame
            or not 0 <= frame < len(self.times)
        ):
            raise ValueError("Invalid saved frame")
        if plane not in PLANES or quantity not in {"velocity", "force"}:
            raise ValueError("Invalid plane or quantity")
        if (
            not np.isfinite(coordinate)
            or not -1 <= coordinate <= 1
            or n not in {64, 128, 256, 512, 1024}
        ):
            raise ValueError("Invalid coordinate or display resolution")
        coordinate = min(float(coordinate), np.nextafter(2.0, 0.0) - 1.0)
        h, v, f = PLANES[plane]
        axis = -1 + 2 * (np.arange(n) + 0.5) / n
        output = np.full((n, n, 3), np.nan)
        level_map = np.full((n, n), -1, dtype=np.int8)
        for index, lev in enumerate(self.levels):
            origin, dx = np.array(lev["origin"]), np.array(lev["spacing"])
            size = np.array(lev["shape"][:3])
            fixed = int(np.floor((coordinate - origin[f]) / dx[f]))
            if not 0 <= fixed < size[f]:
                continue
            ih = np.floor((axis - origin[h]) / dx[h]).astype(int)
            iv = np.floor((axis - origin[v]) / dx[v]).astype(int)
            ph, pv = (
                np.flatnonzero((ih >= 0) & (ih < size[h])),
                np.flatnonzero((iv >= 0) & (iv < size[v])),
            )
            if not len(ph) or not len(pv):
                continue
            values = self.native_plane(int(frame), index, plane, fixed, quantity)
            output[np.ix_(ph, pv)] = values[np.ix_(ih[ph], iv[pv])]
            level_map[np.ix_(ph, pv)] = index
        if not np.isfinite(output).all() or np.any(level_map < 0):
            raise ValueError("Native hierarchy does not cover the requested plane")
        return output, level_map

    def clear_cache(self):
        self.native_plane.cache_clear()
