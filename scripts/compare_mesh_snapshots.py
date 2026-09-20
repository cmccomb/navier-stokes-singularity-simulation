"""Conservatively compare nested composite fields on the coarser hierarchy.

This measures mesh/source sensitivity, not continuum error. Fine-only variation
is averaged out of the comparison; native source probes must check it separately.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.export_mesh_snapshot import sha
from scripts.paper_run import write
from scripts.revolved_mesh_pilot import block_diagnostics


def values(raw, block):
    return np.memmap(
        raw,
        dtype="<f8",
        mode="r",
        offset=block["offset_bytes"],
        shape=tuple(block["shape"]),
    )


def validate_blocks(blocks, byte_count):
    """Fail closed on corrupted geometry/offsets; comparisons assume [-1,1]^3."""
    offset = 0
    for b in blocks:
        shape, lo, dx = (
            np.array(b["shape"]),
            np.array(b["index_lo"]),
            np.array(b["spacing"]),
        )
        if (
            shape.shape != (4,)
            or shape[3] != 6
            or np.any(shape <= 0)
            or lo.shape != (3,)
            or dx.shape != (3,)
            or np.any(dx <= 0)
            or not np.all(dx == dx[0])
            or b["level"] < 0
            or not np.allclose(b["origin"], -1 + lo * dx, rtol=0, atol=1e-14)
            or np.any(lo < 0)
            or np.any((lo + shape[:3]) * dx > 2 + 1e-14)
            or b["offset_bytes"] != offset
        ):
            raise ValueError("Invalid native block geometry/offset")
        offset += int(np.prod(shape)) * 8
    if offset != byte_count:
        raise ValueError("Native block byte count differs")
    for i, a in enumerate(blocks):
        for b in blocks[i + 1 :]:
            if a["level"] == b["level"] and overlap(
                np.array(a["index_lo"]),
                np.array(a["index_lo"]) + a["shape"][:3],
                np.array(b["index_lo"]),
                np.array(b["index_lo"]) + b["shape"][:3],
            ):
                raise ValueError("Overlapping native blocks on the same level")


def overlap(a, b, c, d):
    lo, hi = np.maximum(a, c), np.minimum(b, d)
    return (lo, hi) if np.all(hi > lo) else None


def slices(lo, hi):
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi, strict=True))


def active_mask(block, blocks):
    lo = np.array(block["index_lo"])
    shape = np.array(block["shape"][:3])
    mask = np.ones(shape, dtype=bool)
    for fine in blocks:
        if fine["level"] != block["level"] + 1:
            continue
        flo, fshape = np.array(fine["index_lo"]), np.array(fine["shape"][:3])
        if np.any(flo % 2) or np.any(fshape % 2):
            raise ValueError("Fine blocks are not parent-cell aligned")
        hit = overlap(lo, lo + shape, flo // 2, (flo + fshape) // 2)
        if hit:
            mask[slices(hit[0] - lo, hit[1] - lo)] = False
    return mask


def restrict_to(raw, source_blocks, target, required):
    """Overwrite coarse samples with volume means of aligned finer samples."""
    lo = np.array(target["index_lo"])
    shape = np.array(target["shape"][:3])
    out = np.zeros((*shape, 6))
    owner = np.full(shape, -1)
    for block in sorted(source_blocks, key=lambda b: b["level"]):
        if block["level"] < target["level"]:
            continue
        ratio = 2 ** (block["level"] - target["level"])
        if not np.allclose(
            np.array(block["spacing"]) * ratio, target["spacing"], rtol=0, atol=1e-14
        ):
            raise ValueError("Grid spacing mismatch")
        a, n = np.array(block["index_lo"]), np.array(block["shape"][:3])
        # Only overlapping active target cells matter. Descendant boxes can be
        # misaligned to this coarser grid inside an already excluded core.
        hit = overlap(lo, lo + shape, a // ratio, (a + n + ratio - 1) // ratio)
        if not hit or not required[slices(hit[0] - lo, hit[1] - lo)].any():
            continue
        if np.any(a % ratio) or np.any(n % ratio):
            raise ValueError("Cannot conservatively restrict misaligned active boxes")
        x, y = hit
        selected = values(raw, block)[slices(x * ratio - a, y * ratio - a)]
        q = y - x
        averaged = selected.reshape(q[0], ratio, q[1], ratio, q[2], ratio, 6).mean(
            axis=(1, 3, 5)
        )
        dest = slices(x - lo, y - lo)
        out[dest] = averaged
        owner[dest] = block["level"]
    if np.any(owner[required] < target["level"]):
        raise ValueError("Candidate is not at least as refined as the reference")
    if not np.isfinite(out[required]).all():
        raise ValueError("Nonfinite candidate")
    return out, owner


def interface_mask(xyz, dx, widths):
    """Two-local-cell collar of baseline cube surfaces, not patch edges."""
    radius = np.maximum.reduce([np.abs(v) for v in xyz])
    return np.any(np.abs(radius[..., None] - np.array(widths)) <= 2 * dx, axis=-1)


def compare(left_raw, left, right_raw, right, widths):
    stats, tops = {}, {"velocity": [], "force": []}
    for block in left:
        active = active_mask(block, left)
        if not active.any():
            continue
        reference = values(left_raw, block)
        candidate, owner = restrict_to(right_raw, right, block, active)
        dx = block["spacing"][0]
        xyz = np.meshgrid(
            *[
                o + (np.arange(n) + 0.5) * d
                for o, n, d in zip(
                    block["origin"], block["shape"][:3], block["spacing"], strict=True
                )
            ],
            indexing="ij",
        )
        collar = (
            interface_mask(xyz, dx, widths) if widths else np.zeros(active.shape, bool)
        )
        masks = {
            "all": active,
            f"level_{block['level']}": active,
            "baseline_interface_collar": active & collar,
            "away_from_baseline_interfaces": active & ~collar,
            "newly_refined": active & (owner > block["level"]),
            "unchanged_spacing": active & (owner == block["level"]),
        }
        for name, sl in (("velocity", slice(0, 3)), ("force", slice(3, 6))):
            delta = np.sum((candidate[..., sl] - reference[..., sl]) ** 2, axis=-1)
            magnitude = np.sum(reference[..., sl] ** 2, axis=-1)
            for region, mask in masks.items():
                key = region + "/" + name
                row = stats.setdefault(
                    key,
                    {
                        "volume": 0.0,
                        "delta_squared_integral": 0.0,
                        "reference_squared_integral": 0.0,
                        "maximum_vector_difference": 0.0,
                    },
                )
                row["volume"] += float(mask.sum() * dx**3)
                row["delta_squared_integral"] += float(delta[mask].sum() * dx**3)
                row["reference_squared_integral"] += float(
                    magnitude[mask].sum() * dx**3
                )
                if mask.any():
                    row["maximum_vector_difference"] = max(
                        row["maximum_vector_difference"],
                        float(np.sqrt(delta[mask].max())),
                    )
            index = np.unravel_index(
                np.argmax(np.where(active, delta, -1)), active.shape
            )
            tops[name].append(
                {
                    "difference": float(np.sqrt(delta[index])),
                    "xyz": [float(v[index]) for v in xyz],
                    "reference_level": block["level"],
                    "candidate_level": int(owner[index]),
                    "dx": dx,
                    "baseline_interface_collar": bool(collar[index]),
                }
            )
    if not np.isclose(stats["all/velocity"]["volume"], 8, rtol=0, atol=1e-12):
        raise ValueError("Comparison does not cover the full domain once")
    for key, row in stats.items():
        row["rms_difference"] = (
            (row["delta_squared_integral"] / row["volume"]) ** 0.5
            if row["volume"]
            else None
        )
        row["relative_l2_difference"] = (
            (row["delta_squared_integral"] / row["reference_squared_integral"]) ** 0.5
            if row["reference_squared_integral"]
            else None
        )
        total = stats["all/" + key.split("/")[1]]["delta_squared_integral"]
        row["fraction_of_squared_difference"] = (
            row["delta_squared_integral"] / total if total else None
        )
    return {
        "regions": stats,
        "largest_block_discrepancies": {
            k: sorted(v, key=lambda r: -r["difference"])[:12] for k, v in tops.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("baseline", "candidate", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    reports = [
        json.loads((p / "manifest.json").read_text())
        for p in (args.baseline, args.candidate)
    ]
    if (
        not all(r["complete"] for r in reports)
        or reports[0]["profile_sha256"] != reports[1]["profile_sha256"]
    ):
        raise ValueError("Require completed snapshots of the same profile")
    if [f["step"] for f in reports[0]["frames"]] != [
        f["step"] for f in reports[1]["frames"]
    ]:
        raise ValueError("Matched steps required")
    for key in ("base_n", "widths", "max_dt", "epsilon_tau_ratio"):
        if reports[0]["parameters"][key] != reports[1]["parameters"][key]:
            raise ValueError(f"Controlled comparison parameter differs: {key}")
    args.output.mkdir(exist_ok=False)
    output = {
        "schema_version": 1,
        "complete": False,
        "snapshot_manifest_sha256": [
            sha(p / "manifest.json") for p in (args.baseline, args.candidate)
        ],
        "frames": [],
        "scope": "Full-domain native vector comparison after conservative restriction to baseline active cells. Not a convergence certificate. Fine-only structure is averaged out. Regions overlap; only each stated partition is additive. Interface collar refers to baseline cubes, not new band interfaces.",
    }
    for a, b in zip(reports[0]["frames"], reports[1]["frames"], strict=True):
        if a["export"]["time"] != b["export"]["time"]:
            raise ValueError("Snapshot clocks differ")
        raw_paths = [
            p / f["path"]
            for p, f in zip((args.baseline, args.candidate), (a, b), strict=True)
        ]
        for p, f in zip(raw_paths, (a, b), strict=True):
            if sha(p) != f["sha256"] or p.stat().st_size != f["export"]["bytes"]:
                raise ValueError("Native export hash/size differs")
            validate_blocks(f["export"]["blocks"], f["export"]["bytes"])
            diagnostic = block_diagnostics(p, f["export"]["blocks"])
            for key in ("volume", "energy", "peak_speed"):
                if not np.isclose(
                    diagnostic[key], f["diagnostic"][key], rtol=1e-10, atol=1e-12
                ):
                    raise ValueError(f"Native audit mismatch: {key}")
        result = compare(
            raw_paths[0],
            a["export"]["blocks"],
            raw_paths[1],
            b["export"]["blocks"],
            reports[0]["parameters"]["widths"],
        )
        output["frames"].append(
            {"step": a["step"], "time": a["export"]["time"], **result}
        )
        print(
            json.dumps(
                {"time": a["export"]["time"], **result["regions"]["all/velocity"]}
            ),
            flush=True,
        )
    output["complete"] = True
    write(args.output / "comparison.json", output)


if __name__ == "__main__":
    main()
