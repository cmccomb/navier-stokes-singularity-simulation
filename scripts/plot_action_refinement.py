"""Source-bound r-z activity maps for the fixed 3D mesh-design record."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from scripts.paper_run import sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads((args.audit / "manifest.json").read_text())
    design = json.loads((args.design / "design.json").read_text())
    if (
        design["audit_manifest_sha256"] != sha(args.audit / "manifest.json")
        or not audit["validated"]
        or not audit["complete_history"]
    ):
        raise ValueError("Complete matching activity audit required")
    if (
        sha(args.design / "refinement.bands") != design["band_sha256"]
        or sha(args.audit / "union.npz") != audit["union_sha256"]
    ):
        raise ValueError("Plot source changed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "text.color": "#e5eef4",
            "axes.labelcolor": "#bacbd7",
            "xtick.color": "#bacbd7",
            "ytick.color": "#bacbd7",
            "axes.edgecolor": "#40546a",
            "axes.facecolor": "#07111f",
            "figure.facecolor": "#07111f",
            "savefig.facecolor": "#07111f",
        }
    )
    fig, axes = plt.subplots(1, 4, figsize=(17, 7), layout="constrained", sharey=True)
    r, z = np.array(audit["r_edges"]), np.array(audit["z_edges"])
    for ax, index in zip(axes[:3], (40, 100, 279), strict=True):
        row = audit["records"][index]
        if sha(args.audit / row["path"]) != row["sha256"]:
            raise ValueError("Native action map changed")
        with np.load(args.audit / row["path"]) as frame:
            values = np.ma.masked_where(frame["volume"] == 0, frame["speed"])
            field = ax.pcolormesh(
                r,
                z,
                values.T,
                norm=LogNorm(vmin=0.01, vmax=8),
                cmap="magma",
                rasterized=True,
            )
        ax.set_title(
            f"t = {row['time']:.3f}\nPeak speed {row['maxima']['speed']:.2f}",
            color="#e5eef4",
            fontsize=13,
            pad=14,
        )
    fig.colorbar(
        field,
        ax=axes[:3],
        location="bottom",
        shrink=0.8,
        pad=0.07,
        label="Maximum speed around each ring · model units · common logarithmic scale",
        ticks=[0.01, 0.1, 1, 8],
    )
    with np.load(args.audit / "union.npz") as union:
        union_field = np.ma.masked_where(union["force"] <= 0, union["force"])
        field = axes[3].pcolormesh(
            r, z, union_field.T, cmap="viridis", vmin=0, vmax=1, rasterized=True
        )
    bands = np.loadtxt(args.design / "refinement.bands", skiprows=2, ndmin=2)
    qr, qz = np.linspace(0, 1, 512), np.linspace(-1, 1, 1024)
    mask = np.zeros((len(qr), len(qz)), dtype=bool)
    for _, a, b, c, d in bands:
        mask |= ((qr >= a) & (qr <= b))[:, None] & ((qz >= c) & (qz <= d))[None, :]
    axes[3].contour(qr, qz, mask.T, levels=[0.5], colors=["#ffd88c"], linewidths=1.5)
    axes[3].set_title(
        "All 280 saved states\nGold: outer-refinement pilot",
        color="#e5eef4",
        fontsize=13,
        pad=14,
    )
    fig.colorbar(
        field,
        ax=axes[3],
        location="bottom",
        pad=0.07,
        label="Force / that frame’s peak\nmaximum over time",
        ticks=[0, 0.5, 1],
    )
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect("equal")
        ax.set_xlabel("Radius r = √(x² + y²)")
        ax.set_xticks([0, 0.5, 1])
        ax.axhline(0, color="#91a5b6", lw=0.5, alpha=0.4)
    axes[0].set_ylabel("Height z")
    fig.suptitle(
        "The active region contracts; the final core is not the whole history",
        fontsize=20,
        color="#e5eef4",
    )
    fig.get_layout_engine().set(rect=(0, 0.055, 1, 0.95))
    fig.text(
        0.02,
        0.025,
        "Azimuthal maxima retain localized 3D activity. Empty bins are unsampled, not zero. Gold is a buffered test region, not an accuracy guarantee.",
        fontsize=11,
        color="#aabfce",
    )
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    write(
        args.output.with_suffix(".json"),
        {
            "source_record_sha256": audit["source_record_sha256"],
            "audit_manifest_sha256": sha(args.audit / "manifest.json"),
            "design_sha256": sha(args.design / "design.json"),
            "plotter_sha256": sha(Path(__file__)),
            "image_sha256": sha(args.output),
            "selected_indices": [40, 100, 279],
            "union_indices": list(range(len(audit["records"]))),
            "scope": "Native-cell r-z histograms; maxima over azimuth, not a 2D or axisymmetric solve. No temporal interpolation.",
        },
    )


if __name__ == "__main__":
    main()
