#!/usr/bin/env python3
"""Plot the numbered terminal FES and organize its sampled exemplar structures.
Inputs: FES arrays, exemplar CSV and structures; outputs: plots and numbered copies.
Supports the landscape panel of V12 Figure 6; molecular rendering is separate.
Run from the repository root; see this folder's README for commands.
"""

# Plot the five-replicate terminal FES with numbered sampled exemplars.
# 
# The numbering follows increasing D49--R102 distance. Within each distance
# group, minima are ordered from the top to the bottom of the displayed periodic
# Q107 chi1 axis: trans, g+, then g-.

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap


BASIN_ORDER = [
    (1, "D15_trans"),
    (2, "D15_gplus"),
    (3, "D15_gminus"),
    (4, "D28_trans"),
    (5, "D28_gplus"),
    (6, "D28_gminus"),
    (7, "D36_trans"),
    (8, "D36_gplus"),
    (9, "D36_gminus"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrays", type=Path, required=True)
    parser.add_argument("--representatives-csv", type=Path, required=True)
    parser.add_argument("--structure-dir", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    parser.add_argument("--output-unlabeled-png", type=Path, required=True)
    parser.add_argument("--output-pdf", type=Path)
    parser.add_argument("--output-key", type=Path, required=True)
    parser.add_argument("--numbered-structure-dir", type=Path, required=True)
    parser.add_argument("--color-cap", type=float, default=15.0)
    parser.add_argument("--color-interval", type=float, default=1.0)
    return parser.parse_args()


def configure_matplotlib() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.0,
        "axes.labelsize": 10.0,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.linewidth": 0.9,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })


def shifted_periodic_view(
    chi: np.ndarray, values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    display_chi = np.arange(-120.0, 241.0, 1.0)
    wrapped = ((display_chi + 180.0) % 360.0) - 180.0
    lookup = {float(angle): index for index, angle in enumerate(chi)}
    indices = np.array([lookup[float(angle)] for angle in wrapped], dtype=int)
    return display_chi, values[..., indices]


def display_chi(chi_deg: float) -> float:
    return chi_deg + 360.0 if chi_deg < -120.0 else chi_deg


def one_matching_file(directory: Path, center_id: str, suffix: str) -> Path:
    matches = sorted(directory.glob(f"{center_id}_*{suffix}"))
    if len(matches) != 1:
        raise ValueError(
            f"expected one {suffix} for {center_id} in {directory}; found {len(matches)}"
        )
    return matches[0]


def main() -> None:
    args = parse_args()
    configure_matplotlib()

    with np.load(args.arrays) as arrays:
        distance = arrays["distance"]
        chi = arrays["chi"]
        fes = arrays["combined_probability_mixture_fes"]

    with args.representatives_csv.open(newline="") as handle:
        representatives = {
            row["center_id"]: row for row in csv.DictReader(handle)
        }
    expected = {center_id for _, center_id in BASIN_ORDER}
    if set(representatives) != expected:
        raise ValueError(
            f"representative center IDs differ from expected IDs: "
            f"missing={sorted(expected - set(representatives))}; "
            f"extra={sorted(set(representatives) - expected)}"
        )

    cap = args.color_cap
    interval = args.color_interval
    band_count = int(round(cap / interval))
    if band_count < 1 or not np.isclose(band_count * interval, cap):
        raise ValueError("color cap must be an integer multiple of color interval")
    boundaries = np.linspace(0.0, cap, band_count + 1)
    # Match the supplied OPES reference with exactly one discrete color per
    # free-energy interval: dark blue at low free energy, transitioning
    # through cyan to pale green at high free energy.
    base_cmap = mpl.colormaps["GnBu_r"]
    cmap = ListedColormap(
        base_cmap(np.linspace(0.0, 1.0, band_count)),
        name=f"GnBu_r_{band_count}_bands",
    )
    norm = BoundaryNorm(boundaries, cmap.N, clip=True)
    display_angles, display_fes = shifted_periodic_view(chi, np.minimum(fes, cap))

    fig, ax = plt.subplots(figsize=(7.2, 5.35), constrained_layout=False)
    fig.subplots_adjust(left=0.14, right=0.81, bottom=0.16, top=0.96)
    filled = ax.contourf(
        distance,
        display_angles,
        display_fes.T,
        levels=boundaries,
        cmap=cmap,
        norm=norm,
        antialiased=True,
    )
    ax.contour(
        distance,
        display_angles,
        display_fes.T,
        levels=np.arange(3.0, cap, 3.0),
        colors="#5E5E5E",
        linewidths=0.55,
        alpha=0.75,
    )
    ax.set_xlim(float(distance[0]), float(distance[-1]))
    ax.set_ylim(-120.0, 240.0)
    ax.set_xticks([6, 10, 15, 20, 25, 30, 35, 39])
    ax.set_yticks([-120, -60, 0, 60, 120, 180, 240])
    ax.set_xlabel("D49–R102 residue COM distance (Å)", labelpad=7)
    ax.set_ylabel("Q107 χ₁ dihedral angle (°)", labelpad=8)
    ax.grid(False)
    ax.tick_params(top=False, right=False, length=4)
    for spine in ax.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(0.9)

    args.numbered_structure_dir.mkdir(parents=True, exist_ok=True)
    key_rows = []
    for number, center_id in BASIN_ORDER:
        row = representatives[center_id]
        x = float(row["center_minimum_image_distance_A"])
        canonical_y = float(row["center_chi1_canonical_deg"])
        y = display_chi(canonical_y)
        source_pdb = one_matching_file(args.structure_dir, center_id, ".pdb")
        source_maegz = one_matching_file(args.structure_dir, center_id, ".maegz")
        numbered_pdb = args.numbered_structure_dir / f"{number:02d}_{source_pdb.name}"
        numbered_maegz = args.numbered_structure_dir / f"{number:02d}_{source_maegz.name}"
        shutil.copy2(source_pdb, numbered_pdb)
        shutil.copy2(source_maegz, numbered_maegz)
        key_rows.append({
            "basin_number": number,
            "center_id": center_id,
            "distance_A": f"{x:.6f}",
            "chi1_canonical_deg": f"{canonical_y:.6f}",
            "chi1_display_deg": f"{y:.6f}",
            "smoothed_relative_fes_kcal_mol": row[
                "center_smoothed_relative_fes_kcal_mol"
            ],
            "replicate": row["replicate"],
            "trajectory_frame_zero_based": row["trajectory_frame_zero_based"],
            "time_ns": row["time_ns"],
            "numbered_pdb": str(numbered_pdb.resolve()),
            "numbered_maegz": str(numbered_maegz.resolve()),
        })

    colorbar = fig.colorbar(
        filled, ax=ax, pad=0.045, fraction=0.06,
        ticks=np.arange(0, cap + 0.1, 3.0),
        boundaries=boundaries, drawedges=True,
    )
    colorbar.set_label("Relative free energy (kcal mol⁻¹)", labelpad=9)
    colorbar.outline.set_linewidth(0.8)
    colorbar.dividers.set_linewidth(0.25)
    colorbar.dividers.set_color((0.25, 0.32, 0.32, 0.30))
    tick_labels = [f"{value:g}" for value in np.arange(0, cap + 0.1, 3.0)]
    tick_labels[-1] = f"≥{cap:g}"
    colorbar.set_ticklabels(tick_labels)
    ax.text(
        1.0, 1.018, "periodic seam: −120° ≡ 240°",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=7.7, color="#555555",
    )

    args.output_unlabeled_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        args.output_unlabeled_png, dpi=600,
        bbox_inches="tight", pad_inches=0.035,
    )

    for row in key_rows:
        x = float(row["distance_A"])
        y = float(row["chi1_display_deg"])
        number = int(row["basin_number"])
        ax.scatter(
            [x], [y], s=190, marker="o", facecolor="white",
            edgecolor="#202020", linewidth=1.0, zorder=8,
        )
        ax.text(
            x, y, str(number), ha="center", va="center",
            fontsize=8.5, fontweight="bold", color="#111111", zorder=9,
        )

    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=600, bbox_inches="tight", pad_inches=0.035)
    if args.output_pdf:
        args.output_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output_pdf, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)

    args.output_key.parent.mkdir(parents=True, exist_ok=True)
    with args.output_key.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(key_rows[0]))
        writer.writeheader()
        writer.writerows(key_rows)


if __name__ == "__main__":
    main()
