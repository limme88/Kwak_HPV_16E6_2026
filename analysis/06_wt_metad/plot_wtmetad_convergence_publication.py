#!/usr/bin/env python3
"""Plot the five matched WT-MetaD runs' convergence diagnostics.
Inputs: per-run convergence JSON; outputs: PDF, PNG and source-data CSVs.
Supports V12 Figure S5; diagnostic generation requires external software.
Run from the repository root; see this folder's README for commands.
"""

# Create a publication-ready convergence figure for the five matched WT-MetaD runs.
# 
# The script consumes the per-replicate JSON files written by
# ``metadynamics_convergence_analysis_batch.py``.  It does not require a
# Schrodinger installation or license.

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".mpl-cache"))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


COLORS = ["#D55E00", "#0072B2", "#009E73", "#CC79A7", "#E69F00"]
LINESTYLES = ["-"] * 5


def moving_median(values: np.ndarray, window: int = 5) -> np.ndarray:
    """Centered moving median with shortened windows at both ends."""
    half = window // 2
    return np.asarray([
        np.median(values[max(0, i - half):min(len(values), i + half + 1)])
        for i in range(len(values))
    ])


def load_runs(input_dir: Path) -> list[dict]:
    paths = sorted(input_dir.glob("rep*/rep*_convergence_metrics.json"))
    if len(paths) != 5:
        raise ValueError(
            f"Expected five convergence-metrics JSON files under {input_dir}; "
            f"found {len(paths)}."
        )

    runs = []
    for path in paths:
        match = re.search(r"rep(\d+)_([0-9.]+)", path.parent.name)
        if match is None:
            raise ValueError(f"Could not parse replicate identifier from {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        times = np.asarray(data["block_end_time_ps_from_kerseq"], dtype=float)
        times = times[1:] / 1000.0
        global_rms = np.asarray(
            data["consecutive_cumulative_fes_grid_rms_kcal_mol"], dtype=float
        )
        low_rms = np.asarray(
            data["consecutive_terminal_low_energy_fes_grid_rms_kcal_mol"],
            dtype=float,
        )
        density_rms = np.asarray(
            data["consecutive_block_hill_center_density_rms"], dtype=float
        )
        if not (len(times) == len(global_rms) == len(low_rms) == len(density_rms)):
            raise ValueError(f"Time-series length mismatch in {path}")
        hills = data["hill_height_kcal_mol"]
        initial = float(hills["initial"])
        runs.append({
            "index": int(match.group(1)),
            "date": match.group(2),
            "label": f"Replicate {int(match.group(1))}",
            "path": path,
            "times_ns": times,
            "global_rms": global_rms,
            "low_rms": low_rms,
            "density_rms": density_rms,
            "initial_hill": initial,
            "final_hill_ratio": float(hills["final"]) / initial,
            "late_median_hill_ratio": float(hills["late_10_percent_median"]) / initial,
            "late_max_hill_ratio": float(hills["late_10_percent_maximum"]) / initial,
            "low_energy_window": float(data["terminal_low_energy_window_kcal_mol"]),
            "temperature_K": float(data["temperature_K"]),
            "kTemp_kcal_mol": float(data["kTemp_kcal_mol"]),
            "blocks": int(data["number_of_blocks"]),
            "hills": int(data["total_nonzero_hills_in_file"]),
        })

    runs.sort(key=lambda item: item["index"])
    if [run["index"] for run in runs] != [1, 2, 3, 4, 5]:
        raise ValueError("Replicate identifiers must be 1 through 5.")
    for key in ("temperature_K", "kTemp_kcal_mol", "blocks", "hills"):
        values = {run[key] for run in runs}
        if len(values) != 1:
            raise ValueError(f"Matched-run invariant differs for {key}: {values}")
    return runs


def write_source_data(runs: list[dict], timeseries_csv: Path, hills_csv: Path) -> None:
    timeseries_csv.parent.mkdir(parents=True, exist_ok=True)
    with timeseries_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "replicate",
            "run_date",
            "time_ns",
            "consecutive_cumulative_fes_rms_kcal_mol",
            "consecutive_terminal_low_energy_fes_rms_kcal_mol",
            "consecutive_block_hill_center_density_rms",
        ])
        for run in runs:
            for values in zip(
                run["times_ns"],
                run["global_rms"],
                run["low_rms"],
                run["density_rms"],
            ):
                writer.writerow([run["index"], run["date"], *values])

    with hills_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "replicate",
            "run_date",
            "initial_hill_kcal_mol",
            "final_hill_over_initial",
            "late_10_percent_median_hill_over_initial",
            "late_10_percent_max_hill_over_initial",
        ])
        for run in runs:
            writer.writerow([
                run["index"],
                run["date"],
                run["initial_hill"],
                run["final_hill_ratio"],
                run["late_median_hill_ratio"],
                run["late_max_hill_ratio"],
            ])


def style_axis(ax: mpl.axes.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(direction="out", length=3.0, width=0.8, pad=2.5)
    ax.grid(False)


def plot_series_panel(
    ax: mpl.axes.Axes,
    runs: list[dict],
    field: str,
    title: str,
    ylabel: str,
) -> None:
    for i, run in enumerate(runs):
        x = run["times_ns"]
        y = run[field]
        ax.plot(
            x / 1000.0,
            moving_median(y, window=5),
            color=COLORS[i],
            linestyle=LINESTYLES[i],
            lw=1.45,
            zorder=2,
        )
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.25, 0.50, 0.75, 1.00])
    ymax = max(float(np.max(run[field])) for run in runs)
    ax.set_ylim(0, ymax * 1.06)
    ax.set_xlabel(r"WT-MetaD time ($\mu$s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", pad=5, fontsize=9)
    style_axis(ax)


def add_panel_label(ax: mpl.axes.Axes, label: str) -> None:
    ax.text(
        -0.15,
        1.09,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="left",
    )


def create_figure(runs: list[dict], output_pdf: Path, output_png: Path) -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": True,
    })

    fig, axes = plt.subplots(2, 2, figsize=(7.15, 6.15))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    plot_series_panel(
        ax_a,
        runs,
        "global_rms",
        "Whole CV grid",
        "Consecutive FES RMS change\n(kcal mol$^{-1}$)",
    )
    plot_series_panel(
        ax_b,
        runs,
        "low_rms",
        "Terminal low-energy region",
        "Consecutive FES RMS change\n(kcal mol$^{-1}$)",
    )
    plot_series_panel(
        ax_c,
        runs,
        "density_rms",
        "Hill-center sampling distribution",
        "Consecutive RMS density change",
    )

    x = np.arange(1, 6, dtype=float)
    final_values = 100.0 * np.asarray([run["final_hill_ratio"] for run in runs])
    median_values = 100.0 * np.asarray([
        run["late_median_hill_ratio"] for run in runs
    ])
    max_values = 100.0 * np.asarray([run["late_max_hill_ratio"] for run in runs])
    for i in range(5):
        ax_d.plot(
            [x[i], x[i]],
            [median_values[i], max_values[i]],
            color=COLORS[i],
            lw=1.2,
            alpha=0.8,
            zorder=1,
        )
        ax_d.scatter(
            x[i], median_values[i], marker="o", s=28,
            facecolor=COLORS[i], edgecolor="white", linewidth=0.6, zorder=3,
        )
        ax_d.scatter(
            x[i], final_values[i], marker="D", s=27,
            facecolor="white", edgecolor=COLORS[i], linewidth=1.1, zorder=4,
        )
        ax_d.scatter(
            x[i], max_values[i], marker="^", s=31,
            facecolor="white", edgecolor=COLORS[i], linewidth=1.1, zorder=3,
        )
    ax_d.set_xlim(0.5, 5.5)
    ax_d.set_xticks(x, [f"Replicate {i}" for i in range(1, 6)], rotation=25, ha="right")
    ax_d.set_ylim(0, max(max_values) * 1.18)
    ax_d.set_ylabel("Hill height (% of initial)")
    ax_d.set_title("Late-stage Gaussian hill heights", loc="left", pad=5, fontsize=9)
    style_axis(ax_d)
    hill_legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#555555",
               markeredgecolor="white", markersize=5.5,
               label="Median, final 10%"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="white",
               markeredgecolor="#555555", markersize=5,
               label="Final hill"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="white",
               markeredgecolor="#555555", markersize=5.5,
               label="Maximum, final 10%"),
    ]
    ax_d.legend(
        handles=hill_legend,
        loc="upper right",
        frameon=False,
        handletextpad=0.4,
        borderaxespad=0.2,
        labelspacing=0.35,
    )

    for label, ax in zip("abcd", (ax_a, ax_b, ax_c, ax_d)):
        add_panel_label(ax, label)

    replicate_handles = [
        Line2D(
            [0], [0],
            color=COLORS[i],
            linestyle=LINESTYLES[i],
            lw=1.5,
            label=run["label"],
        )
        for i, run in enumerate(runs)
    ]
    fig.legend(
        handles=replicate_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=5,
        frameon=False,
        handlelength=2.8,
        columnspacing=1.1,
        handletextpad=0.45,
    )
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.09, top=0.91,
                        wspace=0.34, hspace=0.42)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(output_png, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-pdf", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    parser.add_argument("--timeseries-csv", type=Path, required=True)
    parser.add_argument("--hills-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = load_runs(args.input_dir)
    write_source_data(runs, args.timeseries_csv, args.hills_csv)
    create_figure(runs, args.output_pdf, args.output_png)
    print(f"Wrote {args.output_pdf}")
    print(f"Wrote {args.output_png}")
    print(f"Wrote {args.timeseries_csv}")
    print(f"Wrote {args.hills_csv}")


if __name__ == "__main__":
    main()
