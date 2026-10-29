#!/usr/bin/env python3
"""Plot structural observables for the two saved deeptime macrostates.
Inputs: frame-level metrics CSV and matching MSM assignment bundle.
Outputs: PNG/PDF panels, bootstrap summary CSV and run manifest.
Supports V12 Figure 5d–f; historical Figure4 output names are retained.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FIGSIZE_IN = (7.2, 2.5)
PNG_DPI = 600
BLOCK_LENGTH_NS = 5000.0
N_BOOTSTRAP = 20_000
BASE_SEED = 20_260_908
STATES = ("Closed", "Open")
COLORS = ("#842277", "#FF7F0E")
OBSERVABLES = (
    (
        "tail_extension_A",
        "tail_ca130_ca151_distance_A",
        "C-tail extension (Å)",
        (0.0, 34.0),
    ),
    (
        "e6hl_helix_fraction",
        "e6hl_helix_fraction_H_or_G",
        "E6HL helicity",
        (0.0, 0.68),
    ),
    (
        "groove_aperture_A",
        "lxxll_aperture_all_heavy_A",
        "Pocket aperture (Å)",
        (0.0, 17.5),
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_and_validate(metrics_csv: Path, msm_bundle: Path) -> tuple[pd.DataFrame, np.ndarray]:
    data = pd.read_csv(metrics_csv)
    required = {
        "frame_index_0based",
        "manuscript_replica",
        "replica_time_ns_declared",
        "msm_macrostate",
        *(column for _, column, _, _ in OBSERVABLES),
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"metrics CSV lacks required columns: {missing}")
    if data[list(required)].isna().any().any():
        raise ValueError("required columns contain missing values")
    expected_index = np.arange(len(data), dtype=int)
    if not np.array_equal(data["frame_index_0based"].to_numpy(dtype=int), expected_index):
        raise ValueError("metrics rows are not in contiguous trajectory frame order")

    with np.load(msm_bundle) as bundle:
        hard_assignment = np.asarray(
            bundle["hard_macrostate_trajectory_closed0_open1"], dtype=int
        )
    if len(hard_assignment) != len(data):
        raise ValueError(
            f"deeptime assignment has {len(hard_assignment)} frames; CSV has {len(data)}"
        )
    if not set(np.unique(hard_assignment)).issubset({0, 1}):
        raise ValueError("deeptime hard assignment contains labels other than Closed=0/Open=1")
    csv_assignment = data["msm_macrostate"].map({"Closed": 0, "Open": 1})
    if csv_assignment.isna().any():
        raise ValueError("metrics CSV contains unrecognized macrostate labels")
    csv_assignment = csv_assignment.to_numpy(dtype=int)
    mismatch_count = int(np.count_nonzero(csv_assignment != hard_assignment))
    if mismatch_count:
        raise ValueError(
            f"metrics macrostate labels disagree with deeptime for {mismatch_count} frames"
        )
    return data, hard_assignment


def calculate_statistics(data: pd.DataFrame, assignment: np.ndarray) -> list[dict]:
    data = data.copy()
    data["hard_macrostate_index"] = assignment
    # Keep exact 5 or 10 µs endpoint frames in the preceding physical block.
    shifted_time = np.maximum(data["replica_time_ns_declared"].to_numpy() - 1e-9, 0.0)
    data["block_5us"] = np.floor(shifted_time / BLOCK_LENGTH_NS).astype(int)

    rows: list[dict] = []
    for observable_index, (name, column, _, _) in enumerate(OBSERVABLES):
        for state_index, state in enumerate(STATES):
            state_data = data.loc[data["hard_macrostate_index"] == state_index]
            block_means = (
                state_data.groupby(["manuscript_replica", "block_5us"], sort=True)[column]
                .mean()
                .to_numpy(dtype=float)
            )
            if len(block_means) < 2:
                raise ValueError(f"too few state-containing blocks for {state} {name}")
            rng = np.random.default_rng(BASE_SEED + state_index * 100 + observable_index)
            draws = rng.integers(
                0,
                len(block_means),
                size=(N_BOOTSTRAP, len(block_means)),
            )
            bootstrap_means = block_means[draws].mean(axis=1)
            rows.append(
                {
                    "macrostate": state,
                    "observable": name,
                    "mean": float(state_data[column].mean()),
                    "bootstrap_sd": float(bootstrap_means.std(ddof=1)),
                    "hard_assigned_frames": int(len(state_data)),
                    "state_containing_5us_blocks": int(len(block_means)),
                }
            )
    return rows


def render_figure(rows: list[dict], png_path: Path, pdf_path: Path) -> None:
    lookup = {(row["macrostate"], row["observable"]): row for row in rows}
    style = {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with plt.rc_context(style):
        figure, axes = plt.subplots(1, 3, figsize=FIGSIZE_IN, layout="constrained")
        figure.get_layout_engine().set(w_pad=0.035, h_pad=0.02, wspace=0.10, hspace=0.02)
        for axis, (name, _, ylabel, ylim) in zip(axes, OBSERVABLES):
            means = [lookup[(state, name)]["mean"] for state in STATES]
            errors = [lookup[(state, name)]["bootstrap_sd"] for state in STATES]
            positions = np.arange(2)
            axis.bar(
                positions,
                means,
                yerr=errors,
                width=0.62,
                color=COLORS,
                edgecolor="black",
                linewidth=0.8,
                error_kw={
                    "ecolor": "black",
                    "elinewidth": 0.9,
                    "capsize": 3.0,
                    "capthick": 0.9,
                },
            )
            axis.set_xticks(positions, STATES)
            axis.set_ylabel(ylabel)
            axis.set_ylim(*ylim)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.tick_params(length=3)
            axis.grid(False)
        figure.savefig(
            png_path,
            dpi=PNG_DPI,
            bbox_inches=None,
            facecolor="white",
            metadata={"Title": "Figure 4d-f: apo-16E6 MSM structural observables"},
        )
        figure.savefig(
            pdf_path,
            bbox_inches=None,
            facecolor="white",
            metadata={"Title": "Figure 4d-f: apo-16E6 MSM structural observables"},
        )
        plt.close(figure)


def write_values(rows: list[dict], path: Path) -> None:
    fields = [
        "macrostate",
        "observable",
        "mean",
        "bootstrap_sd",
        "hard_assigned_frames",
        "state_containing_5us_blocks",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-csv", required=True, type=Path)
    parser.add_argument("--msm-bundle", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data, assignment = load_and_validate(args.metrics_csv, args.msm_bundle)
    rows = calculate_statistics(data, assignment)

    stem = args.output_dir / "Figure4d-f_MSM_structural_observables"
    png_path = stem.with_suffix(".png")
    pdf_path = stem.with_suffix(".pdf")
    values_path = args.output_dir / "Figure4d-f_values.csv"
    render_figure(rows, png_path, pdf_path)
    write_values(rows, values_path)

    manifest = {
        "metrics_csv": str(args.metrics_csv.resolve()),
        "metrics_csv_sha256": sha256(args.metrics_csv),
        "deeptime_msm_bundle": str(args.msm_bundle.resolve()),
        "deeptime_msm_bundle_sha256": sha256(args.msm_bundle),
        "deeptime_assignment_key": "hard_macrostate_trajectory_closed0_open1",
        "deeptime_assignment_matches_metrics_csv": True,
        "frame_count": int(len(data)),
        "state_order": list(STATES),
        "uncertainty": {
            "definition": "SD of state-specific means over bootstrap resamples of nonoverlapping 5-us blocks",
            "block_length_ns": BLOCK_LENGTH_NS,
            "bootstrap_replicates": N_BOOTSTRAP,
            "base_seed": BASE_SEED,
        },
        "figure_size_inches": list(FIGSIZE_IN),
        "png_dpi": PNG_DPI,
        "outputs": [str(png_path.resolve()), str(pdf_path.resolve()), str(values_path.resolve())],
    }
    (args.output_dir / "Figure4d-f_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"Wrote Figure 4d-f PNG/PDF and source values to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
