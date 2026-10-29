#!/usr/bin/env python3
"""Compute full DSSP and 100 ns secondary-structure propensities.
Inputs: four replica XTC files and the matching 151-residue topology.
Outputs: raw assignments, binned CSV matrices, and a four-panel plot.
Supports V12 Figure S2; final manuscript typography was assembled separately.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
from matplotlib.colors import to_rgb


REPLICAS = (
    {"replicate": 1, "filename": "replicate_1_10us.xtc", "duration_us": 10.0},
    {"replicate": 2, "filename": "replicate_2_10us.xtc", "duration_us": 10.0},
    {"replicate": 3, "filename": "replicate_3_10us.xtc", "duration_us": 10.0},
    {"replicate": 4, "filename": "replicate_4_5us.xtc", "duration_us": 5.0},
)

HELIX_CODES = ("H", "G", "I")
STRAND_CODES = ("E", "B")
KNOWN_DSSP_CODES = {" ", "B", "E", "G", "H", "I", "S", "T", "NA"}
HELIX_COLOR = "#F15A40"
STRAND_COLOR = "#3C78D8"

# Canonical HPV16 E6 element ranges follow the 4GIZ-based nomenclature.  The
# y positions are annotations only; DSSP itself is calculated independently.
NATIVE_ELEMENTS = (
    {"label": "alpha1", "display": r"$\alpha$1", "ranges": ((11, 20),), "y": 15.5},
    {"label": "beta1", "display": r"$\beta$1", "ranges": ((29, 30),), "y": 29.5},
    {"label": "alpha2", "display": r"$\alpha$2", "ranges": ((38, 48),), "y": 43.0},
    {
        "label": "beta2,3",
        "display": r"$\beta$2,3",
        "ranges": ((53, 55), (58, 60)),
        "y": 56.5,
    },
    {"label": "alpha3", "display": r"$\alpha$3", "ranges": ((63, 78),), "y": 70.5},
    {"label": "beta4", "display": r"$\beta$4", "ranges": ((79, 83),), "y": 81.0},
    {"label": "alpha4", "display": r"$\alpha$4", "ranges": ((84, 93),), "y": 88.5},
    {"label": "beta5", "display": r"$\beta$5", "ranges": ((101, 103),), "y": 102.0},
    {"label": "alpha5", "display": r"$\alpha$5", "ranges": ((111, 121),), "y": 116.0},
    {
        "label": "beta6,7",
        "display": r"$\beta$6,7",
        "ranges": ((125, 128), (131, 134)),
        "y": 129.5,
    },
    {"label": "alpha6", "display": r"$\alpha$6", "ranges": ((136, 142),), "y": 139.0},
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def protein_atom_indices(topology: md.Topology, requested_chain_id: str | None) -> tuple[np.ndarray, str]:
    protein_chains = []
    for chain in topology.chains:
        residues = [residue for residue in chain.residues if residue.is_protein]
        if residues:
            protein_chains.append((chain, residues))

    if requested_chain_id is None:
        if len(protein_chains) != 1:
            available = [chain.chain_id for chain, _ in protein_chains]
            raise ValueError(
                "Topology contains more than one protein chain; pass --chain-id. "
                f"Available protein chain IDs: {available}"
            )
        selected_chain = protein_chains[0][0]
    else:
        matches = [chain for chain, _ in protein_chains if chain.chain_id == requested_chain_id]
        if len(matches) != 1:
            raise ValueError(f"Could not uniquely select protein chain ID {requested_chain_id!r}")
        selected_chain = matches[0]

    atoms = np.asarray(
        [atom.index for atom in selected_chain.atoms if atom.residue.is_protein], dtype=int
    )
    return atoms, selected_chain.chain_id


def validate_e6_topology(topology: md.Topology) -> list[int]:
    residues = list(topology.residues)
    residue_numbers = [int(residue.resSeq) for residue in residues]
    if topology.n_chains != 1:
        raise ValueError(f"Expected one selected E6 chain, found {topology.n_chains}")
    if len(residues) != 151:
        raise ValueError(f"Expected 151 E6 residues, found {len(residues)}")
    if residue_numbers != list(range(1, 152)):
        raise ValueError("Selected E6 chain is not consecutively numbered 1-151")

    for residue in residues:
        names = Counter(atom.name for atom in residue.atoms)
        missing = [name for name in ("N", "CA", "C", "O") if names[name] != 1]
        if missing:
            raise ValueError(f"Residue {residue.resSeq} lacks a unique backbone atom: {missing}")
    return residue_numbers


def bin_dssp(
    dssp: np.ndarray,
    duration_us: float,
    frame_spacing_ns: float,
    bin_ns: float,
) -> dict[str, np.ndarray]:
    duration_ns = duration_us * 1000.0
    n_bins_float = duration_ns / bin_ns
    n_bins = int(round(n_bins_float))
    if not np.isclose(n_bins_float, n_bins, rtol=0, atol=1e-9):
        raise ValueError(f"Duration {duration_us} us is not divisible by {bin_ns} ns")

    frame_times_ns = np.arange(dssp.shape[0], dtype=float) * frame_spacing_ns
    if frame_times_ns[-1] > duration_ns + 1e-6:
        raise ValueError("Trajectory extends beyond its declared duration")
    if duration_ns - frame_times_ns[-1] > frame_spacing_ns + 1e-6:
        raise ValueError("Trajectory ends more than one saved-frame interval before its duration")

    bin_edges_ns = np.linspace(0.0, duration_ns, n_bins + 1)
    assignments = np.searchsorted(bin_edges_ns, frame_times_ns, side="right") - 1
    # Include a coordinate saved exactly at the nominal endpoint in the final
    # window, avoiding a misleading singleton terminal bin.
    assignments[assignments == n_bins] = n_bins - 1
    if np.any(assignments < 0) or np.any(assignments >= n_bins):
        raise RuntimeError("At least one frame was not assigned to a time bin")

    helix = np.full((n_bins, dssp.shape[1]), np.nan, dtype=np.float32)
    strand = np.full_like(helix, np.nan)
    other = np.full_like(helix, np.nan)
    valid_counts = np.zeros((n_bins, dssp.shape[1]), dtype=np.int32)
    frame_counts = np.bincount(assignments, minlength=n_bins).astype(np.int32)

    for bin_index in range(n_bins):
        values = dssp[assignments == bin_index]
        valid = values != "NA"
        counts = valid.sum(axis=0)
        valid_counts[bin_index] = counts
        usable = counts > 0
        helix_counts = (np.isin(values, HELIX_CODES) & valid).sum(axis=0)
        strand_counts = (np.isin(values, STRAND_CODES) & valid).sum(axis=0)
        other_counts = (valid & ~np.isin(values, HELIX_CODES + STRAND_CODES)).sum(axis=0)
        helix[bin_index, usable] = helix_counts[usable] / counts[usable]
        strand[bin_index, usable] = strand_counts[usable] / counts[usable]
        other[bin_index, usable] = other_counts[usable] / counts[usable]

    observed = np.stack((helix, strand, other))
    if np.nanmin(observed) < -1e-7 or np.nanmax(observed) > 1.0 + 1e-7:
        raise RuntimeError("A calculated propensity lies outside [0, 1]")
    if not np.allclose(helix + strand + other, 1.0, atol=2e-6, equal_nan=True):
        raise RuntimeError("Helix, strand, and other propensities do not sum to one")
    if np.nanmax(helix + strand) > 1.0 + 2e-6:
        raise RuntimeError("Helix plus strand propensity exceeds one")
    if int(frame_counts.sum()) != dssp.shape[0]:
        raise RuntimeError("Binning did not allocate every frame exactly once")

    return {
        "helix": helix,
        "strand": strand,
        "other": other,
        "valid_counts": valid_counts,
        "frame_counts": frame_counts,
        "bin_edges_us": bin_edges_ns / 1000.0,
        "frame_times_us": frame_times_ns / 1000.0,
    }


def bivariate_rgb(helix: np.ndarray, strand: np.ndarray) -> np.ndarray:
    """Map mutually exclusive helix/strand occupancy to a white/orange/blue simplex."""
    h = np.nan_to_num(np.asarray(helix, dtype=float), nan=0.0)
    s = np.nan_to_num(np.asarray(strand, dtype=float), nan=0.0)
    white = np.asarray(to_rgb("#FFFFFF"))
    orange = np.asarray(to_rgb(HELIX_COLOR))
    blue = np.asarray(to_rgb(STRAND_COLOR))
    rgb = (1.0 - h - s)[..., None] * white + h[..., None] * orange + s[..., None] * blue
    return np.clip(rgb, 0.0, 1.0)


def plot_figure(
    results: list[dict], png_path: Path, pdf_path: Path, dpi: int, bin_ns: float
) -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.0,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(8.4, 6.0), facecolor="white")
    outer = fig.add_gridspec(
        3,
        2,
        width_ratios=(35.0, 4.4),
        height_ratios=(151.0, 12.0, 28.0),
        left=0.085,
        right=0.975,
        bottom=0.095,
        top=0.955,
        wspace=0.07,
        hspace=0.10,
    )
    heatmap_grid = outer[0, 0].subgridspec(
        1,
        4,
        width_ratios=[result["duration_us"] for result in results],
        wspace=0.055,
    )

    axes = []
    for index, result in enumerate(results):
        ax = fig.add_subplot(heatmap_grid[0, index], sharey=axes[0] if axes else None)
        axes.append(ax)
        rgb = bivariate_rgb(result["helix"].T, result["strand"].T)
        ax.imshow(
            rgb,
            origin="upper",
            interpolation="nearest",
            aspect="auto",
            extent=(0.0, result["duration_us"], 151.5, 0.5),
            rasterized=True,
        )
        ax.set_xlim(0.0, result["duration_us"])
        ax.set_ylim(151.5, 0.5)
        duration = result["duration_us"]
        ax.set_xticks(np.linspace(0.0, duration, 3))
        tick_labels = ["0", f"{duration / 2:g}", f"{duration:g}"]
        if index > 0:
            tick_labels[0] = ""
        ax.set_xticklabels(tick_labels)
        ax.set_title(
            f"Replicate {result['replicate']}  ({duration:g} µs)",
            fontsize=8.5,
            pad=5.0,
        )
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_linewidth(0.7)

        if index == 0:
            y_ticks = [1, 20, 40, 60, 80, 100, 120, 140, 151]
            ax.set_yticks(y_ticks)
            ax.set_yticklabels([str(value) for value in y_ticks])
            ax.set_ylabel("E6 residue", fontsize=9)
        else:
            ax.tick_params(axis="y", which="both", left=False, labelleft=False)

    label_ax = fig.add_subplot(outer[0, 1], sharey=axes[0])
    label_ax.set_xlim(0.0, 1.0)
    label_ax.set_ylim(151.5, 0.5)
    label_ax.axis("off")
    for element in NATIVE_ELEMENTS:
        label_ax.annotate(
            element["display"],
            xy=(0.05, element["y"]),
            xytext=(0.43, element["y"]),
            ha="left",
            va="center",
            fontsize=9.5,
            arrowprops={
                "arrowstyle": "-|>",
                "color": "#222222",
                "lw": 0.75,
                "shrinkA": 0,
                "shrinkB": 0,
            },
        )

    x_label_ax = fig.add_subplot(outer[1, 0])
    x_label_ax.axis("off")
    x_label_ax.text(
        0.5,
        0.32,
        "Time within replicate (µs)",
        ha="center",
        va="center",
        fontsize=9,
    )

    legend_ax = fig.add_subplot(outer[1:, 1])
    steps = 151
    h_values = np.linspace(0.0, 1.0, steps)
    s_values = np.linspace(0.0, 1.0, steps)
    h_grid, s_grid = np.meshgrid(h_values, s_values)
    feasible = h_grid + s_grid <= 1.0 + 1e-12
    key_rgba = np.zeros((steps, steps, 4), dtype=float)
    key_rgba[feasible, :3] = bivariate_rgb(h_grid[feasible], s_grid[feasible])
    key_rgba[feasible, 3] = 1.0
    legend_ax.imshow(
        key_rgba,
        origin="lower",
        interpolation="nearest",
        extent=(0.0, 1.0, 0.0, 1.0),
        aspect="equal",
        rasterized=True,
    )
    legend_ax.plot((0.0, 1.0), (1.0, 0.0), color="#555555", lw=0.6)
    legend_ax.set_xlim(0.0, 1.0)
    legend_ax.set_ylim(0.0, 1.0)
    legend_ax.set_xticks((0.0, 0.5, 1.0))
    legend_ax.set_yticks((0.0, 0.5, 1.0))
    legend_ax.set_xlabel("Helix", fontsize=7.0, labelpad=1.0)
    legend_ax.set_ylabel("Strand", fontsize=7.0, labelpad=1.0)
    legend_ax.tick_params(labelsize=6.0, pad=1.0, length=2.0)
    legend_ax.grid(False)
    legend_ax.spines["top"].set_visible(False)
    legend_ax.spines["right"].set_visible(False)
    legend_ax.spines["left"].set_linewidth(0.6)
    legend_ax.spines["bottom"].set_linewidth(0.6)

    note_ax = fig.add_subplot(outer[2, 0])
    note_ax.axis("off")
    note_ax.text(
        0.0,
        0.56,
        "DSSP helix: H/G/I     DSSP strand: E/B     White: other (turn, bend, or coil)",
        ha="left",
        va="center",
        fontsize=7.5,
    )
    note_ax.text(
        0.0,
        0.24,
        f"Values are fractions of saved frames within each {bin_ns:g}-ns window.",
        ha="left",
        va="center",
        fontsize=7.5,
    )

    fig.savefig(png_path, dpi=dpi, facecolor="white")
    fig.savefig(
        pdf_path,
        dpi=dpi,
        facecolor="white",
        metadata={
            "Title": "Time-resolved secondary-structure propensity of apo-16E6",
            "Subject": "100-ns-binned DSSP helix and strand occupancy across four replicates",
            "Creator": f"Matplotlib {matplotlib.__version__}; MDTraj {md.__version__}",
        },
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument("--topology", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chain-id")
    parser.add_argument("--frame-spacing-ns", type=float, default=1.0)
    parser.add_argument("--bin-ns", type=float, default=100.0)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()

    trajectory_dir = args.trajectory_dir.resolve()
    topology_path = (
        args.topology.resolve()
        if args.topology is not None
        else trajectory_dir / "topology_apo16e6.pdb"
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    topology_frame = md.load(str(topology_path))
    selected_atom_indices, selected_chain_id = protein_atom_indices(
        topology_frame.topology, args.chain_id
    )
    selected_topology_frame = topology_frame.atom_slice(selected_atom_indices)
    residue_numbers = validate_e6_topology(selected_topology_frame.topology)

    results = []
    raw_npz = {"residue_numbers": np.asarray(residue_numbers, dtype=np.int16)}
    propensity_npz = {"residue_numbers": np.asarray(residue_numbers, dtype=np.int16)}
    summary_rows = []

    for replica_spec in REPLICAS:
        replica = replica_spec["replicate"]
        trajectory_path = trajectory_dir / replica_spec["filename"]
        trajectory = md.load(str(trajectory_path), top=str(topology_path))
        if len(selected_atom_indices) != trajectory.n_atoms:
            trajectory = trajectory.atom_slice(selected_atom_indices)
        validate_e6_topology(trajectory.topology)

        dssp = md.compute_dssp(trajectory, simplified=False).astype("U2")
        if dssp.shape != (trajectory.n_frames, 151):
            raise RuntimeError(f"Replicate {replica}: unexpected DSSP shape {dssp.shape}")
        observed_codes = set(np.unique(dssp).tolist())
        unexpected = observed_codes - KNOWN_DSSP_CODES
        if unexpected:
            raise RuntimeError(f"Replicate {replica}: unexpected DSSP codes {unexpected}")

        binned = bin_dssp(
            dssp,
            duration_us=replica_spec["duration_us"],
            frame_spacing_ns=args.frame_spacing_ns,
            bin_ns=args.bin_ns,
        )
        valid = dssp != "NA"
        n_valid = int(valid.sum())
        overall_helix = float((np.isin(dssp, HELIX_CODES) & valid).sum() / n_valid)
        overall_strand = float((np.isin(dssp, STRAND_CODES) & valid).sum() / n_valid)
        overall_other = 1.0 - overall_helix - overall_strand
        code_counts = Counter(dssp.ravel().tolist())

        result = {
            "replicate": replica,
            "duration_us": replica_spec["duration_us"],
            "trajectory_path": trajectory_path,
            "n_frames": trajectory.n_frames,
            "dssp": dssp,
            **binned,
            "overall_helix": overall_helix,
            "overall_strand": overall_strand,
            "overall_other": overall_other,
            "code_counts": dict(sorted(code_counts.items())),
        }
        results.append(result)

        prefix = f"rep{replica}"
        raw_npz[f"{prefix}_dssp"] = dssp.astype("U2")
        raw_npz[f"{prefix}_time_us"] = binned["frame_times_us"].astype(np.float32)
        for field in ("helix", "strand", "other"):
            propensity_npz[f"{prefix}_{field}"] = binned[field]
        propensity_npz[f"{prefix}_valid_counts"] = binned["valid_counts"]
        propensity_npz[f"{prefix}_frame_counts"] = binned["frame_counts"]
        propensity_npz[f"{prefix}_bin_edges_us"] = binned["bin_edges_us"].astype(np.float32)

        summary_rows.append(
            {
                "replicate": replica,
                "trajectory_file": trajectory_path.name,
                "n_frames": trajectory.n_frames,
                "frame_spacing_ns": args.frame_spacing_ns,
                "nominal_duration_us": replica_spec["duration_us"],
                "n_time_bins": len(binned["frame_counts"]),
                "bin_width_ns": args.bin_ns,
                "overall_helix_fraction_H_G_I": overall_helix,
                "overall_strand_fraction_E_B": overall_strand,
                "overall_other_fraction": overall_other,
                "dssp_code_counts_json": json.dumps(dict(sorted(code_counts.items()))),
            }
        )

    bin_label = f"{args.bin_ns:g}ns".replace(".", "p")
    raw_path = output_dir / "dssp_assignments_all_replicates.npz"
    propensity_path = output_dir / f"dssp_propensity_{bin_label}_matrices.npz"
    np.savez_compressed(raw_path, **raw_npz)
    np.savez_compressed(propensity_path, **propensity_npz)

    long_csv_path = output_dir / f"dssp_propensity_{bin_label}_long.csv"
    with long_csv_path.open("w", newline="") as handle:
        fieldnames = (
            "replicate",
            "nominal_duration_us",
            "bin_index_0_based",
            "bin_start_us",
            "bin_end_us",
            "bin_center_us",
            "residue",
            "helix_propensity_H_G_I",
            "strand_propensity_E_B",
            "other_propensity",
            "valid_frames",
            "total_frames",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            edges = result["bin_edges_us"]
            for bin_index in range(len(edges) - 1):
                for residue_index, residue in enumerate(residue_numbers):
                    writer.writerow(
                        {
                            "replicate": result["replicate"],
                            "nominal_duration_us": result["duration_us"],
                            "bin_index_0_based": bin_index,
                            "bin_start_us": f"{edges[bin_index]:.6f}",
                            "bin_end_us": f"{edges[bin_index + 1]:.6f}",
                            "bin_center_us": f"{(edges[bin_index] + edges[bin_index + 1]) / 2:.6f}",
                            "residue": residue,
                            "helix_propensity_H_G_I": f"{result['helix'][bin_index, residue_index]:.6f}",
                            "strand_propensity_E_B": f"{result['strand'][bin_index, residue_index]:.6f}",
                            "other_propensity": f"{result['other'][bin_index, residue_index]:.6f}",
                            "valid_frames": int(result["valid_counts"][bin_index, residue_index]),
                            "total_frames": int(result["frame_counts"][bin_index]),
                        }
                    )

    summary_csv_path = output_dir / "dssp_summary_by_replicate.csv"
    with summary_csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    png_path = output_dir / f"apo16e6_dssp_propensity_{bin_label}.png"
    pdf_path = output_dir / f"apo16e6_dssp_propensity_{bin_label}.pdf"
    plot_figure(results, png_path, pdf_path, args.dpi, args.bin_ns)

    caption_path = output_dir / "supplementary_figure_caption.txt"
    caption_path.write_text(
        "Supplementary Figure 2. Time-resolved secondary-structure propensities of apo-16E6 "
        "across four independent trajectories. DSSP assignments for residues 1–151 were "
        f"averaged within {args.bin_ns:g}-ns windows. For each residue and window, orange intensity "
        "represents the fraction of frames assigned as helix (H, G, or I), and blue intensity "
        "represents the fraction assigned as strand (E or B); windows sampling both classes are "
        "shown by blended colors, while other assignments are white. The four adjacent replicate "
        "blocks retain their respective simulated durations of 10, 10, 10, and 5 µs. Canonical "
        "E6 α-helical and β-strand elements are labeled once along the right margin. Values "
        "represent time-local occupancy within each sampled trajectory and should not be "
        "interpreted as equilibrium probabilities.\n"
    )

    manifest = {
        "software": {
            "python": platform.python_version(),
            "mdtraj": md.__version__,
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "topology": {
            "path": str(topology_path),
            "sha256": sha256(topology_path),
            "selected_chain_id": selected_chain_id,
            "protein_chains_in_selected_topology": 1,
            "residues": 151,
            "residue_numbering": "1-151",
        },
        "dssp": {
            "mdtraj_simplified": False,
            "helix_codes": HELIX_CODES,
            "strand_codes": STRAND_CODES,
            "other_codes_observed": ["blank", "S", "T"],
            "missing_code": "NA (excluded from denominators; none observed)",
        },
        "binning": {
            "frame_spacing_ns": args.frame_spacing_ns,
            "bin_width_ns": args.bin_ns,
            "endpoint_policy": (
                "A frame exactly at the nominal final time is included in the final bin. "
                f"Thus the final {args.bin_ns:g}-ns bin contains "
                f"{int(round(args.bin_ns / args.frame_spacing_ns)) + 1} frames for replicas with "
                "a saved nominal-endpoint coordinate; "
                "no singleton endpoint bin is created."
            ),
        },
        "color_mapping": {
            "helix_color": HELIX_COLOR,
            "strand_color": STRAND_COLOR,
            "white": "#FFFFFF",
            "formula": "C = (1-h-s)*white + h*orange + s*blue; h+s<=1",
        },
        "native_element_annotations": [
            {key: value for key, value in element.items() if key != "display"}
            for element in NATIVE_ELEMENTS
        ],
        "native_annotation_note": (
            "Canonical ranges follow 4GIZ-based HPV16 E6 nomenclature. They are margin "
            "annotations and do not determine DSSP assignments. Alpha6 is a canonical element "
            "label even when residues 136-142 are nonhelical in a sampled frame."
        ),
        "replicates": [
            {
                "replicate": result["replicate"],
                "trajectory": str(result["trajectory_path"]),
                "trajectory_sha256": sha256(result["trajectory_path"]),
                "n_frames": result["n_frames"],
                "nominal_duration_us": result["duration_us"],
                "dssp_shape": list(result["dssp"].shape),
                "time_bins": len(result["frame_counts"]),
                "frames_per_bin_min": int(result["frame_counts"].min()),
                "frames_per_bin_max": int(result["frame_counts"].max()),
                "overall_helix_fraction": result["overall_helix"],
                "overall_strand_fraction": result["overall_strand"],
                "overall_other_fraction": result["overall_other"],
                "dssp_code_counts": result["code_counts"],
            }
            for result in results
        ],
        "outputs": {
            "png": {"file": png_path.name, "sha256": sha256(png_path)},
            "pdf": {"file": pdf_path.name, "sha256": sha256(pdf_path)},
            "raw_dssp_npz": {"file": raw_path.name, "sha256": sha256(raw_path)},
            "propensity_npz": {"file": propensity_path.name, "sha256": sha256(propensity_path)},
            "long_csv": {"file": long_csv_path.name, "sha256": sha256(long_csv_path)},
            "summary_csv": {"file": summary_csv_path.name, "sha256": sha256(summary_csv_path)},
            "caption": {"file": caption_path.name, "sha256": sha256(caption_path)},
        },
    }
    manifest_path = output_dir / "dssp_analysis_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Analyzed chain {selected_chain_id}: 151 residues")
    for row in summary_rows:
        print(
            f"Replicate {row['replicate']}: {row['n_frames']} frames, "
            f"helix={row['overall_helix_fraction_H_G_I']:.4f}, "
            f"strand={row['overall_strand_fraction_E_B']:.4f}, "
            f"other={row['overall_other_fraction']:.4f}"
        )
    print(f"Figure: {png_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
