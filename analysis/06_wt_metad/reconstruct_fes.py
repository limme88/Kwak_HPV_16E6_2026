#!/usr/bin/env python3
"""Reconstruct the equal-replicate terminal WT-MetaD probability-mixture FES.
Inputs: five matched Desmond FES grids; outputs: grids, plots and minima.
Supports V12 Figure 6; this is a descriptive terminal-FES mixture.
Run from the repository root; see this folder's README for commands.
"""

# Reconstruct and plot periodic apo-16E6 WT-MetaD free-energy surfaces.
# 
# The Desmond ``*.fes`` files contain the negative accumulated bias on a grid.
# For well-tempered metadynamics, the asymptotic free-energy estimate is
# 
#     F(s) = -gamma/(gamma - 1) * V(s),
# 
# where gamma = 1 + kTemp/(k_B T).  Desmond's exported third column is -V(s),
# so the relative values are multiplied by gamma/(gamma - 1).
# 
# Multiple terminal grids are summarized as an equal-replicate probability mixture.
# This is a transparent descriptive combination of independently normalized
# surfaces; it is not a substitute for trajectory-level reweighting or a
# convergence analysis.

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter, gaussian_filter1d, minimum_filter
from scipy.signal import find_peaks
from scipy.special import logsumexp


KB_KCAL_PER_MOL_K = 0.00198720425864083
PALETTE = ["#7F93D1", "#BDD0F1", "#F3EFEA", "#F2BEAA", "#D97F80"]
ATOMIC_MASSES = {
    "H": 1.008,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "P": 30.973761998,
    "S": 32.06,
    "ZN": 65.38,
}


@dataclass(frozen=True)
class Surface:
    path: Path
    distance: np.ndarray
    chi: np.ndarray
    raw: np.ndarray
    declared_shape: tuple[int, int] | None
    declared_units: str | None
    columns: str | None


@dataclass(frozen=True)
class PdbAtom:
    serial: int
    name: str
    altloc: str
    residue_name: str
    chain: str
    residue_number: int
    insertion_code: str
    occupancy: float
    element: str
    xyz: np.ndarray


@dataclass(frozen=True)
class ReferenceMetrics:
    pdb_path: Path
    chain: str
    d49_com: np.ndarray
    r102_com: np.ndarray
    distance_A: float
    chi1_deg: float
    d49_atom_count: int
    r102_atom_count: int
    d49_total_mass_Da: float
    r102_total_mass_Da: float
    heavy_atom_distance_A: float
    ca_distance_A: float


def parse_pdb_atoms(path: Path) -> list[PdbAtom]:
    atoms: list[PdbAtom] = []
    active_model = True
    saw_model = False
    for line in path.read_text().splitlines():
        if line.startswith("MODEL"):
            if saw_model:
                active_model = False
            else:
                saw_model = True
                active_model = True
            continue
        if line.startswith("ENDMDL") and active_model:
            break
        if not active_model or not line.startswith(("ATOM  ", "HETATM")):
            continue
        element = line[76:78].strip().upper()
        if not element:
            element = re.sub(r"[^A-Za-z]", "", line[12:16]).upper()[:1]
        atoms.append(PdbAtom(
            serial=int(line[6:11]),
            name=line[12:16].strip(),
            altloc=line[16].strip(),
            residue_name=line[17:20].strip(),
            chain=line[21].strip(),
            residue_number=int(line[22:26]),
            insertion_code=line[26].strip(),
            occupancy=float(line[54:60]),
            element=element,
            xyz=np.array([
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
            ]),
        ))
    if not atoms:
        raise ValueError(f"{path}: no PDB atoms found")
    return atoms


def select_altloc_atoms(atoms: Sequence[PdbAtom]) -> list[PdbAtom]:
    grouped: dict[tuple[str, int, str, str, str], list[PdbAtom]] = {}
    for atom in atoms:
        key = (
            atom.chain, atom.residue_number, atom.insertion_code,
            atom.residue_name, atom.name,
        )
        grouped.setdefault(key, []).append(atom)
    selected = []
    for candidates in grouped.values():
        blank = [atom for atom in candidates if not atom.altloc]
        if blank:
            selected.append(blank[0])
        else:
            selected.append(max(
                candidates,
                key=lambda atom: (atom.occupancy, atom.altloc == "A", -atom.serial),
            ))
    return sorted(selected, key=lambda atom: atom.serial)


def residue_atoms(
    atoms: Sequence[PdbAtom], chain: str, residue_number: int,
) -> list[PdbAtom]:
    selected = [
        atom for atom in atoms
        if atom.chain == chain and atom.residue_number == residue_number
        and not atom.insertion_code and atom.residue_name != "ZN"
    ]
    if not selected:
        raise ValueError(f"residue {chain}:{residue_number} is absent")
    return selected


def center_of_mass(atoms: Sequence[PdbAtom]) -> tuple[np.ndarray, float]:
    try:
        masses = np.array([ATOMIC_MASSES[atom.element] for atom in atoms])
    except KeyError as exc:
        raise ValueError(f"unsupported PDB element for COM: {exc.args[0]}") from exc
    xyz = np.array([atom.xyz for atom in atoms])
    total_mass = float(np.sum(masses))
    return np.sum(xyz * masses[:, None], axis=0) / total_mass, total_mass


def schrodinger_signed_dihedral_deg(
    atom1: np.ndarray, atom2: np.ndarray,
    atom3: np.ndarray, atom4: np.ndarray,
) -> float:
    """Match the signed four-point convention used by Desmond analysis."""
    r12 = atom1 - atom2
    r32 = atom3 - atom2
    r42 = atom4 - atom2
    cross_1232 = np.cross(r12, r32)
    cross_4232 = np.cross(r42, r32)
    denominator = np.linalg.norm(cross_1232) * np.linalg.norm(cross_4232)
    if denominator == 0.0:
        raise ValueError("cannot calculate dihedral from collinear coordinates")
    cosine = float(np.clip(np.dot(cross_1232, cross_4232) / denominator, -1.0, 1.0))
    angle = math.degrees(math.acos(cosine))
    if float(np.dot(r12, cross_4232)) < 0.0:
        angle *= -1.0
    return angle


def compute_reference_metrics(path: Path, requested_chain: str | None) -> ReferenceMetrics:
    atoms = select_altloc_atoms(parse_pdb_atoms(path))
    candidate_chains = sorted({atom.chain for atom in atoms})
    if requested_chain:
        chain = requested_chain
    else:
        compatible = []
        for candidate in candidate_chains:
            residue_numbers = {
                atom.residue_number for atom in atoms if atom.chain == candidate
            }
            if {49, 102, 107}.issubset(residue_numbers):
                compatible.append(candidate)
        if len(compatible) != 1:
            raise ValueError(
                "reference chain is ambiguous; use --reference-chain "
                f"(compatible chains: {compatible})"
            )
        chain = compatible[0]

    d49 = residue_atoms(atoms, chain, 49)
    r102 = residue_atoms(atoms, chain, 102)
    q107 = residue_atoms(atoms, chain, 107)
    if d49[0].residue_name != "ASP" or r102[0].residue_name != "ARG" or q107[0].residue_name != "GLN":
        raise ValueError(
            f"reference residue identities are not ASP49/ARG102/GLN107 in chain {chain}"
        )

    d49_com, d49_mass = center_of_mass(d49)
    r102_com, r102_mass = center_of_mass(r102)
    d49_heavy_com, _ = center_of_mass([atom for atom in d49 if atom.element != "H"])
    r102_heavy_com, _ = center_of_mass([atom for atom in r102 if atom.element != "H"])
    q107_by_name = {atom.name: atom for atom in q107}
    d49_by_name = {atom.name: atom for atom in d49}
    r102_by_name = {atom.name: atom for atom in r102}
    chi_atoms = ["N", "CA", "CB", "CG"]
    missing = [name for name in chi_atoms if name not in q107_by_name]
    if missing:
        raise ValueError(f"Q107 is missing chi1 atoms: {missing}")
    chi1 = schrodinger_signed_dihedral_deg(
        *[q107_by_name[name].xyz for name in chi_atoms]
    )

    return ReferenceMetrics(
        pdb_path=path,
        chain=chain,
        d49_com=d49_com,
        r102_com=r102_com,
        distance_A=float(np.linalg.norm(d49_com - r102_com)),
        chi1_deg=float(chi1),
        d49_atom_count=len(d49),
        r102_atom_count=len(r102),
        d49_total_mass_Da=d49_mass,
        r102_total_mass_Da=r102_mass,
        heavy_atom_distance_A=float(np.linalg.norm(d49_heavy_com - r102_heavy_com)),
        ca_distance_A=float(np.linalg.norm(
            d49_by_name["CA"].xyz - r102_by_name["CA"].xyz
        )),
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_surface(path: Path) -> Surface:
    header_lines: list[str] = []
    with path.open() as handle:
        for line in handle:
            if line.startswith("#"):
                header_lines.append(line.strip())
            elif line.strip():
                break

    shape = None
    units = None
    columns = None
    for line in header_lines:
        shape_match = re.match(r"# FES shape:\s+(\d+)\s+(\d+)", line)
        if shape_match:
            shape = (int(shape_match.group(1)), int(shape_match.group(2)))
        units_match = re.match(r"# FES units:\s+(.+)", line)
        if units_match:
            units = units_match.group(1).strip()
        if line.startswith("# dist "):
            columns = line.removeprefix("# ")

    data = np.loadtxt(path, comments="#")
    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError(f"{path}: expected three numeric columns")
    if not np.isfinite(data).all():
        raise ValueError(f"{path}: grid contains non-finite values")

    distance = np.unique(data[:, 0])
    chi = np.unique(data[:, 1])
    expected = distance.size * chi.size
    if data.shape[0] != expected:
        raise ValueError(f"{path}: grid is incomplete ({data.shape[0]} != {expected})")
    if shape is not None and shape != (distance.size, chi.size):
        raise ValueError(f"{path}: declared shape {shape} disagrees with data")

    order = np.lexsort((data[:, 1], data[:, 0]))
    sorted_data = data[order]
    raw = sorted_data[:, 2].reshape(distance.size, chi.size)
    expected_distance, expected_chi = np.meshgrid(distance, chi, indexing="ij")
    if not np.allclose(sorted_data[:, 0], expected_distance.ravel()):
        raise ValueError(f"{path}: distance grid ordering is inconsistent")
    if not np.allclose(sorted_data[:, 1], expected_chi.ravel()):
        raise ValueError(f"{path}: angular grid ordering is inconsistent")

    if not (math.isclose(chi[0], -180.0, abs_tol=1e-8) and
            math.isclose(chi[-1], 180.0, abs_tol=1e-8)):
        raise ValueError(f"{path}: expected periodic -180/+180 endpoints")
    endpoint_error = float(np.max(np.abs(raw[:, 0] - raw[:, -1])))
    if endpoint_error > 1e-8:
        raise ValueError(f"{path}: periodic endpoint mismatch {endpoint_error:g}")

    return Surface(path, distance, chi, raw, shape, units, columns)


def configure_matplotlib() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.0,
        "axes.labelsize": 10.0,
        "axes.titlesize": 10.0,
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


def build_common_surfaces(
    surfaces: Sequence[Surface],
    temperature_k: float,
    ktemp_kcal_mol: float,
    n_distance: int,
) -> dict[str, np.ndarray | float]:
    distance_min = max(surface.distance[0] for surface in surfaces)
    distance_max = min(surface.distance[-1] for surface in surfaces)
    if distance_min >= distance_max:
        raise ValueError("replicate distance grids have no common interval")

    distance = np.linspace(distance_min, distance_max, n_distance)
    # Canonical integration grid has one copy of every periodic angular point.
    chi = np.arange(-180.0, 180.0, 1.0)
    d_grid, c_grid = np.meshgrid(distance, chi, indexing="ij")
    query = np.column_stack((d_grid.ravel(), c_grid.ravel()))

    kbt = KB_KCAL_PER_MOL_K * temperature_k
    gamma = 1.0 + ktemp_kcal_mol / kbt
    wt_scale = gamma / (gamma - 1.0)

    relative = []
    log_probabilities = []
    for surface in surfaces:
        interpolator = RegularGridInterpolator(
            (surface.distance, surface.chi), surface.raw,
            method="linear", bounds_error=True,
        )
        interpolated = interpolator(query).reshape(d_grid.shape)
        rel = (interpolated - np.min(interpolated)) * wt_scale
        relative.append(rel)
        log_probability = -rel / kbt
        log_probability -= logsumexp(log_probability)
        log_probabilities.append(log_probability)

    relative_array = np.asarray(relative)
    log_probability_array = np.asarray(log_probabilities)
    mixture_log_probability = (
        logsumexp(log_probability_array, axis=0) - math.log(len(surfaces))
    )
    mixture_fes = -kbt * mixture_log_probability
    mixture_fes -= np.min(mixture_fes)

    mean_fes = np.mean(relative_array, axis=0)
    mean_fes -= np.min(mean_fes)

    return {
        "distance": distance,
        "chi": chi,
        "relative": relative_array,
        "log_probabilities": log_probability_array,
        "mixture_log_probability": mixture_log_probability,
        "mixture_fes": mixture_fes,
        "mean_fes": mean_fes,
        "kbt": kbt,
        "gamma": gamma,
        "wt_scale": wt_scale,
        "distance_min": distance_min,
        "distance_max": distance_max,
    }


def shifted_periodic_view(chi: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Move the angular seam from trans to -120/240 degrees."""
    display_chi = np.arange(-120.0, 241.0, 1.0)
    wrapped = ((display_chi + 180.0) % 360.0) - 180.0
    lookup = {float(angle): idx for idx, angle in enumerate(chi)}
    indices = np.array([lookup[float(angle)] for angle in wrapped], dtype=int)
    return display_chi, values[..., indices]


def color_setup(cap: float, interval: float):
    boundaries = np.arange(0.0, cap + interval * 0.5, interval)
    if boundaries[-1] < cap:
        boundaries = np.append(boundaries, cap)
    cmap = LinearSegmentedColormap.from_list("muted_blue_red", PALETTE, N=256)
    norm = BoundaryNorm(boundaries, cmap.N, clip=False)
    return boundaries, cmap, norm


def style_axis(ax: plt.Axes) -> None:
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(0.9)
    ax.tick_params(top=False, right=False, length=4)


def draw_surface(
    ax: plt.Axes,
    distance: np.ndarray,
    display_chi: np.ndarray,
    display_fes: np.ndarray,
    boundaries: np.ndarray,
    cmap: LinearSegmentedColormap,
    norm: BoundaryNorm,
    cap: float,
):
    filled = ax.contourf(
        distance, display_chi, display_fes.T,
        levels=boundaries, cmap=cmap, norm=norm, extend="max",
        antialiased=True,
    )
    contour_levels = np.arange(2.5, cap + 0.01, 2.5)
    ax.contour(
        distance, display_chi, display_fes.T,
        levels=contour_levels, colors="#5E5E5E", linewidths=0.55,
        alpha=0.75,
    )
    ax.set_xlim(distance[0], distance[-1])
    ax.set_ylim(-120.0, 240.0)
    ax.set_xticks([6, 10, 15, 20, 25, 30, 35, 39])
    ax.set_yticks([-120, -60, 0, 60, 120, 180, 240])
    style_axis(ax)
    return filled


def display_chi_value(chi_deg: float) -> float:
    displayed = ((chi_deg + 120.0) % 360.0) - 120.0
    if math.isclose(displayed, -120.0, abs_tol=1e-10) and chi_deg > 0:
        return 240.0
    return displayed


def draw_reference(ax: plt.Axes, reference: ReferenceMetrics) -> None:
    chi_display = display_chi_value(reference.chi1_deg)
    dash = (0, (5, 4))
    # Dark under-strokes keep the white reference visible in pale regions.
    ax.axvline(reference.distance_A, color="#333333", linewidth=1.8,
               linestyle=dash, alpha=0.70, zorder=7)
    ax.axhline(chi_display, color="#333333", linewidth=1.8,
               linestyle=dash, alpha=0.70, zorder=7)
    ax.axvline(reference.distance_A, color="white", linewidth=1.05,
               linestyle=dash, zorder=8)
    ax.axhline(chi_display, color="white", linewidth=1.05,
               linestyle=dash, zorder=8)
    ax.plot(reference.distance_A, chi_display, marker="o", markersize=4.2,
            markerfacecolor="white", markeredgecolor="#333333",
            markeredgewidth=0.75, zorder=9)
    label = ax.text(
        reference.distance_A + 0.45, chi_display + 10.0, "4XR8",
        color="white", fontsize=8.0, fontweight="semibold",
        ha="left", va="bottom", zorder=10,
    )
    label.set_path_effects([
        path_effects.withStroke(linewidth=1.5, foreground="#333333"),
    ])


def save_main_figure(
    out_png: Path,
    out_pdf: Path,
    distance: np.ndarray,
    chi: np.ndarray,
    mixture_fes: np.ndarray,
    reference: ReferenceMetrics | None,
    cap: float,
    interval: float,
) -> None:
    display_chi, display_fes = shifted_periodic_view(chi, mixture_fes)
    boundaries, cmap, norm = color_setup(cap, interval)

    fig, ax = plt.subplots(figsize=(7.2, 5.35), constrained_layout=False)
    fig.subplots_adjust(left=0.14, right=0.81, bottom=0.16, top=0.96)
    filled = draw_surface(
        ax, distance, display_chi, display_fes,
        boundaries, cmap, norm, cap,
    )
    ax.set_xlabel("D49–R102 residue COM distance (Å)", labelpad=7)
    ax.set_ylabel("Q107 χ₁ dihedral angle (°)", labelpad=8)
    if reference is not None:
        draw_reference(ax, reference)

    colorbar = fig.colorbar(
        filled, ax=ax, pad=0.045, fraction=0.06,
        ticks=np.arange(0, cap + 0.1, 3.0), extendfrac=0.05,
        extendrect=True,
    )
    colorbar.set_label("Relative free energy (kcal mol⁻¹)", labelpad=9)
    colorbar.outline.set_linewidth(0.8)
    ax.text(
        1.0, 1.018, "periodic seam: −120° ≡ 240°",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=7.7, color="#555555",
    )

    fig.savefig(out_png, dpi=600, bbox_inches="tight", pad_inches=0.035)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)


def save_diagnostic_figure(
    out_png: Path,
    distance: np.ndarray,
    chi: np.ndarray,
    relative: np.ndarray,
    mixture_fes: np.ndarray,
    cap: float,
    interval: float,
) -> None:
    display_chi, replicate_display = shifted_periodic_view(chi, relative)
    _, mixture_display = shifted_periodic_view(chi, mixture_fes)
    boundaries, cmap, norm = color_setup(cap, interval)

    surfaces = [*replicate_display, mixture_display]
    panel_count = len(surfaces)
    ncols = min(4, panel_count)
    nrows = math.ceil(panel_count / ncols)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(3.35 * ncols, 3.25 * nrows + 0.3),
        sharex=True, sharey=True, squeeze=False,
    )
    fig.subplots_adjust(left=0.065, right=0.90, bottom=0.09, top=0.94,
                        wspace=0.12, hspace=0.20)
    titles = [f"Replicate {idx}" for idx in range(1, len(relative) + 1)]
    titles.append("Equal-replicate mixture")
    filled = None
    for idx, (ax, panel_fes, title) in enumerate(zip(axes.flat, surfaces, titles)):
        filled = draw_surface(
            ax, distance, display_chi, panel_fes,
            boundaries, cmap, norm, cap,
        )
        ax.set_title(
            title, pad=5,
            fontweight="semibold" if idx == panel_count - 1 else "normal",
        )
        if idx % ncols == 0:
            ax.set_ylabel("Q107 χ₁ (°)")
        if idx // ncols == nrows - 1:
            ax.set_xlabel("D49–R102 COM distance (Å)")

    for ax in axes.flat[panel_count:]:
        ax.set_visible(False)

    colorbar_ax = fig.add_axes([0.92, 0.20, 0.018, 0.60])
    colorbar = fig.colorbar(
        filled, cax=colorbar_ax,
        ticks=np.arange(0, cap + 0.1, 3.0), extendfrac=0.05,
        extendrect=True,
    )
    colorbar.set_label("Relative free energy (kcal mol⁻¹)", labelpad=8)
    colorbar.outline.set_linewidth(0.8)
    fig.text(0.90, 0.963, "−120° ≡ 240°", ha="right", va="top",
             fontsize=7.7, color="#555555")
    fig.savefig(out_png, dpi=500, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def distance_profiles(
    distance: np.ndarray,
    kbt: float,
    replicate_log_probabilities: np.ndarray,
    mixture_log_probability: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    replicate_profiles = []
    for log_probability in replicate_log_probabilities:
        profile = -kbt * logsumexp(log_probability, axis=1)
        profile -= np.min(profile)
        replicate_profiles.append(profile)
    mixture_profile = -kbt * logsumexp(mixture_log_probability, axis=1)
    mixture_profile -= np.min(mixture_profile)
    return np.asarray(replicate_profiles), mixture_profile


def find_distance_minima(distance: np.ndarray, profile: np.ndarray) -> list[dict[str, float]]:
    spacing = float(np.mean(np.diff(distance)))
    smooth = gaussian_filter1d(profile, sigma=0.7 / spacing, mode="nearest")
    peaks, properties = find_peaks(
        -smooth,
        prominence=0.25,
        distance=max(1, int(round(2.5 / spacing))),
    )
    minima = []
    for peak, prominence in zip(peaks, properties["prominences"]):
        minima.append({
            "distance_A": float(distance[peak]),
            "smoothed_relative_free_energy_kcal_mol": float(smooth[peak]),
            "prominence_kcal_mol": float(prominence),
        })
    return minima


def find_periodic_2d_minima(
    distance: np.ndarray,
    chi: np.ndarray,
    fes: np.ndarray,
) -> list[dict[str, float]]:
    distance_spacing = float(np.mean(np.diff(distance)))
    chi_spacing = float(np.mean(np.diff(chi)))
    smoothed = gaussian_filter(
        fes,
        sigma=(0.5 / distance_spacing, 7.2 / chi_spacing),
        mode=("nearest", "wrap"),
    )
    distance_window = max(3, int(round(4.0 / distance_spacing)))
    if distance_window % 2 == 0:
        distance_window += 1
    chi_window = max(3, int(round(80.0 / chi_spacing)))
    if chi_window % 2 == 0:
        chi_window += 1
    local = smoothed == minimum_filter(
        smoothed,
        size=(distance_window, chi_window),
        mode=("nearest", "wrap"),
    )
    indices = np.argwhere(local & (smoothed < 6.0))
    minima = []
    for d_idx, c_idx in sorted(indices, key=lambda item: smoothed[tuple(item)]):
        canonical_chi = float(chi[c_idx])
        display_chi = canonical_chi if canonical_chi >= -120.0 else canonical_chi + 360.0
        minima.append({
            "distance_A": float(distance[d_idx]),
            "chi1_canonical_deg": canonical_chi,
            "chi1_display_deg": display_chi,
            "smoothed_relative_free_energy_kcal_mol": float(smoothed[d_idx, c_idx]),
        })
    return minima


def write_grid_csv(path: Path, distance: np.ndarray, chi: np.ndarray, fes: np.ndarray) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["distance_A", "chi1_deg", "relative_free_energy_kcal_mol"])
        for d_idx, d_value in enumerate(distance):
            for c_idx, c_value in enumerate(chi):
                writer.writerow([f"{d_value:.8f}", f"{c_value:.1f}", f"{fes[d_idx, c_idx]:.8f}"])


def write_profile_csv(
    path: Path,
    distance: np.ndarray,
    replicate_profiles: np.ndarray,
    mixture_profile: np.ndarray,
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        replicate_headers = [
            f"replicate_{idx}" for idx in range(1, len(replicate_profiles) + 1)
        ]
        writer.writerow(["distance_A", *replicate_headers, "equal_replicate_mixture"])
        for idx, d_value in enumerate(distance):
            writer.writerow([
                f"{d_value:.8f}",
                *[f"{profile[idx]:.8f}" for profile in replicate_profiles],
                f"{mixture_profile[idx]:.8f}",
            ])


def write_dict_csv(path: Path, rows: Sequence[dict[str, float]]) -> None:
    if not rows:
        raise ValueError(f"no rows available for {path.name}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_manifest(
    surfaces: Sequence[Surface],
    common: dict[str, np.ndarray | float],
    distance_minima: Sequence[dict[str, float]],
    minima_2d: Sequence[dict[str, float]],
    temperature_k: float,
    ktemp_kcal_mol: float,
    cap: float,
    reference: ReferenceMetrics | None,
) -> dict:
    input_rows = []
    for idx, surface in enumerate(surfaces, start=1):
        minimum_index = np.unravel_index(np.argmin(surface.raw), surface.raw.shape)
        input_rows.append({
            "replicate": idx,
            "path": str(surface.path),
            "sha256": sha256(surface.path),
            "declared_shape": surface.declared_shape,
            "declared_units": surface.declared_units,
            "columns": surface.columns,
            "distance_range": [float(surface.distance[0]), float(surface.distance[-1])],
            "distance_spacing": float(np.mean(np.diff(surface.distance))),
            "chi_range": [float(surface.chi[0]), float(surface.chi[-1])],
            "chi_spacing": float(np.mean(np.diff(surface.chi))),
            "raw_value_range": [float(np.min(surface.raw)), float(np.max(surface.raw))],
            "raw_global_minimum": {
                "distance": float(surface.distance[minimum_index[0]]),
                "chi1_deg": float(surface.chi[minimum_index[1]]),
            },
            "periodic_endpoint_max_abs_difference": float(
                np.max(np.abs(surface.raw[:, 0] - surface.raw[:, -1]))
            ),
        })

    distance = common["distance"]
    mixture_profile_value_at_21 = None
    if float(distance[0]) <= 21.0 <= float(distance[-1]):
        replicate_profiles, mixture_profile = distance_profiles(
            distance,
            float(common["kbt"]),
            common["log_probabilities"],
            common["mixture_log_probability"],
        )
        del replicate_profiles
        mixture_profile_value_at_21 = float(np.interp(21.0, distance, mixture_profile))

    replicate_count = len(surfaces)
    manifest = {
        "inputs": input_rows,
        "coordinate_labels": {
            "distance": "D49-R102 residue center-of-mass distance",
            "dihedral": "Q107 chi1",
        },
        "well_tempered_conversion": {
            "temperature_K": temperature_k,
            "kTemp_kcal_mol": ktemp_kcal_mol,
            "kBT_kcal_mol": float(common["kbt"]),
            "gamma": float(common["gamma"]),
            "negative_bias_to_fes_scale_gamma_over_gamma_minus_one": float(common["wt_scale"]),
        },
        "combination": {
            "name": "equal-replicate probability mixture",
            "formula": f"normalize exp(-F_i/kBT) on common grid; average {replicate_count} probabilities; convert with -kBT ln(p)",
            "weights": [1.0 / replicate_count] * replicate_count,
            "common_distance_range_A": [float(common["distance_min"]), float(common["distance_max"])],
            "distance_grid_points": int(common["distance"].size),
            "chi_grid": "-180 through 179 degrees at 1 degree spacing; +180 duplicate dropped",
            "display_seam": "-120 degrees equivalent to 240 degrees",
        },
        "plot": {
            "palette": PALETTE,
            "low_energy_color": "blue",
            "high_energy_color": "red",
            "color_cap_kcal_mol": cap,
            "values_above_cap": "shown with the colorbar over-range extension",
        },
        "distance_profile_minima": list(distance_minima),
        "periodic_2d_minima_below_6_kcal_mol": list(minima_2d),
        "approximately_21_A_check": {
            "local_minimum_detected_between_20_and_22_A": any(
                20.0 <= row["distance_A"] <= 22.0 for row in distance_minima
            ),
            "mixture_profile_relative_free_energy_at_21_A_kcal_mol": mixture_profile_value_at_21,
        },
        "limitations": [
            "The terminal Desmond grids do not contain trajectory-level reweighting factors or time-resolved convergence information.",
            "The combined surface is a descriptive equal-replicate probability mixture, not a formally pooled trajectory FES.",
            f"The {replicate_count} replicate surfaces should be inspected because their global minima occur in different distance regions.",
            "The files identify angular output as degrees but do not explicitly state the energy unit; kcal/mol follows the production kTemp and Desmond energy convention and should be checked against the archived production input.",
        ],
    }
    if reference is not None:
        manifest["prepared_4XR8_reference"] = {
            "path": str(reference.pdb_path),
            "sha256": sha256(reference.pdb_path),
            "chain": reference.chain,
            "selection": {
                "distance": "all atoms present in ASP49 and ARG102 after selecting one altloc per atom; standard element masses",
                "chi1": "GLN107 N-CA-CB-CG",
                "altloc_policy": "blank altloc preferred; otherwise highest occupancy, with A preferred on a tie",
            },
            "D49_atom_count": reference.d49_atom_count,
            "R102_atom_count": reference.r102_atom_count,
            "D49_total_mass_Da": reference.d49_total_mass_Da,
            "R102_total_mass_Da": reference.r102_total_mass_Da,
            "D49_COM_A": reference.d49_com.tolist(),
            "R102_COM_A": reference.r102_com.tolist(),
            "D49_R102_mass_weighted_all_atom_COM_distance_A": reference.distance_A,
            "diagnostic_heavy_atom_COM_distance_A": reference.heavy_atom_distance_A,
            "diagnostic_CA_distance_A": reference.ca_distance_A,
            "Q107_chi1_deg": reference.chi1_deg,
            "chi1_atom_order": ["N", "CA", "CB", "CG"],
            "chi1_sign_convention": "Schrodinger/Desmond signed four-point torsion",
        }
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Desmond .fes files in replicate order")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-pdb", type=Path)
    parser.add_argument("--reference-chain", type=str)
    parser.add_argument("--temperature", type=float, default=310.0, help="simulation temperature in K")
    parser.add_argument("--ktemp", type=float, default=8.4, help="Desmond kTemp = k_B DeltaT in kcal/mol")
    parser.add_argument("--distance-points", type=int, default=333)
    parser.add_argument("--color-cap", type=float, default=15.0)
    parser.add_argument("--color-interval", type=float, default=1.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.temperature <= 0 or args.ktemp <= 0:
        raise ValueError("temperature and kTemp must be positive")
    if args.distance_points < 51:
        raise ValueError("use at least 51 distance grid points")
    if len(args.inputs) < 2:
        raise ValueError("provide at least two replicate FES files")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    surfaces = [parse_surface(path.resolve()) for path in args.inputs]
    reference = None
    if args.reference_pdb is not None:
        reference = compute_reference_metrics(
            args.reference_pdb.resolve(), args.reference_chain,
        )
    common = build_common_surfaces(
        surfaces, args.temperature, args.ktemp, args.distance_points,
    )

    stem = f"apo16e6_wtmetad_{len(surfaces)}rep"
    reference_suffix = "_4xr8_reference" if reference is not None else ""
    main_png = args.output_dir / f"{stem}_combined_fes_blue_red{reference_suffix}.png"
    main_pdf = args.output_dir / f"{stem}_combined_fes_blue_red{reference_suffix}.pdf"
    diagnostics_png = args.output_dir / f"{stem}_replicate_diagnostics_blue_red.png"
    grid_csv = args.output_dir / f"{stem}_combined_grid.csv"
    profiles_csv = args.output_dir / f"{stem}_distance_profiles.csv"
    distance_minima_csv = args.output_dir / f"{stem}_distance_minima.csv"
    minima_2d_csv = args.output_dir / f"{stem}_periodic_2d_minima.csv"
    arrays_npz = args.output_dir / f"{stem}_processed_arrays.npz"
    manifest_json = args.output_dir / f"{stem}_manifest.json"
    reference_json = args.output_dir / f"{stem}_prepared_4xr8_reference_metrics.json"

    save_main_figure(
        main_png, main_pdf,
        common["distance"], common["chi"], common["mixture_fes"],
        reference, args.color_cap, args.color_interval,
    )
    save_diagnostic_figure(
        diagnostics_png,
        common["distance"], common["chi"], common["relative"],
        common["mixture_fes"], args.color_cap, args.color_interval,
    )

    replicate_profiles, mixture_profile = distance_profiles(
        common["distance"], common["kbt"], common["log_probabilities"],
        common["mixture_log_probability"],
    )
    distance_minima = find_distance_minima(common["distance"], mixture_profile)
    minima_2d = find_periodic_2d_minima(
        common["distance"], common["chi"], common["mixture_fes"],
    )

    write_grid_csv(grid_csv, common["distance"], common["chi"], common["mixture_fes"])
    write_profile_csv(profiles_csv, common["distance"], replicate_profiles, mixture_profile)
    write_dict_csv(distance_minima_csv, distance_minima)
    write_dict_csv(minima_2d_csv, minima_2d)
    np.savez_compressed(
        arrays_npz,
        distance=common["distance"],
        chi=common["chi"],
        replicate_relative_fes=common["relative"],
        combined_probability_mixture_fes=common["mixture_fes"],
        arithmetic_mean_fes_sensitivity=common["mean_fes"],
        replicate_distance_profiles=replicate_profiles,
        combined_distance_profile=mixture_profile,
    )

    manifest = build_manifest(
        surfaces, common, distance_minima, minima_2d,
        args.temperature, args.ktemp, args.color_cap, reference,
    )
    manifest["outputs"] = {
        "main_png": str(main_png.resolve()),
        "main_pdf": str(main_pdf.resolve()),
        "diagnostics_png": str(diagnostics_png.resolve()),
        "combined_grid_csv": str(grid_csv.resolve()),
        "distance_profiles_csv": str(profiles_csv.resolve()),
        "distance_minima_csv": str(distance_minima_csv.resolve()),
        "periodic_2d_minima_csv": str(minima_2d_csv.resolve()),
        "processed_arrays_npz": str(arrays_npz.resolve()),
    }
    if reference is not None:
        reference_payload = manifest["prepared_4XR8_reference"]
        reference_json.write_text(json.dumps(reference_payload, indent=2) + "\n")
        manifest["outputs"]["reference_metrics_json"] = str(reference_json.resolve())
    manifest_json.write_text(json.dumps(manifest, indent=2) + "\n")

    print(json.dumps({
        "main_png": str(main_png),
        "main_pdf": str(main_pdf),
        "diagnostics_png": str(diagnostics_png),
        "distance_minima": distance_minima,
        "periodic_2d_minima_count": len(minima_2d),
        "gamma": common["gamma"],
        "wt_scale": common["wt_scale"],
        "reference": None if reference is None else {
            "chain": reference.chain,
            "D49_R102_COM_distance_A": reference.distance_A,
            "Q107_chi1_deg": reference.chi1_deg,
        },
    }, indent=2))


if __name__ == "__main__":
    main()
