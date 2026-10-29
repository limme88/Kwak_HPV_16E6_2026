#!/usr/bin/env python3
"""Compute common E6 structural descriptors with the original residue mappings.
Inputs: unbiased MD, MSM assignments, BioEmu conformers, 4XR8 and 7UAJ.
Outputs: per-frame descriptors, ensemble summaries and representative structures.
Supports V12 Table S1 and numerical comparisons in Figures 5 and 7.
Run from the repository root; see command-line options for all input paths.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import mdtraj as md
import numpy as np


E6HL = tuple(range(62, 80))
N_WALL = (31, 32, 33, 45, 48, 49, 50, 51)
C_WALL = (100, 101, 102, 106, 107, 109, *range(127, 134))
COMMON_APERTURE_ATOMS = frozenset({"N", "CA", "C", "O", "CB"})
METRIC_ORDER = (
    "e6hl_helix_fraction_H_or_G",
    "d49_r102_ca_distance_A",
    "q107_chi1_deg",
    "tail_ca130_ca151_distance_A",
    "lxxll_aperture_all_heavy_A",
    "lxxll_aperture_common_N_CA_C_O_CB_A",
)
ANGLE_METRIC = "q107_chi1_deg"


@dataclass(frozen=True)
class ResidueMap:
    name: str
    chain_id: str
    resseq_offset: int

    def resseq(self, e6_residue: int) -> int:
        return e6_residue + self.resseq_offset


MD_MAP = ResidueMap("151-residue MD numbering", "A", 0)
FOUR_XR8_MAP = ResidueMap("4XR8 E6 chain H", "H", 0)
SEVEN_UAJ_MAP = ResidueMap("7UAJ fusion chain D", "D", 378)
BIOEMU_MAP = ResidueMap("158-residue BioEmu sequence", "A", 6)


METRIC_DEFINITIONS = [
    {
        "metric": "e6hl_helix_fraction_H_or_G",
        "display_name": "E6HL helix fraction",
        "units": "fraction",
        "definition": (
            "Number of manuscript E6 residues 62-79 assigned DSSP H or G, "
            "divided by 18. DSSP I is not counted in this common metric."
        ),
    },
    {
        "metric": "d49_r102_ca_distance_A",
        "display_name": "D49-R102 C-alpha distance",
        "units": "angstrom",
        "definition": "Euclidean distance between the D49 and R102 C-alpha atoms.",
    },
    {
        "metric": "q107_chi1_deg",
        "display_name": "Q107 chi1",
        "units": "degree",
        "definition": (
            "Signed N-CA-CB-CG torsion in [-180, 180] degrees using the "
            "Desmond/Schrodinger sign convention. Ensemble means are circular."
        ),
    },
    {
        "metric": "tail_ca130_ca151_distance_A",
        "display_name": "C-terminal-tail extension",
        "units": "angstrom",
        "definition": (
            "Euclidean C-alpha(130)-C-alpha(151) extension coordinate. The "
            "structural tail is residues 140-151; residue 130 is the core anchor."
        ),
    },
    {
        "metric": "lxxll_aperture_all_heavy_A",
        "display_name": "LxxLL groove aperture (angstroms)",
        "units": "angstrom",
        "definition": (
            "Mean of 104 per-residue-pair minimum heavy-atom distances: all "
            "8 N-wall residues x all 13 C-wall residues. N wall = 31,32,33,45,"
            "48-51; C wall = 100,101,102,106,107,109,127-133."
        ),
    },
    {
        "metric": "lxxll_aperture_common_N_CA_C_O_CB_A",
        "display_name": "Common-atom aperture surrogate (angstroms)",
        "units": "angstrom",
        "definition": (
            "Same 104 residue pairs as the primary aperture, but distances use "
            "only N, CA, C, O, and CB atoms (CB absent for glycine). This is an "
            "atom-matched surrogate, not the manuscript all-heavy-atom metric."
        ),
    },
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def protein_chain(topology: md.Topology, chain_id: str):
    matches = []
    for chain in topology.chains:
        residues = [r for r in chain.residues if r.is_protein]
        if chain.chain_id == chain_id and residues:
            matches.append((chain, residues))
    if len(matches) != 1:
        raise ValueError(
            f"Expected one protein chain with author chain ID {chain_id!r}; "
            f"found {len(matches)}"
        )
    return matches[0]


def protein_sequence(topology: md.Topology, chain_id: str) -> str:
    _, residues = protein_chain(topology, chain_id)
    sequence = []
    for residue in residues:
        code = residue.code
        if code is None or len(code) != 1:
            raise ValueError(f"No one-letter code for protein residue {residue}")
        sequence.append(code)
    return "".join(sequence)


def find_residue(topology: md.Topology, mapping: ResidueMap, e6_residue: int):
    target = mapping.resseq(e6_residue)
    matches = [
        residue
        for residue in topology.residues
        if residue.is_protein
        and residue.chain.chain_id == mapping.chain_id
        and residue.resSeq == target
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one mapped E6 residue {e6_residue} at "
            f"{mapping.chain_id}:{target}; found {len(matches)}"
        )
    return matches[0]


def optional_residue(
    topology: md.Topology, mapping: ResidueMap, e6_residue: int
):
    try:
        return find_residue(topology, mapping, e6_residue)
    except ValueError:
        return None


def atom_index(residue, atom_name: str) -> int:
    matches = [atom.index for atom in residue.atoms if atom.name == atom_name]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one {atom_name} atom in {residue}; found {len(matches)}"
        )
    return matches[0]


def optional_atom_index(residue, atom_name: str) -> int | None:
    if residue is None:
        return None
    matches = [atom.index for atom in residue.atoms if atom.name == atom_name]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        return None
    raise ValueError(f"Multiple {atom_name} atoms in {residue}")


def heavy_atom_names(residue) -> set[str]:
    return {
        atom.name
        for atom in residue.atoms
        if atom.element is not None and atom.element.symbol != "H"
    }


def verify_sequences(
    md_topology: md.Topology,
    four_xr8_topology: md.Topology,
    seven_uaj_topology: md.Topology,
    bioemu_topology: md.Topology,
) -> dict:
    reference = protein_sequence(md_topology, MD_MAP.chain_id)
    four_xr8 = protein_sequence(four_xr8_topology, FOUR_XR8_MAP.chain_id)
    seven_uaj = protein_sequence(seven_uaj_topology, SEVEN_UAJ_MAP.chain_id)
    bioemu = protein_sequence(bioemu_topology, BIOEMU_MAP.chain_id)

    if len(reference) != 151:
        raise ValueError(f"MD reference sequence has {len(reference)} residues, not 151")
    if four_xr8 != reference:
        raise ValueError("4XR8 E6 chain H does not exactly match the MD sequence")
    if len(bioemu) != 158 or bioemu[7:] != reference:
        raise ValueError(
            "BioEmu sequence is not the expected seven-residue extension plus "
            "the exact 151-residue MD sequence"
        )

    _, seven_residues = protein_chain(seven_uaj_topology, SEVEN_UAJ_MAP.chain_id)
    seven_start = seven_residues[0].resSeq - SEVEN_UAJ_MAP.resseq_offset
    seven_stop = seven_residues[-1].resSeq - SEVEN_UAJ_MAP.resseq_offset
    seven_missing = sorted(
        set(range(seven_start, seven_stop + 1))
        - {
            residue.resSeq - SEVEN_UAJ_MAP.resseq_offset
            for residue in seven_residues
        }
    )
    seven_mismatches = []
    for residue in seven_residues:
        e6_number = residue.resSeq - SEVEN_UAJ_MAP.resseq_offset
        if residue.code != reference[e6_number - 1]:
            seven_mismatches.append(
                {
                    "e6_residue": e6_number,
                    "observed": residue.code,
                    "expected": reference[e6_number - 1],
                }
            )
    # 7UAJ uses the known C80S/C97S/C111S/C140S soluble E6 variant.  Preserve
    # these resolved sequence differences in provenance rather than treating
    # the crystal file as the exact wild-type sequence.

    return {
        "reference_length": len(reference),
        "reference_sequence": reference,
        "four_xr8_exact_match": True,
        "bioemu_length": len(bioemu),
        "bioemu_n_terminal_extension": bioemu[:7],
        "bioemu_suffix_exact_match": True,
        "bioemu_mapping": "manuscript E6 residue r -> BioEmu PDB resSeq r+6",
        "seven_uaj_resolved_E6_range": [seven_start, seven_stop],
        "seven_uaj_missing_residues_within_range": seven_missing,
        "seven_uaj_resolved_sequence_differences": seven_mismatches,
    }


def verify_bioemu_reconstruction_link(
    backbone_topology_path: Path,
    backbone_trajectory_path: Path,
    reconstructed_topology_path: Path,
    reconstructed_trajectory_path: Path,
) -> dict:
    """Verify that H-Packer output preserves the supplied BioEmu frame order.

    BioEmu's reconstruction code rebuilds side chains independently on each
    backbone frame.  The N/CA/C/O coordinates therefore provide a strict
    frame-identity check between the reduced and reconstructed trajectories.
    """

    backbone = md.load(
        str(backbone_trajectory_path), top=str(backbone_topology_path)
    )
    reconstructed = md.load(
        str(reconstructed_trajectory_path), top=str(reconstructed_topology_path)
    )
    if backbone.n_frames != reconstructed.n_frames:
        raise ValueError(
            "BioEmu backbone and reconstructed trajectories have different "
            f"frame counts: {backbone.n_frames} versus {reconstructed.n_frames}"
        )

    def mapped_indices(trajectory: md.Trajectory, names: set[str]) -> dict:
        return {
            (atom.residue.resSeq, atom.name): atom.index
            for atom in trajectory.topology.atoms
            if atom.name in names
        }

    backbone_names = {"N", "CA", "C", "O"}
    backbone_map = mapped_indices(backbone, backbone_names)
    reconstructed_map = mapped_indices(reconstructed, backbone_names)
    if set(backbone_map) != set(reconstructed_map):
        raise ValueError(
            "BioEmu reduced and reconstructed trajectories do not have the "
            "same N/CA/C/O atom keys"
        )
    keys = sorted(backbone_map)
    old_indices = [backbone_map[key] for key in keys]
    new_indices = [reconstructed_map[key] for key in keys]
    delta = reconstructed.xyz[:, new_indices, :] - backbone.xyz[:, old_indices, :]
    per_frame_rmsd_a = np.sqrt(np.mean(np.sum(delta * delta, axis=2), axis=1)) * 10.0
    max_abs_a = float(np.max(np.abs(delta)) * 10.0)
    if max_abs_a > 1.0e-4:
        raise ValueError(
            "H-Packer trajectory does not preserve the BioEmu N/CA/C/O "
            f"coordinates in frame order (maximum difference {max_abs_a:.6g} A)"
        )

    common_names = {"N", "CA", "C", "O", "CB"}
    common_old = mapped_indices(backbone, common_names)
    common_new = mapped_indices(reconstructed, common_names)
    common_keys = sorted(set(common_old) & set(common_new))
    common_delta = (
        reconstructed.xyz[:, [common_new[key] for key in common_keys], :]
        - backbone.xyz[:, [common_old[key] for key in common_keys], :]
    )
    common_per_frame_rmsd_a = (
        np.sqrt(np.mean(np.sum(common_delta * common_delta, axis=2), axis=1)) * 10.0
    )

    topology_frame = md.load(str(reconstructed_topology_path))
    pdb_xtc_delta = topology_frame.xyz[0] - reconstructed.xyz[0]

    return {
        "reduced_frame_count": backbone.n_frames,
        "reconstructed_frame_count": reconstructed.n_frames,
        "n_ca_c_o_atom_count": len(keys),
        "n_ca_c_o_max_abs_difference_A": max_abs_a,
        "n_ca_c_o_rmsd_A_min_mean_max": [
            float(per_frame_rmsd_a.min()),
            float(per_frame_rmsd_a.mean()),
            float(per_frame_rmsd_a.max()),
        ],
        "same_frame_order_confirmed": True,
        "n_ca_c_o_cb_atom_count": len(common_keys),
        "n_ca_c_o_cb_rmsd_A_min_mean_max": [
            float(common_per_frame_rmsd_a.min()),
            float(common_per_frame_rmsd_a.mean()),
            float(common_per_frame_rmsd_a.max()),
        ],
        "reconstructed_pdb_vs_xtc_frame0_max_abs_A": float(
            np.max(np.abs(pdb_xtc_delta)) * 10.0
        ),
    }


def atom_completeness(
    topology: md.Topology,
    mapping: ResidueMap,
    reference_topology: md.Topology,
    reference_mapping: ResidueMap = MD_MAP,
) -> dict:
    missing_by_residue: dict[str, list[str]] = {}
    for e6_residue in (*N_WALL, *C_WALL):
        residue = find_residue(topology, mapping, e6_residue)
        reference = find_residue(reference_topology, reference_mapping, e6_residue)
        missing = sorted(heavy_atom_names(reference) - heavy_atom_names(residue))
        if missing:
            missing_by_residue[str(e6_residue)] = missing

    q107 = find_residue(topology, mapping, 107)
    q107_missing = [name for name in ("N", "CA", "CB", "CG") if optional_atom_index(q107, name) is None]
    return {
        "wall_missing_heavy_atoms_by_E6_residue": missing_by_residue,
        "full_heavy_aperture_available": not missing_by_residue,
        "q107_chi1_missing_atoms": q107_missing,
        "q107_chi1_available": not q107_missing,
    }


def desmond_signed_dihedral_deg(
    atom1: np.ndarray,
    atom2: np.ndarray,
    atom3: np.ndarray,
    atom4: np.ndarray,
) -> np.ndarray:
    """Vectorized form of the signed torsion convention used by Desmond."""
    r12 = atom1 - atom2
    r32 = atom3 - atom2
    r42 = atom4 - atom2
    cross_1232 = np.cross(r12, r32)
    cross_4232 = np.cross(r42, r32)
    denominator = np.linalg.norm(cross_1232, axis=1) * np.linalg.norm(
        cross_4232, axis=1
    )
    if np.any(denominator == 0.0):
        raise ValueError("Cannot calculate Q107 chi1 from collinear coordinates")
    cosine = np.clip(
        np.einsum("ij,ij->i", cross_1232, cross_4232) / denominator,
        -1.0,
        1.0,
    )
    angle = np.degrees(np.arccos(cosine))
    sign_test = np.einsum("ij,ij->i", r12, cross_4232)
    angle[sign_test < 0.0] *= -1.0
    return angle


def common_atom_indices(residue) -> list[int]:
    indices = [
        atom.index
        for atom in residue.atoms
        if atom.name in COMMON_APERTURE_ATOMS
        and atom.element is not None
        and atom.element.symbol != "H"
    ]
    if not indices:
        raise ValueError(f"No common aperture atoms found in {residue}")
    return indices


def common_atom_aperture_A(
    trajectory: md.Trajectory, wall_pairs: Sequence[tuple[list[int], list[int]]]
) -> np.ndarray:
    values_nm = np.empty((trajectory.n_frames, len(wall_pairs)), dtype=np.float32)
    xyz = trajectory.xyz
    for pair_index, (left, right) in enumerate(wall_pairs):
        delta = xyz[:, left, None, :] - xyz[:, None, right, :]
        distances = np.linalg.norm(delta, axis=3)
        values_nm[:, pair_index] = distances.min(axis=(1, 2))
    return values_nm.mean(axis=1, dtype=np.float64) * 10.0


def prepare_indices(
    topology: md.Topology,
    mapping: ResidueMap,
    full_heavy_aperture_available: bool,
) -> dict:
    e6hl_residue_indices = [
        find_residue(topology, mapping, residue).index for residue in E6HL
    ]
    d49 = find_residue(topology, mapping, 49)
    r102 = find_residue(topology, mapping, 102)
    q107 = find_residue(topology, mapping, 107)
    r130 = optional_residue(topology, mapping, 130)
    r151 = optional_residue(topology, mapping, 151)

    n_wall = [find_residue(topology, mapping, residue) for residue in N_WALL]
    c_wall = [find_residue(topology, mapping, residue) for residue in C_WALL]
    residue_pairs = [(a.index, b.index) for a in n_wall for b in c_wall]
    common_pairs = [
        (common_atom_indices(a), common_atom_indices(b))
        for a in n_wall
        for b in c_wall
    ]

    q_atoms = [optional_atom_index(q107, atom) for atom in ("N", "CA", "CB", "CG")]
    return {
        "e6hl_residue_indices": e6hl_residue_indices,
        "d49_ca": atom_index(d49, "CA"),
        "r102_ca": atom_index(r102, "CA"),
        "q107_atoms": q_atoms if all(index is not None for index in q_atoms) else None,
        "r130_ca": optional_atom_index(r130, "CA"),
        "r151_ca": optional_atom_index(r151, "CA"),
        "residue_pairs": residue_pairs if full_heavy_aperture_available else None,
        "common_pairs": common_pairs,
    }


def calculate_chunk(
    trajectory: md.Trajectory,
    indices: dict,
    periodic_contacts: bool,
) -> dict[str, np.ndarray]:
    xyz = trajectory.xyz
    dssp = md.compute_dssp(trajectory, simplified=False)
    e6hl = dssp[:, indices["e6hl_residue_indices"]]
    helix = np.isin(e6hl, np.asarray(["H", "G"])).mean(axis=1)
    d49_r102 = (
        np.linalg.norm(
            xyz[:, indices["d49_ca"], :] - xyz[:, indices["r102_ca"], :],
            axis=1,
        )
        * 10.0
    )

    if indices["q107_atoms"] is None:
        chi1 = np.full(trajectory.n_frames, np.nan, dtype=float)
    else:
        n_atom, ca_atom, cb_atom, cg_atom = indices["q107_atoms"]
        chi1 = desmond_signed_dihedral_deg(
            xyz[:, n_atom, :],
            xyz[:, ca_atom, :],
            xyz[:, cb_atom, :],
            xyz[:, cg_atom, :],
        )

    if indices["r130_ca"] is None or indices["r151_ca"] is None:
        tail = np.full(trajectory.n_frames, np.nan, dtype=float)
    else:
        tail = (
            np.linalg.norm(
                xyz[:, indices["r130_ca"], :]
                - xyz[:, indices["r151_ca"], :],
                axis=1,
            )
            * 10.0
        )

    if indices["residue_pairs"] is None:
        aperture = np.full(trajectory.n_frames, np.nan, dtype=float)
    else:
        wall_distances_nm, _ = md.compute_contacts(
            trajectory,
            contacts=np.asarray(indices["residue_pairs"], dtype=int),
            scheme="closest-heavy",
            periodic=periodic_contacts,
        )
        aperture = wall_distances_nm.mean(axis=1, dtype=np.float64) * 10.0

    common_aperture = common_atom_aperture_A(
        trajectory, indices["common_pairs"]
    )

    return {
        "e6hl_helix_fraction_H_or_G": np.asarray(helix, dtype=float),
        "d49_r102_ca_distance_A": np.asarray(d49_r102, dtype=float),
        "q107_chi1_deg": np.asarray(chi1, dtype=float),
        "tail_ca130_ca151_distance_A": np.asarray(tail, dtype=float),
        "lxxll_aperture_all_heavy_A": np.asarray(aperture, dtype=float),
        "lxxll_aperture_common_N_CA_C_O_CB_A": np.asarray(
            common_aperture, dtype=float
        ),
    }


def analyze_pdb(
    path: Path,
    mapping: ResidueMap,
    reference_topology: md.Topology,
) -> tuple[dict[str, np.ndarray], dict]:
    trajectory = md.load_pdb(str(path))
    if trajectory.n_frames != 1:
        raise ValueError(f"Expected one model in {path}; found {trajectory.n_frames}")
    completeness = atom_completeness(
        trajectory.topology, mapping, reference_topology
    )
    indices = prepare_indices(
        trajectory.topology,
        mapping,
        completeness["full_heavy_aperture_available"],
    )
    # A crystallographic unit cell must not turn an intramolecular static
    # descriptor into a crystal-image distance.
    return calculate_chunk(trajectory, indices, periodic_contacts=False), completeness


def analyze_xtc(
    trajectory_path: Path,
    topology_path: Path,
    mapping: ResidueMap,
    reference_topology: md.Topology,
    chunk_size: int,
    periodic_contacts: bool,
) -> tuple[dict[str, np.ndarray], dict, md.Topology]:
    topology = md.load_topology(str(topology_path))
    completeness = atom_completeness(topology, mapping, reference_topology)
    indices = prepare_indices(
        topology, mapping, completeness["full_heavy_aperture_available"]
    )
    parts: dict[str, list[np.ndarray]] = {metric: [] for metric in METRIC_ORDER}
    for chunk in md.iterload(
        str(trajectory_path), top=str(topology_path), chunk=chunk_size
    ):
        chunk_metrics = calculate_chunk(
            chunk, indices, periodic_contacts=periodic_contacts
        )
        for metric in METRIC_ORDER:
            parts[metric].append(chunk_metrics[metric])
    metrics = {
        metric: np.concatenate(parts[metric]) if parts[metric] else np.array([])
        for metric in METRIC_ORDER
    }
    frame_counts = {len(values) for values in metrics.values()}
    if len(frame_counts) != 1:
        raise RuntimeError(f"Inconsistent trajectory metric lengths: {frame_counts}")
    return metrics, completeness, topology


def circular_mean_deg(values: np.ndarray) -> float:
    radians = np.radians(values)
    return float(
        np.degrees(math.atan2(float(np.sin(radians).mean()), float(np.cos(radians).mean())))
    )


def circular_difference_deg(values: np.ndarray, reference: float) -> np.ndarray:
    return ((values - reference + 180.0) % 360.0) - 180.0


def finite_values(values: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = mask & np.isfinite(values)
    return values[valid], np.flatnonzero(valid)


def summarize_subset(
    source: str,
    subset: str,
    metrics: dict[str, np.ndarray],
    mask: np.ndarray,
    note: str,
) -> list[dict]:
    rows = []
    definitions = {item["metric"]: item for item in METRIC_DEFINITIONS}
    for metric in METRIC_ORDER:
        values, _ = finite_values(metrics[metric], mask)
        definition = definitions[metric]
        if len(values) == 0:
            minimum = mean = maximum = None
            mean_method = "unavailable"
            status = "unavailable"
        else:
            minimum = float(values.min())
            maximum = float(values.max())
            if metric == ANGLE_METRIC:
                mean = circular_mean_deg(values)
                mean_method = "circular mean; min/max are wrapped [-180,180] extrema"
            else:
                mean = float(values.mean())
                mean_method = "arithmetic mean"
            status = (
                "surrogate"
                if metric == "lxxll_aperture_common_N_CA_C_O_CB_A"
                else "computed"
            )
        rows.append(
            {
                "source": source,
                "subset": subset,
                "n_frames_total": int(mask.sum()),
                "metric": metric,
                "display_name": definition["display_name"],
                "units": definition["units"],
                "minimum": minimum,
                "mean": mean,
                "maximum": maximum,
                "n_valid": int(len(values)),
                "mean_method": mean_method,
                "status": status,
                "note": note,
            }
        )
    return rows


def descriptor_centroid_representative(
    label: str,
    source: str,
    metrics: dict[str, np.ndarray],
    mask: np.ndarray,
    frame_metadata: Sequence[dict],
) -> dict:
    candidate_indices = np.flatnonzero(mask)
    usable_metrics = []
    score = np.zeros(len(candidate_indices), dtype=float)
    centers: dict[str, float] = {}
    scales: dict[str, float] = {}

    for metric in METRIC_ORDER:
        values = metrics[metric][candidate_indices]
        if not np.all(np.isfinite(values)):
            continue
        if metric == "lxxll_aperture_common_N_CA_C_O_CB_A" and np.all(
            np.isfinite(metrics["lxxll_aperture_all_heavy_A"][candidate_indices])
        ):
            # Use the manuscript aperture when it is available; the common-atom
            # surrogate is reserved for sources lacking complete side chains.
            continue
        if metric == ANGLE_METRIC:
            center = circular_mean_deg(values)
            deviations = circular_difference_deg(values, center)
            scale = float(np.std(deviations))
        else:
            center = float(values.mean())
            deviations = values - center
            scale = float(np.std(values))
        if not np.isfinite(scale) or scale <= 1.0e-12:
            continue
        score += (deviations / scale) ** 2
        usable_metrics.append(metric)
        centers[metric] = center
        scales[metric] = scale

    if not usable_metrics:
        raise ValueError(f"No usable metrics for representative {label}")
    local_index = int(np.argmin(score))
    frame_index = int(candidate_indices[local_index])
    row = {
        "label": label,
        "source": source,
        "selection": (
            "actual frame minimizing summed squared standardized deviations "
            "from subset means; Q107 uses circular deviation"
        ),
        "selection_metrics": usable_metrics,
        "descriptor_score": float(score[local_index]),
        "centers": centers,
        "scales": scales,
        **frame_metadata[frame_index],
    }
    for metric in METRIC_ORDER:
        value = metrics[metric][frame_index]
        row[metric] = float(value) if np.isfinite(value) else None
    return row


def extrema_rows(
    source: str,
    subset: str,
    metrics: dict[str, np.ndarray],
    mask: np.ndarray,
    frame_metadata: Sequence[dict],
) -> list[dict]:
    rows = []
    for metric in METRIC_ORDER:
        values, global_indices = finite_values(metrics[metric], mask)
        if len(values) == 0:
            continue
        mean = circular_mean_deg(values) if metric == ANGLE_METRIC else float(values.mean())
        if metric == ANGLE_METRIC:
            mean_nearest_local = int(
                np.argmin(np.abs(circular_difference_deg(values, mean)))
            )
        else:
            mean_nearest_local = int(np.argmin(np.abs(values - mean)))
        selections = (
            ("minimum", int(np.argmin(values))),
            ("mean-nearest frame", mean_nearest_local),
            ("maximum", int(np.argmax(values))),
        )
        for statistic, local_index in selections:
            global_index = int(global_indices[local_index])
            rows.append(
                {
                    "source": source,
                    "subset": subset,
                    "metric": metric,
                    "statistic": statistic,
                    "summary_mean": mean,
                    "frame_value": float(values[local_index]),
                    **frame_metadata[global_index],
                }
            )
    return rows


def md_frame_metadata(n_frames: int, macrostates: np.ndarray) -> list[dict]:
    # Coordinate-supported boundaries and manuscript labels from the validated
    # trajectory-splitting manifest.  These labels do not alter MSM assignments.
    segments = (
        (0, 10001, 1),
        (10001, 20001, 2),
        (20001, 25001, 4),
        (25001, 35002, 3),
    )
    if n_frames != 35002:
        raise ValueError(f"Expected 35002 MD frames; found {n_frames}")
    if len(macrostates) != n_frames:
        raise ValueError(
            f"MSM macrostate array has {len(macrostates)} values for {n_frames} frames"
        )
    metadata: list[dict | None] = [None] * n_frames
    for start, stop, replica in segments:
        for global_index in range(start, stop):
            local_index = global_index - start
            metadata[global_index] = {
                "frame_index_0based": global_index,
                "frame_index_1based": global_index + 1,
                "manuscript_replica": replica,
                "replica_frame_index_0based": local_index,
                "replica_time_ns_declared": float(local_index),
                "msm_macrostate": "Closed" if macrostates[global_index] == 0 else "Open",
            }
    if any(item is None for item in metadata):
        raise RuntimeError("Incomplete MD frame metadata mapping")
    return list(metadata)  # type: ignore[arg-type]


def bioemu_frame_metadata(n_frames: int) -> list[dict]:
    return [
        {
            "frame_index_0based": index,
            "frame_index_1based": index + 1,
            "manuscript_replica": None,
            "replica_frame_index_0based": None,
            "replica_time_ns_declared": None,
            "msm_macrostate": None,
            "bioemu_cluster_id": None,
            "bioemu_cluster_weight": None,
        }
        for index in range(n_frames)
    ]


def frame_rows(
    metadata: Sequence[dict], metrics: dict[str, np.ndarray]
) -> list[dict]:
    rows = []
    for index, item in enumerate(metadata):
        row = dict(item)
        for metric in METRIC_ORDER:
            value = metrics[metric][index]
            row[metric] = float(value) if np.isfinite(value) else None
        if row["q107_chi1_deg"] is None:
            row["q107_chi1_sin"] = None
            row["q107_chi1_cos"] = None
        else:
            radians = math.radians(row["q107_chi1_deg"])
            row["q107_chi1_sin"] = math.sin(radians)
            row["q107_chi1_cos"] = math.cos(radians)
        rows.append(row)
    return rows


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def extract_representative_pdb(
    representative: dict,
    trajectory_path: Path,
    topology_path: Path,
    output_path: Path,
) -> None:
    frame = md.load_frame(
        str(trajectory_path),
        int(representative["frame_index_0based"]),
        top=str(topology_path),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.save_pdb(str(output_path))


def static_row(label: str, source_file: Path, metrics: dict[str, np.ndarray]) -> dict:
    row = {
        "label": label,
        "source": "Static structure",
        "selection": "supplied crystallographic structure",
        "source_file": str(source_file.resolve()),
        "frame_index_0based": 0,
        "frame_index_1based": 1,
        "manuscript_replica": None,
        "replica_frame_index_0based": None,
        "replica_time_ns_declared": None,
        "msm_macrostate": None,
    }
    for metric in METRIC_ORDER:
        value = metrics[metric][0]
        row[metric] = float(value) if np.isfinite(value) else None
    return row


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--md-topology",
        type=Path,
        default=Path('structures/topology_apo16e6.pdb'),
    )
    parser.add_argument(
        "--md-trajectory",
        type=Path,
        default=Path('trajectories/trajectory_35us.xtc'),
    )
    parser.add_argument(
        "--msm-bundle",
        type=Path,
        default=Path(
            'trajectories/msm/apo16e6_msm_numeric_bundle.npz'
        ),
    )
    parser.add_argument(
        "--bioemu-topology",
        type=Path,
        default=Path(
            'structures/hpacker-openmm_sidechain_rec.pdb'
        ),
    )
    parser.add_argument(
        "--bioemu-trajectory",
        type=Path,
        default=Path(
            'trajectories/bioemu/hpacker-openmm_sidechain_rec.xtc'
        ),
    )
    parser.add_argument(
        "--bioemu-backbone-topology",
        type=Path,
        default=Path('structures/clustered_topology.pdb'),
    )
    parser.add_argument(
        "--bioemu-backbone-trajectory",
        type=Path,
        default=Path('trajectories/bioemu/clustered_samples.xtc'),
    )
    parser.add_argument(
        "--pdb-4xr8",
        type=Path,
        default=Path(
            'structures/4XR8_clean.pdb'
        ),
    )
    parser.add_argument(
        "--pdb-7uaj",
        type=Path,
        default=Path(
            'structures/7UAJ_clean.pdb'
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            'results/common_metrics'
        ),
    )
    parser.add_argument("--chunk-size", type=int, default=1000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_paths = (
        args.md_topology,
        args.md_trajectory,
        args.msm_bundle,
        args.bioemu_topology,
        args.bioemu_trajectory,
        args.bioemu_backbone_topology,
        args.bioemu_backbone_trajectory,
        args.pdb_4xr8,
        args.pdb_7uaj,
    )
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    md_topology = md.load_topology(str(args.md_topology))
    four_xr8_topology = md.load_topology(str(args.pdb_4xr8))
    seven_uaj_topology = md.load_topology(str(args.pdb_7uaj))
    bioemu_topology = md.load_topology(str(args.bioemu_topology))
    sequence_checks = verify_sequences(
        md_topology, four_xr8_topology, seven_uaj_topology, bioemu_topology
    )
    bioemu_reconstruction_link = verify_bioemu_reconstruction_link(
        args.bioemu_backbone_topology,
        args.bioemu_backbone_trajectory,
        args.bioemu_topology,
        args.bioemu_trajectory,
    )

    four_xr8_metrics, four_xr8_completeness = analyze_pdb(
        args.pdb_4xr8, FOUR_XR8_MAP, md_topology
    )
    seven_uaj_metrics, seven_uaj_completeness = analyze_pdb(
        args.pdb_7uaj, SEVEN_UAJ_MAP, md_topology
    )
    md_metrics, md_completeness, _ = analyze_xtc(
        args.md_trajectory,
        args.md_topology,
        MD_MAP,
        md_topology,
        args.chunk_size,
        True,
    )
    bioemu_metrics, bioemu_completeness, _ = analyze_xtc(
        args.bioemu_trajectory,
        args.bioemu_topology,
        BIOEMU_MAP,
        md_topology,
        args.chunk_size,
        False,
    )

    msm_bundle = np.load(args.msm_bundle, allow_pickle=False)
    macrostates = np.asarray(
        msm_bundle["hard_macrostate_trajectory_closed0_open1"], dtype=int
    )
    md_metadata = md_frame_metadata(len(md_metrics[METRIC_ORDER[0]]), macrostates)
    bioemu_metadata = bioemu_frame_metadata(len(bioemu_metrics[METRIC_ORDER[0]]))

    summary_rows: list[dict] = []
    summary_rows.extend(
        summarize_subset(
            "4XR8",
            "bound-reference chain H",
            four_xr8_metrics,
            np.array([True]),
            "Single supplied prepared structure; min=mean=max.",
        )
    )
    summary_rows.extend(
        summarize_subset(
            "7UAJ",
            "fusion chain D",
            seven_uaj_metrics,
            np.array([True]),
            "Single crystal copy; E6 residues 144-151 are unresolved.",
        )
    )

    md_all = np.ones(len(macrostates), dtype=bool)
    md_closed = macrostates == 0
    md_open = macrostates == 1
    summary_rows.extend(
        summarize_subset(
            "Unbiased MD",
            "all four trajectories",
            md_metrics,
            md_all,
            (
                "Frame-weighted descriptive summary over 35,002 serially "
                "correlated frames; not an equilibrium uncertainty estimate."
            ),
        )
    )
    summary_rows.extend(
        summarize_subset(
            "MSM-labelled MD frames",
            "Closed hard macrostate",
            md_metrics,
            md_closed,
            (
                "Descriptive statistics of hard-assigned MD frames; not "
                "stationary-distribution-weighted and model-dependent."
            ),
        )
    )
    summary_rows.extend(
        summarize_subset(
            "MSM-labelled MD frames",
            "Open hard macrostate",
            md_metrics,
            md_open,
            (
                "Descriptive statistics of hard-assigned MD frames; not "
                "stationary-distribution-weighted and model-dependent."
            ),
        )
    )
    bio_all = np.ones(len(bioemu_metadata), dtype=bool)
    summary_rows.extend(
        summarize_subset(
            "BioEmu",
            "72 H-Packer-reconstructed conformers",
            bioemu_metrics,
            bio_all,
            (
                "Unweighted over supplied conformers. Q107 chi1 and the "
                "all-heavy aperture use H-Packer-reconstructed side chains. "
                "Cluster IDs/weights are absent, timestamps are all zero, and "
                "the manuscript states 73 clusters, so this is not a "
                "cluster-weighted ensemble."
            ),
        )
    )

    representatives = [
        static_row("4XR8 bound reference", args.pdb_4xr8, four_xr8_metrics),
        static_row("7UAJ chain D", args.pdb_7uaj, seven_uaj_metrics),
        descriptor_centroid_representative(
            "Unbiased MD descriptor-centroid frame",
            "Unbiased MD",
            md_metrics,
            md_all,
            md_metadata,
        ),
        descriptor_centroid_representative(
            "MSM Closed descriptor-centroid frame",
            "MSM-labelled MD frames",
            md_metrics,
            md_closed,
            md_metadata,
        ),
        descriptor_centroid_representative(
            "MSM Open descriptor-centroid frame",
            "MSM-labelled MD frames",
            md_metrics,
            md_open,
            md_metadata,
        ),
        descriptor_centroid_representative(
            "BioEmu H-Packer descriptor-centroid frame",
            "BioEmu",
            bioemu_metrics,
            bio_all,
            bioemu_metadata,
        ),
    ]

    representative_dir = output_dir / "representative_pdbs"
    for representative in representatives[2:5]:
        safe = representative["label"].lower().replace(" ", "_").replace("-", "_")
        output_path = representative_dir / f"{safe}.pdb"
        extract_representative_pdb(
            representative, args.md_trajectory, args.md_topology, output_path
        )
        representative["extracted_pdb"] = str(output_path.resolve())
    bio_representative = representatives[5]
    bio_output = representative_dir / "bioemu_supplied_set_descriptor_centroid_frame.pdb"
    extract_representative_pdb(
        bio_representative,
        args.bioemu_trajectory,
        args.bioemu_topology,
        bio_output,
    )
    bio_representative["extracted_pdb"] = str(bio_output.resolve())

    extrema: list[dict] = []
    extrema.extend(extrema_rows("Unbiased MD", "all", md_metrics, md_all, md_metadata))
    extrema.extend(
        extrema_rows(
            "MSM-labelled MD frames", "Closed", md_metrics, md_closed, md_metadata
        )
    )
    extrema.extend(
        extrema_rows(
            "MSM-labelled MD frames", "Open", md_metrics, md_open, md_metadata
        )
    )
    extrema.extend(
        extrema_rows("BioEmu", "supplied frames", bioemu_metrics, bio_all, bioemu_metadata)
    )

    md_rows = frame_rows(md_metadata, md_metrics)
    bioemu_rows = frame_rows(bioemu_metadata, bioemu_metrics)
    write_csv(output_dir / "md_frame_metrics.csv", md_rows)
    write_csv(output_dir / "bioemu_frame_metrics.csv", bioemu_rows)
    write_csv(output_dir / "summary_precomputed.csv", summary_rows)
    write_csv(output_dir / "representatives.csv", representatives)
    write_csv(output_dir / "extrema_frames.csv", extrema)
    write_csv(output_dir / "metric_definitions.csv", METRIC_DEFINITIONS)

    provenance = {
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "mdtraj": md.__version__,
        },
        "inputs": {
            key: {"path": str(path.resolve()), "sha256": sha256(path)}
            for key, path in {
                "md_topology": args.md_topology,
                "md_trajectory": args.md_trajectory,
                "msm_bundle": args.msm_bundle,
                "bioemu_hpacker_topology": args.bioemu_topology,
                "bioemu_hpacker_trajectory": args.bioemu_trajectory,
                "bioemu_reduced_topology": args.bioemu_backbone_topology,
                "bioemu_reduced_trajectory": args.bioemu_backbone_trajectory,
                "pdb_4xr8": args.pdb_4xr8,
                "pdb_7uaj": args.pdb_7uaj,
            }.items()
        },
        "sequence_checks": sequence_checks,
        "bioemu_reconstruction": {
            "method": "H-Packer side-chain reconstruction on BioEmu backbones",
            "frame_link_checks": bioemu_reconstruction_link,
            "official_documentation": {
                "bioemu_readme": (
                    "https://github.com/microsoft/bioemu#side-chain-"
                    "reconstruction-and-md-relaxation"
                ),
                "bioemu_reconstruction_source": (
                    "https://github.com/microsoft/bioemu/blob/main/src/"
                    "bioemu/sidechain_relax.py#L67-L104"
                ),
                "bioemu_output_source": (
                    "https://github.com/microsoft/bioemu/blob/main/src/"
                    "bioemu/sidechain_relax.py#L269-L282"
                ),
            },
        },
        "atom_completeness": {
            "4XR8": four_xr8_completeness,
            "7UAJ": seven_uaj_completeness,
            "MD": md_completeness,
            "BioEmu": bioemu_completeness,
        },
        "frame_counts": {
            "MD": len(md_rows),
            "MSM_Closed": int(md_closed.sum()),
            "MSM_Open": int(md_open.sum()),
            "BioEmu_supplied": len(bioemu_rows),
        },
        "metric_definitions": METRIC_DEFINITIONS,
        "distance_conventions": {
            "md_all_heavy_aperture": (
                "Minimum-image distances using the XTC unit-cell metadata, "
                "matching MDTraj's periodic=True contact convention."
            ),
            "static_all_heavy_aperture": (
                "Direct Cartesian distances; the supplied PDB structures do "
                "not define a periodic comparison box."
            ),
            "bioemu_hpacker_all_heavy_aperture": (
                "Direct Cartesian distances; the reconstructed BioEmu XTC "
                "contains no unit cell and is not a periodic trajectory."
            ),
            "common_atom_surrogate": (
                "Direct Cartesian distances for every source so the reduced "
                "BioEmu topology can be compared on the same atom set."
            ),
            "d49_r102_and_tail": "Direct Cartesian C-alpha distances.",
        },
        "limitations": [
            "The MD summaries are descriptive over serially correlated frames.",
            "MSM Closed/Open summaries use hard frame assignments and are not stationary-population-weighted.",
            "The supplied BioEmu XTC has 72 frames, no cluster IDs or weights, and zero timestamps, while the manuscript states 73 clusters.",
            "BioEmu has a seven-residue N-terminal extension; metrics use an exact sequence-aligned +6 PDB-resSeq mapping.",
            *(
                [
                    "BioEmu Q107 chi1 and the all-heavy aperture use H-Packer-predicted side-chain coordinates and are model-dependent reconstructed observables.",
                    "The H-Packer sidechain_rec files contain no Zn atoms or validated Cys4-Zn coordination.",
                    "In the official BioEmu workflow, the sidechain_rec output is written before optional OpenMM minimization/equilibration; relaxed output uses the md_equil suffix.",
                    "The common-atom aperture remains an explicitly labelled surrogate and is retained to connect the reconstructed ensemble to the original reduced topology.",
                ]
                if bioemu_completeness["full_heavy_aperture_available"]
                and bioemu_completeness["q107_chi1_available"]
                else [
                    "The supplied BioEmu files lack atoms required for Q107 chi1 and the manuscript all-heavy-atom groove aperture.",
                    "BioEmu common-atom aperture is an explicitly labelled surrogate, not a substitute for side-chain-reconstructed values.",
                ]
            ),
            "7UAJ chain D lacks E6 residues 144-151, so C-alpha(130)-C-alpha(151) tail extension is unavailable.",
            "Q107 angular min/max are wrapped-coordinate extrema; ensemble means are circular and should not be interpreted linearly across the periodic seam.",
        ],
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    compact_results = {
        "summary": summary_rows,
        "representatives": representatives,
        "extrema": extrema,
        "provenance": provenance,
    }
    (output_dir / "comparison_results.json").write_text(
        json.dumps(compact_results, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps({
        "output_dir": str(output_dir.resolve()),
        "frame_counts": provenance["frame_counts"],
        "sequence_checks": sequence_checks,
        "bioemu_atom_completeness": bioemu_completeness,
        "summary_rows": len(summary_rows),
        "representatives": [
            {"label": row["label"], "frame_index_0based": row.get("frame_index_0based")}
            for row in representatives
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
