#!/usr/bin/env python3
"""Extract CV-nearest sampled structures for nine terminal-FES minima.
Inputs: native runs, pinned centers and manifest; outputs: structures and QC tables.
Supports V12 Figure 6; requires the companion Desmond parser module.
Run from the repository root; see this folder's README for commands.
"""

# Extract coordinate-matched structures for the five-run terminal Desmond FES.
# 
# This script does not calculate or apply trajectory-reweighting factors.  It uses
# the nine algorithm-detected minima previously identified on the smoothed,
# equal-replicate mixture of five terminal-time Desmond FES grids (runs
# 8.10.25--8.14.25), maps every CVSEQ record to its unique same-time saved
# trajectory frame, and extracts the closest sampled structure to each minimum.
# Extra trajectory-only endpoint frames are reported and excluded; trajectory
# and CVSEQ record counts therefore need not be identical.
# 
# The Desmond group-COM distance is a minimum-image coordinate.  To avoid
# exporting an extended protein with a misleading compact label, every local
# candidate is made whole and checked for a D49--R102 periodic lattice shift.
# Only a zero-shift candidate can be exported. This is not a general proof that
# the protein lacks self-image contacts or other periodic-boundary artifacts.

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.spatial import cKDTree

try:
    import wtmetad_5rep_reweight_basins as common
except ImportError as exc:
    # In the source tree the validated Desmond parsers live in a sibling
    # analysis directory.  In the portable HPC package the helper is placed
    # beside this script, so the normal import above succeeds.
    sibling = Path(__file__).resolve().parents[2] / "trajectory_reweighting"
    if (sibling / "wtmetad_5rep_reweight_basins.py").is_file():
        sys.path.insert(0, str(sibling))
        import wtmetad_5rep_reweight_basins as common
    else:
        raise SystemExit(
            "ERROR: wtmetad_5rep_reweight_basins.py must be beside this script"
        ) from exc


EXPECTED_CENTERS_SHA256 = (
    "8ca975c6d3e4927c5e51b0be01324ce6e2039b1bfd5821a56df534fb9af1ae1d"
)
EXPECTED_FES_MANIFEST_SHA256 = (
    "3780f8a73da5aaa008b9a2b6ac286dbd9a322873283ac84ae3c5ea53e2a57a11"
)
EXPECTED_PROTOCOL_SCALARS = {
    "temperature_K": 310.0,
    "kTemp_kcal_mol": 8.4,
    "initial_hill_height_kcal_mol": 0.3,
    "first_hill_time_ps": 0.0,
    "hill_interval_ps": 10.0,
    "kernel_cutoff_widths": 9.0,
}
DEFAULT_INPUT_ROOT = Path("trajectories/wt_metad")
E6_STEMS = (
    "desmond_WT_MetaD_4xr8_chainH_E6_1us_d49r102_q107chi1_8.10.25",
    "desmond_WT_MetaD_4xr8_chainH_E6_1us_d49r102_q107chi1_8.11.25",
    "desmond_WT_MetaD_4xr8_chainH_E6_1us_d49r102_q107chi1_8.12.25",
    "desmond_WT_MetaD_4xr8_chainH_E6_1us_d49r102_q107chi1_8.13.25",
    "desmond_WT_MetaD_4xr8_chainH_E6_1us_d49r102_q107chi1_8.14.25",
)


@dataclass(frozen=True)
class RunInput:
    replicate: str
    stem: str
    cfg: Path
    cvseq: Path
    cms: Path
    trajectory: Path


@dataclass(frozen=True)
class Center:
    identifier: str
    distance_A: float
    chi_deg: float
    display_chi_deg: float
    smoothed_relative_fes_kcal_mol: float


@dataclass(frozen=True)
class Selections:
    distance_gids: tuple[list[int], list[int]]
    distance_masses: tuple[np.ndarray, np.ndarray]
    chi_gids: list[int]
    structure_aids: list[int]
    structure_gids: list[int]
    heavy_structure_gids: list[int]
    ca_gids: list[int]
    zinc_gids: list[int]
    cysteine_sulfur_gids: list[int]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_csv_gz(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write("")
        return
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_output_checksums(output_dir: Path) -> None:
    excluded = {"OUTPUT_SHA256SUMS.txt", "STATUS.json"}
    lines = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name not in excluded:
            lines.append(f"{sha256(path)}  {path.relative_to(output_dir)}")
    (output_dir / "OUTPUT_SHA256SUMS.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def locate_file(directory: Path, names: Sequence[str], description: str) -> Path:
    found = [directory / name for name in names if (directory / name).is_file()]
    found = [path for path in found if path.stat().st_size > 0]
    if not found:
        raise ValueError(
            f"{description} not found; expected one of: "
            + ", ".join(str(directory / name) for name in names)
        )
    return found[0].resolve()


def locate_trajectory(directory: Path, stem: str) -> Path:
    candidates = [directory / f"{stem}_trj", directory / f"{stem}-out_trj"]
    found = [path for path in candidates if path.is_dir()]
    if len(found) > 1:
        raise ValueError(
            f"multiple native trajectories found for {stem}: "
            + ", ".join(str(path) for path in found)
        )
    if found:
        return found[0].resolve()
    xtc = directory / f"{stem}.xtc"
    if xtc.is_file() and xtc.stat().st_size > 0:
        return xtc.resolve()
    raise ValueError(
        f"trajectory not found for {stem}; expected "
        + ", ".join(str(path) for path in (*candidates, xtc))
    )


def auto_runs(root: Path) -> list[RunInput]:
    root = root.expanduser().resolve()
    runs = []
    for index, stem in enumerate(E6_STEMS, start=1):
        directory = root / stem
        if not directory.is_dir():
            raise ValueError(f"missing run directory: {directory}")
        date_token = stem.rsplit("_", 1)[-1]
        runs.append(RunInput(
            replicate=f"rep{index:02d}_{date_token}",
            stem=stem,
            cfg=locate_file(directory, [f"{stem}-out.cfg"], "configuration"),
            cvseq=locate_file(directory, [f"{stem}.cvseq"], "CV sequence"),
            cms=locate_file(
                directory,
                [f"{stem}-out.cms", f"{stem}.cms", f"{stem}-in.cms"],
                "CMS topology",
            ),
            trajectory=locate_trajectory(directory, stem),
        ))
    return runs


def read_manifest(path: Path) -> list[RunInput]:
    path = path.expanduser().resolve()
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"replicate", "cfg_file", "cvseq_file", "cms_file", "trajectory"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                "input manifest is missing columns: "
                + ", ".join(sorted(required.difference(reader.fieldnames or [])))
            )
        rows = list(reader)
    runs = []
    for line_number, row in enumerate(rows, start=2):
        resolved = {}
        for key in required.difference({"replicate"}):
            raw = (row.get(key) or "").strip()
            if not raw:
                raise ValueError(f"manifest line {line_number}: empty {key}")
            item = Path(raw).expanduser()
            if not item.is_absolute():
                item = path.parent / item
            item = item.resolve()
            if key == "trajectory":
                if not item.exists():
                    raise ValueError(f"manifest line {line_number}: missing {key}: {item}")
            elif not item.is_file() or item.stat().st_size == 0:
                raise ValueError(f"manifest line {line_number}: missing/empty {key}: {item}")
            resolved[key] = item
        replicate = (row.get("replicate") or "").strip()
        if not replicate:
            raise ValueError(f"manifest line {line_number}: empty replicate")
        stem = (row.get("stem") or resolved["cvseq_file"].stem).strip()
        runs.append(RunInput(
            replicate=replicate,
            stem=stem,
            cfg=resolved["cfg_file"],
            cvseq=resolved["cvseq_file"],
            cms=resolved["cms_file"],
            trajectory=resolved["trajectory"],
        ))
    return runs


def validate_runs(runs: Sequence[RunInput], expected_ktemp: float) -> tuple[dict, list[dict]]:
    if len(runs) != 5:
        raise ValueError(f"expected exactly five runs, found {len(runs)}")
    if len({run.replicate for run in runs}) != len(runs):
        raise ValueError("replicate identifiers must be unique")
    if tuple(run.stem for run in runs) != E6_STEMS:
        raise ValueError(
            "runs must be the ordered 8.10.25--8.14.25 five-run protocol group "
            "used to construct the pinned terminal FES"
        )
    seen_cv_hashes = {}
    seen_trajectory_paths = {}
    protocols = []
    reference = None
    for run in runs:
        protocol = common.protocol_signature(run.cfg)
        common.validate_protocol(protocol, expected_ktemp)
        for key, expected in EXPECTED_PROTOCOL_SCALARS.items():
            observed = protocol.get(key)
            if observed is None or not math.isclose(
                float(observed), expected, rel_tol=0.0, abs_tol=1e-8
            ):
                raise ValueError(
                    f"{run.replicate}: protocol {key} is {observed}, expected {expected}"
                )
        if reference is None:
            reference = protocol
        elif protocol != reference:
            differences = [key for key in reference if protocol.get(key) != reference.get(key)]
            raise ValueError(
                f"{run.replicate}: protocol differs from {runs[0].replicate} in {differences}"
            )
        digest = sha256(run.cvseq)
        if digest in seen_cv_hashes:
            raise ValueError(
                f"{run.replicate}: CV sequence duplicates {seen_cv_hashes[digest]}"
            )
        seen_cv_hashes[digest] = run.replicate
        trajectory_key = str(run.trajectory.resolve())
        if trajectory_key in seen_trajectory_paths:
            raise ValueError(
                f"{run.replicate}: trajectory is reused from "
                f"{seen_trajectory_paths[trajectory_key]}: {run.trajectory}"
            )
        seen_trajectory_paths[trajectory_key] = run.replicate
        protocols.append(protocol)
    assert reference is not None
    return reference, protocols


def center_id(distance_A: float, chi_deg: float) -> str:
    if distance_A < 21.0:
        distance_family = "D15"
    elif distance_A < 32.0:
        distance_family = "D28"
    else:
        distance_family = "D36"
    if chi_deg < -120.0 or chi_deg >= 120.0:
        rotamer = "trans"
    elif chi_deg < 0.0:
        rotamer = "gminus"
    else:
        rotamer = "gplus"
    return f"{distance_family}_{rotamer}"


def read_centers(path: Path) -> list[Center]:
    path = path.expanduser().resolve()
    digest = sha256(path)
    if digest != EXPECTED_CENTERS_SHA256:
        raise ValueError(
            f"center file SHA-256 is {digest}, expected {EXPECTED_CENTERS_SHA256}; "
            "use the pinned five-run minima file supplied with this package"
        )
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "distance_A", "chi1_canonical_deg", "chi1_display_deg",
        "smoothed_relative_free_energy_kcal_mol",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError("center file does not have the expected columns")
    centers = []
    for row in rows:
        distance_A = float(row["distance_A"])
        chi_deg = math.degrees(float(common.wrap_radians(
            math.radians(float(row["chi1_canonical_deg"]))
        )))
        values = [distance_A, chi_deg, float(row["chi1_display_deg"]),
                  float(row["smoothed_relative_free_energy_kcal_mol"])]
        if not np.isfinite(values).all():
            raise ValueError("center file contains a non-finite value")
        centers.append(Center(
            identifier=center_id(distance_A, chi_deg),
            distance_A=distance_A,
            chi_deg=chi_deg,
            display_chi_deg=float(row["chi1_display_deg"]),
            smoothed_relative_fes_kcal_mol=float(
                row["smoothed_relative_free_energy_kcal_mol"]
            ),
        ))
    expected_ids = {
        f"{distance}_{rotamer}"
        for distance in ("D15", "D28", "D36")
        for rotamer in ("gminus", "trans", "gplus")
    }
    observed_ids = {center.identifier for center in centers}
    if len(centers) != 9 or observed_ids != expected_ids:
        raise ValueError(
            f"expected one center for each of nine distance/rotamer combinations; "
            f"found {sorted(observed_ids)}"
        )
    order = {"D15": 0, "D28": 1, "D36": 2,
             "gminus": 0, "trans": 1, "gplus": 2}
    centers.sort(key=lambda item: (
        order[item.identifier.split("_")[0]],
        order[item.identifier.split("_")[1]],
    ))
    return centers


def validate_source_fes_manifest(path: Path) -> dict:
    """Pin the centers to the exact equal-replicate terminal-FES mixture."""
    path = path.expanduser().resolve()
    digest = sha256(path)
    if digest != EXPECTED_FES_MANIFEST_SHA256:
        raise ValueError(
            f"FES manifest SHA-256 is {digest}, expected "
            f"{EXPECTED_FES_MANIFEST_SHA256}; use the supplied source manifest"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    combination = payload.get("combination", {})
    inputs = payload.get("inputs", [])
    weights = combination.get("weights", [])
    if combination.get("name") != "equal-replicate probability mixture":
        raise ValueError("source FES manifest is not the expected equal-replicate mixture")
    if len(inputs) != 5 or len(weights) != 5 or not np.allclose(weights, 0.2):
        raise ValueError("source FES manifest does not describe five equally weighted runs")
    if payload.get("coordinate_labels", {}).get("dihedral") != "Q107 chi1":
        raise ValueError("source FES manifest has an unexpected dihedral coordinate")
    return payload


def circular_delta_radians(values: np.ndarray | float, target: float):
    return np.arctan2(np.sin(np.asarray(values) - target), np.cos(np.asarray(values) - target))


def build_selections(
    msys_model, cms_model, protocol: dict, structure_asl: str,
    expected_protein_residues: int,
) -> Selections:
    from schrodinger.application.desmond.packages import topo

    common.validate_e6_cv_semantics(msys_model, cms_model, protocol)
    distance_groups = protocol["collective_variables"][0]["atom"]
    distance_aids = [
        sorted(common._selection_aids(cms_model, selection))
        for selection in distance_groups
    ]
    distance_gids = tuple(
        topo.aids2gids(cms_model, aids, include_pseudoatoms=False)
        for aids in distance_aids
    )
    distance_masses = tuple(
        common._msys_masses(msys_model, gids) for gids in distance_gids
    )
    chi_aids = common._selection_aids(
        cms_model, protocol["collective_variables"][1]["atom"]
    )
    chi_gids = topo.aids2gids(cms_model, chi_aids, include_pseudoatoms=False)
    structure_aids = sorted(int(value) for value in cms_model.select_atom(structure_asl))
    if not structure_aids:
        raise ValueError(f"structure ASL selected no atoms: {structure_asl}")
    structure_gids = topo.aids2gids(
        cms_model, structure_aids, include_pseudoatoms=False
    )

    protein_aids = {int(value) for value in cms_model.select_atom("protein")}
    selected_protein_aids = protein_aids.intersection(structure_aids)
    if selected_protein_aids != protein_aids:
        raise ValueError("structure selection must contain every protein atom")
    selected_residue_keys = {
        (
            str(cms_model.fsys_ct.atom[aid].chain).strip(),
            int(cms_model.fsys_ct.atom[aid].resnum),
            str(cms_model.fsys_ct.atom[aid].inscode).strip(),
            str(cms_model.fsys_ct.atom[aid].pdbres).strip(),
        )
        for aid in selected_protein_aids
    }
    if len(selected_residue_keys) != expected_protein_residues:
        raise ValueError(
            f"structure selection contains {len(selected_residue_keys)} protein residues, "
            f"expected {expected_protein_residues}"
        )
    protein_chains = {record[0] for record in selected_residue_keys}
    if len(protein_chains) != 1:
        raise ValueError(
            f"structure selection contains {len(protein_chains)} protein chains, expected 1"
        )

    heavy_aids = []
    ca_records = []
    zinc_aids = []
    cysteine_sulfur_aids = []
    for aid in structure_aids:
        atom = cms_model.fsys_ct.atom[aid]
        element = str(atom.element).strip().upper()
        residue = str(atom.pdbres).strip().upper()
        atom_name = str(atom.pdbname).strip().upper()
        if element != "H":
            heavy_aids.append(aid)
        if atom_name == "CA" and aid in protein_aids:
            ca_records.append((
                str(atom.chain).strip(), int(atom.resnum),
                str(atom.inscode).strip(), str(atom.pdbres).strip(), aid,
            ))
        if element == "ZN":
            zinc_aids.append(aid)
        if element == "S" and residue in {"CYS", "CYX"}:
            cysteine_sulfur_aids.append(aid)
    ca_records.sort(key=lambda value: value[:4])
    if len(ca_records) != expected_protein_residues:
        raise ValueError(
            f"structure selection contains {len(ca_records)} protein CA atoms, "
            f"expected {expected_protein_residues}"
        )
    if {record[:4] for record in ca_records} != selected_residue_keys:
        raise ValueError("protein CA atoms are not one-to-one with selected residues")
    if len(zinc_aids) != 2:
        raise ValueError(f"structure selection contains {len(zinc_aids)} Zn atoms, expected 2")
    if len(cysteine_sulfur_aids) < 8:
        raise ValueError("structure selection contains fewer than eight cysteine sulfurs")
    return Selections(
        distance_gids=(list(distance_gids[0]), list(distance_gids[1])),
        distance_masses=(distance_masses[0], distance_masses[1]),
        chi_gids=list(chi_gids),
        structure_aids=structure_aids,
        structure_gids=list(structure_gids),
        heavy_structure_gids=topo.aids2gids(
            cms_model, heavy_aids, include_pseudoatoms=False
        ),
        ca_gids=topo.aids2gids(
            cms_model, [record[4] for record in ca_records], include_pseudoatoms=False
        ),
        zinc_gids=topo.aids2gids(
            cms_model, zinc_aids, include_pseudoatoms=False
        ),
        cysteine_sulfur_gids=topo.aids2gids(
            cms_model, cysteine_sulfur_aids, include_pseudoatoms=False
        ),
    )


def mass_center(frame, gids: Sequence[int], masses: np.ndarray) -> np.ndarray:
    xyz = np.asarray(frame.pos(gids), dtype=float)
    return np.sum(xyz * masses[:, None], axis=0) / float(np.sum(masses))


def box_face_heights(box: np.ndarray) -> tuple[float, float, float]:
    box = np.asarray(box, dtype=float).reshape(3, 3)
    volume = abs(float(np.linalg.det(box)))
    if volume <= 0.0:
        raise ValueError("trajectory frame has a singular box")
    heights = []
    for index in range(3):
        other = [value for value in range(3) if value != index]
        area = float(np.linalg.norm(np.cross(box[other[0]], box[other[1]])))
        if area <= 0.0:
            raise ValueError("trajectory frame has a degenerate box face")
        heights.append(volume / area)
    return tuple(heights)


def evaluate_local_frame(
    original_frame, whole_frame, msys_model, cms_model, protocol: dict,
    selections: Selections, logged_distance_A: float, logged_chi_rad: float,
    args,
) -> dict:
    from schrodinger.application.desmond.packages.analysis import Pbc

    recomputed_distance, recomputed_chi = common.recompute_frame_cvs(
        msys_model, cms_model, original_frame, protocol
    )
    distance_error = recomputed_distance - logged_distance_A
    chi_error_deg = math.degrees(float(circular_delta_radians(
        recomputed_chi, logged_chi_rad
    )))
    if abs(distance_error) > args.coordinate_distance_tolerance_A:
        raise ValueError(
            f"minimum-image distance differs from CVSEQ by {distance_error:.6g} A"
        )
    if abs(chi_error_deg) > args.coordinate_chi_tolerance_deg:
        raise ValueError(f"Q107 chi1 differs from CVSEQ by {chi_error_deg:.6g} degrees")

    center_0 = mass_center(
        whole_frame, selections.distance_gids[0], selections.distance_masses[0]
    )
    center_1 = mass_center(
        whole_frame, selections.distance_gids[1], selections.distance_masses[1]
    )
    direct_vector = center_1 - center_0
    direct_distance = float(np.linalg.norm(direct_vector))
    box = np.asarray(original_frame.box, dtype=float).reshape(3, 3)
    pbc = Pbc(box)
    minimum_vector = np.asarray(pbc.calcMinimumDiff(center_0, center_1), dtype=float)
    minimum_distance_from_whole = float(np.linalg.norm(minimum_vector))
    if abs(minimum_distance_from_whole - logged_distance_A) > args.coordinate_distance_tolerance_A:
        raise ValueError(
            "minimum-image distance from molecule-whole frame does not reproduce CVSEQ"
        )
    shift_vector = direct_vector - minimum_vector
    shift_coefficients = np.linalg.solve(box.T, shift_vector)
    integer_shift = np.rint(shift_coefficients).astype(int)
    lattice_residual = shift_vector - integer_shift @ box
    lattice_residual_norm = float(np.linalg.norm(lattice_residual))
    if lattice_residual_norm > args.lattice_residual_tolerance_A:
        raise ValueError(
            f"direct/minimum-image displacement is not a box lattice vector; "
            f"residual {lattice_residual_norm:.6g} A"
        )
    shift_vector_norm = float(np.linalg.norm(shift_vector))
    nonzero_d49r102_shift = bool(
        shift_vector_norm > args.d49r102_shift_tolerance_A
    )

    whole_chi = common.signed_dihedral_radians(
        np.asarray(whole_frame.pos(selections.chi_gids), dtype=float)
    )
    whole_chi_error_deg = math.degrees(float(circular_delta_radians(
        whole_chi, logged_chi_rad
    )))
    if abs(whole_chi_error_deg) > args.coordinate_chi_tolerance_deg:
        raise ValueError(
            f"molecule-whole Q107 chi1 differs from CVSEQ by "
            f"{whole_chi_error_deg:.6g} degrees"
        )

    ca_xyz = np.asarray(whole_frame.pos(selections.ca_gids), dtype=float)
    max_ca_step = float(np.max(np.linalg.norm(np.diff(ca_xyz, axis=0), axis=1)))
    quality_failures = []
    if max_ca_step > args.maximum_consecutive_ca_A:
        quality_failures.append(
            f"molecule-whole chain has a consecutive CA step of {max_ca_step:.6g} A"
        )

    zinc_xyz = np.asarray(whole_frame.pos(selections.zinc_gids), dtype=float)
    sulfur_xyz = np.asarray(
        whole_frame.pos(selections.cysteine_sulfur_gids), dtype=float
    )
    fourth_sulfur_distances = []
    for zinc in zinc_xyz:
        distances = np.sort(np.linalg.norm(sulfur_xyz - zinc, axis=1))
        fourth = float(distances[3])
        if fourth > args.zinc_fourth_sulfur_cutoff_A:
            quality_failures.append(
                f"Zn fourth-nearest cysteine S distance is {fourth:.6g} A"
            )
        fourth_sulfur_distances.append(fourth)

    face_heights = box_face_heights(box)
    return {
        "recomputed_minimum_image_distance_A": recomputed_distance,
        "minimum_image_distance_error_A": distance_error,
        "direct_whole_distance_A": direct_distance,
        "direct_minus_minimum_image_scalar_A": direct_distance - logged_distance_A,
        "periodic_shift_vector_norm_A": shift_vector_norm,
        "periodic_shift_nx": int(integer_shift[0]),
        "periodic_shift_ny": int(integer_shift[1]),
        "periodic_shift_nz": int(integer_shift[2]),
        "lattice_reconstruction_residual_A": lattice_residual_norm,
        "d49r102_nonzero_lattice_shift": nonzero_d49r102_shift,
        "recomputed_chi1_deg": math.degrees(recomputed_chi),
        "chi1_circular_error_deg": chi_error_deg,
        "direct_whole_chi1_deg": math.degrees(whole_chi),
        "direct_whole_chi1_error_deg": whole_chi_error_deg,
        "box_ax_A": float(box[0, 0]), "box_ay_A": float(box[0, 1]),
        "box_az_A": float(box[0, 2]), "box_bx_A": float(box[1, 0]),
        "box_by_A": float(box[1, 1]), "box_bz_A": float(box[1, 2]),
        "box_cx_A": float(box[2, 0]), "box_cy_A": float(box[2, 1]),
        "box_cz_A": float(box[2, 2]),
        "box_face_height_a_A": face_heights[0],
        "box_face_height_b_A": face_heights[1],
        "box_face_height_c_A": face_heights[2],
        "shortest_box_face_height_A": min(face_heights),
        "maximum_consecutive_ca_distance_A": max_ca_step,
        "candidate_structural_qc_pass": not quality_failures,
        "candidate_structural_qc_failure": "; ".join(quality_failures),
        "zinc_fourth_nearest_cysteine_sulfur_distances_A": ";".join(
            f"{value:.6f}" for value in fourth_sulfur_distances
        ),
    }


def candidate_row(
    run: RunInput, center: Center, frame_index: int, cv_index: int,
    trajectory_time_ps: float, cvseq_time_ps: float,
    logged_distance_A: float, logged_chi_rad: float,
    metrics: dict, args,
) -> dict:
    delta_distance = logged_distance_A - center.distance_A
    delta_chi_rad = float(circular_delta_radians(
        logged_chi_rad, math.radians(center.chi_deg)
    ))
    delta_chi_deg = math.degrees(delta_chi_rad)
    score = math.sqrt(
        (delta_distance / args.distance_score_scale_A) ** 2
        + (delta_chi_deg / args.chi_score_scale_deg) ** 2
    )
    preferred = (
        abs(delta_distance) <= args.preferred_distance_window_A
        and abs(delta_chi_deg) <= args.preferred_chi_window_deg
    )
    return {
        "center_id": center.identifier,
        "center_minimum_image_distance_A": center.distance_A,
        "center_chi1_canonical_deg": center.chi_deg,
        "center_smoothed_relative_fes_kcal_mol": (
            center.smoothed_relative_fes_kcal_mol
        ),
        "replicate": run.replicate,
        "trajectory_frame_zero_based": frame_index,
        "trajectory_frame_one_based": frame_index + 1,
        "cvseq_index_zero_based": cv_index,
        "time_ps": trajectory_time_ps,
        "time_ns": trajectory_time_ps / 1000.0,
        "trajectory_time_ps": trajectory_time_ps,
        "cvseq_time_ps": cvseq_time_ps,
        "dtr_cvseq_time_error_ps": abs(trajectory_time_ps - cvseq_time_ps),
        "logged_minimum_image_distance_A": logged_distance_A,
        "logged_q107_chi1_deg": math.degrees(logged_chi_rad),
        "distance_residual_from_center_A": delta_distance,
        "chi1_periodic_residual_from_center_deg": delta_chi_deg,
        "scaled_periodic_cv_distance": score,
        "within_preferred_half_cell_window": preferred,
        **metrics,
    }


def compact_index_time_records(indices: np.ndarray, times: np.ndarray) -> str:
    """Compactly serialize endpoint-only unmatched records for CSV/log output."""
    indices = np.asarray(indices, dtype=int)
    if not indices.size:
        return ""
    if indices.size <= 6:
        shown = indices
    else:
        shown = np.concatenate((indices[:3], indices[-3:]))
    records = [f"{int(index)}@{float(times[index]):.9g}" for index in shown]
    if indices.size > shown.size:
        records.insert(3, "...")
    return ";".join(records)


def map_all_cvseq_records_to_trajectory(
    run: RunInput, trajectory_times: np.ndarray, cvseq, protocol: dict, args,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Map each CVSEQ record to a unique exact-time DTR frame.

    Desmond can write a final trajectory-only endpoint frame. Such records are
    safe to exclude, but an unmatched CVSEQ record or an unmatched DTR frame
    inside the CVSEQ time range would make the coordinate/structure join
    incomplete and is therefore rejected.
    """
    trajectory_times = np.asarray(trajectory_times, dtype=float)
    cv_times = np.asarray(cvseq.time_ps, dtype=float)
    for label, times in (("trajectory", trajectory_times), ("CVSEQ", cv_times)):
        if not np.isfinite(times).all() or np.any(np.diff(times) <= 0.0):
            raise ValueError(f"{run.replicate}: {label} times are not finite/increasing")

    cv_spacing = float(np.median(np.diff(cv_times)))
    trajectory_spacing = float(np.median(np.diff(trajectory_times)))
    endpoint_slack_ps = 1.1 * max(cv_spacing, trajectory_spacing)
    expected_duration_ps = args.expected_duration_ns * 1000.0
    trajectory_duration_ps = float(trajectory_times[-1] - trajectory_times[0])
    cv_duration_ps = float(cv_times[-1] - cv_times[0])
    for label, duration_ps in (
        ("CVSEQ", cv_duration_ps), ("trajectory", trajectory_duration_ps),
    ):
        if abs(duration_ps - expected_duration_ps) > endpoint_slack_ps:
            raise ValueError(
                f"{run.replicate}: {label} duration is {duration_ps / 1000.0:.6g} ns, "
                f"expected approximately {args.expected_duration_ns:g} ns"
            )
    if (
        abs(trajectory_times[0] - cv_times[0]) > endpoint_slack_ps
        or abs(trajectory_times[-1] - cv_times[-1]) > endpoint_slack_ps
    ):
        raise ValueError(
            f"{run.replicate}: trajectory/CVSEQ endpoints differ by more than one "
            f"output interval (DTR {trajectory_times[0]:.9g}--"
            f"{trajectory_times[-1]:.9g} ps; CVSEQ {cv_times[0]:.9g}--"
            f"{cv_times[-1]:.9g} ps)"
        )

    time_tolerance_ps = min(
        args.maximum_time_tolerance_ps,
        0.001 * float(protocol["hill_interval_ps"]),
    )
    matched_frames, matched_cv, errors = common.exact_time_match_subset(
        trajectory_times, cv_times, time_tolerance_ps
    )
    all_frames = np.arange(trajectory_times.size, dtype=int)
    all_cv = np.arange(cv_times.size, dtype=int)
    unmatched_frames = np.setdiff1d(all_frames, matched_frames, assume_unique=True)
    unmatched_cv = np.setdiff1d(all_cv, matched_cv, assume_unique=True)
    internal_unmatched_frames = unmatched_frames[
        (trajectory_times[unmatched_frames] >= cv_times[0] - time_tolerance_ps)
        & (trajectory_times[unmatched_frames] <= cv_times[-1] + time_tolerance_ps)
    ]
    if (
        matched_cv.size != cv_times.size
        or not np.array_equal(matched_cv, all_cv)
        or internal_unmatched_frames.size
    ):
        raise ValueError(
            f"{run.replicate}: incomplete exact DTR/CVSEQ join: "
            f"DTR={trajectory_times.size}, CVSEQ={cv_times.size}, "
            f"matches={matched_frames.size}, unmatched_DTR={unmatched_frames.size} "
            f"[{compact_index_time_records(unmatched_frames, trajectory_times)}], "
            f"unmatched_CVSEQ={unmatched_cv.size} "
            f"[{compact_index_time_records(unmatched_cv, cv_times)}]"
        )

    frame_for_cv = np.empty(cv_times.size, dtype=int)
    frame_for_cv[matched_cv] = matched_frames
    summary = {
        "raw_trajectory_frames": int(trajectory_times.size),
        "cvseq_records": int(cv_times.size),
        "exact_time_matches": int(matched_frames.size),
        "unmatched_trajectory_endpoint_frames_excluded": int(unmatched_frames.size),
        "unmatched_trajectory_endpoint_index_time_ps": compact_index_time_records(
            unmatched_frames, trajectory_times
        ),
        "unmatched_cvseq_records": int(unmatched_cv.size),
        "trajectory_start_ps": float(trajectory_times[0]),
        "trajectory_end_ps": float(trajectory_times[-1]),
        "cvseq_start_ps": float(cv_times[0]),
        "cvseq_end_ps": float(cv_times[-1]),
        "trajectory_median_spacing_ps": trajectory_spacing,
        "cvseq_median_spacing_ps": cv_spacing,
        "maximum_time_error_ps": float(np.max(errors)),
        "time_match_tolerance_ps": time_tolerance_ps,
    }
    return frame_for_cv, matched_frames, matched_cv, summary


def process_run(run: RunInput, protocol: dict, centers: Sequence[Center], args):
    from schrodinger.application.desmond.packages import topo, traj

    print(f"[{run.replicate}] reading topology, trajectory, and CVSEQ", flush=True)
    cvseq = common.parse_cvseq(run.cvseq)
    msys_model, cms_model = topo.read_cms(str(run.cms))
    frames = traj.read_traj(str(run.trajectory))
    if len(frames) < 2:
        raise ValueError(f"{run.replicate}: trajectory has fewer than two frames")
    frame_particle_counts = {int(frame.natoms) for frame in frames}
    if frame_particle_counts != {int(msys_model.natoms)}:
        raise ValueError(
            f"{run.replicate}: trajectory particle counts {sorted(frame_particle_counts)} "
            f"do not match MSYS particle count {int(msys_model.natoms)}"
        )
    consistency = topo.check_consistency(cms_model, frames[0])
    if consistency:
        raise ValueError(f"{run.replicate}: CMS/trajectory mismatch: {consistency}")
    signature, system_signature, residue_count, zinc_count = common.topology_signature(
        msys_model, cms_model, frames[0].natoms
    )
    if residue_count != args.expected_protein_residues or zinc_count != args.expected_zinc:
        raise ValueError(
            f"{run.replicate}: found {residue_count} protein residues and {zinc_count} Zn; "
            f"expected {args.expected_protein_residues} and {args.expected_zinc}"
        )
    selections = build_selections(
        msys_model, cms_model, protocol, args.structure_asl,
        args.expected_protein_residues,
    )
    trajectory_times = np.asarray([float(frame.time) for frame in frames], dtype=float)
    frame_for_cv, matched_frames, matched_cv, alignment = (
        map_all_cvseq_records_to_trajectory(
            run, trajectory_times, cvseq, protocol, args
        )
    )
    active = np.arange(cvseq.time_ps.size, dtype=int)
    cv_for_frame = {
        int(frame_index): int(cv_index)
        for cv_index, frame_index in enumerate(frame_for_cv)
    }

    frame_centers: dict[int, list[Center]] = {}
    for center in centers:
        distance_delta = np.abs(cvseq.distance_A[active] - center.distance_A)
        chi_delta_deg = np.abs(np.rad2deg(circular_delta_radians(
            cvseq.chi_rad[active], math.radians(center.chi_deg)
        )))
        local = active[
            (distance_delta <= args.fallback_distance_window_A)
            & (chi_delta_deg <= args.fallback_chi_window_deg)
        ]
        for cv_index in local:
            frame_index = int(frame_for_cv[int(cv_index)])
            frame_centers.setdefault(frame_index, []).append(center)

    print(
        f"[{run.replicate}] validating {len(frame_centers)} local candidate frames",
        flush=True,
    )
    rows = []
    frame_indices = sorted(frame_centers)
    for start in range(0, len(frame_indices), args.glue_batch_size):
        batch_indices = frame_indices[start:start + args.glue_batch_size]
        originals = [frames[index] for index in batch_indices]
        whole_frames = topo.glue(
            msys_model, selections.structure_gids,
            [frame.copy() for frame in originals],
        )
        for frame_index, original, whole in zip(batch_indices, originals, whole_frames):
            cv_index = cv_for_frame[frame_index]
            metrics = evaluate_local_frame(
                original, whole, msys_model, cms_model, protocol, selections,
                float(cvseq.distance_A[cv_index]), float(cvseq.chi_rad[cv_index]), args,
            )
            for center in frame_centers[frame_index]:
                rows.append(candidate_row(
                    run, center, frame_index, cv_index,
                    float(trajectory_times[frame_index]),
                    float(cvseq.time_ps[cv_index]),
                    float(cvseq.distance_A[cv_index]),
                    float(cvseq.chi_rad[cv_index]), metrics, args,
                ))
    del frames, cms_model, msys_model

    summaries = []
    best_rows = []
    for center in centers:
        local = [row for row in rows if row["center_id"] == center.identifier]
        local.sort(key=lambda row: (
            row["scaled_periodic_cv_distance"],
            row["trajectory_frame_zero_based"],
        ))
        eligible = [
            row for row in local
            if not row["d49r102_nonzero_lattice_shift"]
            and row["candidate_structural_qc_pass"]
        ]
        preferred_eligible = [
            row for row in eligible if row["within_preferred_half_cell_window"]
        ]
        raw_best = local[0] if local else None
        safe_best = (
            preferred_eligible[0] if preferred_eligible
            else eligible[0] if eligible
            else None
        )
        if safe_best is not None:
            candidate_type = (
                "nearest_zero_shift_qc_pass_preferred"
                if preferred_eligible
                else "nearest_zero_shift_qc_pass_fallback"
            )
            best_rows.append({"candidate_type": candidate_type, **safe_best})
        if raw_best is not None:
            best_rows.append({"candidate_type": "nearest_regardless_of_shift_or_qc", **raw_best})
        summaries.append({
            "center_id": center.identifier,
            "replicate": run.replicate,
            "raw_local_candidate_count_not_population": len(local),
            "raw_local_nonzero_d49r102_shift_count_not_population": sum(
                bool(row["d49r102_nonzero_lattice_shift"]) for row in local
            ),
            "raw_local_nonzero_d49r102_shift_fraction_not_population": (
                sum(bool(row["d49r102_nonzero_lattice_shift"]) for row in local) / len(local)
                if local else math.nan
            ),
            "raw_local_structural_qc_rejection_count_not_population": sum(
                not row["candidate_structural_qc_pass"] for row in local
            ),
            "has_zero_shift_qc_pass_candidate": bool(eligible),
            "has_preferred_zero_shift_qc_pass_candidate": bool(preferred_eligible),
            "selected_zero_shift_qc_pass_scaled_cv_distance": (
                safe_best["scaled_periodic_cv_distance"] if safe_best else math.nan
            ),
            "selected_zero_shift_qc_pass_within_preferred_half_cell_window": (
                safe_best["within_preferred_half_cell_window"] if safe_best else False
            ),
            "minimum_direct_whole_distance_A_in_local_window": (
                min(row["direct_whole_distance_A"] for row in local) if local else math.nan
            ),
            "maximum_direct_whole_distance_A_in_local_window": (
                max(row["direct_whole_distance_A"] for row in local) if local else math.nan
            ),
        })
    return rows, best_rows, summaries, {
        "replicate": run.replicate,
        **alignment,
        "local_candidate_frames_validated": len(frame_indices),
        "protein_signature_sha256": signature,
        "system_signature_sha256": system_signature,
        "protein_residue_count": residue_count,
        "zinc_count": zinc_count,
    }


def preflight_run(run: RunInput, protocol: dict, args) -> dict:
    """Exercise trajectory/CV/topology/PBC APIs on early, middle, and late matches."""
    from schrodinger.application.desmond.packages import topo, traj

    cvseq = common.parse_cvseq(run.cvseq)
    msys_model, cms_model = topo.read_cms(str(run.cms))
    frames = traj.read_traj(str(run.trajectory))
    if len(frames) < 3:
        raise ValueError(f"{run.replicate}: trajectory has fewer than three frames")
    consistency = topo.check_consistency(cms_model, frames[0])
    if consistency:
        raise ValueError(f"{run.replicate}: CMS/trajectory mismatch: {consistency}")
    frame_particle_counts = {int(frame.natoms) for frame in frames}
    if frame_particle_counts != {int(msys_model.natoms)}:
        raise ValueError(f"{run.replicate}: MSYS/trajectory particle counts differ")
    _, _, residue_count, zinc_count = common.topology_signature(
        msys_model, cms_model, frames[0].natoms
    )
    if residue_count != args.expected_protein_residues or zinc_count != args.expected_zinc:
        raise ValueError(
            f"{run.replicate}: found {residue_count} protein residues and {zinc_count} Zn; "
            f"expected {args.expected_protein_residues} and {args.expected_zinc}"
        )
    selections = build_selections(
        msys_model, cms_model, protocol, args.structure_asl,
        args.expected_protein_residues,
    )
    trajectory_times = np.asarray([float(frame.time) for frame in frames], dtype=float)
    frame_for_cv, matched_frames, matched_cv, alignment = (
        map_all_cvseq_records_to_trajectory(
            run, trajectory_times, cvseq, protocol, args
        )
    )
    sample_cv_indices = sorted({
        0, len(cvseq.time_ps) // 2, len(cvseq.time_ps) - 1,
    })
    sample_frame_indices = [int(frame_for_cv[index]) for index in sample_cv_indices]
    originals = [frames[index] for index in sample_frame_indices]
    whole_frames = topo.glue(
        msys_model, selections.structure_gids,
        [frame.copy() for frame in originals],
    )
    sample_shift_flags = []
    sample_qc_flags = []
    for frame_index, cv_index, original, whole in zip(
        sample_frame_indices, sample_cv_indices, originals, whole_frames
    ):
        metrics = evaluate_local_frame(
            original, whole, msys_model, cms_model, protocol, selections,
            float(cvseq.distance_A[cv_index]), float(cvseq.chi_rad[cv_index]), args,
        )
        sample_shift_flags.append(bool(metrics["d49r102_nonzero_lattice_shift"]))
        sample_qc_flags.append(bool(metrics["candidate_structural_qc_pass"]))
    return {
        "replicate": run.replicate,
        **alignment,
        "sampled_preflight_cvseq_indices": ";".join(
            str(index) for index in sample_cv_indices
        ),
        "sampled_preflight_trajectory_frame_indices": ";".join(
            str(index) for index in sample_frame_indices
        ),
        "sampled_preflight_nonzero_d49r102_shift_flags": sample_shift_flags,
        "sampled_preflight_structural_qc_pass_flags": sample_qc_flags,
    }


def minimum_solute_image_distance(frame, heavy_gids: Sequence[int]) -> float:
    """Diagnostic over the 26 adjacent lattice translations.

    This is suitable for the near-orthogonal production boxes here; it is not a
    general closest-vector solver for an arbitrarily skewed unreduced lattice.
    """
    xyz = np.asarray(frame.pos(heavy_gids), dtype=float)
    box = np.asarray(frame.box, dtype=float).reshape(3, 3)
    translations = [
        np.asarray(coefficients, dtype=float) @ box
        for coefficients in product((-1, 0, 1), repeat=3)
        if coefficients != (0, 0, 0)
    ]
    images = np.concatenate([xyz + shift for shift in translations], axis=0)
    tree = cKDTree(images)
    distances, _ = tree.query(xyz, k=1)
    return float(np.min(distances))


def extract_structures(
    chosen: Sequence[dict], runs: Sequence[RunInput], protocols: Sequence[dict],
    output_dir: Path, args,
) -> list[dict]:
    from schrodinger import structure
    from schrodinger.application.desmond.packages import topo, traj

    representatives_dir = output_dir / "representatives"
    representatives_dir.mkdir(parents=True, exist_ok=True)
    combined_path = representatives_dir / "all_D49R102_zero_shift_terminal_fes_exemplars.maegz"
    combined_writer = structure.MaestroWriter(str(combined_path))
    output_rows = []
    try:
        for run_index, (run, protocol) in enumerate(zip(runs, protocols)):
            selected = [row for row in chosen if row["run_index"] == run_index]
            if not selected:
                continue
            msys_model, cms_model = topo.read_cms(str(run.cms))
            frames = traj.read_traj(str(run.trajectory))
            selections = build_selections(
                msys_model, cms_model, protocol, args.structure_asl,
                args.expected_protein_residues,
            )
            expected_identity = [
                (
                    str(cms_model.fsys_ct.atom[aid].chain).strip(),
                    int(cms_model.fsys_ct.atom[aid].resnum),
                    str(cms_model.fsys_ct.atom[aid].inscode).strip(),
                    str(cms_model.fsys_ct.atom[aid].pdbres).strip(),
                    str(cms_model.fsys_ct.atom[aid].pdbname).strip(),
                    str(cms_model.fsys_ct.atom[aid].element).strip(),
                )
                for aid in selections.structure_aids
            ]
            for row in selected:
                frame_index = int(row["trajectory_frame_zero_based"])
                if frame_index < 0 or frame_index >= len(frames):
                    raise ValueError(
                        f"{row['center_id']}: trajectory frame {frame_index} is out of range"
                    )
                original = frames[frame_index]
                if not math.isclose(
                    float(original.time), float(row["time_ps"]),
                    rel_tol=0.0, abs_tol=1e-8,
                ):
                    raise ValueError(
                        f"{row['center_id']}: frame time changed between selection and extraction"
                    )
                whole = topo.glue(
                    msys_model, selections.structure_gids, [original.copy()]
                )[0]
                metrics = evaluate_local_frame(
                    original, whole, msys_model, cms_model, protocol, selections,
                    float(row["logged_minimum_image_distance_A"]),
                    math.radians(float(row["logged_q107_chi1_deg"])), args,
                )
                if metrics["d49r102_nonzero_lattice_shift"]:
                    raise ValueError(
                        f"{row['center_id']}: selected frame has a nonzero D49-R102 "
                        "lattice shift during extraction"
                    )
                if not metrics["candidate_structural_qc_pass"]:
                    raise ValueError(
                        f"{row['center_id']}: selected frame fails structural QC during "
                        f"extraction: {metrics['candidate_structural_qc_failure']}"
                    )
                st = cms_model.extract(selections.structure_aids)
                xyz = np.asarray(whole.pos(selections.structure_gids), dtype=float)
                if int(st.atom_total) != xyz.shape[0]:
                    raise ValueError("extracted atom count does not match structure selection")
                observed_identity = [
                    (
                        str(atom.chain).strip(), int(atom.resnum),
                        str(atom.inscode).strip(), str(atom.pdbres).strip(),
                        str(atom.pdbname).strip(), str(atom.element).strip(),
                    )
                    for atom in st.atom
                ]
                if observed_identity != expected_identity:
                    raise ValueError("extracted atom identity/order is inconsistent")
                st.setXYZ(xyz)
                zinc_atom_indices = {
                    int(atom.index)
                    for atom in st.atom
                    if str(atom.element).strip().upper() == "ZN"
                }
                zinc_zero_order_counts = {
                    atom_index: 0 for atom_index in zinc_atom_indices
                }
                for bond in st.bond:
                    if bond.order != 0:
                        continue
                    atom_1, atom_2 = list(bond.atom)
                    pair = ((atom_1, atom_2), (atom_2, atom_1))
                    for zinc_atom, partner in pair:
                        if int(zinc_atom.index) not in zinc_atom_indices:
                            continue
                        if (
                            str(partner.element).strip().upper() == "S"
                            and str(partner.pdbres).strip().upper() in {"CYS", "CYX"}
                        ):
                            zinc_zero_order_counts[int(zinc_atom.index)] += 1
                if sorted(zinc_zero_order_counts.values()) != [4, 4]:
                    raise ValueError(
                        "extracted MAEGZ structure does not contain four zero-order "
                        "cysteine-S bonds for each Zn"
                    )
                topo.update_ct_box(st, np.asarray(whole.box, dtype=float))
                st.title = (
                    f"{row['center_id']} D49-R102-zero-shift terminal-FES exemplar; "
                    f"{run.replicate}; {row['time_ns']:.6f} ns"
                )
                st.property["s_user_center_id"] = row["center_id"]
                st.property["s_user_replicate"] = run.replicate
                st.property["i_user_trajectory_frame_zero_based"] = frame_index
                st.property["r_user_time_ns"] = float(row["time_ns"])
                st.property["r_user_biased_minimum_image_com_A"] = float(
                    row["logged_minimum_image_distance_A"]
                )
                st.property["r_user_direct_whole_com_A"] = float(
                    metrics["direct_whole_distance_A"]
                )
                st.property["r_user_q107_chi1_deg"] = float(row["logged_q107_chi1_deg"])
                safe_id = row["center_id"].replace("+", "plus").replace("-", "minus")
                base = (
                    f"{safe_id}_CV-nearest_D49R102-zero-shift_{run.replicate}_"
                    f"frame{frame_index:06d}_{row['time_ns']:.3f}ns"
                )
                maegz = representatives_dir / f"{base}.maegz"
                pdb = representatives_dir / f"{base}.pdb"
                st.write(str(maegz))
                st.write(str(pdb))
                if not maegz.is_file() or maegz.stat().st_size == 0:
                    raise ValueError(f"failed to write {maegz}")
                if not pdb.is_file() or pdb.stat().st_size == 0:
                    raise ValueError(f"failed to write {pdb}")
                pdb_has_cryst1 = any(
                    line.startswith("CRYST1")
                    for line in pdb.read_text(
                        encoding="utf-8", errors="replace"
                    ).splitlines()
                )
                if not pdb_has_cryst1:
                    raise ValueError(f"exported PDB does not retain its CRYST1 box: {pdb}")
                combined_writer.append(st)
                solute_image_distance = minimum_solute_image_distance(
                    whole, selections.heavy_structure_gids
                )
                output_rows.append({
                    **{key: value for key, value in row.items() if key != "run_index"},
                    **metrics,
                    "diagnostic_minimum_protein_heavy_atom_to_26_neighbor_images_A": (
                        solute_image_distance
                    ),
                    "pdb_contains_cryst1_record": pdb_has_cryst1,
                    "maegz_zinc_zero_order_cysteine_s_bond_counts": ";".join(
                        str(value) for value in sorted(zinc_zero_order_counts.values())
                    ),
                    "output_maegz": str(maegz.resolve()),
                    "output_pdb": str(pdb.resolve()),
                })
            del frames, cms_model, msys_model
    finally:
        combined_writer.close()
    return output_rows


def packaged_or_source_fes_file(filename: str) -> Path:
    packaged = Path(__file__).with_name(filename)
    if packaged.is_file():
        return packaged
    source_tree = Path("data/wt_metad") / filename
    return source_tree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--e6-auto", action="store_true")
    source.add_argument("--manifest", type=Path)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument(
        "--centers", type=Path,
        default=packaged_or_source_fes_file(
            "apo16e6_wtmetad_periodic_2d_minima.csv"
        ),
    )
    parser.add_argument(
        "--fes-manifest", type=Path,
        default=packaged_or_source_fes_file("apo16e6_wtmetad_manifest.json"),
        help="Pinned provenance manifest for the five-run terminal Desmond FES mixture",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--expected-ktemp", type=float, default=8.4)
    parser.add_argument("--structure-asl", default="(protein) or a.e Zn")
    parser.add_argument("--expected-protein-residues", type=int, default=151)
    parser.add_argument("--expected-zinc", type=int, default=2)
    parser.add_argument("--expected-duration-ns", type=float, default=1000.0)
    parser.add_argument("--distance-score-scale-A", type=float, default=0.68)
    parser.add_argument("--chi-score-scale-deg", type=float, default=7.2)
    parser.add_argument("--preferred-distance-window-A", type=float, default=0.34)
    parser.add_argument("--preferred-chi-window-deg", type=float, default=3.6)
    parser.add_argument("--fallback-distance-window-A", type=float, default=0.70)
    parser.add_argument("--fallback-chi-window-deg", type=float, default=7.2)
    parser.add_argument("--coordinate-distance-tolerance-A", type=float, default=0.01)
    parser.add_argument("--coordinate-chi-tolerance-deg", type=float, default=0.1)
    parser.add_argument("--d49r102-shift-tolerance-A", type=float, default=0.05)
    parser.add_argument("--lattice-residual-tolerance-A", type=float, default=0.01)
    parser.add_argument("--maximum-consecutive-ca-A", type=float, default=4.5)
    parser.add_argument("--zinc-fourth-sulfur-cutoff-A", type=float, default=3.0)
    parser.add_argument("--maximum-time-tolerance-ps", type=float, default=0.01)
    parser.add_argument("--glue-batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    numeric_positive = [
        "expected_ktemp", "expected_duration_ns", "distance_score_scale_A",
        "chi_score_scale_deg",
        "preferred_distance_window_A", "preferred_chi_window_deg",
        "fallback_distance_window_A", "fallback_chi_window_deg",
        "coordinate_distance_tolerance_A", "coordinate_chi_tolerance_deg",
        "d49r102_shift_tolerance_A", "lattice_residual_tolerance_A",
        "maximum_consecutive_ca_A", "zinc_fourth_sulfur_cutoff_A",
        "maximum_time_tolerance_ps", "glue_batch_size",
    ]
    for name in numeric_positive:
        if float(getattr(args, name)) <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.preferred_distance_window_A > args.fallback_distance_window_A:
        raise ValueError("preferred distance window exceeds fallback window")
    if args.preferred_chi_window_deg > args.fallback_chi_window_deg:
        raise ValueError("preferred chi window exceeds fallback window")

    centers = read_centers(args.centers)
    source_fes_manifest = validate_source_fes_manifest(args.fes_manifest)
    runs = auto_runs(args.input_root) if args.e6_auto else read_manifest(args.manifest)
    reference_protocol, protocols = validate_runs(runs, args.expected_ktemp)
    if args.dry_run:
        summaries = [
            preflight_run(run, protocol, args)
            for run, protocol in zip(runs, protocols)
        ]
        for summary in summaries:
            print(
                f"{summary['replicate']}: DTR={summary['raw_trajectory_frames']}; "
                f"CVSEQ={summary['cvseq_records']}; "
                f"exact matches={summary['exact_time_matches']}; "
                f"max time error={summary['maximum_time_error_ps']:.3g} ps; "
                f"DTR range={summary['trajectory_start_ps']:.9g}--"
                f"{summary['trajectory_end_ps']:.9g} ps "
                f"(median dt={summary['trajectory_median_spacing_ps']:.9g} ps); "
                f"CVSEQ range={summary['cvseq_start_ps']:.9g}--"
                f"{summary['cvseq_end_ps']:.9g} ps "
                f"(median dt={summary['cvseq_median_spacing_ps']:.9g} ps); "
                f"excluded endpoint-only DTR frames="
                f"{summary['unmatched_trajectory_endpoint_frames_excluded']} "
                f"[{summary['unmatched_trajectory_endpoint_index_time_ps']}]; "
                f"unmatched CVSEQ={summary['unmatched_cvseq_records']}; "
                f"sampled nonzero-shift flags "
                f"{summary['sampled_preflight_nonzero_d49r102_shift_flags']}; "
                f"sampled structural-QC pass flags "
                f"{summary['sampled_preflight_structural_qc_pass_flags']}"
            )
        print(
            "DRY RUN PASS: pinned nine centers, five protocol-matched runs, all "
            "CVSEQ records uniquely mapped, and early/middle/late structures validated."
        )
        return 0
    if args.output_dir is None:
        raise ValueError("--output-dir is required unless --dry-run is used")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory must be new or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "STATUS.json", {
        "status": "RUNNING",
        "started_utc": datetime.now(timezone.utc).isoformat(),
    })
    try:
        all_candidates = []
        per_run_best = []
        local_summaries = []
        run_summaries = []
        for run, protocol in zip(runs, protocols):
            candidates, best, summaries, run_summary = process_run(
                run, protocol, centers, args
            )
            all_candidates.extend(candidates)
            per_run_best.extend(best)
            local_summaries.extend(summaries)
            run_summaries.append(run_summary)

        if len({row["protein_signature_sha256"] for row in run_summaries}) != 1:
            raise ValueError("the five runs do not contain the same protein atom identity/order")
        if len({row["system_signature_sha256"] for row in run_summaries}) != 1:
            raise ValueError("the five runs do not contain the same full simulated system")

        chosen = []
        missing = []
        fallback_selected = []
        center_coverage = []
        for center in centers:
            local = [
                row for row in all_candidates
                if row["center_id"] == center.identifier
            ]
            eligible = [
                row for row in local
                if not row["d49r102_nonzero_lattice_shift"]
                and row["candidate_structural_qc_pass"]
            ]
            preferred = [
                row for row in eligible if row["within_preferred_half_cell_window"]
            ]
            candidates = preferred if preferred else eligible
            candidates.sort(key=lambda row: (
                row["scaled_periodic_cv_distance"], row["replicate"],
                row["trajectory_frame_zero_based"],
            ))
            if not candidates:
                missing.append(center.identifier)
                if not local:
                    missing_reason = "NO_LOCAL_CANDIDATE"
                elif not any(
                    not row["d49r102_nonzero_lattice_shift"] for row in local
                ):
                    missing_reason = "NO_ZERO_D49R102_SHIFT_LOCAL_CANDIDATE"
                else:
                    missing_reason = "NO_ZERO_SHIFT_STRUCTURAL_QC_PASS_CANDIDATE"
                center_coverage.append({
                    "center_id": center.identifier,
                    "center_minimum_image_distance_A": center.distance_A,
                    "center_chi1_canonical_deg": center.chi_deg,
                    "exemplar_status": missing_reason,
                    "selected_replicate": "",
                    "selected_frame_zero_based": "",
                    "selected_time_ns": "",
                })
                continue
            selected = dict(candidates[0])
            selected["run_index"] = next(
                index for index, run in enumerate(runs)
                if run.replicate == selected["replicate"]
            )
            chosen.append(selected)
            selection_tier = "SELECTED_PREFERRED" if preferred else "SELECTED_FALLBACK"
            if not preferred:
                fallback_selected.append(center.identifier)
            center_coverage.append({
                "center_id": center.identifier,
                "center_minimum_image_distance_A": center.distance_A,
                "center_chi1_canonical_deg": center.chi_deg,
                "exemplar_status": selection_tier,
                "selected_replicate": selected["replicate"],
                "selected_frame_zero_based": selected["trajectory_frame_zero_based"],
                "selected_time_ns": selected["time_ns"],
            })
        write_csv_gz(output_dir / "candidate_diagnostics.csv.gz", all_candidates)
        write_csv(output_dir / "per_run_best_candidates.csv", per_run_best)
        write_csv(output_dir / "local_d49r102_shift_summary.csv", local_summaries)
        combined_shift_summaries = []
        for center in centers:
            local = [
                row for row in all_candidates if row["center_id"] == center.identifier
            ]
            per_run_fractions = [
                row["raw_local_nonzero_d49r102_shift_fraction_not_population"]
                for row in local_summaries
                if row["center_id"] == center.identifier
                and math.isfinite(
                    row["raw_local_nonzero_d49r102_shift_fraction_not_population"]
                )
            ]
            shifted_count = sum(
                row["d49r102_nonzero_lattice_shift"] for row in local
            )
            combined_shift_summaries.append({
                "center_id": center.identifier,
                "pooled_raw_candidate_count_not_population": len(local),
                "pooled_raw_nonzero_shift_count_not_population": shifted_count,
                "pooled_raw_nonzero_shift_fraction_not_population": (
                    shifted_count / len(local) if local else math.nan
                ),
                "equal_replicate_mean_raw_shift_fraction_not_population": (
                    float(np.mean(per_run_fractions)) if per_run_fractions else math.nan
                ),
                "replicates_with_local_candidates": len(per_run_fractions),
                "zero_shift_structural_qc_pass_candidate_count_not_population": sum(
                    not row["d49r102_nonzero_lattice_shift"]
                    and row["candidate_structural_qc_pass"]
                    for row in local
                ),
            })
        write_csv(
            output_dir / "combined_d49r102_shift_summary.csv",
            combined_shift_summaries,
        )
        write_csv(output_dir / "run_summary.csv", run_summaries)
        write_csv(output_dir / "center_coverage.csv", center_coverage)

        representatives = extract_structures(
            chosen, runs, protocols, output_dir, args
        )
        write_csv(output_dir / "representatives.csv", representatives)
        mixed_shift_centers = []
        for center in centers:
            local = [
                row for row in all_candidates if row["center_id"] == center.identifier
            ]
            if local and any(
                row["d49r102_nonzero_lattice_shift"] for row in local
            ) and any(
                not row["d49r102_nonzero_lattice_shift"] for row in local
            ):
                mixed_shift_centers.append(center.identifier)

        input_rows = []
        for run in runs:
            input_rows.append({
                "replicate": run.replicate,
                "cfg": str(run.cfg), "cfg_sha256": sha256(run.cfg),
                "cvseq": str(run.cvseq), "cvseq_sha256": sha256(run.cvseq),
                "cms": str(run.cms), "cms_sha256": sha256(run.cms),
                "trajectory": str(run.trajectory),
            })
        manifest = {
            "analysis": "CV-nearest structures for the five-run terminal Desmond FES mixture",
            "trajectory_reweighting_used": False,
            "source_fes_manifest": str(args.fes_manifest.resolve()),
            "source_fes_manifest_sha256": sha256(args.fes_manifest.resolve()),
            "source_fes_combination": source_fes_manifest["combination"],
            "center_file": str(args.centers.resolve()),
            "center_file_sha256": sha256(args.centers.resolve()),
            "extractor_script_sha256": sha256(Path(__file__).resolve()),
            "parser_helper_sha256": sha256(Path(common.__file__).resolve()),
            "centers": [center.__dict__ for center in centers],
            "inputs": input_rows,
            "protocol": reference_protocol,
            "run_summaries": run_summaries,
            "selection": {
                "structure_cv_join": (
                    "every CVSEQ record is mapped to one unique DTR frame within "
                    f"{min(args.maximum_time_tolerance_ps, 0.001 * float(reference_protocol['hill_interval_ps'])):g} ps; "
                    "trajectory-only endpoint frames outside the CVSEQ time range are "
                    "reported and excluded"
                ),
                "ranking": (
                    "Euclidean distance in D49-R102/Q107 periodic CV space, scaled by "
                    f"{args.distance_score_scale_A:g} A and {args.chi_score_scale_deg:g} deg"
                ),
                "preferred_window": [
                    args.preferred_distance_window_A, args.preferred_chi_window_deg,
                ],
                "fallback_window": [
                    args.fallback_distance_window_A, args.fallback_chi_window_deg,
                ],
                "selection_priority": (
                    "nearest zero-D49-R102-shift structural-QC-pass candidate in the "
                    "preferred window; use the fallback window only if none exists"
                ),
                "d49r102_shift_rule": (
                    "only candidates whose direct-minus-minimum-image displacement vector "
                    f"is <= {args.d49r102_shift_tolerance_A:g} A are exportable"
                ),
            },
            "mixed_zero_and_nonzero_d49r102_shift_centers": mixed_shift_centers,
            "centers_without_zero_shift_qc_pass_exemplar": missing,
            "centers_selected_from_fallback_window": fallback_selected,
            "scientific_scope": (
                "Descriptive coordinate-matched exemplars only. The nine centers are "
                "algorithm-detected minima of a smoothed descriptive mixture, not validated "
                "equilibrium basins, populations, or kinetic states. A zero D49-R102 shift "
                "does not prove absence of protein self-image contacts or remove periodic-image "
                "ambiguity from the underlying FES region."
            ),
            "outputs": {
                "representatives_csv": str((output_dir / "representatives.csv").resolve()),
                "candidate_diagnostics_csv_gz": str(
                    (output_dir / "candidate_diagnostics.csv.gz").resolve()
                ),
                "local_d49r102_shift_summary_csv": str(
                    (output_dir / "local_d49r102_shift_summary.csv").resolve()
                ),
                "combined_d49r102_shift_summary_csv": str(
                    (output_dir / "combined_d49r102_shift_summary.csv").resolve()
                ),
                "center_coverage_csv": str(
                    (output_dir / "center_coverage.csv").resolve()
                ),
                "run_summary_csv": str(
                    (output_dir / "run_summary.csv").resolve()
                ),
                "combined_maegz": str((
                    output_dir
                    / "representatives/all_D49R102_zero_shift_terminal_fes_exemplars.maegz"
                ).resolve()),
                "output_checksums": str(
                    (output_dir / "OUTPUT_SHA256SUMS.txt").resolve()
                ),
            },
        }
        write_json(output_dir / "analysis_manifest.json", manifest)
        read_before = (
            "These structures are CV-nearest sampled exemplars near nine algorithm-detected "
            "minima of the smoothed equal-replicate terminal Desmond FES mixture. No trajectory "
            "weights were used to select them. Every selected structure was joined to its "
            "CVSEQ coordinate by an exact timestamp; any trajectory-only endpoint frame was "
            "reported and excluded. They are not equilibrium-basin representatives, "
            "medoids, or population estimates. Each export has a zero D49-R102 lattice shift; "
            "this does not prove that the protein lacks self-image contacts or that its FES "
            "region lacks periodic-coordinate mixing. A center with no qualifying local frame "
            "is intentionally left without an exemplar. Review center_coverage.csv and "
            "local_d49r102_shift_summary.csv, and compare the 26-neighbor protein-image-distance "
            "diagnostic with the actual production interaction cutoff before use. The MAEGZ "
            "files are topology-authoritative; PDB files are for visualization and cannot "
            "preserve the zero-order Zn coordination bonds.\n"
        )
        (output_dir / "READ_BEFORE_USE.txt").write_text(read_before, encoding="utf-8")
        write_output_checksums(output_dir)
        completion_status = (
            "COMPLETE_WITH_MISSING_EXEMPLARS" if missing
            else "COMPLETE_WITH_FALLBACK_EXEMPLARS" if fallback_selected
            else "COMPLETE"
        )
        status = {
            "status": completion_status,
            "representatives_extracted": len(representatives),
            "centers_without_zero_shift_qc_pass_exemplar": missing,
            "centers_selected_from_fallback_window": fallback_selected,
            "individual_representative_coordinate_validation": (
                "PASS_CV_MATCH_AND_ZERO_D49R102_SHIFT"
            ),
            "physical_basin_interpretation": (
                "BLOCKED_MIXED_D49R102_PERIODIC_SHIFT_REGIONS" if mixed_shift_centers
                else "BLOCKED_MISSING_PHYSICAL_EXEMPLARS" if missing
                else "PENDING_FULL_TRAJECTORY_PBC_REVIEW"
            ),
            "mixed_zero_and_nonzero_d49r102_shift_centers": mixed_shift_centers,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_json(output_dir / "STATUS.json", status)
        print(
            f"{completion_status}: extracted {len(representatives)} zero-D49-R102-shift "
            f"exemplars to {output_dir}"
        )
        if missing:
            print("NO SAFE EXEMPLAR for: " + ", ".join(missing))
        if fallback_selected:
            print("FALLBACK-WINDOW EXEMPLAR for: " + ", ".join(fallback_selected))
        if mixed_shift_centers:
            print(
                "CAUTION: local regions with both zero and nonzero D49-R102 shifts: "
                + ", ".join(mixed_shift_centers)
            )
        return 0
    except Exception as exc:
        write_json(output_dir / "STATUS.json", {
            "status": "FAILED",
            "error": str(exc),
            "finished_utc": datetime.now(timezone.utc).isoformat(),
        })
        raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
