#!/usr/bin/env python3
"""Provide Desmond parsers and optional five-run trajectory reweighting.
Inputs: native CFG, CMS, CVSEQ, KERSEQ and trajectories; outputs: FES and structures.
Parser dependency for V12 Figure 6; its reweighted FES is not the plotted surface.
Run from the repository root; see this folder's README for commands.
"""

# Reweight five matched Desmond WT-MetaD runs and extract basin frames.
# 
# This script is specialized for the five apo-16E6 runs dated 8.10.25 through
# 8.14.25, but a manifest mode is provided for equivalent data sets.  Each
# independent bias history is reweighted separately with the time-dependent
# well-tempered offset c(t).  Only the five normalized probability surfaces are
# then averaged; kernel histories and free-energy values are never concatenated
# or averaged.
# 
# Required per run
# ----------------
# * Desmond ``*-out.cfg``
# * ``*.kerseq``
# * ``*.cvseq``
# * matching ``*-out.cms``
# * matching native ``*_trj`` directory (or an explicitly supplied XTC)
# 
# Run with Schrödinger's Python, for example::
# 
#     "$SCHRODINGER/run" wtmetad_5rep_reweight_basins.py         --e6-auto --input-root trajectories/wt_metad         --ct-distance-min-A 0 --ct-distance-max-A 40         --output-dir wtmetad_5rep_reweighted_fes
# 
# The automatically detected basin count is deliberately not fixed at 9 or 12.
# Representatives are actual sampled frames nearest the basin minima in the two
# biased collective variables; they are not all-atom structural medoids.

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import heapq
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import NoReturn, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter, minimum_filter
from scipy.special import expit, logsumexp


KB_KCAL_PER_MOL_K = 0.00198720425864083
DEFAULT_INPUT_ROOT = Path("trajectories/wt_metad")
E6_STEMS = (
    "desmond_WT_MetaD_4xr8_chainH_E6_1us_d49r102_q107chi1_8.10.25",
    "desmond_WT_MetaD_4xr8_chainH_E6_1us_d49r102_q107chi1_8.11.25",
    "desmond_WT_MetaD_4xr8_chainH_E6_1us_d49r102_q107chi1_8.12.25",
    "desmond_WT_MetaD_4xr8_chainH_E6_1us_d49r102_q107chi1_8.13.25",
    "desmond_WT_MetaD_4xr8_chainH_E6_1us_d49r102_q107chi1_8.14.25",
)
PALETTE = ["#7589CB", "#B7C9EE", "#F3EFEA", "#F2B9A4", "#D67878"]
CVSEQ_FIELD = re.compile(r"\[\s*([A-Za-z0-9_$.-]+)\s+\[([^\[\]]*)\]\s*\]")


@dataclass(frozen=True)
class RunSpec:
    replicate: str
    stem: str
    cfg: Path
    kerseq: Path
    cvseq: Path
    cms: Path
    trajectory: Path


@dataclass(frozen=True)
class CVSequence:
    time_ps: np.ndarray
    distance_A: np.ndarray
    chi_rad: np.ndarray
    potential_kcal_mol: np.ndarray


@dataclass(frozen=True)
class Hills:
    time_ps: np.ndarray
    height_kcal_mol: np.ndarray
    centers: np.ndarray
    scales: np.ndarray
    cutoff: float


@dataclass
class RunData:
    spec: RunSpec
    cvseq_index: np.ndarray
    time_ps: np.ndarray
    distance_A: np.ndarray
    chi_rad: np.ndarray
    total_potential_kcal_mol: np.ndarray
    meta_bias_kcal_mol: np.ndarray
    wall_kcal_mol: np.ndarray
    c_t_kcal_mol: np.ndarray
    log_weight: np.ndarray
    weight: np.ndarray
    probability: np.ndarray | None = None
    fes: np.ndarray | None = None
    basin: np.ndarray | None = None
    trajectory_frame_index: np.ndarray | None = None
    trajectory_time_ps: np.ndarray | None = None
    trajectory_distance_A: np.ndarray | None = None
    trajectory_chi_rad: np.ndarray | None = None
    trajectory_basin: np.ndarray | None = None
    importance_ess: float = math.nan
    max_weight_fraction: float = math.nan
    time_match_max_error_ps: float = math.nan
    matched_trajectory_frame_count: int = 0
    bias_validation_max_error: float = math.nan
    distance_validation_max_error_A: float = math.nan
    chi_validation_max_error_deg: float = math.nan
    ct_omitted_kernel_tail_upper_bound_kcal_mol: float = math.nan
    protein_signature: str = ""
    system_signature: str = ""
    protein_residue_count: int = 0
    zinc_count: int = 0


def fail(message: str) -> NoReturn:
    raise SystemExit(f"ERROR: {message}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(json_safe(payload), indent=2) + "\n", encoding="utf-8")


def _sea_plain(value):
    if hasattr(value, "val"):
        return _sea_plain(value.val)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return float(value)
    if isinstance(value, (list, tuple)):
        return [_sea_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _sea_plain(item) for key, item in value.items()}
    return str(value)


def _sea_value(mapping, key, default=None):
    try:
        if key in mapping:
            return _sea_plain(mapping[key])
    except (KeyError, TypeError, AttributeError):
        pass
    return default


def protocol_signature(cfg_path: Path) -> dict:
    """Read the material WT-MetaD settings with Schrödinger SEA."""
    try:
        from schrodinger.utils import sea
    except ImportError as exc:
        fail("Schrödinger modules are unavailable; run with $SCHRODINGER/run")
        raise exc

    try:
        cfg = sea.Map(cfg_path.read_text(encoding="utf-8"))
        meta_cfg = cfg.ORIG_CFG.meta
        temperature = float(cfg.integrator.temperature.T_ref.val)
    except Exception as exc:
        raise ValueError(f"{cfg_path}: cannot read ORIG_CFG.meta and temperature") from exc

    cvs = []
    try:
        cv_entries = meta_cfg.cv
    except AttributeError as exc:
        raise ValueError(f"{cfg_path}: ORIG_CFG.meta.cv is missing") from exc
    for index, cv in enumerate(cv_entries, start=1):
        cv_type = _sea_value(cv, "type")
        atoms = _sea_value(cv, "atom")
        width = _sea_value(cv, "width")
        if cv_type is None or atoms is None or width is None:
            raise ValueError(f"{cfg_path}: CV {index} is incomplete")
        cvs.append({
            "type": cv_type,
            "atom": atoms,
            "width": width,
            "range": _sea_value(cv, "range"),
            "wall": _sea_value(cv, "wall"),
            "floor": _sea_value(cv, "floor"),
            "nbin": _sea_value(cv, "nbin"),
        })
    return {
        "temperature_K": temperature,
        "kTemp_kcal_mol": _sea_value(meta_cfg, "kTemp"),
        "initial_hill_height_kcal_mol": _sea_value(meta_cfg, "height"),
        "first_hill_time_ps": _sea_value(meta_cfg, "first"),
        "last_hill_time_ps": _sea_value(meta_cfg, "last"),
        "hill_interval_ps": _sea_value(meta_cfg, "interval"),
        "kernel_cutoff_widths": _sea_value(meta_cfg, "cutoff", 9.0),
        "initial_bias": _sea_value(meta_cfg, "initial"),
        "collective_variables": cvs,
    }


def validate_protocol(protocol: dict, expected_ktemp: float | None) -> None:
    cvs = protocol["collective_variables"]
    if len(cvs) != 2 or [entry["type"] for entry in cvs] != ["dist", "dihedral"]:
        raise ValueError("expected exactly a distance CV followed by a dihedral CV")
    if protocol["kTemp_kcal_mol"] is None:
        raise ValueError("meta.kTemp is absent; this is not a WT-MetaD configuration")
    if expected_ktemp is not None and not math.isclose(
        float(protocol["kTemp_kcal_mol"]), expected_ktemp, rel_tol=0.0, abs_tol=1e-8
    ):
        raise ValueError(
            f"expected kTemp={expected_ktemp:g} kcal/mol, found "
            f"{protocol['kTemp_kcal_mol']} kcal/mol"
        )
    if not math.isclose(float(cvs[0]["width"]), 0.8, abs_tol=1e-8):
        raise ValueError(f"distance width is {cvs[0]['width']}, expected 0.8 Å")
    if not math.isclose(float(cvs[1]["width"]), 12.0, abs_tol=1e-8):
        raise ValueError(f"dihedral width is {cvs[1]['width']}, expected 12 degrees")
    if cvs[0]["wall"] is None or not math.isclose(
        float(cvs[0]["wall"]), 40.0, abs_tol=1e-8
    ):
        raise ValueError(f"distance wall is {cvs[0]['wall']}, expected 40 Å")
    if cvs[0]["floor"] is not None:
        raise ValueError("distance CV has a floor that this script does not model")
    if cvs[1]["wall"] is not None or cvs[1]["floor"] is not None:
        raise ValueError("dihedral CV has a wall/floor that this script does not model")
    if protocol.get("initial_bias") not in (None, "", []):
        raise ValueError(
            "the run declares an initial metadynamics bias; this script requires the "
            "complete deposited history in the supplied kerseq"
        )


def locate_file(directory: Path, names: Sequence[str], description: str) -> Path:
    candidates = [directory / name for name in names]
    found = [path for path in candidates if path.is_file() and path.stat().st_size > 0]
    if not found:
        raise ValueError(
            f"{description} not found; expected one of: "
            + ", ".join(str(path) for path in candidates)
        )
    return found[0].resolve()


def locate_trajectory(directory: Path, stem: str) -> Path:
    native_candidates = [
        directory / f"{stem}_trj",
        directory / f"{stem}-out_trj",
    ]
    native_found = [path for path in native_candidates if path.is_dir()]
    if len(native_found) > 1:
        raise ValueError(
            f"found multiple native DTR trajectories for {stem}: "
            + ", ".join(str(path) for path in native_found)
        )
    if native_found:
        return native_found[0].resolve()
    xtc = directory / f"{stem}.xtc"
    if xtc.is_file() and xtc.stat().st_size > 0:
        return xtc.resolve()
    raise ValueError(
        f"trajectory not found for {stem}; expected one of: "
        + ", ".join(str(path) for path in (*native_candidates, xtc))
    )


def auto_run_specs(root: Path) -> list[RunSpec]:
    root = root.expanduser().resolve()
    runs = []
    for index, stem in enumerate(E6_STEMS, start=1):
        directory = root / stem
        if not directory.is_dir():
            raise ValueError(f"missing run directory: {directory}")
        date_token = stem.rsplit("_", 1)[-1]
        runs.append(RunSpec(
            replicate=f"rep{index:02d}_{date_token}",
            stem=stem,
            cfg=locate_file(directory, [f"{stem}-out.cfg"], "configuration"),
            kerseq=locate_file(directory, [f"{stem}.kerseq"], "kernel sequence"),
            cvseq=locate_file(directory, [f"{stem}.cvseq"], "CV sequence"),
            cms=locate_file(
                directory,
                [f"{stem}-out.cms", f"{stem}.cms", f"{stem}-in.cms"],
                "CMS topology",
            ),
            trajectory=locate_trajectory(directory, stem),
        ))
    return runs


def read_manifest(path: Path) -> list[RunSpec]:
    path = path.expanduser().resolve()
    text = path.read_text(encoding="utf-8-sig")
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",\t")
    except csv.Error:
        dialect = csv.excel_tab if path.suffix.lower() == ".tsv" else csv.excel
    reader = csv.DictReader(text.splitlines(), dialect=dialect)
    required = {
        "replicate", "cfg_file", "kerseq_file", "cvseq_file", "cms_file", "trajectory"
    }
    if not reader.fieldnames or not required.issubset(reader.fieldnames):
        missing = sorted(required.difference(reader.fieldnames or []))
        raise ValueError(f"manifest is missing columns: {missing}")

    runs = []
    seen = set()
    for line_number, row in enumerate(reader, start=2):
        replicate = (row.get("replicate") or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", replicate):
            raise ValueError(f"manifest line {line_number}: invalid replicate {replicate!r}")
        if replicate in seen:
            raise ValueError(f"manifest line {line_number}: duplicate replicate {replicate}")
        seen.add(replicate)

        resolved = {}
        for column in required.difference({"replicate"}):
            raw = (row.get(column) or "").strip()
            if not raw:
                raise ValueError(f"manifest line {line_number}: empty {column}")
            item = Path(raw).expanduser()
            if not item.is_absolute():
                item = path.parent / item
            item = item.resolve()
            if column == "trajectory":
                if not item.exists():
                    raise ValueError(f"manifest line {line_number}: missing {column}: {item}")
            elif not item.is_file() or item.stat().st_size == 0:
                raise ValueError(f"manifest line {line_number}: missing/empty {column}: {item}")
            resolved[column] = item
        stem = (row.get("stem") or resolved["kerseq_file"].stem).strip()
        runs.append(RunSpec(
            replicate=replicate,
            stem=stem,
            cfg=resolved["cfg_file"],
            kerseq=resolved["kerseq_file"],
            cvseq=resolved["cvseq_file"],
            cms=resolved["cms_file"],
            trajectory=resolved["trajectory"],
        ))
    if not runs:
        raise ValueError("manifest contains no runs")
    return runs


def validate_input_set(runs: Sequence[RunSpec], expected_runs: int,
                       expected_ktemp: float | None) -> tuple[dict, list[dict]]:
    if len(runs) != expected_runs:
        raise ValueError(f"expected {expected_runs} runs, found {len(runs)}")
    protocols = []
    reference = None
    seen_paths: dict[tuple[str, str], str] = {}
    seen_hashes: dict[tuple[str, str], str] = {}
    for run in runs:
        protocol = protocol_signature(run.cfg)
        validate_protocol(protocol, expected_ktemp)
        if reference is None:
            reference = protocol
        elif protocol != reference:
            differences = [key for key in reference if protocol.get(key) != reference.get(key)]
            raise ValueError(
                f"{run.replicate}: protocol differs from {runs[0].replicate} in {differences}"
            )
        protocols.append(protocol)
        for role, path in (
            ("cfg", run.cfg), ("kerseq", run.kerseq), ("cvseq", run.cvseq),
            ("cms", run.cms), ("trajectory", run.trajectory),
        ):
            key = (role, str(path))
            if role in {"kerseq", "cvseq", "trajectory"}:
                if key in seen_paths:
                    raise ValueError(
                        f"{run.replicate}: {role} path is reused from "
                        f"{seen_paths[key]}: {path}"
                    )
                seen_paths[key] = run.replicate
            # Identical configuration and CMS content is legitimate for independent
            # replicas.  Identical CV or hill histories, however, indicate accidental
            # reuse rather than independent trajectories.
            if role in {"kerseq", "cvseq"}:
                digest = sha256(path)
                hash_key = (role, digest)
                if hash_key in seen_hashes:
                    raise ValueError(
                        f"{run.replicate}: {role} content duplicates "
                        f"{seen_hashes[hash_key]}"
                    )
                seen_hashes[hash_key] = run.replicate
    assert reference is not None
    return reference, protocols


def parse_cvseq(path: Path) -> CVSequence:
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = {}
            for key, values_text in CVSEQ_FIELD.findall(stripped):
                try:
                    values = [float(value) for value in values_text.split()]
                except ValueError as exc:
                    raise ValueError(f"{path}: nonnumeric {key} on line {line_number}") from exc
                if not values:
                    raise ValueError(f"{path}: empty {key} on line {line_number}")
                fields[key] = values
            missing = [key for key in ("time", "cv_00", "cv_01", "potential") if key not in fields]
            if missing:
                raise ValueError(f"{path}: line {line_number} lacks {missing}")
            rows.append(tuple(fields[key][0] for key in ("time", "cv_00", "cv_01", "potential")))
    if len(rows) < 3:
        raise ValueError(f"{path}: fewer than three CV records")
    array = np.asarray(rows, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(f"{path}: CV sequence contains NaN or infinity")
    if np.any(np.diff(array[:, 0]) <= 0.0):
        raise ValueError(f"{path}: CV times are not strictly increasing")
    return CVSequence(
        time_ps=array[:, 0],
        distance_A=array[:, 1],
        chi_rad=wrap_radians(array[:, 2]),
        potential_kcal_mol=array[:, 3],
    )


def load_hills(run: RunSpec) -> Hills:
    try:
        from schrodinger.application.desmond import meta
    except ImportError as exc:
        fail("Schrödinger modules are unavailable; run with $SCHRODINGER/run")
        raise exc
    analysis = meta.MetaDynamicsAnalysis(str(run.kerseq), inp_fname=str(run.cfg))
    if list(analysis.cv) != ["dist", "dihedral"]:
        raise ValueError(f"{run.replicate}: parsed CV types are {analysis.cv}")
    times = np.asarray(analysis.time, dtype=float)
    heights = np.asarray(analysis.height, dtype=float)
    centers = np.asarray(analysis.centers, dtype=float)
    scales = np.asarray(analysis.scales, dtype=float)
    if times.size < 2 or centers.shape != scales.shape or centers.shape != (times.size, 2):
        raise ValueError(f"{run.replicate}: malformed parsed kernel arrays")
    cutoff = float(analysis.cutoff)
    if (
        heights.shape != times.shape
        or not np.isfinite(times).all()
        or not np.isfinite(heights).all()
        or not np.isfinite(centers).all()
        or not np.isfinite(scales).all()
        or not math.isfinite(cutoff)
        or cutoff <= 0.0
        or np.any(heights <= 0.0)
        or np.any(scales <= 0.0)
    ):
        raise ValueError(f"{run.replicate}: invalid hill heights or widths")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError(f"{run.replicate}: nonzero hill times are not strictly increasing")
    return Hills(times, heights, centers, scales, cutoff)


def wrap_radians(angle: np.ndarray | float) -> np.ndarray:
    return (np.asarray(angle, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def circular_difference_radians(a, b):
    return wrap_radians(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))


def desmond_wall(distance_A, location_A: float, width_A: float = 0.2,
                 height_kcal_mol: float = 1000.0):
    """Standard wall generated by Desmond's metadynamics panel."""
    return height_kcal_mol * expit((np.asarray(distance_A, dtype=float) - location_A) / width_A)


def evaluate_raw_bias(hills: Hills, distance_A: float, chi_rad: float,
                      time_ps: float) -> float:
    """Evaluate V_meta from hills deposited strictly before ``time_ps``."""
    end = int(np.searchsorted(hills.time_ps, time_ps, side="left"))
    if end == 0:
        return 0.0
    centers = hills.centers[:end]
    scales = hills.scales[:end]
    z2 = ((distance_A - centers[:, 0]) * scales[:, 0]) ** 2
    z2 += ((math.cos(chi_rad) - np.cos(centers[:, 1])) * scales[:, 1]) ** 2
    z2 += ((math.sin(chi_rad) - np.sin(centers[:, 1])) * scales[:, 1]) ** 2
    selected = z2 < hills.cutoff ** 2
    return float(np.sum(hills.height_kcal_mol[:end][selected] * np.exp(-0.5 * z2[selected])))


def validate_cvseq_potential(cvseq: CVSequence, hills: Hills, wall_A: float,
                             points: int, tolerance: float) -> tuple[float, list[dict]]:
    indices = np.unique(np.linspace(0, len(cvseq.time_ps) - 1, points, dtype=int))
    records = []
    for index in indices:
        calculated = evaluate_raw_bias(
            hills,
            float(cvseq.distance_A[index]),
            float(cvseq.chi_rad[index]),
            float(cvseq.time_ps[index]),
        )
        wall = float(desmond_wall(cvseq.distance_A[index], wall_A))
        reported_meta = float(cvseq.potential_kcal_mol[index] - wall)
        error = reported_meta - calculated
        records.append({
            "cvseq_index": int(index),
            "time_ps": float(cvseq.time_ps[index]),
            "reported_total_potential_kcal_mol": float(cvseq.potential_kcal_mol[index]),
            "wall_kcal_mol": wall,
            "reported_meta_bias_kcal_mol": reported_meta,
            "reconstructed_meta_bias_kcal_mol": calculated,
            "error_kcal_mol": error,
        })
    maximum = max(abs(row["error_kcal_mol"]) for row in records)
    if maximum > tolerance:
        raise ValueError(
            f"cvseq potential does not equal reconstructed metadynamics bias plus the "
            f"standard wall (maximum error {maximum:.6g} kcal/mol > {tolerance:g}); "
            "check for extra restraints, restarted kernel history, or mismatched files"
        )
    return maximum, records


def add_hill_block_to_grid(bias_grid: np.ndarray, hills: Hills, start: int, end: int,
                           distance_grid: np.ndarray, chi_grid: np.ndarray,
                           chunk_size: int = 500) -> None:
    """Increment a bias grid using separability of distance and circular hills.

    This retains Gaussian tails outside Desmond's joint cutoff.  Their strict
    pointwise upper bound is recorded and checked by the caller; at cutoff 9 it
    is normally far below floating-point-relevant energy scales.
    """
    cos_grid = np.cos(chi_grid)
    sin_grid = np.sin(chi_grid)
    for block_start in range(start, end, chunk_size):
        block_end = min(block_start + chunk_size, end)
        centers = hills.centers[block_start:block_end]
        scales = hills.scales[block_start:block_end]
        heights = hills.height_kcal_mol[block_start:block_end]
        d_basis = np.exp(
            -0.5 * ((distance_grid[None, :] - centers[:, 0, None]) * scales[:, 0, None]) ** 2
        )
        angular_z2 = (
            ((cos_grid[None, :] - np.cos(centers[:, 1, None])) * scales[:, 1, None]) ** 2
            + ((sin_grid[None, :] - np.sin(centers[:, 1, None])) * scales[:, 1, None]) ** 2
        )
        a_basis = np.exp(-0.5 * angular_z2)
        bias_grid += (d_basis.T * heights[None, :]) @ a_basis


def compute_c_t(hills: Hills, checkpoint_times_ps: np.ndarray,
                distance_min_A: float, distance_max_A: float,
                distance_step_A: float, angle_step_deg: float,
                kbt: float, ktemp: float) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                                   np.ndarray, np.ndarray]:
    """Compute c(t) at exact requested times on one fixed integration grid."""
    distance_cells_float = (distance_max_A - distance_min_A) / distance_step_A
    distance_cells = int(round(distance_cells_float))
    if not math.isclose(
        distance_cells_float, distance_cells, rel_tol=0.0, abs_tol=1e-8
    ):
        raise ValueError("c(t) distance step must divide its integration range exactly")
    distance_grid = distance_min_A + (
        np.arange(distance_cells, dtype=float) + 0.5
    ) * distance_step_A
    angle_cells = int(round(360.0 / angle_step_deg))
    chi_grid = np.deg2rad(
        -180.0 + (np.arange(angle_cells, dtype=float) + 0.5) * angle_step_deg
    )
    if distance_grid.size < 3 or chi_grid.size < 12:
        raise ValueError("c(t) integration grid is too coarse")
    checkpoints = np.asarray(checkpoint_times_ps, dtype=float)
    if checkpoints.ndim != 1 or checkpoints.size < 2:
        raise ValueError("c(t) requires at least two checkpoint times")
    if not np.isfinite(checkpoints).all() or np.any(np.diff(checkpoints) <= 0.0):
        raise ValueError("c(t) checkpoint times must be finite and strictly increasing")

    bias_grid = np.zeros((distance_grid.size, chi_grid.size), dtype=float)
    c_values = np.zeros(checkpoints.size, dtype=float)
    previous_end = 0
    # beta*gamma/(gamma-1) = 1/kBT + 1/kTemp; beta/(gamma-1) = 1/kTemp.
    numerator_coefficient = 1.0 / kbt + 1.0 / ktemp
    denominator_coefficient = 1.0 / ktemp
    # c(t) is identical for consecutive CV records that see the same number of
    # previously deposited hills. Evaluate each distinct bias state only once.
    hill_counts = np.searchsorted(hills.time_ps, checkpoints, side="left")
    unique_counts, first_indices, repetitions = np.unique(
        hill_counts, return_index=True, return_counts=True
    )
    for end_raw, first_index, repetition_count in zip(
        unique_counts, first_indices, repetitions
    ):
        end = int(end_raw)
        if end > previous_end:
            add_hill_block_to_grid(
                bias_grid, hills, previous_end, end, distance_grid, chi_grid
            )
            previous_end = end
        c_value = kbt * (
            logsumexp(numerator_coefficient * bias_grid)
            - logsumexp(denominator_coefficient * bias_grid)
        )
        c_values[int(first_index):int(first_index + repetition_count)] = c_value
    return checkpoints, c_values, distance_grid, chi_grid, bias_grid


def exact_time_match_subset(trajectory_times: np.ndarray, cv_times: np.ndarray,
                            tolerance_ps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return only DTR frames with a same-time cvseq record; do not force pairing."""
    insertions = np.searchsorted(cv_times, trajectory_times)
    right = np.clip(insertions, 0, cv_times.size - 1)
    left = np.clip(insertions - 1, 0, cv_times.size - 1)
    choose_right = np.abs(cv_times[right] - trajectory_times) < np.abs(
        cv_times[left] - trajectory_times
    )
    matched = np.where(choose_right, right, left)
    errors = np.abs(cv_times[matched] - trajectory_times)
    keep = errors <= tolerance_ps
    trajectory_indices = np.flatnonzero(keep).astype(int)
    cv_indices = matched[keep].astype(int)
    if np.unique(cv_indices).size != cv_indices.size:
        raise ValueError("multiple trajectory frames map to one same-time cvseq record")
    return trajectory_indices, cv_indices, errors[keep]


def signed_dihedral_radians(points: np.ndarray) -> float:
    """Signed four-point convention used in the existing E6 coordinate audit."""
    atom1, atom2, atom3, atom4 = np.asarray(points, dtype=float)
    r12 = atom1 - atom2
    r32 = atom3 - atom2
    r42 = atom4 - atom2
    cross_a = np.cross(r12, r32)
    cross_b = np.cross(r42, r32)
    denominator = np.linalg.norm(cross_a) * np.linalg.norm(cross_b)
    if denominator == 0.0:
        raise ValueError("cannot calculate a dihedral from collinear coordinates")
    cosine = float(np.clip(np.dot(cross_a, cross_b) / denominator, -1.0, 1.0))
    angle = math.acos(cosine)
    if float(np.dot(r12, cross_b)) < 0.0:
        angle *= -1.0
    return float(wrap_radians(angle))


def _msys_masses(msys_model, gids: Sequence[int]) -> np.ndarray:
    """Return particle masses without relying on Cms.particle_total/id_maps."""
    masses = np.asarray(
        [float(msys_model.atom(int(gid)).mass) for gid in gids], dtype=float
    )
    if not np.isfinite(masses).all():
        raise ValueError("MSYS contains non-finite particle masses")
    return masses


def topology_signature(
    msys_model, cms_model, trajectory_particle_total: int
) -> tuple[str, str, int, int]:
    protein_aids = list(cms_model.select_atom("protein"))
    zinc_aids = list(cms_model.select_atom("a.e Zn"))
    if not protein_aids:
        raise ValueError("CMS contains no atoms selected by 'protein'")
    descriptors = []
    residues = set()
    for aid in protein_aids:
        atom = cms_model.fsys_ct.atom[int(aid)]
        descriptor = (
            str(atom.chain).strip(), int(atom.resnum), str(atom.inscode).strip(),
            str(atom.pdbres).strip(), str(atom.pdbname).strip(), str(atom.element).strip(),
        )
        descriptors.append(descriptor)
        residues.add(descriptor[:4])
    digest = hashlib.sha256(repr(descriptors).encode("utf-8")).hexdigest()
    particle_total = int(msys_model.natoms)
    trajectory_particle_total = int(trajectory_particle_total)
    if particle_total != trajectory_particle_total:
        raise ValueError(
            "MSYS/trajectory particle-count mismatch: "
            f"{particle_total} versus {trajectory_particle_total}"
        )
    particles = [msys_model.atom(gid) for gid in range(particle_total)]
    atomic_numbers = np.asarray(
        [int(atom.atomic_number) for atom in particles], dtype="<i4"
    )
    masses = np.asarray([float(atom.mass) for atom in particles], dtype="<f8")
    charges = np.asarray([float(atom.charge) for atom in particles], dtype="<f8")
    if not np.isfinite(masses).all() or not np.isfinite(charges).all():
        raise ValueError("MSYS contains non-finite particle masses or charges")

    system_digest = hashlib.sha256()
    system_digest.update(
        f"atoms={cms_model.fsys_ct.atom_total};particles={particle_total}\n".encode()
    )
    for atom in cms_model.fsys_ct.atom:
        identity = (
            int(atom.index), str(atom.chain).strip(), int(atom.resnum),
            str(atom.inscode).strip(), str(atom.pdbres).strip(),
            str(atom.pdbname).strip(), str(atom.element).strip(),
            int(getattr(atom, "formal_charge", 0)),
        )
        system_digest.update((repr(identity) + "\n").encode())
    for bond in cms_model.fsys_ct.bond:
        bond_record = (
            min(int(bond.atom1.index), int(bond.atom2.index)),
            max(int(bond.atom1.index), int(bond.atom2.index)),
            float(bond.order),
        )
        system_digest.update((repr(bond_record) + "\n").encode())
    system_digest.update(atomic_numbers.tobytes())
    system_digest.update(masses.tobytes())
    system_digest.update(charges.tobytes())
    return digest, system_digest.hexdigest(), len(residues), len(zinc_aids)


def _selection_aids(cms_model, selection) -> list[int]:
    if isinstance(selection, str):
        return [int(value) for value in cms_model.select_atom(selection)]
    if isinstance(selection, (int, float)):
        return [int(selection)]
    return [int(value) for value in selection]


def validate_e6_cv_semantics(msys_model, cms_model, protocol: dict) -> dict:
    """Prove that the generic configured CV groups are the labeled E6 coordinates."""
    protein_aids = [int(value) for value in cms_model.select_atom("protein")]
    distance_groups = protocol["collective_variables"][0]["atom"]
    if len(distance_groups) != 2:
        raise ValueError("distance CV must contain exactly two atom groups")
    expected_residues = ((49, "ASP"), (102, "ARG"))
    distance_records = []
    for group_number, (selection, expected) in enumerate(
        zip(distance_groups, expected_residues), start=1
    ):
        aids = sorted(_selection_aids(cms_model, selection))
        if not aids:
            raise ValueError(f"distance CV group {group_number} selects no atoms")
        residue_keys = {
            (
                str(cms_model.fsys_ct.atom[aid].chain).strip(),
                int(cms_model.fsys_ct.atom[aid].resnum),
                str(cms_model.fsys_ct.atom[aid].inscode).strip(),
                str(cms_model.fsys_ct.atom[aid].pdbres).strip(),
            )
            for aid in aids
        }
        if len(residue_keys) != 1:
            raise ValueError(
                f"distance CV group {group_number} does not select exactly one residue"
            )
        residue_key = next(iter(residue_keys))
        if (residue_key[1], residue_key[3]) != expected:
            raise ValueError(
                f"distance CV group {group_number} is {residue_key[3]}{residue_key[1]}, "
                f"expected {expected[1]}{expected[0]}"
            )
        whole_residue_aids = sorted(
            aid for aid in protein_aids
            if (
                str(cms_model.fsys_ct.atom[aid].chain).strip(),
                int(cms_model.fsys_ct.atom[aid].resnum),
                str(cms_model.fsys_ct.atom[aid].inscode).strip(),
                str(cms_model.fsys_ct.atom[aid].pdbres).strip(),
            ) == residue_key
        )
        if aids != whole_residue_aids:
            missing = sorted(set(whole_residue_aids).difference(aids))
            extra = sorted(set(aids).difference(whole_residue_aids))
            raise ValueError(
                f"distance CV group {group_number} is not the complete prepared residue "
                f"{expected[1]}{expected[0]} (missing AIDs {missing}; extra AIDs {extra})"
            )
        gids = topo_aids2gids_no_pseudo(cms_model, aids)
        distance_records.append({
            "group": group_number,
            "residue": f"{expected[1]}{expected[0]}",
            "chain": residue_key[0],
            "atom_count": len(aids),
            "aids": aids,
            "atom_names": [
                str(cms_model.fsys_ct.atom[aid].pdbname).strip() for aid in aids
            ],
            "total_mass_Da": float(np.sum(_msys_masses(msys_model, gids))),
        })

    dihedral_aids = _selection_aids(
        cms_model, protocol["collective_variables"][1]["atom"]
    )
    if len(dihedral_aids) != 4:
        raise ValueError("Q107 chi1 CV must contain exactly four atoms")
    dihedral_atoms = [cms_model.fsys_ct.atom[aid] for aid in dihedral_aids]
    atom_names = [str(atom.pdbname).strip() for atom in dihedral_atoms]
    residue_ids = {
        (str(atom.chain).strip(), int(atom.resnum), str(atom.inscode).strip(),
         str(atom.pdbres).strip())
        for atom in dihedral_atoms
    }
    if len(residue_ids) != 1:
        raise ValueError("Q107 chi1 atoms do not belong to one residue")
    residue_id = next(iter(residue_ids))
    if (residue_id[1], residue_id[3]) != (107, "GLN"):
        raise ValueError(
            f"dihedral CV is {residue_id[3]}{residue_id[1]}, expected GLN107"
        )
    if atom_names != ["N", "CA", "CB", "CG"]:
        raise ValueError(
            f"Q107 chi1 atoms are {atom_names}, expected ['N', 'CA', 'CB', 'CG']"
        )
    return {
        "distance_definition": "mass-weighted complete prepared residues ASP49 and ARG102",
        "distance_groups": distance_records,
        "dihedral_definition": "GLN107 N-CA-CB-CG",
        "dihedral_aids": dihedral_aids,
        "dihedral_atom_names": atom_names,
    }


def topo_aids2gids_no_pseudo(cms_model, aids: Sequence[int]) -> list[int]:
    """Late import keeps pure numerical tests independent of Schrödinger."""
    from schrodinger.application.desmond.packages import topo

    return topo.aids2gids(cms_model, aids, include_pseudoatoms=False)


def recompute_frame_cvs(msys_model, cms_model, frame, protocol: dict) -> tuple[float, float]:
    from schrodinger.application.desmond.packages import topo
    from schrodinger.application.desmond.packages.analysis import Pbc

    pbc = Pbc(frame.box)
    distance_groups = protocol["collective_variables"][0]["atom"]
    if len(distance_groups) != 2:
        raise ValueError("distance CV does not contain two groups")
    centers = []
    for group in distance_groups:
        aids = _selection_aids(cms_model, group)
        gids = topo.aids2gids(cms_model, aids, include_pseudoatoms=False)
        xyz = np.asarray(frame.pos(gids), dtype=float)
        reference = np.repeat(xyz[:1], xyz.shape[0], axis=0)
        xyz = reference + np.asarray(pbc.calcMinimumDiff(reference, xyz), dtype=float)
        masses = _msys_masses(msys_model, gids)
        centers.append(np.sum(xyz * masses[:, None], axis=0) / np.sum(masses))
    difference = np.asarray(pbc.calcMinimumDiff(centers[0], centers[1]))
    distance = float(np.linalg.norm(difference))

    dihedral_aids = _selection_aids(
        cms_model, protocol["collective_variables"][1]["atom"]
    )
    if len(dihedral_aids) != 4:
        raise ValueError("dihedral CV does not contain four atoms")
    dihedral_gids = topo.aids2gids(cms_model, dihedral_aids, include_pseudoatoms=False)
    raw = np.asarray(frame.pos(dihedral_gids), dtype=float)
    unwrapped = np.empty_like(raw)
    unwrapped[0] = raw[0]
    for index in range(1, 4):
        unwrapped[index] = unwrapped[index - 1] + np.asarray(
            pbc.calcMinimumDiff(raw[index - 1], raw[index]), dtype=float
        )
    chi = signed_dihedral_radians(unwrapped)
    return distance, chi


def trapezoid_time_weights(times_ps: np.ndarray) -> np.ndarray:
    if times_ps.size == 1:
        return np.ones(1)
    gaps = np.diff(times_ps)
    if np.any(gaps <= 0.0):
        raise ValueError("selected frame times are not strictly increasing")
    weights = np.empty(times_ps.size, dtype=float)
    weights[0] = gaps[0] / 2.0
    weights[-1] = gaps[-1] / 2.0
    if times_ps.size > 2:
        weights[1:-1] = (times_ps[2:] - times_ps[:-2]) / 2.0
    return weights


def process_run(run: RunSpec, protocol: dict, args, output_dir: Path) -> RunData:
    from schrodinger.application.desmond.packages import topo, traj

    print(f"[{run.replicate}] reading CV and hill histories", flush=True)
    cvseq = parse_cvseq(run.cvseq)
    hills = load_hills(run)
    hill_distance_min = float(np.min(hills.centers[:, 0]))
    hill_distance_max = float(np.max(hills.centers[:, 0]))
    if (
        hill_distance_min < args.ct_distance_min_A
        or hill_distance_max > args.ct_distance_max_A
    ):
        raise ValueError(
            f"{run.replicate}: deposited distance hill centers span "
            f"[{hill_distance_min:.6g}, {hill_distance_max:.6g}] Å, outside the "
            f"c(t) integration bounds [{args.ct_distance_min_A:g}, "
            f"{args.ct_distance_max_A:g}] Å"
        )
    wall_A = float(protocol["collective_variables"][0]["wall"])
    bias_error, bias_records = validate_cvseq_potential(
        cvseq, hills, wall_A, args.bias_validation_points, args.bias_validation_tolerance
    )

    print(f"[{run.replicate}] reading trajectory and validating topology/CV semantics", flush=True)
    msys_model, cms_model = topo.read_cms(str(run.cms))
    frames = traj.read_traj(str(run.trajectory))
    if len(frames) < 2:
        raise ValueError(f"{run.replicate}: trajectory has fewer than two frames")
    expected_particle_count = int(msys_model.natoms)
    frame_particle_counts = {int(frame.natoms) for frame in frames}
    if frame_particle_counts != {expected_particle_count}:
        raise ValueError(
            f"{run.replicate}: trajectory frame particle counts "
            f"{sorted(frame_particle_counts)} do not match MSYS particle count "
            f"{expected_particle_count}"
        )
    consistency_error = topo.check_consistency(cms_model, frames[0])
    if consistency_error:
        raise ValueError(f"{run.replicate}: CMS/trajectory mismatch: {consistency_error}")
    trajectory_times = np.asarray([float(frame.time) for frame in frames], dtype=float)
    if not np.isfinite(trajectory_times).all() or np.any(np.diff(trajectory_times) <= 0.0):
        raise ValueError(f"{run.replicate}: trajectory times are not finite/increasing")
    cv_spacing = float(np.median(np.diff(cvseq.time_ps)))
    trajectory_spacing = float(np.median(np.diff(trajectory_times)))
    endpoint_slack = 1.1 * max(cv_spacing, trajectory_spacing)
    expected_duration_ps = args.expected_duration_ns * 1000.0
    cv_duration_ps = float(cvseq.time_ps[-1] - cvseq.time_ps[0])
    trajectory_duration_ps = float(trajectory_times[-1] - trajectory_times[0])
    if (
        abs(cv_duration_ps - expected_duration_ps) > endpoint_slack
        or abs(trajectory_duration_ps - expected_duration_ps) > endpoint_slack
    ):
        raise ValueError(
            f"{run.replicate}: expected an approximately {args.expected_duration_ns:g} ns "
            f"run, but cvseq and trajectory durations are {cv_duration_ps / 1000.0:.6g} "
            f"and {trajectory_duration_ps / 1000.0:.6g} ns"
        )
    if (
        abs(trajectory_times[0] - cvseq.time_ps[0]) > endpoint_slack
        or abs(trajectory_times[-1] - cvseq.time_ps[-1]) > endpoint_slack
    ):
        raise ValueError(
            f"{run.replicate}: the structure trajectory endpoints do not match the cvseq "
            f"range within one output interval (DTR {trajectory_times[0]:.6g}–"
            f"{trajectory_times[-1]:.6g} ps; cvseq {cvseq.time_ps[0]:.6g}–"
            f"{cvseq.time_ps[-1]:.6g} ps)"
        )
    signature, system_signature, residue_count, zinc_count = topology_signature(
        msys_model, cms_model, frames[0].natoms
    )
    cv_semantics = validate_e6_cv_semantics(msys_model, cms_model, protocol)
    if args.expected_protein_residues and residue_count != args.expected_protein_residues:
        raise ValueError(
            f"{run.replicate}: found {residue_count} protein residues, expected "
            f"{args.expected_protein_residues}"
        )
    if args.expected_zinc and zinc_count != args.expected_zinc:
        raise ValueError(
            f"{run.replicate}: found {zinc_count} Zn atoms, expected {args.expected_zinc}"
        )

    print(f"[{run.replicate}] calculating CVs for all saved structure frames", flush=True)
    trajectory_distances = np.empty(len(frames), dtype=float)
    trajectory_chis = np.empty(len(frames), dtype=float)
    for frame_index, frame in enumerate(frames):
        trajectory_distances[frame_index], trajectory_chis[frame_index] = (
            recompute_frame_cvs(msys_model, cms_model, frame, protocol)
        )

    # Exact same-time records, when present, are an independent numerical check.
    matched_traj, matched_cv, matched_errors = exact_time_match_subset(
        trajectory_times, cvseq.time_ps, args.time_tolerance_ps
    )
    if matched_traj.size < args.minimum_same_time_validation_frames:
        raise ValueError(
            f"{run.replicate}: only {matched_traj.size} saved trajectory frames have "
            f"same-time cvseq records within {args.time_tolerance_ps:g} ps; at least "
            f"{args.minimum_same_time_validation_frames} are required to validate that "
            "the structure trajectory and CV history use the same coordinate definitions"
        )
    match_progress = (
        (cvseq.time_ps[matched_cv] - cvseq.time_ps[0])
        / (cvseq.time_ps[-1] - cvseq.time_ps[0])
    )
    covered_thirds = set(
        np.minimum((3.0 * match_progress).astype(int), 2).tolist()
    )
    if covered_thirds != {0, 1, 2}:
        raise ValueError(
            f"{run.replicate}: exact-time DTR/cvseq validation matches do not cover "
            "the early, middle, and late thirds of the run"
        )
    coordinate_records = []
    if matched_traj.size:
        distance_error_values = (
            trajectory_distances[matched_traj] - cvseq.distance_A[matched_cv]
        )
        chi_error_values_deg = np.rad2deg(circular_difference_radians(
            trajectory_chis[matched_traj], cvseq.chi_rad[matched_cv]
        ))
        for traj_index, cv_index, time_error, distance_error, chi_error in zip(
            matched_traj, matched_cv, matched_errors,
            distance_error_values, chi_error_values_deg,
        ):
            coordinate_records.append({
                "trajectory_frame_zero_based": int(traj_index),
                "cvseq_index_zero_based": int(cv_index),
                "trajectory_time_ps": float(trajectory_times[traj_index]),
                "cvseq_time_ps": float(cvseq.time_ps[cv_index]),
                "time_error_ps": float(time_error),
                "logged_distance_A": float(cvseq.distance_A[cv_index]),
                "recomputed_distance_A": float(trajectory_distances[traj_index]),
                "distance_error_A": float(distance_error),
                "logged_chi_deg": math.degrees(float(cvseq.chi_rad[cv_index])),
                "recomputed_chi_deg": math.degrees(float(trajectory_chis[traj_index])),
                "chi_circular_error_deg": float(chi_error),
            })
        distance_validation_error = float(np.max(np.abs(distance_error_values)))
        chi_validation_error = float(np.max(np.abs(chi_error_values_deg)))
        if distance_validation_error > args.coordinate_distance_tolerance_A:
            raise ValueError(
                f"{run.replicate}: recomputed distance differs from same-time cvseq "
                f"records by up to {distance_validation_error:.6g} Å"
            )
        if chi_validation_error > args.coordinate_chi_tolerance_deg:
            raise ValueError(
                f"{run.replicate}: recomputed chi1 differs from same-time cvseq "
                f"records by up to {chi_validation_error:.6g} degrees"
            )
        match_error = float(np.max(matched_errors))
    else:
        distance_validation_error = math.nan
        chi_validation_error = math.nan
        match_error = math.nan

    print(f"[{run.replicate}] computing c(t)", flush=True)
    checkpoints, c_values, c_distance, c_chi, terminal_bias = compute_c_t(
        hills=hills,
        checkpoint_times_ps=cvseq.time_ps,
        distance_min_A=args.ct_distance_min_A,
        distance_max_A=args.ct_distance_max_A,
        distance_step_A=args.ct_distance_step_A,
        angle_step_deg=args.ct_angle_step_deg,
        kbt=args.kbt,
        ktemp=args.ktemp,
    )
    omitted_tail_bound = float(
        np.sum(hills.height_kcal_mol) * math.exp(-0.5 * hills.cutoff ** 2)
    )
    if omitted_tail_bound > args.ct_omitted_tail_tolerance:
        raise ValueError(
            f"{run.replicate}: the upper bound from retaining Gaussian tails outside "
            f"the Desmond kernel cutoff is {omitted_tail_bound:.6g} kcal/mol, above "
            f"--ct-omitted-tail-tolerance {args.ct_omitted_tail_tolerance:g}"
        )

    cv_indices = np.arange(cvseq.time_ps.size, dtype=int)
    times = cvseq.time_ps.copy()
    distances = cvseq.distance_A.copy()
    chis = cvseq.chi_rad.copy()
    total_potential = cvseq.potential_kcal_mol.copy()
    wall = desmond_wall(distances, wall_A)
    meta_bias = total_potential - wall
    # c(t) was evaluated at each exact cvseq time using only earlier hills.
    c_at_frames = c_values.copy()

    # ``start_ns`` is an elapsed-time discard applied independently to every run.
    # Build quadrature weights before the spatial-domain filter so that a removed
    # out-of-domain frame does not lend its elapsed time to a neighboring frame.
    start_ps = float(times[0]) + args.start_ns * 1000.0
    after_start = times >= start_ps
    if np.count_nonzero(after_start) < 10:
        raise ValueError(
            f"{run.replicate}: fewer than ten frames remain after --start-ns"
        )
    temporal_weight_after_start = trapezoid_time_weights(times[after_start])
    in_domain_after_start = (
        (distances[after_start] >= args.distance_min_A)
        & (distances[after_start] <= args.distance_max_A)
    )
    selected = after_start.copy()
    selected[after_start] = in_domain_after_start
    if np.count_nonzero(selected) < 10:
        raise ValueError(f"{run.replicate}: fewer than ten frames remain after filters")
    cv_indices = cv_indices[selected]
    times = times[selected]
    distances = distances[selected]
    chis = chis[selected]
    total_potential = total_potential[selected]
    wall = wall[selected]
    meta_bias = meta_bias[selected]
    c_at_frames = c_at_frames[selected]

    temporal_weight = temporal_weight_after_start[in_domain_after_start]
    bias_for_target = total_potential if args.target == "unwalled" else meta_bias
    log_weight = np.log(temporal_weight) + (bias_for_target - c_at_frames) / args.kbt
    log_weight -= logsumexp(log_weight)
    weight = np.exp(log_weight)
    ess = float(1.0 / np.sum(weight * weight))
    max_weight = float(np.max(weight))

    trajectory_start_ps = float(cvseq.time_ps[0]) + args.start_ns * 1000.0
    trajectory_end_ps = float(cvseq.time_ps[-1])
    trajectory_selected = (
        (trajectory_times >= trajectory_start_ps)
        & (trajectory_times <= trajectory_end_ps)
        & (trajectory_distances >= args.distance_min_A)
        & (trajectory_distances <= args.distance_max_A)
    )
    if np.count_nonzero(trajectory_selected) == 0:
        raise ValueError(
            f"{run.replicate}: no structure frames remain for exemplar selection"
        )
    trajectory_frame_indices = np.flatnonzero(trajectory_selected).astype(int)

    run_dir = output_dir / "per_replicate" / run.replicate
    run_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(run_dir / "bias_validation.csv", bias_records)
    write_csv_rows(run_dir / "coordinate_validation.csv", coordinate_records)
    write_json(run_dir / "cv_semantics.json", cv_semantics)
    write_csv_columns(
        run_dir / "c_t.csv",
        {
            "time_ps": checkpoints,
            "time_ns": checkpoints / 1000.0,
            "c_t_kcal_mol": c_values,
        },
    )
    np.savez_compressed(
        run_dir / "terminal_bias_and_ct_grid.npz",
        c_distance_A=c_distance,
        c_chi_rad=c_chi,
        c_chi_deg=np.rad2deg(c_chi),
        terminal_raw_meta_bias_kcal_mol=terminal_bias,
        c_time_ps=checkpoints,
        c_t_kcal_mol=c_values,
    )
    del frames, cms_model, msys_model

    return RunData(
        spec=run,
        cvseq_index=cv_indices,
        time_ps=times,
        distance_A=distances,
        chi_rad=chis,
        total_potential_kcal_mol=total_potential,
        meta_bias_kcal_mol=meta_bias,
        wall_kcal_mol=wall,
        c_t_kcal_mol=c_at_frames,
        log_weight=log_weight,
        weight=weight,
        trajectory_frame_index=trajectory_frame_indices,
        trajectory_time_ps=trajectory_times[trajectory_selected],
        trajectory_distance_A=trajectory_distances[trajectory_selected],
        trajectory_chi_rad=trajectory_chis[trajectory_selected],
        importance_ess=ess,
        max_weight_fraction=max_weight,
        time_match_max_error_ps=match_error,
        matched_trajectory_frame_count=int(matched_traj.size),
        bias_validation_max_error=bias_error,
        distance_validation_max_error_A=distance_validation_error,
        chi_validation_max_error_deg=chi_validation_error,
        ct_omitted_kernel_tail_upper_bound_kcal_mol=omitted_tail_bound,
        protein_signature=signature,
        system_signature=system_signature,
        protein_residue_count=residue_count,
        zinc_count=zinc_count,
    )


def write_csv_rows(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_csv_columns(path: Path, columns: dict[str, Sequence]) -> None:
    names = list(columns)
    lengths = {len(columns[name]) for name in names}
    if len(lengths) != 1:
        raise ValueError(f"columns for {path} have different lengths")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(names)
        writer.writerows(zip(*(columns[name] for name in names)))


def histogram_probability(run: RunData, distance_edges: np.ndarray,
                          chi_edges: np.ndarray, smooth_distance_A: float,
                          smooth_chi_deg: float) -> np.ndarray:
    histogram, _, _ = np.histogram2d(
        run.distance_A,
        np.rad2deg(run.chi_rad),
        bins=(distance_edges, chi_edges),
        weights=run.weight,
    )
    distance_step = float(np.diff(distance_edges)[0])
    chi_step = float(np.diff(chi_edges)[0])
    probability = gaussian_filter(
        histogram,
        sigma=(smooth_distance_A / distance_step, smooth_chi_deg / chi_step),
        mode=("constant", "wrap"),
        cval=0.0,
    )
    total = float(np.sum(probability))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError(f"{run.spec.replicate}: probability grid cannot be normalized")
    return probability / total


def probability_to_fes(probability: np.ndarray, kbt: float) -> np.ndarray:
    if np.any(probability < 0.0) or not np.any(probability > 0.0):
        raise ValueError("invalid probability surface")
    tiny = np.finfo(float).tiny
    log_probability = np.log(np.maximum(probability, tiny))
    return -kbt * (log_probability - np.max(log_probability))


def surface_sensitivity_metrics(
    five_probability: np.ndarray,
    five_fes: np.ndarray,
    four_probability: np.ndarray,
    four_fes: np.ndarray,
    label_grid: np.ndarray,
    centers: Sequence[dict],
    low_energy_cap: float,
) -> dict:
    """Quantify—but do not automatically adjudicate—the run-exclusion result."""
    five_probability = np.asarray(five_probability, dtype=float)
    four_probability = np.asarray(four_probability, dtype=float)
    mixture = 0.5 * (five_probability + four_probability)
    positive_five = five_probability > 0.0
    positive_four = four_probability > 0.0
    js_nats = 0.5 * float(np.sum(
        five_probability[positive_five]
        * np.log(five_probability[positive_five] / mixture[positive_five])
    ))
    js_nats += 0.5 * float(np.sum(
        four_probability[positive_four]
        * np.log(four_probability[positive_four] / mixture[positive_four])
    ))

    low_energy = (five_fes <= low_energy_cap) & (four_fes <= low_energy_cap)
    if np.any(low_energy):
        relative_fes_difference = four_fes[low_energy] - five_fes[low_energy]
        rms_fes_difference = float(np.sqrt(np.mean(relative_fes_difference ** 2)))
        mean_abs_fes_difference = float(np.mean(np.abs(relative_fes_difference)))
        max_abs_fes_difference = float(np.max(np.abs(relative_fes_difference)))
    else:
        rms_fes_difference = math.nan
        mean_abs_fes_difference = math.nan
        max_abs_fes_difference = math.nan

    region_changes = []
    for label, center in enumerate(centers, start=1):
        five_population = float(np.sum(five_probability[label_grid == label]))
        four_population = float(np.sum(four_probability[label_grid == label]))
        region_changes.append({
            "region_id": center["basin_id"],
            "five_run_integrated_smoothed_probability": five_population,
            "four_run_integrated_smoothed_probability": four_population,
            "four_minus_five_probability": four_population - five_population,
            "absolute_probability_change": abs(four_population - five_population),
        })

    return {
        "comparison": "four-run exclusion surface minus all-five primary surface",
        "relative_fes_zero_reference": (
            "each surface is independently shifted so its minimum is zero"
        ),
        "low_energy_comparison_cap_kcal_mol": low_energy_cap,
        "low_energy_grid_cell_count": int(np.count_nonzero(low_energy)),
        "low_energy_relative_fes_rms_difference_kcal_mol": rms_fes_difference,
        "low_energy_relative_fes_mean_absolute_difference_kcal_mol": (
            mean_abs_fes_difference
        ),
        "low_energy_relative_fes_max_absolute_difference_kcal_mol": (
            max_abs_fes_difference
        ),
        "probability_total_variation_distance": float(
            0.5 * np.sum(np.abs(five_probability - four_probability))
        ),
        "probability_jensen_shannon_divergence_nats": js_nats,
        "maximum_region_probability_absolute_change": (
            max((row["absolute_probability_change"] for row in region_changes), default=0.0)
        ),
        "region_probability_changes": region_changes,
        "automatic_scientific_pass_fail": False,
        "review_requirement": (
            "Inspect the numerical differences, surface plot, region identities, and "
            "representative assignments before manuscript use."
        ),
    }


def periodic_angle_distance_deg(a, b):
    return np.rad2deg(np.abs(circular_difference_radians(np.deg2rad(a), np.deg2rad(b))))


def read_basin_centers(path: Path, distance: np.ndarray,
                       chi_deg: np.ndarray) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"distance_A", "chi1_deg"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"basin-center CSV requires columns {sorted(required)}")
        centers = []
        seen_ids = set()
        seen_grid_cells = set()
        for index, row in enumerate(reader, start=1):
            d = float(row["distance_A"])
            chi = float(wrap_radians(math.radians(float(row["chi1_deg"]))))
            if not math.isfinite(d) or not math.isfinite(chi):
                raise ValueError(f"basin-center row {index + 1} contains a nonfinite value")
            if d < float(distance[0]) or d > float(distance[-1]):
                raise ValueError(
                    f"basin-center row {index + 1} distance {d:g} Å is outside "
                    f"the analyzed grid [{distance[0]:g}, {distance[-1]:g}] Å"
                )
            d_index = int(np.argmin(np.abs(distance - d)))
            c_index = int(np.argmin(periodic_angle_distance_deg(chi_deg, math.degrees(chi))))
            d_half_bin = 0.5 * float(np.mean(np.diff(distance)))
            chi_half_bin = 0.5 * float(np.mean(np.diff(chi_deg)))
            if abs(float(distance[d_index]) - d) > d_half_bin + 1e-8:
                raise ValueError(
                    f"basin-center row {index + 1} is farther than half a distance "
                    "bin from the nearest grid center"
                )
            if float(periodic_angle_distance_deg(
                chi_deg[c_index], math.degrees(chi)
            )) > chi_half_bin + 1e-8:
                raise ValueError(
                    f"basin-center row {index + 1} is farther than half an angular "
                    "bin from the nearest grid center"
                )
            basin_id = (row.get("basin_id") or f"B{index:02d}").strip()
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", basin_id):
                raise ValueError(f"basin-center row {index + 1}: invalid basin_id {basin_id!r}")
            if basin_id in seen_ids:
                raise ValueError(f"duplicate basin_id in basin-center CSV: {basin_id}")
            if (d_index, c_index) in seen_grid_cells:
                raise ValueError(
                    f"basin-center row {index + 1} maps to a duplicate grid cell"
                )
            seen_ids.add(basin_id)
            seen_grid_cells.add((d_index, c_index))
            centers.append({
                "basin_id": basin_id,
                "distance_index": d_index,
                "chi_index": c_index,
            })
    if not centers:
        raise ValueError("basin-center CSV contains no centers")
    return centers


def detect_basin_centers(distance: np.ndarray, chi_deg: np.ndarray, fes: np.ndarray,
                         max_free_energy: float, min_distance_separation_A: float,
                         min_chi_separation_deg: float, max_basins: int,
                         distance_edge_exclusion_A: float) -> list[dict]:
    d_step = float(np.mean(np.diff(distance)))
    c_step = float(np.mean(np.diff(chi_deg)))
    d_window = max(3, int(round(min_distance_separation_A / d_step)))
    c_window = max(3, int(round(min_chi_separation_deg / c_step)))
    if d_window % 2 == 0:
        d_window += 1
    if c_window % 2 == 0:
        c_window += 1
    local = fes == minimum_filter(
        fes, size=(d_window, c_window), mode=("nearest", "wrap")
    )
    # Exclude one smoothing bandwidth at nonperiodic distance boundaries.
    edge_bins = max(1, int(math.ceil(distance_edge_exclusion_A / d_step)))
    local[:edge_bins, :] = False
    local[-edge_bins:, :] = False
    candidates = np.argwhere(local & np.isfinite(fes) & (fes <= max_free_energy))
    candidates = sorted(candidates, key=lambda item: float(fes[tuple(item)]))
    selected: list[tuple[int, int]] = []
    for d_index, c_index in candidates:
        d = float(distance[d_index])
        chi = float(chi_deg[c_index])
        too_close = False
        for old_d_index, old_c_index in selected:
            scaled = math.sqrt(
                ((d - float(distance[old_d_index])) / min_distance_separation_A) ** 2
                + (
                    float(periodic_angle_distance_deg(chi, chi_deg[old_c_index]))
                    / min_chi_separation_deg
                ) ** 2
            )
            if scaled < 1.0:
                too_close = True
                break
        if not too_close:
            selected.append((int(d_index), int(c_index)))
    if not selected:
        raise ValueError("automatic basin detection found no minima")
    if len(selected) > max_basins:
        raise ValueError(
            f"automatic basin detection found {len(selected)} minima (limit {max_basins}); "
            "increase smoothing/separation or provide --basin-centers"
        )
    return [
        {
            "basin_id": f"B{index:02d}",
            "distance_index": d_index,
            "chi_index": c_index,
        }
        for index, (d_index, c_index) in enumerate(selected, start=1)
    ]


def periodic_watershed(fes: np.ndarray, centers: Sequence[dict],
                       assignment_cap: float) -> np.ndarray:
    """Marker-based minimax watershed with periodic angular connectivity."""
    n_distance, n_chi = fes.shape
    mask = np.isfinite(fes) & (fes <= assignment_cap)
    labels = np.zeros(fes.shape, dtype=np.int16)
    best_cost = np.full(fes.shape, np.inf, dtype=float)
    heap: list[tuple[float, int, int, int]] = []
    for label, center in enumerate(centers, start=1):
        d_index = int(center["distance_index"])
        c_index = int(center["chi_index"])
        if not mask[d_index, c_index]:
            raise ValueError(
                f"basin center {center['basin_id']} is outside the assignment cap"
            )
        cost = float(fes[d_index, c_index])
        labels[d_index, c_index] = label
        best_cost[d_index, c_index] = cost
        heapq.heappush(heap, (cost, label, d_index, c_index))

    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1),
    ]
    tolerance = 1e-12
    while heap:
        cost, label, d_index, c_index = heapq.heappop(heap)
        if cost > best_cost[d_index, c_index] + tolerance:
            continue
        if labels[d_index, c_index] != label and math.isclose(
            cost, best_cost[d_index, c_index], abs_tol=tolerance
        ):
            continue
        for d_offset, c_offset in neighbors:
            new_d = d_index + d_offset
            if new_d < 0 or new_d >= n_distance:
                continue
            new_c = (c_index + c_offset) % n_chi
            if not mask[new_d, new_c]:
                continue
            new_cost = max(cost, float(fes[new_d, new_c]))
            if (
                new_cost < best_cost[new_d, new_c] - tolerance
                or (
                    math.isclose(new_cost, best_cost[new_d, new_c], abs_tol=tolerance)
                    and (labels[new_d, new_c] == 0 or label < labels[new_d, new_c])
                )
            ):
                best_cost[new_d, new_c] = new_cost
                labels[new_d, new_c] = label
                heapq.heappush(heap, (new_cost, label, new_d, new_c))
    return labels


def coordinate_grid_labels(distance_A: np.ndarray, chi_rad: np.ndarray,
                           distance_edges: np.ndarray, chi_edges: np.ndarray,
                           label_grid: np.ndarray) -> np.ndarray:
    output = np.zeros(distance_A.size, dtype=label_grid.dtype)
    valid = (
        np.isfinite(distance_A)
        & np.isfinite(chi_rad)
        & (distance_A >= distance_edges[0])
        & (distance_A <= distance_edges[-1])
    )
    if not np.any(valid):
        return output
    d_index = np.searchsorted(distance_edges, distance_A[valid], side="right") - 1
    d_index = np.minimum(d_index, label_grid.shape[0] - 1)
    chi_values = np.rad2deg(wrap_radians(chi_rad))
    c_index = np.searchsorted(chi_edges, chi_values[valid], side="right") - 1
    c_index %= label_grid.shape[1]
    output[valid] = label_grid[d_index, c_index]
    return output


def basin_statistics(runs: Sequence[RunData], centers: Sequence[dict],
                     distance: np.ndarray, chi_deg: np.ndarray,
                     combined_probability: np.ndarray, fes: np.ndarray,
                     label_grid: np.ndarray,
                     support_threshold: float, bootstrap: int,
                     seed: int, kbt: float) -> tuple[list[dict], np.ndarray]:
    populations = np.zeros((len(runs), len(centers)), dtype=float)
    for run_index, run in enumerate(runs):
        assert run.basin is not None
        for basin_index in range(len(centers)):
            populations[run_index, basin_index] = float(
                np.sum(run.weight[run.basin == basin_index + 1])
            )
    rng = np.random.default_rng(seed)
    if bootstrap > 0:
        resamples = rng.integers(0, len(runs), size=(bootstrap, len(runs)))
        boot = np.mean(populations[resamples, :], axis=1)
    else:
        boot = np.empty((0, len(centers)))

    rows = []
    mean_populations = np.mean(populations, axis=0)
    positive = mean_populations[mean_populations > 0.0]
    maximum_population = float(np.max(positive)) if positive.size else 1.0
    center_energies = np.asarray([
        float(fes[int(center["distance_index"]), int(center["chi_index"])])
        for center in centers
    ])
    energy_order = np.argsort(center_energies, kind="stable")
    energy_ranks = np.empty(len(centers), dtype=int)
    energy_ranks[energy_order] = np.arange(1, len(centers) + 1)
    majority = len(runs) // 2 + 1
    integrated_probabilities = np.asarray([
        float(np.sum(combined_probability[label_grid == basin_index + 1]))
        for basin_index in range(len(centers))
    ])
    assigned_grid_probability = float(np.sum(integrated_probabilities))
    for basin_index, center in enumerate(centers):
        label = basin_index + 1
        weights = []
        raw_count = 0
        for run in runs:
            selected = run.basin == label
            raw_count += int(np.count_nonzero(selected))
            weights.extend((run.weight[selected] / len(runs)).tolist())
        weight_array = np.asarray(weights, dtype=float)
        total_weight = float(np.sum(weight_array))
        ess = (
            float(total_weight * total_weight / np.sum(weight_array * weight_array))
            if weight_array.size and np.sum(weight_array * weight_array) > 0.0 else 0.0
        )
        population = float(mean_populations[basin_index])
        ci_low = float(np.quantile(boot[:, basin_index], 0.025)) if bootstrap else math.nan
        ci_high = float(np.quantile(boot[:, basin_index], 0.975)) if bootstrap else math.nan
        d_index = int(center["distance_index"])
        c_index = int(center["chi_index"])
        row = {
            "basin_id": center["basin_id"],
            "energy_rank": int(energy_ranks[basin_index]),
            "minimum_distance_A": float(distance[d_index]),
            "minimum_chi1_deg": float(chi_deg[c_index]),
            "minimum_fes_kcal_mol": float(fes[d_index, c_index]),
            "assigned_cvseq_weight_probability": population,
            "integrated_smoothed_grid_probability": float(
                integrated_probabilities[basin_index]
            ),
            "conditional_smoothed_probability_among_assigned_regions": (
                float(integrated_probabilities[basin_index] / assigned_grid_probability)
                if assigned_grid_probability > 0.0 else math.nan
            ),
            "population_bootstrap_95pct_low": ci_low,
            "population_bootstrap_95pct_high": ci_high,
            "population_relative_free_energy_kcal_mol": (
                -kbt * math.log(population / maximum_population) if population > 0.0 else math.inf
            ),
            "replicates_with_population_ge_threshold": int(np.count_nonzero(
                populations[:, basin_index] >= support_threshold
            )),
            "support_threshold_per_replicate": support_threshold,
            "majority_support_required": majority,
            "majority_replicate_support": bool(np.count_nonzero(
                populations[:, basin_index] >= support_threshold
            ) >= majority),
            "raw_cvseq_record_count": raw_count,
            "importance_weight_ess_not_autocorrelation_corrected": ess,
            "maximum_within_basin_combined_cvseq_weight_fraction": (
                float(np.max(weight_array) / total_weight) if total_weight > 0.0 else math.nan
            ),
        }
        for run_index, run in enumerate(runs):
            row[f"population_{run.spec.replicate}"] = float(populations[run_index, basin_index])
            selected = run.basin == label
            selected_weights = run.weight[selected]
            row[f"raw_cvseq_records_{run.spec.replicate}"] = int(
                np.count_nonzero(selected)
            )
            row[f"importance_weight_ess_{run.spec.replicate}"] = (
                float(np.sum(selected_weights) ** 2 / np.sum(selected_weights ** 2))
                if selected_weights.size and np.sum(selected_weights ** 2) > 0.0 else 0.0
            )
        rows.append(row)
    return rows, populations


def select_representatives(runs: Sequence[RunData], centers: Sequence[dict],
                           distance: np.ndarray, chi_deg: np.ndarray,
                           fes: np.ndarray, distance_edges: np.ndarray,
                           chi_edges: np.ndarray, distance_width_A: float,
                           chi_width_deg: float, max_delta_fes: float,
                           top_candidates: int) -> tuple[list[dict], list[dict]]:
    representatives = []
    candidates_output = []
    for basin_index, center in enumerate(centers):
        label = basin_index + 1
        d_min = float(distance[int(center["distance_index"])])
        chi_min = math.radians(float(chi_deg[int(center["chi_index"])]))
        minimum_fes = float(fes[
            int(center["distance_index"]), int(center["chi_index"])
        ])
        candidates = []
        for run_index, run in enumerate(runs):
            assert run.trajectory_basin is not None
            assert run.trajectory_frame_index is not None
            assert run.trajectory_time_ps is not None
            assert run.trajectory_distance_A is not None
            assert run.trajectory_chi_rad is not None
            indices = np.flatnonzero(run.trajectory_basin == label)
            if indices.size == 0:
                continue
            d_grid_index = np.searchsorted(
                distance_edges, run.trajectory_distance_A[indices], side="right"
            ) - 1
            d_grid_index = np.clip(d_grid_index, 0, fes.shape[0] - 1)
            c_grid_index = np.searchsorted(
                chi_edges,
                np.rad2deg(wrap_radians(run.trajectory_chi_rad[indices])),
                side="right",
            ) - 1
            c_grid_index %= fes.shape[1]
            frame_fes = fes[d_grid_index, c_grid_index]
            local = frame_fes <= minimum_fes + max_delta_fes
            indices = indices[local]
            d_grid_index = d_grid_index[local]
            c_grid_index = c_grid_index[local]
            frame_fes = frame_fes[local]
            if indices.size == 0:
                continue
            delta_d_A = run.trajectory_distance_A[indices] - d_min
            delta_chi_rad = circular_difference_radians(
                run.trajectory_chi_rad[indices], chi_min
            )
            delta_d = delta_d_A / distance_width_A
            angular_chord = 2.0 * np.sin(delta_chi_rad / 2.0) / math.radians(
                chi_width_deg
            )
            scores = delta_d * delta_d + angular_chord * angular_chord
            for local_index, d_cell, c_cell, local_fes, delta_distance, delta_chi, score in zip(
                indices, d_grid_index, c_grid_index, frame_fes,
                delta_d_A, delta_chi_rad, scores,
            ):
                candidates.append({
                    "basin_id": center["basin_id"],
                    "run_index": run_index,
                    "replicate": run.spec.replicate,
                    "trajectory_frame_zero_based": int(
                        run.trajectory_frame_index[local_index]
                    ),
                    "trajectory_frame_one_based": int(
                        run.trajectory_frame_index[local_index]
                    ) + 1,
                    "time_ps": float(run.trajectory_time_ps[local_index]),
                    "time_ns": float(run.trajectory_time_ps[local_index] / 1000.0),
                    "distance_A": float(run.trajectory_distance_A[local_index]),
                    "chi1_deg": math.degrees(
                        float(run.trajectory_chi_rad[local_index])
                    ),
                    "distance_residual_from_minimum_A": float(delta_distance),
                    "chi1_periodic_residual_from_minimum_deg": math.degrees(
                        float(delta_chi)
                    ),
                    "frame_grid_distance_index": int(d_cell),
                    "frame_grid_chi_index": int(c_cell),
                    "frame_grid_fes_kcal_mol": float(local_fes),
                    "delta_fes_from_basin_minimum_kcal_mol": float(
                        local_fes - minimum_fes
                    ),
                    "scaled_desmond_periodic_cv_distance_squared": float(score),
                })
        if not candidates:
            raise ValueError(
                f"no sampled frame in {center['basin_id']} lies within "
                f"{max_delta_fes:g} kcal/mol of its minimum; inspect the grid or "
                "increase --representative-max-delta-fes"
            )
        candidates.sort(key=lambda row: (
            row["scaled_desmond_periodic_cv_distance_squared"],
            row["replicate"],
            row["trajectory_frame_zero_based"],
        ))
        representatives.append(candidates[0])
        for rank, row in enumerate(candidates[:top_candidates], start=1):
            candidates_output.append({"candidate_rank": rank, **row})
    return representatives, candidates_output


def extract_representative_structures(representatives: list[dict], runs: Sequence[RunData],
                                      protocol: dict, output_dir: Path,
                                      structure_asl: str,
                                      distance_tolerance_A: float,
                                      chi_tolerance_deg: float,
                                      expected_protein_residues: int,
                                      expected_zinc: int,
                                      zinc_coordination_cutoff_A: float) -> list[dict]:
    from schrodinger import structure
    from schrodinger.application.desmond.packages import topo, traj

    representatives_dir = output_dir / "representatives"
    representatives_dir.mkdir(parents=True, exist_ok=True)
    combined_writer = structure.MaestroWriter(
        str(representatives_dir / "all_basin_representatives.maegz")
    )
    extracted_records = []
    try:
        by_run: dict[int, list[dict]] = {}
        for row in representatives:
            by_run.setdefault(int(row["run_index"]), []).append(row)
        for run_index, selected_rows in by_run.items():
            run = runs[run_index]
            msys_model, cms_model = topo.read_cms(str(run.spec.cms))
            frames = traj.read_traj(str(run.spec.trajectory))
            aids = sorted(int(value) for value in cms_model.select_atom(structure_asl))
            if not aids:
                raise ValueError(
                    f"{run.spec.replicate}: structure ASL selected no atoms: {structure_asl}"
                )
            aid_set = set(aids)
            protein_aids = aid_set.intersection(
                int(value) for value in cms_model.select_atom("protein")
            )
            zinc_aids = aid_set.intersection(
                int(value) for value in cms_model.select_atom("a.e Zn")
            )
            selected_residues = {
                (
                    str(cms_model.fsys_ct.atom[aid].chain).strip(),
                    int(cms_model.fsys_ct.atom[aid].resnum),
                    str(cms_model.fsys_ct.atom[aid].inscode).strip(),
                    str(cms_model.fsys_ct.atom[aid].pdbres).strip(),
                )
                for aid in protein_aids
            }
            if expected_protein_residues and len(selected_residues) != expected_protein_residues:
                raise ValueError(
                    f"{run.spec.replicate}: output ASL contains {len(selected_residues)} "
                    f"protein residues, expected {expected_protein_residues}"
                )
            if expected_zinc and len(zinc_aids) != expected_zinc:
                raise ValueError(
                    f"{run.spec.replicate}: output ASL contains {len(zinc_aids)} Zn atoms, "
                    f"expected {expected_zinc}"
                )
            gids = topo.aids2gids(cms_model, aids, include_pseudoatoms=False)
            for row in selected_rows:
                frame_index = int(row["trajectory_frame_zero_based"])
                if frame_index < 0 or frame_index >= len(frames):
                    raise ValueError(
                        f"{row['basin_id']}: trajectory frame {frame_index} is out of range"
                    )
                if not math.isclose(
                    float(frames[frame_index].time), float(row["time_ps"]),
                    rel_tol=0.0, abs_tol=1e-8,
                ):
                    raise ValueError(
                        f"{row['basin_id']}: trajectory frame time changed between "
                        "selection and extraction"
                    )
                work_frame = frames[frame_index].copy()
                # Glue the protein and structural Zn components into one image;
                # make_whole alone may leave disconnected metal ions elsewhere.
                work_frame = topo.glue(msys_model, gids, [work_frame])[0]
                st = cms_model.extract(aids)
                xyz = np.asarray(work_frame.pos(gids), dtype=float)
                if int(st.atom_total) != xyz.shape[0]:
                    raise ValueError(
                        f"{run.spec.replicate}: extracted atom count does not match frame selection"
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
                    for aid in aids
                ]
                observed_identity = [
                    (
                        str(atom.chain).strip(), int(atom.resnum),
                        str(atom.inscode).strip(), str(atom.pdbres).strip(),
                        str(atom.pdbname).strip(), str(atom.element).strip(),
                    )
                    for atom in st.atom
                ]
                if observed_identity != expected_identity:
                    raise ValueError(
                        f"{run.spec.replicate}: extracted atom identity/order is inconsistent"
                    )
                st.setXYZ(xyz)
                zinc_atoms = [
                    atom for atom in st.atom if str(atom.element).strip().upper() == "ZN"
                ]
                cysteine_sulfurs = [
                    atom for atom in st.atom
                    if str(atom.element).strip().upper() == "S"
                    and str(atom.pdbres).strip().upper() in {"CYS", "CYX"}
                ]
                zinc_fourth_sulfur_distances = []
                for zinc_atom in zinc_atoms:
                    sulfur_distances = sorted(
                        float(np.linalg.norm(np.asarray(zinc_atom.xyz) - np.asarray(sulfur.xyz)))
                        for sulfur in cysteine_sulfurs
                    )
                    if len(sulfur_distances) < 4:
                        raise ValueError(
                            f"{row['basin_id']}: fewer than four cysteine sulfur atoms "
                            "are present for Zn coordination validation"
                        )
                    fourth_distance = sulfur_distances[3]
                    if fourth_distance > zinc_coordination_cutoff_A:
                        raise ValueError(
                            f"{row['basin_id']}: a Zn ion has only three cysteine sulfurs "
                            f"within {zinc_coordination_cutoff_A:g} Å after imaging "
                            f"(fourth-nearest distance {fourth_distance:.6g} Å)"
                        )
                    zinc_fourth_sulfur_distances.append(fourth_distance)
                st.title = (
                    f"{row['basin_id']} CV-nearest sampled exemplar; "
                    f"{run.spec.replicate}; frame {frame_index}; {row['time_ns']:.6f} ns"
                )
                st.property["s_user_basin_id"] = row["basin_id"]
                st.property["s_user_replicate"] = run.spec.replicate
                st.property["i_user_trajectory_frame_zero_based"] = frame_index
                st.property["r_user_time_ns"] = float(row["time_ns"])
                st.property["r_user_d49_r102_com_distance_A"] = float(row["distance_A"])
                st.property["r_user_q107_chi1_deg"] = float(row["chi1_deg"])

                recomputed_distance, recomputed_chi = recompute_frame_cvs(
                    msys_model, cms_model, work_frame, protocol
                )
                distance_error = recomputed_distance - float(row["distance_A"])
                chi_error = math.degrees(float(circular_difference_radians(
                    recomputed_chi, math.radians(float(row["chi1_deg"]))
                )))
                if abs(distance_error) > distance_tolerance_A or abs(chi_error) > chi_tolerance_deg:
                    raise ValueError(
                        f"{row['basin_id']}: extracted-frame CV validation failed "
                        f"(distance error {distance_error:.6g} Å; chi error {chi_error:.6g}°)"
                    )
                base = (
                    f"{row['basin_id']}_CV-nearest_{run.spec.replicate}_"
                    f"frame{frame_index:06d}_"
                    f"{row['time_ns']:.3f}ns"
                )
                mae_path = representatives_dir / f"{base}.maegz"
                pdb_path = representatives_dir / f"{base}.pdb"
                st.write(str(mae_path))
                st.write(str(pdb_path))
                combined_writer.append(st)
                extracted_records.append({
                    **{key: value for key, value in row.items() if key != "run_index"},
                    "output_maegz": str(mae_path.resolve()),
                    "output_pdb": str(pdb_path.resolve()),
                    "recomputed_distance_A": recomputed_distance,
                    "distance_error_A": distance_error,
                    "recomputed_chi1_deg": math.degrees(recomputed_chi),
                    "chi1_circular_error_deg": chi_error,
                    "zinc_fourth_nearest_cysteine_sulfur_distances_A": ";".join(
                        f"{value:.6f}" for value in zinc_fourth_sulfur_distances
                    ),
                })
            del frames, cms_model, msys_model
    finally:
        combined_writer.close()
    return extracted_records


def shifted_periodic_view(chi_deg: np.ndarray, values: np.ndarray,
                          seam_deg: float = -120.0) -> tuple[np.ndarray, np.ndarray]:
    index = int(np.searchsorted(chi_deg, seam_deg, side="left"))
    display_chi = np.concatenate((chi_deg[index:], chi_deg[:index] + 360.0))
    display_values = np.concatenate((values[..., index:], values[..., :index]), axis=-1)
    return display_chi, display_values


def configure_matplotlib() -> None:
    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.labelsize": 10.5,
        "axes.titlesize": 10.0,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.linewidth": 0.9,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })


def plot_surface(ax, distance: np.ndarray, chi_deg: np.ndarray, fes: np.ndarray,
                 cap: float, title: str | None = None, centers: Sequence[dict] | None = None):
    display_chi, display_fes = shifted_periodic_view(chi_deg, fes)
    cmap = LinearSegmentedColormap.from_list("muted_blue_red", PALETTE, N=256)
    image = ax.pcolormesh(
        distance, display_chi, np.minimum(display_fes, cap).T,
        shading="nearest", cmap=cmap, vmin=0.0, vmax=cap, rasterized=True,
    )
    contour_levels = np.arange(2.0, cap, 2.0)
    if contour_levels.size:
        ax.contour(
            distance, display_chi, display_fes.T,
            levels=contour_levels, colors="#5A5A5A", linewidths=0.45, alpha=0.65,
        )
    ax.set_xlim(float(distance[0]), float(distance[-1]))
    ax.set_ylim(float(display_chi[0]), float(display_chi[-1]))
    ax.set_xlabel("D49–R102 residue COM distance (Å)")
    ax.set_ylabel("Q107 χ₁ dihedral angle (°)")
    ax.grid(False)
    if title:
        ax.set_title(title, pad=5)
    if centers:
        for center in centers:
            d = float(distance[int(center["distance_index"])] )
            chi = float(chi_deg[int(center["chi_index"])] )
            if chi < display_chi[0]:
                chi += 360.0
            label = ax.text(
                d, chi, center["basin_id"], ha="center", va="center",
                color="white", fontsize=7.4, fontweight="bold", zorder=6,
            )
            label.set_path_effects([
                path_effects.withStroke(linewidth=1.4, foreground="#333333")
            ])
    return image


def save_figures(output_dir: Path, distance: np.ndarray, chi_deg: np.ndarray,
                 runs: Sequence[RunData], combined_fes: np.ndarray,
                 four_run_fes: np.ndarray, centers: Sequence[dict], cap: float,
                 target: str, excluded_replicate: str) -> None:
    configure_matplotlib()
    target_label = "static wall removed" if target == "unwalled" else "40 Å wall retained"

    fig, ax = plt.subplots(figsize=(7.2, 5.35))
    fig.subplots_adjust(left=0.13, right=0.82, bottom=0.14, top=0.93)
    image = plot_surface(
        ax, distance, chi_deg, combined_fes, cap,
        title=f"Five-replicate trajectory-reweighted FES ({target_label})",
        centers=centers,
    )
    colorbar = fig.colorbar(image, ax=ax, pad=0.04, fraction=0.06)
    colorbar.set_label("Relative free energy (kcal mol⁻¹)")
    ax.text(
        1.0, 1.012, "periodic seam: −120° ≡ 240°",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color="#555555",
    )
    fig.savefig(output_dir / "combined_5rep_reweighted_fes.png", dpi=600,
                bbox_inches="tight", pad_inches=0.04)
    fig.savefig(output_dir / "combined_5rep_reweighted_fes.pdf",
                bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(11.0, 7.2), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.075, right=0.90, bottom=0.09, top=0.93,
                        wspace=0.14, hspace=0.20)
    image = None
    for index, run in enumerate(runs):
        image = plot_surface(
            axes.flat[index], distance, chi_deg, run.fes, cap,
            title=run.spec.replicate,
        )
    image = plot_surface(
        axes.flat[5], distance, chi_deg, combined_fes, cap,
        title="Equal-replicate combination", centers=centers,
    )
    colorbar_ax = fig.add_axes([0.92, 0.20, 0.018, 0.60])
    colorbar = fig.colorbar(image, cax=colorbar_ax)
    colorbar.set_label("Relative free energy (kcal mol⁻¹)")
    fig.savefig(output_dir / "per_replicate_and_combined_fes.png", dpi=500,
                bbox_inches="tight", pad_inches=0.04)
    fig.savefig(output_dir / "per_replicate_and_combined_fes.pdf",
                bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    difference = four_run_fes - combined_fes
    finite = np.isfinite(difference) & (combined_fes <= cap) & (four_run_fes <= cap)
    difference_plot = np.where(finite, difference, np.nan)
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.065, right=0.93, bottom=0.16, top=0.88, wspace=0.18)
    image = plot_surface(axes[0], distance, chi_deg, combined_fes, cap, "All five runs")
    plot_surface(
        axes[1], distance, chi_deg, four_run_fes, cap,
        f"Four runs without {excluded_replicate}",
    )
    display_chi, display_difference = shifted_periodic_view(chi_deg, difference_plot)
    finite_difference = np.abs(display_difference[np.isfinite(display_difference)])
    vmax = (
        max(0.5, float(np.percentile(finite_difference, 95.0)))
        if finite_difference.size else 0.5
    )
    diff_image = axes[2].pcolormesh(
        distance, display_chi, display_difference.T, shading="nearest",
        cmap="RdBu_r", vmin=-vmax, vmax=vmax, rasterized=True,
    )
    axes[2].set_title(
        "Four-run minus five-run relative FES\n(separately zero-referenced)", pad=5
    )
    axes[2].set_xlabel("D49–R102 residue COM distance (Å)")
    axes[2].set_ylabel("Q107 χ₁ dihedral angle (°)")
    axes[2].grid(False)
    fig.colorbar(image, ax=axes[:2], pad=0.02, fraction=0.035,
                 label="Relative free energy (kcal mol⁻¹)")
    fig.colorbar(diff_image, ax=axes[2], pad=0.02, fraction=0.06,
                 label="Relative-FES difference (kcal mol⁻¹)")
    fig.savefig(output_dir / "single_run_exclusion_sensitivity.png", dpi=500,
                bbox_inches="tight", pad_inches=0.04)
    fig.savefig(output_dir / "single_run_exclusion_sensitivity.pdf",
                bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def write_fes_grid(path: Path, distance: np.ndarray, chi_deg: np.ndarray,
                   probability: np.ndarray, fes: np.ndarray,
                   four_probability: np.ndarray, four_fes: np.ndarray,
                   label_grid: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "distance_A", "chi1_deg", "probability_5rep", "fes_5rep_kcal_mol",
            "probability_4run_sensitivity", "fes_4run_sensitivity_kcal_mol",
            "segmentation_region_label",
        ])
        for d_index, d in enumerate(distance):
            for c_index, chi in enumerate(chi_deg):
                writer.writerow([
                    d, chi, probability[d_index, c_index], fes[d_index, c_index],
                    four_probability[d_index, c_index], four_fes[d_index, c_index],
                    int(label_grid[d_index, c_index]),
                ])


def write_single_fes_grid(path: Path, distance: np.ndarray, chi_deg: np.ndarray,
                          probability: np.ndarray, fes: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "distance_A", "chi1_deg", "probability", "fes_kcal_mol",
        ])
        for d_index, d in enumerate(distance):
            for c_index, chi in enumerate(chi_deg):
                writer.writerow([
                    d, chi, probability[d_index, c_index], fes[d_index, c_index],
                ])


def write_cvseq_weights(path: Path, runs: Sequence[RunData], centers: Sequence[dict]) -> None:
    label_names = {index + 1: center["basin_id"] for index, center in enumerate(centers)}
    fieldnames = [
        "replicate", "cvseq_index_zero_based", "time_ps", "time_ns",
        "distance_A", "chi1_deg",
        "reported_total_potential_kcal_mol", "meta_bias_from_total_minus_wall_kcal_mol",
        "static_wall_kcal_mol", "c_t_kcal_mol", "log_normalized_weight",
        "normalized_within_replicate_weight", "combined_equal_replicate_weight",
        "basin_id",
    ]
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            assert run.basin is not None
            for index in range(run.time_ps.size):
                label = int(run.basin[index])
                writer.writerow({
                    "replicate": run.spec.replicate,
                    "cvseq_index_zero_based": int(run.cvseq_index[index]),
                    "time_ps": float(run.time_ps[index]),
                    "time_ns": float(run.time_ps[index] / 1000.0),
                    "distance_A": float(run.distance_A[index]),
                    "chi1_deg": math.degrees(float(run.chi_rad[index])),
                    "reported_total_potential_kcal_mol": float(
                        run.total_potential_kcal_mol[index]
                    ),
                    "meta_bias_from_total_minus_wall_kcal_mol": float(
                        run.meta_bias_kcal_mol[index]
                    ),
                    "static_wall_kcal_mol": float(run.wall_kcal_mol[index]),
                    "c_t_kcal_mol": float(run.c_t_kcal_mol[index]),
                    "log_normalized_weight": float(run.log_weight[index]),
                    "normalized_within_replicate_weight": float(run.weight[index]),
                    "combined_equal_replicate_weight": float(run.weight[index] / len(runs)),
                    "basin_id": label_names.get(label, "unassigned"),
                })


def write_trajectory_cvs(path: Path, runs: Sequence[RunData],
                         centers: Sequence[dict]) -> None:
    label_names = {index + 1: center["basin_id"] for index, center in enumerate(centers)}
    fieldnames = [
        "replicate", "trajectory_frame_zero_based", "trajectory_frame_one_based",
        "time_ps", "time_ns", "distance_A", "chi1_deg", "basin_id",
    ]
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            assert run.trajectory_frame_index is not None
            assert run.trajectory_time_ps is not None
            assert run.trajectory_distance_A is not None
            assert run.trajectory_chi_rad is not None
            assert run.trajectory_basin is not None
            for index in range(run.trajectory_time_ps.size):
                label = int(run.trajectory_basin[index])
                frame_index = int(run.trajectory_frame_index[index])
                writer.writerow({
                    "replicate": run.spec.replicate,
                    "trajectory_frame_zero_based": frame_index,
                    "trajectory_frame_one_based": frame_index + 1,
                    "time_ps": float(run.trajectory_time_ps[index]),
                    "time_ns": float(run.trajectory_time_ps[index] / 1000.0),
                    "distance_A": float(run.trajectory_distance_A[index]),
                    "chi1_deg": math.degrees(float(run.trajectory_chi_rad[index])),
                    "basin_id": label_names.get(label, "unassigned"),
                })


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Trajectory-reweight five matched two-CV Desmond WT-MetaD runs, "
            "construct a periodic combined FES, and extract basin representatives."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--e6-auto", action="store_true",
                        help="use the built-in 8.10.25–8.14.25 E6 job stems")
    source.add_argument("--manifest", type=Path,
                        help="CSV/TSV specifying all five input files per run")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT,
                        help=f"WT_MetaD parent directory (default: {DEFAULT_INPUT_ROOT})")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("wtmetad_5rep_reweighted_fes"))
    parser.add_argument("--expected-ktemp", type=float, default=8.4)
    parser.add_argument("--expected-duration-ns", type=float, default=1000.0)
    parser.add_argument("--expected-protein-residues", type=int, default=151)
    parser.add_argument("--expected-zinc", type=int, default=2)
    parser.add_argument("--zinc-coordination-cutoff-A", type=float, default=3.0)
    parser.add_argument(
        "--sensitivity-exclude-replicate", default="rep01_8.10.25",
        help=(
            "replicate omitted from the automatic four-run sensitivity surface "
            "(default: rep01_8.10.25)"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument(
        "--start-ns", type=float, default=0.0,
        help=(
            "elapsed time discarded from the beginning of every run; "
            "default 0 uses each complete 1 us trajectory"
        ),
    )
    parser.add_argument(
        "--target", choices=("wall-restrained", "unwalled"), default="wall-restrained",
        help=(
            "target distribution: remove only WT-MetaD bias (wall-restrained, default) "
            "or also remove the static distance wall (unwalled)"
        ),
    )
    parser.add_argument("--distance-min-A", type=float, default=5.0)
    parser.add_argument("--distance-max-A", type=float,
                        help="plot/histogram maximum; default derives from --wall-safe-energy")
    parser.add_argument("--distance-bin-A", type=float, default=0.25)
    parser.add_argument("--chi-bin-deg", type=float, default=5.0)
    parser.add_argument("--smooth-distance-A", type=float, default=0.50)
    parser.add_argument("--smooth-chi-deg", type=float, default=7.5)
    parser.add_argument("--wall-safe-energy", type=float, default=0.10,
                        help="derive the maximum distance where the wall is no stronger than this")

    parser.add_argument(
        "--ct-distance-min-A", type=float,
        help="lower c(t) integration bound; inferred from cfg range when available",
    )
    parser.add_argument(
        "--ct-distance-max-A", type=float,
        help="upper c(t) integration bound; inferred from cfg range when available",
    )
    parser.add_argument("--ct-distance-step-A", type=float, default=0.5)
    parser.add_argument("--ct-angle-step-deg", type=float, default=5.0)
    parser.add_argument("--ct-omitted-tail-tolerance", type=float, default=1e-8)
    parser.add_argument("--bias-validation-points", type=int, default=21)
    parser.add_argument("--bias-validation-tolerance", type=float, default=0.005)
    parser.add_argument("--time-tolerance-ps", type=float)
    parser.add_argument(
        "--minimum-same-time-validation-frames", type=int, default=3,
        help=(
            "minimum exact-time DTR/cvseq matches required per run for independent "
            "coordinate validation (default: 3)"
        ),
    )
    parser.add_argument("--coordinate-distance-tolerance-A", type=float, default=0.05)
    parser.add_argument("--coordinate-chi-tolerance-deg", type=float, default=1.0)

    parser.add_argument("--basin-centers", type=Path,
                        help="optional reviewed CSV with basin_id,distance_A,chi1_deg")
    parser.add_argument("--basin-minimum-max-fes", type=float, default=6.0)
    parser.add_argument("--basin-assignment-max-fes", type=float, default=10.0)
    parser.add_argument("--basin-min-distance-separation-A", type=float, default=3.0)
    parser.add_argument("--basin-min-chi-separation-deg", type=float, default=60.0)
    parser.add_argument("--max-basins", type=int, default=24)
    parser.add_argument("--replicate-support-threshold", type=float, default=0.01)
    parser.add_argument(
        "--representative-max-delta-fes", type=float, default=1.5,
        help="maximum grid free energy above a basin minimum for exemplar candidates",
    )
    parser.add_argument("--top-representative-candidates", type=int, default=10)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--structure-asl", default="protein or a.e Zn")
    parser.add_argument("--plot-cap-kcal-mol", type=float, default=12.0)
    args = parser.parse_args(argv)

    positive_fields = (
        "expected_ktemp", "expected_duration_ns", "zinc_coordination_cutoff_A",
        "distance_bin_A", "chi_bin_deg",
        "smooth_distance_A", "smooth_chi_deg", "wall_safe_energy",
        "ct_distance_step_A", "ct_angle_step_deg", "bias_validation_points",
        "minimum_same_time_validation_frames",
        "ct_omitted_tail_tolerance",
        "bias_validation_tolerance",
        "coordinate_distance_tolerance_A", "coordinate_chi_tolerance_deg",
        "basin_min_distance_separation_A", "basin_min_chi_separation_deg",
        "basin_minimum_max_fes", "basin_assignment_max_fes",
        "max_basins", "representative_max_delta_fes",
        "top_representative_candidates", "plot_cap_kcal_mol",
    )
    for field in positive_fields:
        value = getattr(args, field)
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    if args.start_ns < 0.0:
        parser.error("--start-ns cannot be negative")
    if args.wall_safe_energy >= 1000.0:
        parser.error("--wall-safe-energy must be below the fixed 1000 kcal/mol wall height")
    if 360.0 / args.chi_bin_deg % 1.0 > 1e-8:
        parser.error("--chi-bin-deg must divide 360 exactly")
    if 360.0 / args.ct_angle_step_deg % 1.0 > 1e-8:
        parser.error("--ct-angle-step-deg must divide 360 exactly")
    if args.bootstrap < 0:
        parser.error("--bootstrap cannot be negative")
    if args.time_tolerance_ps is not None and (
        not math.isfinite(args.time_tolerance_ps) or args.time_tolerance_ps <= 0.0
    ):
        parser.error("--time-tolerance-ps must be finite and positive")
    if args.basin_assignment_max_fes < args.basin_minimum_max_fes:
        parser.error(
            "--basin-assignment-max-fes must be at least --basin-minimum-max-fes"
        )
    if not 0.0 <= args.replicate_support_threshold <= 1.0:
        parser.error("--replicate-support-threshold must be between 0 and 1")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        runs = auto_run_specs(args.input_root) if args.e6_auto else read_manifest(args.manifest)
        protocol, protocols = validate_input_set(
            runs, 5, args.expected_ktemp
        )
    except (ValueError, OSError) as exc:
        fail(str(exc))

    args.temperature_K = float(protocol["temperature_K"])
    args.ktemp = float(protocol["kTemp_kcal_mol"])
    args.kbt = KB_KCAL_PER_MOL_K * args.temperature_K
    args.gamma = 1.0 + args.ktemp / args.kbt
    hill_interval_ps = float(protocol["hill_interval_ps"])
    if not math.isfinite(hill_interval_ps) or hill_interval_ps <= 0.0:
        fail("the WT-MetaD hill interval is missing or invalid")
    maximum_time_match_tolerance = min(0.01, 0.001 * hill_interval_ps)
    if args.time_tolerance_ps is None:
        args.time_tolerance_ps = maximum_time_match_tolerance
    elif args.time_tolerance_ps > maximum_time_match_tolerance:
        fail(
            f"--time-tolerance-ps cannot exceed {maximum_time_match_tolerance:g} ps "
            "for this hill interval; cvseq values must never be paired across bias updates"
        )
    wall_A = float(protocol["collective_variables"][0]["wall"])
    configured_distance_range = protocol["collective_variables"][0].get("range")
    if args.ct_distance_min_A is None and args.ct_distance_max_A is None:
        if (
            isinstance(configured_distance_range, list)
            and len(configured_distance_range) == 2
        ):
            args.ct_distance_min_A = float(configured_distance_range[0])
            args.ct_distance_max_A = float(configured_distance_range[1])
        else:
            fail(
                "the cfg does not define a finite distance range for c(t); provide "
                "both --ct-distance-min-A and --ct-distance-max-A explicitly "
                "(the E6 analysis uses the physical 0–40 Å interval)"
            )
    elif args.ct_distance_min_A is None or args.ct_distance_max_A is None:
        fail("provide both c(t) distance bounds, or neither when cfg contains a range")
    if (
        not math.isfinite(args.ct_distance_min_A)
        or not math.isfinite(args.ct_distance_max_A)
        or args.ct_distance_min_A < 0.0
        or args.ct_distance_min_A >= args.ct_distance_max_A
    ):
        fail("c(t) distance bounds must be finite, nonnegative, and increasing")
    if not math.isfinite(args.distance_min_A) or args.distance_min_A < 0.0:
        fail("distance minimum must be finite and nonnegative")
    if args.distance_max_A is not None and not math.isfinite(args.distance_max_A):
        fail("distance maximum must be finite")
    if args.distance_max_A is None:
        args.distance_max_A = wall_A + 0.2 * math.log(
            args.wall_safe_energy / (1000.0 - args.wall_safe_energy)
        )
        # Round down to the requested histogram grid.
        bins = math.floor(
            (args.distance_max_A - args.distance_min_A) / args.distance_bin_A
        )
        args.distance_max_A = args.distance_min_A + bins * args.distance_bin_A
    if args.distance_min_A >= args.distance_max_A:
        fail("distance minimum must be below distance maximum")
    distance_bin_count_float = (
        (args.distance_max_A - args.distance_min_A) / args.distance_bin_A
    )
    distance_bin_count = int(round(distance_bin_count_float))
    if not math.isclose(
        distance_bin_count_float, distance_bin_count, rel_tol=0.0, abs_tol=1e-8
    ):
        fail("the requested distance bin width must divide the analysis range exactly")
    if distance_bin_count < 5:
        fail("the analysis range must contain at least five distance bins")
    wall_at_max = float(desmond_wall(args.distance_max_A, wall_A))
    if (
        args.target == "unwalled"
        and wall_at_max > args.wall_safe_energy * (1.0 + 1e-8)
    ):
        fail(
            f"distance maximum {args.distance_max_A:g} Å has wall energy "
            f"{wall_at_max:.6g} kcal/mol, above --wall-safe-energy "
            f"{args.wall_safe_energy:g}"
        )
    if (
        args.target == "wall-restrained"
        and wall_at_max > args.wall_safe_energy * (1.0 + 1e-8)
    ):
        print(
            "WARNING: the requested display includes a region materially shaped by "
            "the retained 40 Å wall.",
            file=sys.stderr,
        )

    print(f"Validated {len(runs)} protocol-matched runs.")
    print(
        f"T={args.temperature_K:g} K; kTemp={args.ktemp:g} kcal/mol; "
        f"gamma={args.gamma:.6f}; target={args.target}"
    )
    for run in runs:
        print(f"  {run.replicate}: {run.trajectory}")
    if args.sensitivity_exclude_replicate not in {run.replicate for run in runs}:
        fail(
            "--sensitivity-exclude-replicate does not match an input replicate; "
            f"available values are {[run.replicate for run in runs]}"
        )
    if args.dry_run:
        print("DRY RUN PASS: paths, file uniqueness, and protocol checks succeeded.")
        return 0

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        fail(f"output directory must be new or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "STATUS.json", {
        "status": "RUNNING",
        "started_utc": datetime.now(timezone.utc).isoformat(),
    })

    manifest = {
        "status": "RUNNING",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "argv": list(sys.argv[1:] if argv is None else argv),
        "python": sys.version,
        "schrodinger_root": os.environ.get("SCHRODINGER"),
        "protocol": protocol,
        "parameters": vars(args).copy(),
        "inputs": [],
        "scientific_definition": {
            "independent_run_weighting": "Tiwary-Parrinello c(t), normalized within run",
            "combination": "arithmetic mean of five normalized run probability grids",
            "reported_domain": (
                "probabilities are conditioned and renormalized within the declared "
                "distance and periodic-dihedral analysis domain"
            ),
            "ct_integration_domain": (
                "fixed distance grid from --ct-distance-min-A to --ct-distance-max-A "
                "and one nonduplicated periodic chi1 cycle"
            ),
            "kernel_cutoff_approximation": (
                "c(t) grid retains Gaussian tails beyond the Desmond joint cutoff; "
                "a strict summed-height tail bound is required below the configured tolerance"
            ),
            "chi_periodicity": "[-180,180) with wrapped smoothing and distances",
            "fes_sample_source": "all selected cvseq records; no forced DTR/cvseq pairing",
            "structure_source": "all selected saved DTR/XTC frames with independently recomputed CVs",
            "same_time_validation": (
                "exact-time DTR/cvseq matches validate coordinate identity only; they are "
                "not the statistical sample used for the FES"
            ),
            "representative_definition": (
                "saved structure frame nearest a candidate FES minimum in the two biased CVs"
            ),
            "importance_ess_caveat": "Kish ESS from importance weights; not corrected for time autocorrelation",
            "basin_count": "data-derived or supplied in reviewed basin-center CSV; never hard-coded",
        },
    }
    for run in runs:
        manifest["inputs"].append({
            "replicate": run.replicate,
            "stem": run.stem,
            "cfg": str(run.cfg),
            "cfg_sha256": sha256(run.cfg),
            "kerseq": str(run.kerseq),
            "kerseq_sha256": sha256(run.kerseq),
            "cvseq": str(run.cvseq),
            "cvseq_sha256": sha256(run.cvseq),
            "cms": str(run.cms),
            "cms_sha256": sha256(run.cms),
            "trajectory": str(run.trajectory),
        })
    write_json(output_dir / "analysis_manifest.json", manifest)

    try:
        processed = [process_run(run, protocols[index], args, output_dir)
                     for index, run in enumerate(runs)]
        signatures = {run.protein_signature for run in processed}
        if len(signatures) != 1:
            raise ValueError("protein atom/residue topology differs among CMS inputs")
        system_signatures = {run.system_signature for run in processed}
        if len(system_signatures) != 1:
            raise ValueError(
                "full-system atom, bond, mass, or charge signature differs among CMS inputs"
            )

        distance_edges = np.linspace(
            args.distance_min_A, args.distance_max_A,
            distance_bin_count + 1,
        )
        chi_edges = np.arange(-180.0, 180.0 + args.chi_bin_deg * 0.5, args.chi_bin_deg)
        distance = (distance_edges[:-1] + distance_edges[1:]) / 2.0
        chi_deg = (chi_edges[:-1] + chi_edges[1:]) / 2.0
        for run in processed:
            run.probability = histogram_probability(
                run, distance_edges, chi_edges,
                args.smooth_distance_A, args.smooth_chi_deg,
            )
            run.fes = probability_to_fes(run.probability, args.kbt)
            per_run_dir = output_dir / "per_replicate" / run.spec.replicate
            write_single_fes_grid(
                per_run_dir / "reweighted_fes.csv",
                distance, chi_deg, run.probability, run.fes,
            )

        probability_stack = np.asarray([run.probability for run in processed])
        combined_probability = np.mean(probability_stack, axis=0)
        combined_probability /= np.sum(combined_probability)
        combined_fes = probability_to_fes(combined_probability, args.kbt)
        excluded_indices = [
            index for index, run in enumerate(processed)
            if run.spec.replicate == args.sensitivity_exclude_replicate
        ]
        if len(excluded_indices) != 1:
            raise ValueError(
                "--sensitivity-exclude-replicate must match exactly one replicate; "
                f"available values are {[run.spec.replicate for run in processed]}"
            )
        excluded_index = excluded_indices[0]
        sensitivity_indices = [
            index for index in range(len(processed)) if index != excluded_index
        ]
        four_probability = np.mean(probability_stack[sensitivity_indices], axis=0)
        four_probability /= np.sum(four_probability)
        four_fes = probability_to_fes(four_probability, args.kbt)

        # The probabilities were already smoothed periodically before conversion
        # to free energy.  A second smoothing of free energy would alter barriers.
        basin_surface = combined_fes.copy()
        if args.basin_centers:
            centers = read_basin_centers(
                args.basin_centers.expanduser().resolve(), distance, chi_deg
            )
            center_source = str(args.basin_centers.expanduser().resolve())
        else:
            centers = detect_basin_centers(
                distance, chi_deg, basin_surface,
                args.basin_minimum_max_fes,
                args.basin_min_distance_separation_A,
                args.basin_min_chi_separation_deg,
                args.max_basins,
                args.smooth_distance_A,
            )
            center_source = "automatic periodic local minima"
        region_definition_status = (
            "reviewed_basin_definition"
            if args.basin_centers else "automatic_candidate_minimum_region"
        )
        label_grid = periodic_watershed(
            basin_surface, centers, args.basin_assignment_max_fes
        )
        for run in processed:
            run.basin = coordinate_grid_labels(
                run.distance_A, run.chi_rad, distance_edges, chi_edges, label_grid
            )
            assert run.trajectory_distance_A is not None
            assert run.trajectory_chi_rad is not None
            run.trajectory_basin = coordinate_grid_labels(
                run.trajectory_distance_A, run.trajectory_chi_rad,
                distance_edges, chi_edges, label_grid,
            )

        center_rows = []
        for center in centers:
            d_index = int(center["distance_index"])
            c_index = int(center["chi_index"])
            center_rows.append({
                "basin_id": center["basin_id"],
                "distance_A": float(distance[d_index]),
                "chi1_deg": float(chi_deg[c_index]),
                "relative_free_energy_kcal_mol": float(basin_surface[d_index, c_index]),
                "source": center_source,
                "region_definition_status": region_definition_status,
            })
        write_csv_rows(output_dir / "candidate_basin_centers.csv", center_rows)

        basin_rows, population_matrix = basin_statistics(
            processed, centers, distance, chi_deg, combined_probability,
            basin_surface, label_grid,
            args.replicate_support_threshold, args.bootstrap, args.seed, args.kbt,
        )
        basin_rows = [
            {"region_definition_status": region_definition_status, **row}
            for row in basin_rows
        ]
        write_csv_rows(output_dir / "basins.csv", basin_rows)
        representatives, candidate_rows = select_representatives(
            processed, centers, distance, chi_deg, basin_surface,
            distance_edges, chi_edges,
            float(protocol["collective_variables"][0]["width"]),
            float(protocol["collective_variables"][1]["width"]),
            args.representative_max_delta_fes,
            args.top_representative_candidates,
        )
        for row in representatives:
            row["region_definition_status"] = region_definition_status
        for row in candidate_rows:
            row["region_definition_status"] = region_definition_status
        write_csv_rows(output_dir / "representative_candidates.csv", [
            {key: value for key, value in row.items() if key != "run_index"}
            for row in candidate_rows
        ])
        extracted = extract_representative_structures(
            representatives, processed, protocol, output_dir,
            args.structure_asl,
            args.coordinate_distance_tolerance_A,
            args.coordinate_chi_tolerance_deg,
            args.expected_protein_residues,
            args.expected_zinc,
            args.zinc_coordination_cutoff_A,
        )
        write_csv_rows(output_dir / "representatives.csv", extracted)

        sensitivity_metrics = surface_sensitivity_metrics(
            combined_probability, combined_fes, four_probability, four_fes,
            label_grid, centers, args.basin_minimum_max_fes,
        )
        write_json(output_dir / "single_run_exclusion_sensitivity_metrics.json",
                   sensitivity_metrics)

        write_fes_grid(
            output_dir / "combined_reweighted_fes_grid.csv",
            distance, chi_deg, combined_probability, combined_fes,
            four_probability, four_fes, label_grid,
        )
        np.savez_compressed(
            output_dir / "combined_reweighted_analysis.npz",
            distance_A=distance,
            chi1_deg=chi_deg,
            per_replicate_probability=probability_stack,
            per_replicate_fes_kcal_mol=np.asarray([run.fes for run in processed]),
            combined_5rep_probability=combined_probability,
            combined_5rep_fes_kcal_mol=combined_fes,
            sensitivity_excluded_replicate=np.asarray(
                [args.sensitivity_exclude_replicate]
            ),
            four_run_sensitivity_probability=four_probability,
            four_run_sensitivity_fes_kcal_mol=four_fes,
            basin_surface_kcal_mol=basin_surface,
            basin_label_grid=label_grid,
            per_replicate_basin_populations=population_matrix,
        )
        write_cvseq_weights(
            output_dir / "cvseq_weights_and_regions.csv.gz", processed, centers
        )
        write_trajectory_cvs(
            output_dir / "trajectory_frame_cvs_and_regions.csv.gz", processed, centers
        )
        save_figures(
            output_dir, distance, chi_deg, processed,
            combined_fes, four_fes, centers, args.plot_cap_kcal_mol, args.target,
            args.sensitivity_exclude_replicate,
        )

        run_summary = []
        for run in processed:
            assert run.trajectory_time_ps is not None
            unassigned = float(np.sum(run.weight[run.basin == 0]))
            run_summary.append({
                "replicate": run.spec.replicate,
                "selected_cvseq_records": int(run.time_ps.size),
                "selected_cvseq_start_ns": float(run.time_ps[0] / 1000.0),
                "selected_cvseq_end_ns": float(run.time_ps[-1] / 1000.0),
                "selected_structure_frames": int(run.trajectory_time_ps.size),
                "selected_structure_start_ns": float(run.trajectory_time_ps[0] / 1000.0),
                "selected_structure_end_ns": float(run.trajectory_time_ps[-1] / 1000.0),
                "importance_weight_ess_not_autocorrelation_corrected": run.importance_ess,
                "maximum_normalized_cvseq_weight": run.max_weight_fraction,
                "unassigned_cvseq_weight_probability": unassigned,
                "time_match_max_abs_error_ps": run.time_match_max_error_ps,
                "same_time_dtr_cvseq_validation_frames": (
                    run.matched_trajectory_frame_count
                ),
                "bias_validation_max_abs_error_kcal_mol": run.bias_validation_max_error,
                "coordinate_distance_max_abs_error_A": run.distance_validation_max_error_A,
                "coordinate_chi_max_abs_error_deg": run.chi_validation_max_error_deg,
                "ct_omitted_kernel_tail_upper_bound_kcal_mol": (
                    run.ct_omitted_kernel_tail_upper_bound_kcal_mol
                ),
                "protein_residue_count": run.protein_residue_count,
                "zinc_count": run.zinc_count,
                "system_signature_sha256": run.system_signature,
            })
        write_csv_rows(output_dir / "run_summary.csv", run_summary)

        warnings = []
        warnings.append(
            "The automatic candidate-region partition is operational and must be reviewed; "
            "it is not "
            "evidence for a predetermined nine- or twelve-basin model."
        )
        warnings.append(
            "Exported structures are CV-nearest sampled exemplars: frames nearest minima in "
            "D49-R102 COM distance and Q107 chi1 space. They are not all-atom medoids, and "
            "hidden structural coordinates may remain heterogeneous."
        )
        warnings.append(
            "Importance-weight ESS values do not correct for trajectory autocorrelation."
        )
        warnings.append(
            "The five-run primary result includes every protocol-matched run. The separate "
            f"sensitivity output omits the user-selected run "
            f"{args.sensitivity_exclude_replicate}; by default this is the 8.10.25 run flagged "
            "by the prior convergence review."
        )
        warnings.append(
            "Computational completion is not a scientific pass. Review the all-five versus "
            "leave-8.10-out sensitivity metrics and plot before using the FES, populations, "
            "region count, or exemplars in the manuscript."
        )
        if args.start_ns == 0.0:
            warnings.append(
                "The primary result uses each complete trajectory. A post-transient cutoff has "
                "not been inferred automatically; rerun with a justified --start-ns for a "
                "publication sensitivity analysis."
            )
        if args.target == "wall-restrained":
            if wall_at_max <= args.wall_safe_energy * (1.0 + 1e-8):
                warnings.append(
                    "The target retains the static 40 A wall while removing WT-MetaD bias. "
                    "The default display is cropped before the wall materially rises."
                )
            else:
                warnings.append(
                    "The target retains the static 40 A wall, and the requested display range "
                    "includes a region materially shaped by that restraint."
                )
        else:
            warnings.append(
                "The static wall was also reweighted out. Interpret only the declared wall-safe "
                "distance region; this script does not by itself establish adequate overlap."
            )
        warnings.append(
            f"The numerical c(t) integration used the explicitly recorded distance bounds "
            f"[{args.ct_distance_min_A:g}, {args.ct_distance_max_A:g}] A. Confirm that the "
            "reported FES is stable to a justified wider and/or finer integration grid before "
            "publication."
        )
        warnings.append(
            "MAEGZ is the topology-authoritative exemplar output. PDB copies are supplied for "
            "visualization and should not be relied on to retain zero-order Zn-cysteine bonds."
        )
        warnings.append(
            "The cross-run system signature checks atom identities, bonds, masses, and charges; "
            "it is not a complete comparison of every force-field or constraint parameter."
        )
        (output_dir / "READ_BEFORE_MANUSCRIPT_USE.txt").write_text(
            "\n".join(f"- {warning}" for warning in warnings) + "\n", encoding="utf-8"
        )

        manifest.update({
            "status": "COMPLETE",
            "scientific_validation_status": "PENDING_REVIEW",
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "automatic_or_reviewed_basin_source": center_source,
            "region_definition_status": region_definition_status,
            "candidate_or_reviewed_region_count": len(centers),
            "wall_energy_at_histogram_max_kcal_mol": wall_at_max,
            "run_summary": run_summary,
            "single_run_exclusion_sensitivity_metrics": sensitivity_metrics,
            "warnings": warnings,
            "outputs": {
                "main_png": str((output_dir / "combined_5rep_reweighted_fes.png").resolve()),
                "main_pdf": str((output_dir / "combined_5rep_reweighted_fes.pdf").resolve()),
                "grid_csv": str((output_dir / "combined_reweighted_fes_grid.csv").resolve()),
                "cvseq_weights_csv_gz": str(
                    (output_dir / "cvseq_weights_and_regions.csv.gz").resolve()
                ),
                "trajectory_cvs_csv_gz": str(
                    (output_dir / "trajectory_frame_cvs_and_regions.csv.gz").resolve()
                ),
                "candidate_basin_centers_csv": str(
                    (output_dir / "candidate_basin_centers.csv").resolve()
                ),
                "basins_csv": str((output_dir / "basins.csv").resolve()),
                "representatives_csv": str((output_dir / "representatives.csv").resolve()),
                "representatives_maegz": str(
                    (output_dir / "representatives/all_basin_representatives.maegz").resolve()
                ),
                "exclusion_sensitivity_png": str(
                    (output_dir / "single_run_exclusion_sensitivity.png").resolve()
                ),
                "exclusion_sensitivity_metrics_json": str(
                    (output_dir / "single_run_exclusion_sensitivity_metrics.json").resolve()
                ),
            },
        })
        write_json(output_dir / "analysis_manifest.json", manifest)
        write_json(output_dir / "STATUS.json", {
            "status": "COMPLETE",
            "scientific_validation_status": "PENDING_REVIEW",
            "finished_utc": manifest["finished_utc"],
            "region_definition_status": region_definition_status,
            "candidate_or_reviewed_region_count": len(centers),
        })
    except Exception as exc:
        write_json(output_dir / "STATUS.json", {
            "status": "FAILED",
            "scientific_validation_status": "NOT_EVALUATED",
            "failed_utc": datetime.now(timezone.utc).isoformat(),
            "error_type": type(exc).__name__,
            "error": str(exc),
        })
        raise

    print(f"COMPLETE: five-run computation written to {output_dir}")
    print(
        f"Processed {len(centers)} candidate or reviewed regions and extracted "
        "CV-nearest sampled exemplars."
    )
    print("SCIENTIFIC VALIDATION: PENDING_REVIEW (see READ_BEFORE_MANUSCRIPT_USE.txt)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
