#!/usr/bin/env python3
"""Compute D49–R102 COM distance and Q107 chi1 for 7UAJ E6 copies.
Input: deposited complete 7UAJ PDB; chain D is the default reference.
Output: JSON containing the reference coordinates and residue mapping.
Supports the V12 Figure 7 discussion and comparison with Figure 6.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


MASSES = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "S": 32.06}
E6_TO_PDB_OFFSET = 378
TARGETS = {"D49": 49, "R102": 102, "Q107": 107}


def vector_sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def norm(a):
    return math.sqrt(dot(a, a))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_first_model(path: Path) -> list[dict]:
    atoms = []
    saw_model = False
    active_model = True
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
            element = "".join(c for c in line[12:16] if c.isalpha())[:1].upper()
        atoms.append({
            "serial": int(line[6:11]),
            "name": line[12:16].strip(),
            "altloc": line[16].strip(),
            "resname": line[17:20].strip(),
            "chain": line[21].strip(),
            "resseq": int(line[22:26]),
            "icode": line[26].strip(),
            "occupancy": float(line[54:60]),
            "element": element,
            "xyz": tuple(float(line[a:b]) for a, b in ((30, 38), (38, 46), (46, 54))),
        })
    return atoms


def select_altlocs(atoms: list[dict]) -> list[dict]:
    grouped = {}
    for atom in atoms:
        key = (
            atom["chain"], atom["resseq"], atom["icode"],
            atom["resname"], atom["name"],
        )
        grouped.setdefault(key, []).append(atom)
    selected = []
    for candidates in grouped.values():
        blanks = [a for a in candidates if not a["altloc"]]
        if blanks:
            selected.append(blanks[0])
        else:
            selected.append(max(
                candidates,
                key=lambda a: (a["occupancy"], a["altloc"] == "A", -a["serial"]),
            ))
    return selected


def residue(atoms: list[dict], chain: str, e6_number: int) -> list[dict]:
    pdb_number = e6_number + E6_TO_PDB_OFFSET
    selected = [
        a for a in atoms
        if a["chain"] == chain and a["resseq"] == pdb_number and not a["icode"]
    ]
    if not selected:
        raise ValueError(f"missing E6 residue {e6_number} (PDB {chain}:{pdb_number})")
    return selected


def center_of_mass(atoms: list[dict]):
    weights = [MASSES[a["element"]] for a in atoms]
    total = sum(weights)
    center = tuple(
        sum(a["xyz"][i] * weight for a, weight in zip(atoms, weights)) / total
        for i in range(3)
    )
    return center, total


def signed_dihedral_deg(p1, p2, p3, p4) -> float:
    """Signed four-point torsion matching the Desmond/Schrodinger convention."""
    r12 = vector_sub(p1, p2)
    r32 = vector_sub(p3, p2)
    r42 = vector_sub(p4, p2)
    c1 = cross(r12, r32)
    c2 = cross(r42, r32)
    cosine = max(-1.0, min(1.0, dot(c1, c2) / (norm(c1) * norm(c2))))
    angle = math.degrees(math.acos(cosine))
    return -angle if dot(r12, c2) < 0.0 else angle


def calculate_copy(atoms: list[dict], chain: str) -> dict:
    d49 = residue(atoms, chain, TARGETS["D49"])
    r102 = residue(atoms, chain, TARGETS["R102"])
    q107 = residue(atoms, chain, TARGETS["Q107"])
    expected = {"D49": "ASP", "R102": "ARG", "Q107": "GLN"}
    observed = {"D49": d49[0]["resname"], "R102": r102[0]["resname"], "Q107": q107[0]["resname"]}
    if observed != expected:
        raise ValueError(f"unexpected target residue identities for chain {chain}: {observed}")
    d49_com, d49_mass = center_of_mass(d49)
    r102_com, r102_mass = center_of_mass(r102)
    q107_by_name = {a["name"]: a for a in q107}
    chi_names = ("N", "CA", "CB", "CG")
    if any(name not in q107_by_name for name in chi_names):
        raise ValueError(f"Q107 chi1 atoms are incomplete in chain {chain}")
    return {
        "D49_R102_mass_weighted_whole_residue_COM_distance_A": norm(
            vector_sub(d49_com, r102_com)
        ),
        "Q107_chi1_deg": signed_dihedral_deg(
            *(q107_by_name[name]["xyz"] for name in chi_names)
        ),
        "D49_COM_A": list(d49_com),
        "R102_COM_A": list(r102_com),
        "D49_atom_count": len(d49),
        "R102_atom_count": len(r102),
        "D49_total_mass_Da": d49_mass,
        "R102_total_mass_Da": r102_mass,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdb", type=Path)
    parser.add_argument("--selected-chain", default="D")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    atoms = select_altlocs(parse_first_model(args.pdb.resolve()))
    copies = {chain: calculate_copy(atoms, chain) for chain in "ABCD"}
    if args.selected_chain not in copies:
        raise ValueError("selected chain must be one of A, B, C, or D")
    payload = {
        "input_pdb": str(args.pdb.resolve()),
        "sha256": sha256(args.pdb.resolve()),
        "structure": "7UAJ; four MBP-E6 fusion copies in chains A-D",
        "numbering": "151-residue E6 construct residue r = 7UAJ PDB residue r + 378",
        "target_mapping": {name: number + E6_TO_PDB_OFFSET for name, number in TARGETS.items()},
        "selected_chain": args.selected_chain,
        "selection_reason": "matches the existing project 7UAJ analysis",
        "atom_selection": (
            "all explicitly modeled atoms in each residue with standard elemental masses; "
            "7UAJ contains no hydrogen atoms"
        ),
        "chi1_definition": "Q107 N-CA-CB-CG; signed Desmond/Schrodinger convention",
        "selected_result": copies[args.selected_chain],
        "all_E6_copies": copies,
        "caveat": (
            "Chain D is a consistency selection, not evidence that it is a uniquely preferred "
            "crystallographic copy; report the chain explicitly."
        ),
    }
    rendered = json.dumps(payload, indent=2) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)


if __name__ == "__main__":
    main()
