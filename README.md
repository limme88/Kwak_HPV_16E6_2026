# Kwak_HPV_16E6_2026

Code accompanying "**Conformational Dynamics and Disorder of HPV-16 E6**" by Liam Kwak and Roberto A. Garza-López (Pomona College).

- Paper: <journal, year, DOI>
- Preprint: ChemRxiv, <https://doi.org/10.26434/chemrxiv-2025-hvzpc>
- Trajectories and large data: Zenodo, <https://doi.org/10.5281/zenodo.XXXXXXX>

## Repository contents

| Folder | Manuscript section |
|---|---|
| `structures/` | Prepared apo-16E6 model (from PDB 4XR8 chain H, C80S/C97S/C111S/C140S reverted) |
| `simulation_inputs/unbiased_md/` | Desmond inputs (`.cms`, `.cfg`, `.msj`) for the 4 unbiased runs (3 × 10 µs, 1 × 5 µs; OPLS4, TIP3P, 310 K) |
| `simulation_inputs/wt_metad/` | Well-tempered metadynamics inputs (PLUMED / Desmond `.cfg`) |
| `analysis/01_rmsd_rmsf/` | RMSD and RMSF (Fig. S1) |
| `analysis/02_structural_observables/` | D49–R102 Cα distance, Q107 χ1, LxxLL pocket SASA and aperture, C-terminal tail extension |
| `analysis/03_secondary_structure/` | DSSP helix propensity of E6HL (residues 62–79) |
| `analysis/04_elvim/` | ELViM landscape and basin representatives |
| `analysis/05_msm_deeptime/` | TICA, Markov state model, Chapman–Kolmogorov test, MFPTs (Fig. 4) |
| `analysis/06_wt_metad/` | WT-MetaD convergence, FES reconstruction, reweighting |
| `analysis/07_bioemu/` | BioEmu ensemble generation and analysis |
| `analysis/08_chemical_shifts/` | SPARTA+ back-calculation vs. BMRB 17967 / PDB 2LJZ, Delta2D comparison |
| `analysis/09_7uaj_comparison/` | Comparison with the partner-free structure (PDB 7UAJ) |
| `figures/` | Notebooks that regenerate the main-text and SI figures |
| `data/` | Small processed data (CSV) used by the figure notebooks |

## Trajectories (Zenodo)

The trajectories are not stored in this repository. Download them from Zenodo into `trajectories/`, which git ignores:

```bash
mkdir -p trajectories && cd trajectories
wget https://zenodo.org/records/XXXXXXX/files/apo16E6_unbiased_35us_protein.tar.gz
tar -xzf apo16E6_unbiased_35us_protein.tar.gz
```

| Data | Method | Length | Frames | File |
|---|---|---|---|---|
| Unbiased MD, 4 replicates (protein + Zn, solvent removed) | Desmond, OPLS4, 310 K | 35 µs | 1 ns stride | `apo16E6_unbiased_35us_protein.tar.gz` |
| WT-MetaD replicates + HILLS/COLVAR | Desmond/PLUMED | <µs> | <stride> | `apo16E6_wtmetad.tar.gz` |
| BioEmu ensemble | BioEmu | – | <n> | `apo16E6_bioemu.tar.gz` |
| Saved MSM and TICA models | deeptime | – | – | `apo16E6_msm_models.tar.gz` |

## Environment

```bash
conda env create -f environment.yml
conda activate e6-md
```

Desmond and Maestro (Schrödinger 2023-1 / 2025-2) were used to set up and run the simulations and are not included. ELViM: <https://github.com/VLeiteGroup/ELViM>. Delta2D and SPARTA+ must be obtained from their authors.

## Reproducing the figures

1. Download the trajectories (above).
2. Run the scripts in `analysis/` in numbered order. Outputs go to `data/`.
3. Open the notebooks in `figures/`.

## Citation

Please cite the paper above if you use this code (see `CITATION.cff`). The code is released under the MIT License (see `LICENSE`); the trajectories on Zenodo are released under CC-BY-4.0.
