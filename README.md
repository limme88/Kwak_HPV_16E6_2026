# Kwak_HPV_16E6_2026

Code accompanying "**Conformational Dynamics and Disorder of HPV-16 E6**" by Liam Kwak and Roberto A. Garza-López (Pomona College).

- Paper: Kwak, L.; Garza-López, R. A. *J. Phys. Chem. B* **2026**. <https://doi.org/10.1021/acs.jpcb.6c04908>
- Trajectories and large data: Zenodo, <https://doi.org/10.5281/zenodo.XXXXXXX>

## Repository contents

| Folder | Contents | Manuscript |
|---|---|---|
| `structures/` | 4XR8 and 7UAJ reference structures and the analysis topology | Methods |
| `simulation_inputs/unbiased_md/` | Desmond `.msj`/`.cfg` (and `.cms` where under 10 MB) for the four unbiased runs (3 × 10 µs, 1 × 5 µs) | Methods |
| `simulation_inputs/wt_metad/` | Desmond well-tempered metadynamics inputs for the five replicates | Methods, Fig. 6 |
| `analysis/01_rmsd_rmsf/` | RMSD/RMSF were computed in Desmond's Trajectory Player (no script) | Fig. S1 |
| `analysis/02_structural_observables/` | D49–R102 distance, Q107 χ1, LxxLL SASA and aperture, C-terminal tail extension (MD and BioEmu) | Table S1, Fig. 7 |
| `analysis/03_secondary_structure/` | DSSP helix propensity | Fig. S2 |
| `analysis/04_elvim/` | ELViM landscape (scripts not yet recovered) | Fig. 4 |
| `analysis/05_msm_deeptime/` | TICA, Markov state model, implied timescales, CK test, macrostate observables | Fig. 5, Fig. S4 |
| `analysis/06_wt_metad/` | WT-MetaD free-energy surfaces, minima, representatives, convergence | Fig. 6, Fig. S5 |
| `analysis/07_bioemu/` | BioEmu ensemble (generation script not yet recovered) | Fig. 7 |
| `analysis/08_chemical_shifts/` | SPARTA+ comparison with BMRB 17967 / PDB 2LJZ, Delta2D helicity | Fig. 2, Table 1 |
| `analysis/09_7uaj_comparison/` | Reference metrics for the partner-free structure PDB 7UAJ | Table S1 |
| `data/` | Small processed data used by the plotting scripts | |

Each `analysis/` folder has its own README with exact commands and required inputs.

### Not included

Scripts for the following were not recovered and are not in this repository: Fig. 1 (structure views), Fig. 3 (time series, KDE, SASA panels), Fig. 4 (ELViM projection), Fig. 5a–c (TICA landscape and basins), Fig. S3, BioEmu generation, and the molecular renderings and panel composition in Figs. 2 and 5–7.

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
| WT-MetaD, 5 replicates (solvent removed) + bias files | Desmond WT-MetaD | 5 × 1 µs | <stride> | `apo16E6_wtmetad.tar.gz` |
| BioEmu ensemble | BioEmu | – | <n> | `apo16E6_bioemu.tar.gz` |
| Saved MSM and TICA models | deeptime | – | – | `apo16E6_msm_models.tar.gz` |

## Environment

```bash
conda env create -f environment.yml
conda activate e6-md
```

Desmond and Maestro (Schrödinger 2023-1 / 2025-2) were used to set up and run the simulations and are not included. ELViM: <https://github.com/VLeiteGroup/ELViM>. Delta2D and SPARTA+ must be obtained from their authors.

## Reproducing the figures

1. Download the trajectories (above) into `trajectories/`.
2. Run `analysis/05_msm_deeptime` before `analysis/02_structural_observables`, which uses its frame assignments. The other folders are independent.
3. Each folder README lists the commands and which figure or table they reproduce. Outputs go to `results/`.

## Citation

Please cite the paper above if you use this code (see `CITATION.cff`). The code is released under the MIT License (see `LICENSE`); the trajectories on Zenodo are released under CC-BY-4.0.
