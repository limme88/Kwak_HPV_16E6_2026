# V12 figure reproduction map

Run from the repository root. Plotting scripts remain beside their analysis dependencies to avoid divergent copies. The analysis READMEs give exact commands and required inputs.

| V12 output | Source code location | Coverage |
|---|---|---|
| Figure 1 | Missing | Molecular views, schematic and composition |
| Figure 2 | `analysis/08_chemical_shifts/make_figure2.py` | Numerical panels; composition separate |
| Figure 3 | Missing | Time series, KDE, SASA and molecular views |
| Figure 4 | Missing | E6 ELViM projection, basins and final panels |
| Figure 5d–f | `analysis/05_msm_deeptime/plot_figure4def_structural_observables.py` | Final structural-observable panels; filename retains old numbering |
| Figure 5g–h | `analysis/05_msm_deeptime/apo-16E6_deeptime_clean.ipynb` | Corrected MSM notebook with original cached assignments or rerun inputs |
| Figure 5a–c | Missing | Final TICA landscape, basins and molecular rendering |
| Figure 6 | `analysis/06_wt_metad/` | FES, minima and representatives; molecular rendering/composite missing |
| Figure 7 | `analysis/02_structural_observables/` | BioEmu metrics only; original generation, rendering and schematic missing |
| S1 | Desmond Trajectory Player | No standalone script recovered |
| S2 | `analysis/03_secondary_structure/make_dssp_propensity_figure.py` | Original patterns; final typography/legend edited separately |
| S3 | Missing | Correlation and uncertainty figure |
| S4 | MSM notebook in folder 05 | Implied timescales |
| S5 | `analysis/06_wt_metad/plot_wtmetad_convergence_publication.py` | Plot from archived diagnostic arrays; copyrighted generator excluded |
| Table 1 | `analysis/08_chemical_shifts/make_table1.py` | Numerical calculation; Cα RMSD rounds to 1.01 versus V12 1.00 |
| Table S1 | `analysis/02_structural_observables/compute_common_metrics.py` | Numerical metrics; Office assembly excluded |

A successful public-file check does not establish end-to-end figure reproduction. Original inputs must be supplied through Zenodo. Missing scientific workflows were not replaced with invented code.
