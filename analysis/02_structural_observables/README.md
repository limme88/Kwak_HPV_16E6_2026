# Common structural observables — V12 Table S1
`compute_common_metrics.py` computes D49–R102 Cα distance, Q107 chi1, E6HL DSSP H/G fraction, Cα130–Cα151 tail extension, and the 104-pair closest-heavy-atom groove aperture. It includes the explicitly labeled reduced-atom BioEmu aperture comparison. It does not compute the 46-residue pocket SASA or reproduce Figure 3 time-series/KDE panels; those original scripts remain missing.

Run from the repository root after obtaining the matching trajectories and generating the MSM bundle (see folder 05):

```bash
python analysis/02_structural_observables/compute_common_metrics.py --output-dir results/common_metrics
```

All input paths can be overridden with `--help`; defaults use `structures/` and `trajectories/`. The combined MD input is `trajectories/trajectory_35us.xtc`; BioEmu trajectories belong in `trajectories/bioemu/`. The MSM bundle belongs at `trajectories/msm/apo16e6_msm_numeric_bundle.npz`. Existing small summary CSVs are in `data/common_metrics/`; the per-frame MD table is in `data/msm/`.

The source run records Python 3.11.14, NumPy 2.4.2, MDTraj 1.11.1. The supplied BioEmu sequence has a seven-residue extension; the original verified suffix mapping is preserved. Its 72 conformers are unweighted representatives. The tail coordinate shares residue 130 with the groove wall; correlations are descriptive.
