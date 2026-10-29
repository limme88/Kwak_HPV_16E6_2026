# DSSP propensity — V12 Figure S2
`make_dssp_propensity_figure.py` computes full DSSP assignments and 100 ns propensity bins for four separate replicas (10, 10, 10, 5 µs). Its saved output matches the secondary-structure patterns in V12 S2, whose typography and legend were edited during final assembly; this script does not recreate that final layout exactly.

Run from the repository root after downloading the four matching XTC files:

```bash
python analysis/03_secondary_structure/make_dssp_propensity_figure.py --trajectory-dir trajectories --topology structures/topology_apo16e6.pdb --output-dir results/secondary_structure --frame-spacing-ns 1 --bin-ns 100
```

Files: `replicate_1_10us.xtc`, `replicate_2_10us.xtc`, `replicate_3_10us.xtc`, `replicate_4_5us.xtc`. Outputs include assignments, CSVs and PDF/PNG plots. Small saved CSVs are in `data/secondary_structure/`. MDTraj implements DSSP; no vendor source is included. Source manifest versions: Python 3.11.14, MDTraj 1.11.1, NumPy 2.4.2, Matplotlib 3.10.8. https://www.mdtraj.org/

This plot counts H/G/I as helix, whereas the Table S1 common metric counts H/G. These original definitions were retained. The split trajectories use coordinate-supported boundaries and declared 1 ns spacing; the original concatenated XTC has mixed stored timestamps. Do not concatenate replicas for kinetic transition counting.
