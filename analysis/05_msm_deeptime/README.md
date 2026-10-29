# TICA and Markov state model

The selected notebook fits the E6HL-contact TICA/MSM, performs the CK test,
calculates MFPTs, and writes an implied-timescale plot. It supports **Revision
V12 Figure 5g–h and Figure S4**. The historical filenames say Figure4 because
these panels were renumbered in the manuscript. The plotting script supports
**Figure 5d–f**. Final scripts for Figure 5a–c were not recovered; this directory
does not reproduce those TICA/basin/structure-rendering panels or the composite.

The project notebook was selected over the Desktop copy because its CK and
implied-timescale plots omit grid lines, matching the final supplied figures.
Their code differs only in those grid settings and output stems. The older
pre-cleanup notebook has a different model workflow and was excluded.

## Inputs and run order

Run shell commands from the repository root. Obtain the exact matching
35,002-frame combined trajectory and topology from the eventual Zenodo deposit:

- `trajectories/trajectory_35us.xtc`
- `structures/topology_apo16e6.pdb`
- `trajectories/msm/apo16e6_msm_numeric_bundle.npz` (saved labels used in the
  recorded run; also needed to reproduce the published assignments exactly).

1. Reproduce the saved assignment branch used in the recorded notebook execution:

```bash
mkdir -p results/msm
cp trajectories/msm/apo16e6_msm_numeric_bundle.npz results/msm/
APO16E6_REUSE_MICROSTATE_CACHE=1 jupyter nbconvert --execute --to notebook --ExecutePreprocessor.timeout=-1 --output apo16e6_executed.ipynb --output-dir results/msm analysis/05_msm_deeptime/apo-16E6_deeptime_clean.ipynb
```

To rebuild features/TICA/k-means instead, run the same `jupyter nbconvert`
command with `APO16E6_REUSE_MICROSTATE_CACHE=0`. The source retains seed 42,
136 nonadjacent E6HL closest-heavy features, TICA lag 10 frames, four TICs,
six microstates, MSM lag 50 frames, effective counting, and 500 posterior samples.
The recorded execution reused labels; exact figure equivalence of a fresh
clustering run has not been established.

2. Generate the structural-observable panels using the published assignment
bundle and the included frame-level CSV:

```bash
python analysis/05_msm_deeptime/plot_figure4def_structural_observables.py --metrics-csv data/msm/md_frame_metrics.csv --msm-bundle trajectories/msm/apo16e6_msm_numeric_bundle.npz --output-dir results/msm/structural_observables
```

The script checks every CSV macrostate assignment against the bundle and retains
5 µs blocks, 20,000 bootstrap draws, and base seed 20260908. It writes PNG/PDF,
summary CSV and a JSON manifest. Output stems retain `Figure4d-f`.

## Software and interpretation

Recorded notebook environment: Python 3.11.14, deeptime 0.4.5, MDTraj 1.11.1,
NumPy 2.4.2, Matplotlib 3.10.8. Pandas is additionally required by the panel
script; its original version was not recorded in this notebook.

The source declares 1 ns/frame but its saved execution detects 0.1 ns/frame in
replicas 1–3 and 1 ns/frame in replica 4. This unresolved source/export timebase
warning is preserved. The feature code includes all 136 nonadjacent E6HL pairs;
it does not remove pairs containing zinc-ligating cysteines, despite the V12
Methods statement about masking. Physical kinetics and the original feature
claim therefore require author resolution. No scientific settings were changed.
