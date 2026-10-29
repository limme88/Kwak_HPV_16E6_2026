# WT-MetaD analysis

These scripts support **Figure 6 and Figure S5 in Revision V12 (9.23.26)**.
The main landscape is an equal-replicate probability mixture of five terminal
FES estimates, with periodic Q107 chi1. It is not the trajectory-reweighted
surface produced by the optional reweighting command below.

Run all commands from the repository root. Download native inputs from the
associated Zenodo deposit into `trajectories/wt_metad/`. That deposit must contain
the five run directories dated 8.10.25–8.14.25, their CFG/CMS/CVSEQ/KERSEQ and
native trajectories, plus the five terminal `*_meta.fes` grids directly in that
folder. The original byte-pinned `apo16e6_wtmetad_manifest.json` is also required
there. It is historical metadata, not an executable input-path configuration.
Do not modify its bytes: the extraction script verifies its recorded hash.

## Run order and exact commands

1. Reconstruct the five-run terminal mixture and its numerical arrays:

```bash
python analysis/06_wt_metad/reconstruct_fes.py \
  trajectories/wt_metad/desmond_WT_MetaD_4xr8_chainH_E6_1us_d49r102_q107chi1_8.{10,11,12,13,14}.25_meta.fes \
  --output-dir results/wt_metad/fes
```

2. Extract the nearest actual sampled conformers. Native Desmond reading uses
Schrödinger's interpreter; set `SCHRODINGER` to your installation directory.
The first command checks inputs; the second writes structures into a new folder.

```bash
"${SCHRODINGER}/run" analysis/06_wt_metad/extract_combined5_terminal_fes_representatives.py \
  --e6-auto --input-root trajectories/wt_metad \
  --centers data/wt_metad/apo16e6_wtmetad_periodic_2d_minima.csv \
  --fes-manifest trajectories/wt_metad/apo16e6_wtmetad_manifest.json --dry-run
"${SCHRODINGER}/run" analysis/06_wt_metad/extract_combined5_terminal_fes_representatives.py \
  --e6-auto --input-root trajectories/wt_metad \
  --centers data/wt_metad/apo16e6_wtmetad_periodic_2d_minima.csv \
  --fes-manifest trajectories/wt_metad/apo16e6_wtmetad_manifest.json \
  --output-dir results/wt_metad/representatives
```

The extractor imports `wtmetad_5rep_reweight_basins.py` as a parsing library.
It does not call that module's trajectory-reweighting calculation.

3. Plot the numbered Figure 6 landscape and arrange the exemplar files:

```bash
python analysis/06_wt_metad/plot_numbered_fes_basins.py \
  --arrays results/wt_metad/fes/apo16e6_wtmetad_5rep_processed_arrays.npz \
  --representatives-csv results/wt_metad/representatives/representatives.csv \
  --structure-dir results/wt_metad/representatives/representatives \
  --output-png results/wt_metad/figure6.png \
  --output-unlabeled-png results/wt_metad/figure6_unlabeled.png \
  --output-pdf results/wt_metad/figure6.pdf \
  --output-key results/wt_metad/figure6_key.csv \
  --numbered-structure-dir results/wt_metad/numbered_structures
```

Molecular rendering and final multi-panel composition are separate; no final
rendering/compositing script was located. The plotting script retains the source
script's numbered labels and does not change them to Roman numerals.

4. Reproduce Figure S5 from the deposited per-run diagnostic JSON files, laid out
as `trajectories/wt_metad/convergence/rep*/rep*_convergence_metrics.json`:

```bash
mkdir -p results/wt_metad
python analysis/06_wt_metad/plot_wtmetad_convergence_publication.py \
  --input-dir trajectories/wt_metad/convergence \
  --output-pdf results/wt_metad/figureS5.pdf \
  --output-png results/wt_metad/figureS5.png \
  --timeseries-csv results/wt_metad/convergence_timeseries.csv \
  --hills-csv results/wt_metad/convergence_hills.csv
```

The two included convergence CSVs are saved source data. The original plotter
reads JSON rather than these CSVs; its numerical logic has been preserved.
Raw diagnostic generation used a project-modified Schrödinger script marked
“All rights reserved,” which is not redistributed here. The ordinary vendor
script is not claimed to reproduce those modified diagnostics unchanged.

## Optional trajectory reweighting

This separate estimator was not used for the displayed Figure 6 landscape:

```bash
"${SCHRODINGER}/run" analysis/06_wt_metad/wtmetad_5rep_reweight_basins.py \
  --e6-auto --input-root trajectories/wt_metad \
  --ct-distance-min-A 0 --ct-distance-max-A 40 \
  --output-dir results/wt_metad/reweighted
```

The source selections, thresholds, random seed and scientific calculations are
unchanged. Source results distinguish computational completion from equilibrium
validation. The mixture, finite-time reweighting, periodic-image effects and
nearest-CV structure selection do not independently establish equilibrium basin
populations or kinetics.

## External tools

- Schrödinger/Desmond **2026-1** was recorded for structure extraction and
  trajectory reweighting: <https://www.schrodinger.com/>. This proprietary
  environment supplies `schrodinger`; install it separately. No vendor source is
  included. The exact separately supplied convergence-script release is unknown.
- NumPy, SciPy and Matplotlib are required. Exact package versions for these
  particular final figures were not recorded in the inspected manifests.

Original WT-MetaD CMS/CFG/MSJ inputs recovered from Pomona are in
`simulation_inputs/wt_metad/`. The original runs used native Desmond CVs, not
PLUMED; no PLUMED file is supplied.
