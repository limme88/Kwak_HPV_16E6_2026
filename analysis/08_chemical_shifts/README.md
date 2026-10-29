# Chemical-shift and helical-propensity comparisons

`make_figure2.py` reproduces the paired helical-propensity and C-alpha residual plots used for Figure 2 from the saved final processed inputs. `make_table1.py` recomputes pooled RMSD and Pearson correlation for Table 1. Run from the repository root:

```bash
python analysis/08_chemical_shifts/make_figure2.py
python analysis/08_chemical_shifts/make_table1.py
```

Outputs go to `results/chemical_shifts/figure2/` and `results/chemical_shifts/table1/`. The small configuration blocks at the top select inputs and outputs. The plot preserves Helvetica styling; install that font for exact typography. Scientific values do not depend on the font. Figure panel letters and manuscript composition are not added by this script.

The final helicity input includes the experimental substitution sites restored on September 9, 2026. Both construct values at residue 80 are retained; R146 and L151 helicity gaps remain. The three original `plot_helical.py` copies were byte-identical; the selected copy belonged to the mutation-inclusive workflow. This script produces an intermediate two-domain layout, not the final Figure 2 layout:

```bash
python analysis/08_chemical_shifts/plot_helical.py --blocking trajectories/chemical_shifts/blocking_results/blocking_combined.csv --out results/chemical_shifts/intermediate_helicity
```

`delta2d_workflow.py` preserves the upstream experimental preparation, Delta2D, DSSP and duration-weighted comparison calculation. Its `plot` command is an older diagnostic layout and is not the final manuscript panel. `block_helical.py` preserves the blocking calculation that determined whether error bars could be reported. These upstream stages require the native archive and metadata from Zenodo; the final panel/table commands above do not. Populate `trajectories/chemical_shifts/inputs/` with `17967.str`, `experimental_all_rows.csv`, `apo16e6_151aa.fasta`, `LARGE_FILES_ON_HPC.json`, `dssp_provenance.json`, `dssp_combined.csv`, and `summary_provenance.json`. The final three files must describe the exact framewise DSSP arrays used for blocking. Preserve the archive's `exports/` layout and frame manifest. No Zenodo record is supplied in these scripts.

Run upstream stages in this order, with the separately installed Delta2D executable/database at the paths shown (or supply your installation paths):

```bash
mkdir -p results/chemical_shifts
python analysis/08_chemical_shifts/delta2d_workflow.py prepare --out results/chemical_shifts/prepared
python analysis/08_chemical_shifts/delta2d_workflow.py nmr --prepared results/chemical_shifts/prepared --binary external/delta2d/delta2d --database external/delta2d --out results/chemical_shifts/nmr
python analysis/08_chemical_shifts/delta2d_workflow.py dssp --native-root trajectories/chemical_shifts/native --out results/chemical_shifts/dssp
python analysis/08_chemical_shifts/delta2d_workflow.py summarize --dssp-dir results/chemical_shifts/dssp --out results/chemical_shifts/summary
python analysis/08_chemical_shifts/block_helical.py --dssp-dir results/chemical_shifts/dssp --out results/chemical_shifts/blocking
```

These commands require fresh output directories where the original code enforces that requirement. The final figure consumes the saved mutation-inclusive helicity table; the diagnostic upstream plot excludes substitution sites. Those policies are deliberately not silently interchanged.

## External tools and recorded environments

- SPARTA+ **2.90 Rev 2017.143.12.12**, recorded in the original predictor environment: https://spin.niddk.nih.gov/bax-apps/software/SPARTA%2B/ . Vendor software is not included. The production back-calculation driver requires excluded pilot/cache-control files and archived validation inputs; it has not been copied or rewritten. This repository therefore reproduces the figure and comparison from processed predictions, but does not yet supply a self-contained raw-frame SPARTA+ rerun.
- Delta2D **2.1.0**, source/database commit `3c390df3a5a934dd9fc9d49932bafe8821334812`: https://github.com/carlocamilloni/d2D/tree/3c390df3a5a934dd9fc9d49932bafe8821334812 . The neutral-pH option and built-in neighbor averaging are retained.
- pyblock commit `a293b5b75dfa862def45a077824aa8df08bcbf46`: https://github.com/jsspencer/pyblock/tree/a293b5b75dfa862def45a077824aa8df08bcbf46 . Install this separately; the original vendor import now imports `pyblock.blocking` without changing either blocking function.
- Recorded native DSSP environment: MDTraj **1.11.1**, NumPy **2.3.5**. Recorded SPARTA+ comparison environment: Python **3.9.23**, NumPy **1.26.4**, pandas **2.3.1**, SciPy **1.13.1**, Matplotlib **3.9.4**, MDAnalysis **2.7.0**. These were separate environments, not one reproducible combined lockfile.

## Interpretation and unresolved discrepancies

The original source identifies the 0.92 ppm lines as a predictor RMS benchmark on eleven proteins outside the training database; the V12 caption calls them a training-dataset mean error. That wording discrepancy is not corrected here. The plots include substitution sites whereas Table 1 excludes them. No combined blocking SEM passed the original screening for the displayed mean-only helicity figure. July retains documented Zn-site parameter differences. The all-run mean is a descriptive mixture.

The preserved pooled calculation gives C-alpha RMSD 1.0091350205563894 ppm, which rounds to 1.01; V12 reports 1.00. Other Table 1 entries match the saved calculation at the displayed precision. No numeric value has been changed to match the manuscript.
