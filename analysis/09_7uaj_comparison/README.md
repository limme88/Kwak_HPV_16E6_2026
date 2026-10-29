# 7UAJ comparison — V12 Figure 7 discussion and Table S1
`compute_7uaj_reference_coordinates.py` computes the D49–R102 mass-weighted COM distance and Q107 chi1 with the original 378-residue fusion-numbering offset; chain D is the default. It requires only Python standard-library modules.

```bash
mkdir -p results/7uaj
python analysis/09_7uaj_comparison/compute_7uaj_reference_coordinates.py structures/7UAJ.pdb --selected-chain D --output results/7uaj/reference_coordinates.json
```

Table S1 uses Cα distance and Cα130–Cα151 tail extension instead. Reproduce those with `analysis/02_structural_observables/compute_common_metrics.py` after its required inputs are available. The alternative `compute_7uaj_metrics.py` used a revised Cα143–Cα151 definition, inconsistent with final Table S1; it is deliberately not included. Missing tail residues remain unavailable.
