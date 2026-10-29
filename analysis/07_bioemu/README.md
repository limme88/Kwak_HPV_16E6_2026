# BioEmu — V12 Figure 7 and Table S1
The original generation, Foldseek clustering and H-Packer reconstruction commands have not been recovered. No replacement generation script has been invented. Figure 7 structure rendering and schematic assembly scripts are also missing.

The numerical analysis of the final 72 supplied conformers is included in folder 02:

```bash
python analysis/02_structural_observables/compute_common_metrics.py --output-dir results/common_metrics
```

Download `hpacker-openmm_sidechain_rec.xtc` and `clustered_samples.xtc` to `trajectories/bioemu/`; their corresponding topologies are in `structures/`. See folder 02 for all other required inputs and order. `data/common_metrics/bioemu_frame_metrics.csv` contains the saved per-conformer observables.

External tools: BioEmu https://github.com/microsoft/bioemu ; Foldseek https://github.com/steineggerlab/foldseek . Exact generation/model/checkpoint, Foldseek and H-Packer versions are not established by the recovered code. The H-Packer reconstruction is a model-dependent input; it is not a validated zinc-coordination model. The 158-residue BioEmu input differs from the 151-residue MD sequence; the existing suffix alignment is retained.
