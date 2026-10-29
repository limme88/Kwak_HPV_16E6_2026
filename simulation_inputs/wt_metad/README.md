# Original WT-MetaD inputs

The five matched runs dated 8.10.25–8.14.25 support V12 Figure 6 and Figure S5.
For each run this folder contains the original setup `.cms`, production `.cfg`,
multistage `.msj` and resolved production `-out.cfg`. The five resolved CFG files
were checked against the hashes in the saved final-exemplar extraction manifest.
Input CMS files are below 10 MB. Checkpoints, trajectories and output CMS files
belong in the associated Zenodo archive.

Only historical filesystem paths in metadata and launch-command comments were
replaced. Atom indices, CV definitions, seeds, temperatures, durations, and all
other simulation settings are unchanged. These are native Desmond inputs;
there is no PLUMED input file.

## Run order

Use a separately installed licensed Schrödinger/Desmond environment and set
`SCHRODINGER` to its installation directory. The `.msj` runs relaxation before
production. The original launch commands are retained in each `.msj` comment.
From the repository root, the following commands prepare separate run folders
and reproduce the recorded multisim invocation without a Maestro project file:

```bash
for day in 10 11 12 13 14; do
  stem="desmond_WT_MetaD_4xr8_chainH_E6_1us_d49r102_q107chi1_8.${day}.25"
  mkdir -p "trajectories/wt_metad/${stem}"
  cp "simulation_inputs/wt_metad/${stem}.cms" \
     "simulation_inputs/wt_metad/${stem}.cfg" \
     "simulation_inputs/wt_metad/${stem}.msj" \
     "trajectories/wt_metad/${stem}/"
  (
    cd "trajectories/wt_metad/${stem}"
    "${SCHRODINGER}/utilities/multisim" \
      -JOBNAME "${stem}" -HOST localhost -maxjob 1 -cpu 1 \
      -m "${stem}.msj" -c "${stem}.cfg" \
      -description Metadynamics "${stem}.cms" -mode umbrella \
      -o "${stem}-out.cms" -lic DESMOND_GPGPU:16
  )
done
```

These commands have not been used to launch new simulations during repository
preparation. Availability of the appropriate licensed backend and GPU resources
is installation-dependent. Original production software version is not established
by the saved extraction environment: extraction/reweighting used Schrödinger
2026-1, whereas the production jobs are dated 2025. Obtain Schrödinger/Desmond
separately from <https://www.schrodinger.com/>. No vendor source is redistributed.

After simulations or archive download, follow `analysis/06_wt_metad/README.md`.
The copied resolved CFG files document actual production settings and are also
required when analyzing the original archived runs.
