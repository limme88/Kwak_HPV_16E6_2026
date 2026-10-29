# Unbiased Desmond inputs

The dated folders preserve the original MSJ relaxation protocols and initial CFG production settings for the four runs. The first three also include the resolved final configuration from their continuation to 10 microseconds. The native-frame manifest identifies this ensemble:

| Dated run | Initial production | Continuation represented | Total used |
|---|---:|---:|---:|
| native_20250414 | 5 microseconds | 5–10 microseconds | 10 microseconds |
| native_20250505 | 5 microseconds | 5–10 microseconds | 10 microseconds |
| native_20250710 | 5 microseconds | 5–10 microseconds | 10 microseconds |
| native_20250520 | 5 microseconds | None | 5 microseconds |

All initial CFG files contain `time = 5000000.0` in ps. The resolved `10us` files retain that frontend value and set backend `last_time = 10000000.0`. They record the continuation settings; they are not interchangeable with an initial CFG for a fresh simulation. A separate 10-microsecond May20 configuration was found but excluded because that continuation is not part of the final 35-microsecond native-frame manifest. Dated identities are used because the source manifest records conflicting numeric replica conventions.

## Initial-stage commands

Install a licensed Schrödinger/Desmond distribution and set `SCHRODINGER` to its installation root. The original version is not encoded in these copied input files; it has not been guessed. Product information: https://www.schrodinger.com/platform/products/desmond/ . No third-party software is included.

April14 and July10 original CMS files are included next to their CFG/MSJ. Copy those two into the archive layout below, and obtain the larger May5 and May20 originals from Zenodo. Place each dated system CMS under `trajectories/simulation_inputs/unbiased_md/<dated-run>/` with its original filename. From the repository root, the following commands reproduce the original initial relaxation/5-microsecond production invocation, with a separate results directory and without the original GUI project path. These are documented commands, not commands executed while preparing this repository.

```bash
mkdir -p results/unbiased_md
for run in native_20250414 native_20250710; do
    mkdir -p "trajectories/simulation_inputs/unbiased_md/$run"
    cp simulation_inputs/unbiased_md/"$run"/*.cms "trajectories/simulation_inputs/unbiased_md/$run/"
done
while read -r run job; do
    mkdir -p "results/unbiased_md/$run"
    cp "simulation_inputs/unbiased_md/$run/$job.cfg" "simulation_inputs/unbiased_md/$run/$job.msj" "results/unbiased_md/$run/"
    (
        cd "results/unbiased_md/$run" || exit 1
        "$SCHRODINGER/utilities/multisim" -JOBNAME "$job" -HOST localhost -maxjob 1 -cpu 1 -m "$job.msj" -c "$job.cfg" -description "Molecular Dynamics" "../../../trajectories/simulation_inputs/unbiased_md/$run/$job.cms" -mode umbrella -o "$job-out.cms" -lic DESMOND_GPGPU:16
    )
done <<'RUNS'
native_20250414 desmond_md_job_4xr8_chainH_E6_5us_4.14.25
native_20250505 desmond_md_job_4xr8_chainH_e6_5us_5.5.25
native_20250710 desmond_md_job_4xr8_chainH_E6_5us_7.10.25
native_20250520 desmond_md_job_4xr8_chainH_E6_5us_5.20.25
RUNS
```

These commands produce only the initial 5 microseconds per run. The exact original continuation command and checkpoint assets have not been recovered for publication; no invented restart command is supplied. Include those assets/instructions with the Zenodo archive before claiming full simulation reproducibility. The prepared input CMS files and initial/final checkpoints are distinct assets: an output CMS is not a substitute for the original starting system or restart checkpoint.

## Scope and preservation

The inputs support the unbiased ensemble underlying the manuscript structural observables, secondary structure, ELViM, MSM and chemical-shift comparisons, including Figure S1. All scientific settings, atom selections, random seeds, relaxation stages and integration parameters are preserved. Only four-line explanatory comments and the personal project path in a commented launcher were changed. Noncomment text was verified identical to the fetched originals.

The initial May5 and May20 CMS files are each over the repository's 10 MB limit and must be on Zenodo. The original April14 and July10 CMS inputs are below that limit and are included. Native trajectory outputs and output CMS files belong in the external archive. July's documented Zn-site parameter differences are not repaired or normalized here. The input settings have not been executed with Desmond during repository preparation.
