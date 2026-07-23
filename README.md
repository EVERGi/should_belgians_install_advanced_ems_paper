
# Should Belgians Install an Advanced Energy Management System in Their Electrified Dwelling?

This repository contains the code, data and results to reproduce the results and plots of the paper:

> **Should Belgians Install an Advanced Energy Management System in Their Electrified Dwelling?**
> 📄 **https://doi.org/10.1145/3744256.3812575**

The accepted version, including the appendix, is also in this repository as
[`article_with_appendix.pdf`](article_with_appendix.pdf).

If you use this code, data or the results, please cite the paper above.

Don't hesitate to open an issue if you encounter any problems.

## Requirements

- **conda**, and **Python 3.13** (conda-only because `treec` needs `pygmo`; 3.14 isn't supported by `gurobi` yet).
- A **Gurobi license** (free for academics) for the MPC simulations.
- **EnergyPlus 24.1**, needed by every EMS (RBC, TreeC and MPC).
- ~**70 GB** of disk for the full Zenodo bundle, much less to just try a few houses.

## Installation

**1. Clone with submodules** (or `git submodule update --init --recursive` if you already cloned):

```bash
git clone --recurse-submodules https://github.com/EVERGi/should_belgians_install_advanced_ems_paper.git
cd should_belgians_install_advanced_ems_paper
```

**2. Create the environment:**

```bash
conda create --name belgian_dwellings python=3.13
conda activate belgian_dwellings
conda install setuptools python-graphviz
conda install -c gurobi gurobi
conda install --file requirements.txt --file submodules/treec/requirements.txt --file submodules/energy-system-simulation/requirements.txt
pip install --no-build-isolation --no-deps . submodules/treec/ submodules/energy-system-simulation/
```

Check it worked with `python -c "import pygmo, gurobipy, simugrid, treec, belgian_dwellings; print('OK')"`.
Re-run the `pip install` line whenever the code under `submodules/` changes, to sync the installed copies.

**3. Gurobi license.** Academic license instructions
[here](https://support.gurobi.com/hc/en-us/articles/14799677517585-Getting-Started-with-Gurobi-Optimizer).
Without one, `plots` and the RBC/TreeC runs still work, but any `MPC_*` EMS will fail.

**4. EnergyPlus 24.1.** From [the v24.1.0 release](https://github.com/NatLabRockies/EnergyPlus/releases/tag/v24.1.0),
download the **archive** for your system — not the `.exe`/`.dmg`/`.run`/`.sh` installers, which install
system-wide where the code won't find it. All names start with `EnergyPlus-24.1.0-9d7789a3ac-`:

| Your system | Ends with |
| --- | --- |
| Linux, x86_64 | `Linux-Ubuntu22.04-x86_64.tar.gz` |
| Linux, arm64 | `Linux-Ubuntu22.04-arm64.tar.gz` |
| macOS, Apple Silicon | `Darwin-macOS12.1-arm64.tar.gz` |
| macOS, Intel | `Darwin-macOS12.1-x86_64.tar.gz` |
| Windows, x86_64 | `Windows-x86_64.zip` |

Extract it in the repository root and rename the extracted folder to exactly `EnergyPlus-24.1.0`, so that
`EnergyPlus-24.1.0/pyenergyplus/` exists:

```bash
tar -xzf EnergyPlus-24.1.0-9d7789a3ac-Linux-Ubuntu22.04-x86_64.tar.gz
mv EnergyPlus-24.1.0-9d7789a3ac-Linux-Ubuntu22.04-x86_64 EnergyPlus-24.1.0
```

(On Windows, extract the `.zip` with Explorer or 7-zip and rename the folder the same way.)

**5. Download data, results and trained models** from [this Zenodo record](https://zenodo.org/records/18417982)
(⚠️ ~70 GB extracted). From the repository root:

```bash
curl "https://zenodo.org/records/18417982/files/data.tar.gz?download=1" --output data.tar.gz
curl "https://zenodo.org/records/18417982/files/treec_train_500.tar.gz?download=1" --output treec_train_500.tar.gz
curl "https://zenodo.org/records/18417982/files/results.tar.gz?download=1" --output results.tar.gz
tar -xvf data.tar.gz
tar -xvf treec_train_500.tar.gz
tar -xvf results.tar.gz
rm data.tar.gz treec_train_500.tar.gz results.tar.gz
```

This gives `data/` (500 house configurations + shared inputs), `treec_train_500/` (the paper's trained TreeC
trees) and `results/` (the paper's raw KPI results). To just try things out, `data.tar.gz` alone is enough to
run simulations and train new models on a few houses.

## Reproducing the paper's results

`belgian_dwellings.cmd_exec` is the single entry point. Add `--help` to any subcommand for all options.

### Regenerate the figures

Reads `results/belgium_usefull_500.csv` and writes every paper figure to `results/figures/`:

```bash
python -m belgian_dwellings.cmd_exec plots
```

### Train the models

TreeC trees are trained per house with an evolutionary optimizer, defaulting to the paper's parameters
(300 generations, population 200, 5 trainings per house). Houses that already have 5 trees are skipped, so
re-running is cheap.

```bash
python -m belgian_dwellings.cmd_exec train treec --houses 0-9
python -m belgian_dwellings.cmd_exec train treec --houses 0-9 --workers 4   # several houses at once
```

The MPC's forecasting model (thermal calibration + EV predictor) is not a saved model: it is refit from a
year-long EnergyPlus run just before each house's MPC simulation. This reproduces that fit standalone and
writes a summary CSV, for inspecting it independently of a full MPC run:

```bash
python -m belgian_dwellings.cmd_exec train mpc-forecast --houses 0-9
```

### Run the EMS simulations

Runs RBC, TreeC and MPC (perfect and realistic forecast), plus the "no enforcement" variants behind the
SOC/charge-completion plots, into a results folder of your choice so the originals stay intact:

```bash
python -m belgian_dwellings.cmd_exec run --houses 0-9 --results-dir results_rerun
python -m belgian_dwellings.cmd_exec run --houses 0-9 --target main --results-dir results_rerun  # skip SOC runs
python -m belgian_dwellings.cmd_exec run --results-dir results_rerun                             # all 500 houses
```

To plot your own results, copy `results_rerun/belgium_usefull_500.csv` into `results/`, or adapt
`belgian_dwellings/plots/result_plots.py` to read another folder.

### Full-scale reproduction on HPC

Running all 500 houses is compute-heavy (EnergyPlus + Gurobi + evolutionary search per house). The paper ran
it as SLURM array jobs, one task per house — adapt these to your own cluster:

- `belgian_dwellings/training/hpc_train.sh` (+ `training/hpc_run.py`) for TreeC training
- `belgian_dwellings/simulation/hpc_get_results_mpc*.sh` (+ `simulation/hpc_get_results_mpc.py`) for MPC results
