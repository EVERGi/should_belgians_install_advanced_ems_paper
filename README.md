
# Should Belgians Install Advanced EMS in Their Dwellings?

This repository contains the code, data and results to reproduce the results and plots of the paper.

It it is currently not fully working but will be updated in the coming weeks.

Don't hesitate to open an issue if you encounter any problems.

## Installation Instructions

> **Note:** Installation is supported only via **conda** because the `treec` submodule requires the `pygmo` package, which is unavailable through `pip`.

> **Important:** Python 3.14 is not supported, as the `gurobi` package does not yet provide compatibility with this version.

First clone the repository with its submodules:

```bash
git clone --recurse-submodules https://github.com/EVERGi/should_belgians_install_advanced_ems_paper.git
```

To create a new conda environment and install all required packages, run the following commands in your terminal:

```bash
conda create --name belgian_dwellings python=3.13
conda activate belgian_dwellings
conda install setuptools python-graphviz
conda install -c gurobi gurobi
conda install --file requirements.txt --file submodules/treec/requirements.txt --file submodules/energy-system-simulation/requirements.txt 
pip install --no-build-isolation --no-deps . submodules/treec/ submodules/energy-system-simulation/
```

To execute the model predictive control examples, you will need a Gurobi license. Instructions for obtaining an academic license can be found [here](https://support.gurobi.com/hc/en-us/articles/14799677517585-Getting-Started-with-Gurobi-Optimizer).

Next you will need to download energyplus version 24.1 to run the simulations. Download the tar.gz or .zip file for your appropriate OS and CPU architecture from [here](https://github.com/NatLabRockies/EnergyPlus/releases/tag/v24.1.0) and extract the file in the root folder of the cloned repository.

Once extracted, rename the folder to `EnergyPlus-24.1.0`.

Then download the results, data and trained decision trees from the following Zenod record:
https://zenodo.org/records/18417982

> ⚠️ **Warning:** When extracted, the total size of the three folders is approximately 70 GB.


On Windows, use 7-zip or a similar tool to extract the downloaded .tar.gz files.

To download and extract the contents automatically, run the following commands in the root folder of the cloned repository:
```bash
curl "https://zenodo.org/records/18417982/files/data.tar.gz?download=1" --output data.tar.gz
curl "https://zenodo.org/records/18417982/files/treec_train_500.tar.gz?download=1" --output treec_train_500.tar.gz
curl "https://zenodo.org/records/18417982/files/results.tar.gz?download=1" --output results.tar.gz
tar -xvf data.tar.gz
tar -xvf treec_train_500.tar.gz
tar -xvf results.tar.gz
rm data.tar.gz treec_train_500.tar.gz results.tar.gz
```
