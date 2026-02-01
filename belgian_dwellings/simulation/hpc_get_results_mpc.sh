#!/bin/bash
#SBATCH --job-name=mpc_results_realistic
#SBATCH --time=36:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --partition=zen4,zen5_mpi
#SBATCH --array=0-499

cd /data/brussel/102/vsc10250/ems_belgium_usefull
module load Python/3.11.3-GCCcore-12.3.0
module load matplotlib/3.7.2-gfbf-2023a
module load EnergyPlus/25.1.0-foss-2023a
module load Gurobi/11.0.2-GCCcore-12.3.0

source venv/bin/activate
python src/simulation/hpc_get_results_mpc.py $SLURM_ARRAY_TASK_ID realistic