#!/bin/bash
#SBATCH --job-name=test_train_100
#SBATCH --time=54:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --ntasks-per-node=1
#SBATCH --partition=zen4,zen5_mpi
#SBATCH --array=0-499

cd /data/brussel/102/vsc10250/ems_belgium_usefull

module load Python/3.11.3-GCCcore-12.3.0
module load matplotlib/3.7.2-gfbf-2023a
module load pagmo/2.19.1-gfbf-2023a
module load EnergyPlus/25.1.0-foss-2023a

source venv/bin/activate
python src/training/hpc_run.py $SLURM_ARRAY_TASK_ID