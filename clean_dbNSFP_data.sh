#!/bin/bash 
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1 
#SBATCH -p gpu_quad
#SBATCH -t 24:00:00
#SBATCH --mem=90G
#SBATCH --output=./slurm_stdout/BPU/slurm-%j.out
#SBATCH --error=./slurm_stdout/BPU/slurm-%j.err
#SBATCH --job-name="EVE"

module load gcc/6.2.0
module load cuda/10.1

conda env update --file /home/pn73/EVE/protein_env.yml
source activate protein_env

srun python clean_dbNSFP_data.py