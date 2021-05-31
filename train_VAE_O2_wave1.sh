#!/bin/bash 
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH -p gpu_quad
#SBATCH -t 96:00:00
#SBATCH --mem=100G
#SBATCH --output=./slurm_stdout/slurm-%j.out
#SBATCH --error=./slurm_stdout/slurm-%j.err
#SBATCH --job-name="EVE"
#SBATCH --array=34

module load gcc/6.2.0
module load cuda/10.1

conda env update --file /home/pn73/EVE/protein_env.yml
source activate protein_env

export MSA_data_folder=/n/groups/marks/projects/marks_lab_and_oatml/DRP_part_2/MSA_files/MSAs_0B1P
export MSA_list='/home/pn73/EVE/data/mappings/mapping_wave1_part1.csv'
export MSA_weights_location=/n/groups/marks/projects/marks_lab_and_oatml/EVE_models/MSA_seq_weights_wave1_Sep21
export VAE_checkpoint_location=/n/groups/marks/projects/marks_lab_and_oatml/EVE_models/all_trained_models_wave1_Sep21
export model_name_suffix='Sep21'
export model_parameters_location='/home/pn73/EVE/EVE/default_model_params.json'
export training_logs_location=/home/pn73/EVE/logs/0B1P

srun \
    python train_VAE.py \
        --MSA_data_folder ${MSA_data_folder} \
        --MSA_list ${MSA_list} \
        --protein_index $SLURM_ARRAY_TASK_ID \
        --MSA_weights_location ${MSA_weights_location} \
        --VAE_checkpoint_location ${VAE_checkpoint_location} \
        --model_name_suffix ${model_name_suffix} \
        --model_parameters_location ${model_parameters_location} \
        --training_logs_location ${training_logs_location} 