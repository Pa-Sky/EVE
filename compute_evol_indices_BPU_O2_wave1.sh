#!/bin/bash 
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1 
#SBATCH -p gpu_quad
#SBATCH -t 96:00:00
#SBATCH --mem=90G
#SBATCH --output=./slurm_stdout/BPU_wave1/slurm-%j.out
#SBATCH --error=./slurm_stdout/BPU_wave1/slurm-%j.err
#SBATCH --job-name="EVE"
#SBATCH --array=0-10

module load gcc/6.2.0
module load cuda/10.1

conda env update --file /home/pn73/EVE/protein_env.yml
source activate protein_env

export MSA_data_folder=/n/groups/marks/projects/marks_lab_and_oatml/DRP_part_2/MSA_files/MSAs_0B1P
export MSA_list='/home/pn73/EVE/data/mappings/mapping_additional_mutations_wave1.csv'
export MSA_weights_location=/n/groups/marks/projects/marks_lab_and_oatml/EVE_models/MSA_seq_weights_wave1_Sep21
export VAE_checkpoint_location=/n/groups/marks/projects/marks_lab_and_oatml/EVE_models/all_trained_models_wave1_Sep21
export model_name_suffix='Sep21'
export model_parameters_location='/home/pn73/EVE/EVE/default_model_params.json'

export computation_mode='input_mutations_list'
export mutations_location=/home/pn73/EVE/data/mutations/BPU
export output_evol_indices_location=/home/pn73/EVE/results/evol_indices/BPU_200k_wave1
export output_evol_indices_filename_suffix='May15'
export num_samples_compute_evol_indices=200000
export batch_size=4096

srun \
    python compute_evol_indices.py \
        --MSA_data_folder ${MSA_data_folder} \
        --MSA_list ${MSA_list} \
        --protein_index $SLURM_ARRAY_TASK_ID \
        --MSA_weights_location ${MSA_weights_location} \
        --VAE_checkpoint_location ${VAE_checkpoint_location} \
        --model_name_suffix ${model_name_suffix} \
        --model_parameters_location ${model_parameters_location} \
        --computation_mode ${computation_mode} \
        --mutations_location ${mutations_location} \
        --output_evol_indices_location ${output_evol_indices_location} \
        --output_evol_indices_filename_suffix ${output_evol_indices_filename_suffix} \
        --num_samples_compute_evol_indices ${num_samples_compute_evol_indices} \
        --batch_size ${batch_size}