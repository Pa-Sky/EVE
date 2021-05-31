#!/bin/bash 
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu_quad
#SBATCH -t 96:00:00
#SBATCH --mem=135G
#SBATCH --output=./slurm_stdout/slurm-%j.out
#SBATCH --error=./slurm_stdout/slurm-%j.err
#SBATCH --job-name="EVE"

module load gcc/6.2.0
module load cuda/10.1

conda env update --file /home/pn73/EVE/protein_env.yml
source activate protein_env

export input_evol_indices_location='/n/groups/marks/projects/marks_lab_and_oatml/EVE_models/evol_indices_all_singles_final_modeling_May23'
export input_evol_indices_filename_suffix='_20000_samples_May23'
export protein_list='./data/mappings/mapping_all.csv'
export output_eve_scores_location='./results/EVE_scores/All_singles'
export output_eve_scores_filename_suffix='May23'

export GMM_parameter_location='./results/GMM_parameters/May23_final_parameters'
export GMM_parameter_filename_suffix='May23'
export protein_GMM_weight=0.3

export default_uncertainty_threshold_file_location='./utils/final_uncertainty_threshold_May23.json'

python train_GMM_and_compute_EVE_scores.py \
    --input_evol_indices_location ${input_evol_indices_location} \
    --input_evol_indices_filename_suffix ${input_evol_indices_filename_suffix} \
    --protein_list ${protein_list} \
    --compute_EVE_scores \
    --output_eve_scores_location ${output_eve_scores_location} \
    --output_eve_scores_filename_suffix ${output_eve_scores_filename_suffix} \
    --load_GMM_models \
    --GMM_parameter_location ${GMM_parameter_location} \
    --GMM_parameter_filename_suffix ${GMM_parameter_filename_suffix} \
    --protein_GMM_weight ${protein_GMM_weight} \
    --default_uncertainty_threshold_file_location ${default_uncertainty_threshold_file_location} \
    --verbose