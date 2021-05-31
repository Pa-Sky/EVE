#!/bin/bash 
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH -t 24:00:00
#SBATCH --mem=90G
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

export GMM_parameter_location='./results/GMM_parameters/May23_final_parameters'
export GMM_parameter_filename_suffix='May23'
export protein_GMM_weight=0.3
export plot_location='./results'

python train_GMM_and_compute_EVE_scores.py \
    --input_evol_indices_location ${input_evol_indices_location} \
    --input_evol_indices_filename_suffix ${input_evol_indices_filename_suffix} \
    --protein_list ${protein_list} \
    --GMM_parameter_location ${GMM_parameter_location} \
    --GMM_parameter_filename_suffix ${GMM_parameter_filename_suffix} \
    --protein_GMM_weight ${protein_GMM_weight} \
    --plot_histograms \
    --plot_location ${plot_location} \
    --verbose

    ##--compute_EVE_scores \
    ##--output_eve_scores_location ${output_eve_scores_location} \
    ##--output_eve_scores_filename_suffix ${output_eve_scores_filename_suffix} \
    ##--load_GMM_models \
    ##--recompute_uncertainty_threshold \
    ##--plot_scores_vs_labels \
    #--labels_file_location ${labels_file_location} \
    ##--default_uncertainty_threshold_file_location ${default_uncertainty_threshold_file_location} \
    #export default_uncertainty_threshold_file_location='./utils/default_uncertainty_threshold.json'
    