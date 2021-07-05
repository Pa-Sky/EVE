import os,sys
import json
import argparse
import pandas as pd
import torch

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Spearman')
    parser.add_argument('--DMS_list', type=str, help='List of DMS files for which to compute spearman correlation')
    parser.add_argument('--DMS_data_location', type=str, help='Location of DMS data')
    parser.add_argument('--evol_indices_location', type=str, help='Location of evol indices')
    parser.add_argument('--evol_indices_suffix', type=str, help='Suffix of evol indices files')
    parser.add_argument('--output_spearman_location', type=str, help='Output location of computed spearman values')
    parser.add_argument('--output_spearman_filename_suffix', default='', type=str, help='(Optional) Suffix to be added to output filename')
    args = parser.parse_args()

    mapping_file = pd.read_csv(args.DMS_list, low_memory=False)

    for idx,pid in enumerate(mapping_file['pid']):
        DMS_file = pd.read_csv(args.DMS_data_location+os.sep+mapping_file['DMS_filename'][idx], low_memory=False)
        DMS_measurement = mapping_file['measurement'][idx]
        DMS_directionality = mapping_file['directionality'][idx]
        evol_indices = pd.read_csv(args.evol_indices_location+os.sep+pid+args.evol_indices_suffix+'.csv', low_memory=False)
        spearman_df = pd.merge(DMS_file,evol_indices, how='inner',left_on='mutant',right_on='mutations')
        spearman, pval = spearmanr(DMS_directionality * spearman_df['measurement'], spearman_df['evol_indices'],nan_policy='raise')
        print(spearman)