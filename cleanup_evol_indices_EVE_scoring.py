import numpy as np
import pandas as pd
import os


mapping_wave1_location = './data/mappings/mapping_wave1.csv'
mapping_wave2_location = './data/mappings/mapping_wave2.csv'

evol_indices_all_wave1 = '/n/groups/marks/projects/marks_lab_and_oatml/EVE_models/evol_indices_wave1_Sep21/evol_indices_wave1_all_singles_20k_samples'
evol_indices_BPU_wave1 = '/n/groups/marks/projects/marks_lab_and_oatml/EVE_models/evol_indices_wave1_Sep21/evol_indices_wave1_BPU_200k_samples'

evol_indices_all_wave2 = '/n/groups/marks/projects/marks_lab_and_oatml/EVE_models/evol_indices_wave2_Mar10/evol_indices_wave2_all_singles_20k_samples'
evol_indices_BPU_wave2 = '/n/groups/marks/projects/marks_lab_and_oatml/EVE_models/evol_indices_wave2_Mar10/evol_indices_wave2_BPU_200k_samples'

all_singles_clean_folder = '/n/groups/marks/projects/marks_lab_and_oatml/EVE_models/evol_indices_all_singles_final_modeling_May23'
BPU_clean_folder = '/n/groups/marks/projects/marks_lab_and_oatml/EVE_models/evol_indices_BPU_final_modeling_May23'

mapping_wave1 = pd.read_csv(mapping_wave1_location,low_memory=False)
mapping_wave2 = pd.read_csv(mapping_wave2_location,low_memory=False)

for pid in mapping_wave1['protein_name']:
    try:
        current = evol_indices_all_wave1 + os.sep + pid + '_20000_samples.csv'
        target = all_singles_clean_folder + os.sep + pid + '_20000_samples_May23.csv'
        os.system('cp '+current+' '+target)
    except:
        print("No evol indices All singles wave 1 for: "+pid)
    
    try:
        current = evol_indices_BPU_wave1 + os.sep + pid + '_200000_samplesMay15.csv'
        target = BPU_clean_folder + os.sep + pid + '_200000_samples_May23.csv'
        os.system('cp '+current+' '+target)
    except:
        print("No evol indices BPU wave 1 for: "+pid)

for pid in mapping_wave2['protein_name']:
    try:
        current = evol_indices_all_wave2 + os.sep + pid + '_20000_samples.csv'
        target = all_singles_clean_folder + os.sep + pid + '_20000_samples_May23.csv'
        os.system('cp '+current+' '+target)
    except:
        print("No evol indices All singles wave 2 for: "+pid)
    
    try:
        current = evol_indices_BPU_wave2 + os.sep + pid + '_200000_samples_200000_samples.csv'
        target = BPU_clean_folder + os.sep + pid + '_200000_samples_May23.csv'
        os.system('cp '+current+' '+target)
    except:
        print("No evol indices BPU wave 2 for: "+pid)