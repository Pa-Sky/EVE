import numpy as np
import pandas as pd
import os


mapping_all_location = './data/mappings/mapping_all_mutations.csv'

labels_folder = '/n/groups/marks/projects/marks_lab_and_oatml/DRP_part_2/variant_files/variant_files_CV_UKB_gnomAD_complete_May21'

mapping_all = pd.read_csv(mapping_all_location,low_memory=False)

mutations_folder = './data/mutations/BPU_w2'

for pid in mapping_all['protein_name']:
    try:
        protein_labels_df = pd.read_csv(labels_folder+os.sep+pid+'.csv',low_memory=False)
        protein_labels_df = protein_labels_df[['mutant','Starry_Coarse_Grained_Clin_Sig']]
        protein_labels_df = protein_labels_df.dropna(axis=0,subset=['Starry_Coarse_Grained_Clin_Sig'])
        protein_labels_df = protein_labels_df.rename({'mutant': 'mutations'}, axis='columns')
        protein_labels_df['ClinVar_labels'] = protein_labels_df['Starry_Coarse_Grained_Clin_Sig'].apply(lambda x: 1 if x=='pathogenic' else (0 if x=='benign' else 0.5))
        protein_labels_df['protein_name'] = [pid] * len(protein_labels_df)
        try:
            all_labels_df = pd.concat([all_labels_df,protein_labels_df], axis=0)
        except:
            all_labels_df = protein_labels_df
        
        protein_labels_df['mutations'].to_csv(mutations_folder+os.sep+pid+'.csv', index=False)
    except:
        print("No labels for: "+pid)

all_labels_df = all_labels_df[['protein_name','mutations','Starry_Coarse_Grained_Clin_Sig','ClinVar_labels']]
all_labels_df.to_csv('./data/labels/All_3k_proteins_ClinVar_labels_May21.csv', index=False)

print("Stats labels all 3k proteins: ")
print(all_labels_df.describe())
#count    1.304527e+06
#mean     5.002445e-01
#std      8.946078e-02
#min      0.000000e+00
#25%      5.000000e-01
#50%      5.000000e-01
#75%      5.000000e-01
#max      1.000000e+00

print(all_labels_df.groupby('ClinVar_labels').count())
#ClinVar_labels                                                         
#0.0                    20,562
#0.5                 1,262,765
#1.0                    21,200
#All:                1,304,527