import torch
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, accuracy_score

#####################################################################################
experiment_name = "Jun3"
mapping_file_location = "./data/mappings/list_raw_baseline_scorefiles.txt"
raw_baseline_scores_location = '/n/groups/marks/projects/marks_lab_and_oatml/DRP_part_2/dbNSFP_single_trascript_files'
concatenated_baseline_scores_location = '/n/groups/marks/projects/marks_lab_and_oatml/EVE_models/all_dbNSFP_baseline_scores/all_dbNSFP_baseline_scores_'+experiment_name+'.csv'
#####################################################################################
mapping_file = pd.read_csv(mapping_file_location,low_memory=False)
list_protein_files = list(mapping_file.filename)
num_proteins_to_score = len(mapping_file.filename)
print("Number of proteins in dbNSFP extract: "+str(num_proteins_to_score))

mapping_baseline_score = pd.read_csv('./data/mappings/mapping_baseline_score_cleanup.csv',low_memory=False)
#####################################################################################
variables_to_keep=[
'pid',
'mutant',
'clinvar_clnsig',
'BayesDel_addAF_score',
'BayesDel_noAF_score',
'CADD_phred',
'CADD_phred_hg19',
'CADD_raw',
'CADD_raw_hg19',
'ClinPred_score',
'DANN_score',
'DEOGEN2_score',
'Eigen-PC-phred_coding',
'Eigen-PC-raw_coding',
'Eigen-phred_coding',
'Eigen-raw_coding',
'FATHMM_score',
'fathmm-MKL_coding_score',
'fathmm-XF_coding_score',
'GenoCanyon_score',
'LIST-S2_score',
'LRT_score',
'M-CAP_score',
'MetaLR_score',
'MetaSVM_score',
'MPC_score',
'MutationAssessor_score',
'MutationTaster_score',
'MutPred_score',
'MVP_score',
'Polyphen2_HDIV_score',
'Polyphen2_HVAR_score',
'PrimateAI_score',
'PROVEAN_score',
'REVEL_score',
'SIFT_score',
'SIFT4G_score',
'VEST4_score',
'BayesDel_addAF_pred',
'BayesDel_noAF_pred',
'ClinPred_pred',
'DEOGEN2_pred',
'FATHMM_pred',
'fathmm-MKL_coding_pred',
'fathmm-XF_coding_pred',
'LIST-S2_pred',
'LRT_pred',
'M-CAP_pred',
'MetaLR_pred',
'MetaSVM_pred',
'MutationAssessor_pred',
'MutationTaster_pred',
'PrimateAI_pred',
'PROVEAN_pred',
'SIFT_pred',
'SIFT4G_pred'
]

scoring_variables=[
'BayesDel_addAF_score',
'BayesDel_noAF_score',
'CADD_phred',
'CADD_phred_hg19',
'CADD_raw',
'CADD_raw_hg19',
'ClinPred_score',
'DANN_score',
'DEOGEN2_score',
'Eigen-PC-phred_coding',
'Eigen-PC-raw_coding',
'Eigen-phred_coding',
'Eigen-raw_coding',
'FATHMM_score',
'fathmm-MKL_coding_score',
'fathmm-XF_coding_score',
'GenoCanyon_score',
'LIST-S2_score',
'LRT_score',
'M-CAP_score',
'MetaLR_score',
'MetaSVM_score',
'MPC_score',
'MutationAssessor_score',
'MutationTaster_score',
'MutPred_score',
'MVP_score',
'Polyphen2_HDIV_score',
'Polyphen2_HVAR_score',
'PrimateAI_score',
'PROVEAN_score',
'REVEL_score',
'SIFT_score',
'SIFT4G_score',
'VEST4_score'
]

pred_variables_mapping_DT=[
'BayesDel_addAF_pred',
'BayesDel_noAF_pred',
'ClinPred_pred',
'DEOGEN2_pred',
'FATHMM_pred',
'LIST-S2_pred',
'M-CAP_pred',
'MetaLR_pred',
'MetaSVM_pred',
'PrimateAI_pred',
'SIFT_pred',
'SIFT4G_pred'
]

pred_variables_mapping_DN=[
'fathmm-MKL_coding_pred',
'fathmm-XF_coding_pred',
'LRT_pred',
'PROVEAN_pred'
]

pred_variables_to_threshold=[
'MVP_score',
'Polyphen2_HDIV_score',
'Polyphen2_HVAR_score'
]

with open('/n/groups/marks/projects/marks_lab_and_oatml/EVE_models/all_dbNSFP_baseline_scores/logs_line_count_dbNSFP_files.txt', "a") as log:
    log.write("protein_name,filename,line_count\n")

list_processed_scoring_files=[]
for filename in tqdm.tqdm(list_protein_files):
    try:
        protein_name='_'.join(filename.split("_")[:2])
        scoring_file = pd.read_csv(raw_baseline_scores_location+os.sep+filename, low_memory=False)
        scoring_file['pid']=[protein_name]*len(scoring_file)
        scoring_file['mutant']=scoring_file['aaref']+scoring_file['aapos'].astype(str)+scoring_file['aaalt']
        scoring_file=scoring_file[variables_to_keep]
        for score_var in scoring_variables:
            scoring_file[score_var]=pd.to_numeric(scoring_file[score_var], errors="coerce") * int(mapping_baseline_score['directionality'][mapping_baseline_score['prediction_name']==score_var].iloc[0])
        for pred_var in pred_variables_mapping_DT:
            scoring_file[pred_var]=scoring_file[pred_var].map({"D":"Pathogenic", "T":"Benign"})
        for pred_var in pred_variables_mapping_DN:    
            scoring_file[pred_var]=scoring_file[pred_var].map({"D":"Pathogenic", "N":"Benign"})
        scoring_file['MutationAssessor_pred']=scoring_file['MutationAssessor_pred'].map({"H":"Pathogenic","M":"Pathogenic", "L":"Benign", "N":"Benign"})
        scoring_file['Polyphen2_HDIV_pred']=(scoring_file['Polyphen2_HDIV_score']>0.5).map({True:"Pathogenic", False:"Benign"})
        scoring_file['Polyphen2_HVAR_pred']=(scoring_file['Polyphen2_HVAR_score']>0.5).map({True:"Pathogenic", False:"Benign"})
        scoring_file['MutationTaster_pred']=(scoring_file['MutationTaster_score']>0.5).map({True:"Pathogenic", False:"Benign"})
        scoring_file['MVP_pred']=(scoring_file['MVP_score']>0.7).map({True:"Pathogenic", False:"Benign"})
        list_processed_scoring_files.append(scoring_file)
        with open('/n/groups/marks/projects/marks_lab_and_oatml/EVE_models/all_dbNSFP_baseline_scores/logs_line_count_dbNSFP_files.txt', "a") as log:
            log.write(protein_name+","+filename+","+str(len(scoring_file))+"\n")
    except:
        print("Problem processing baseline scores for: "+str(protein_name))
        with open('/n/groups/marks/projects/marks_lab_and_oatml/EVE_models/all_dbNSFP_baseline_scores/logs_errors_dbNSFP_files.txt', "a") as log:
            log.write(filename+"\n")
all_baseline_scores = pd.concat(list_processed_scoring_files, axis=0)
all_baseline_scores.to_csv(concatenated_baseline_scores_location,index=False)
