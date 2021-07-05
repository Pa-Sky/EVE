import torch
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, accuracy_score

#####################################################################################
experiment_name = "Jun1"
mapping_file_location = "./data/mappings/mapping_all.csv"
scores_location = "./results/EVE_scores/BPU/all_EVE_scores_May23.csv"
labels_location = "./data/labels/All_3k_proteins_ClinVar_labels_May21.csv"

create_concatenation_baseline_scores = False
raw_baseline_scores_location = '/n/groups/marks/projects/marks_lab_and_oatml/DRP_part_2/dbNSFP_single_trascript_files'
concatenated_baseline_scores_location = './data/baseline_scores/all_baseline_scores_'+experiment_name+'.csv'

merged_file_eve_clinvar_baseline_location = './results/EVE_scores/BPU/all_EVE_Clinvar_baselines_BP_'+experiment_name+'.csv'

AUC_accuracy_all_location = './results/AUC_Accuracy/AUC_accuracy_all_'+experiment_name+'.csv'
AUC_accuracy_75pct_location = './results/AUC_Accuracy/AUC_accuracy_75pct_'+experiment_name+'.csv'
AUC_accuracy_all_position_level_location = './results/AUC_Accuracy/AUC_accuracy_all_position_level_'+experiment_name+'.csv'
AUC_accuracy_75pct_position_level_location = './results/AUC_Accuracy/AUC_accuracy_75pct_position_level_'+experiment_name+'.csv'
#####################################################################################

mapping_file = pd.read_csv(mapping_file_location,low_memory=False)
list_proteins = list(mapping_file.protein_name)
num_proteins_to_score = len(mapping_file.protein_name)
print("Number of proteins to score: "+str(num_proteins_to_score))

#####################################################################################
## Create concatenated file with all baseline scores
#####################################################################################
mapping_pid_filename = pd.read_csv('./data/mappings/mapping_pid_baseline-filename.csv',low_memory=False)
mapping_baseline_score = pd.read_csv('./data/mappings/mapping_baseline_score_cleanup.csv',low_memory=False)

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
#'Aloft_pred'
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


if create_concatenation_baseline_scores:
    list_processed_scoring_files=[]
    for protein_name in tqdm.tqdm(list_proteins):
        try:
            baseline_filename = mapping_pid_filename['filename'][mapping_pid_filename['pid']==protein_name].iloc[0]
            scoring_file = pd.read_csv(raw_baseline_scores_location+os.sep+baseline_filename, low_memory=False)
            scoring_file['pid']=[protein_name]*len(scoring_file)
            scoring_file['mutant']=scoring_file['aaref']+scoring_file['aapos'].astype(str)+scoring_file['aaalt']
            scoring_file=scoring_file[variables_to_keep]
            for score_var in scoring_variables:
                scoring_file[score_var]=pd.to_numeric(scoring_file[score_var], errors="coerce") * int(mapping_baseline_score['directionality'][mapping_baseline_score['prediction_name']==score_var].iloc[0])
            for pred_var in pred_variables_mapping_DT:
                scoring_file[pred_var]=scoring_file[pred_var].map({"D":"Pathogenic", "T":"Benign"})
            for pred_var in pred_variables_mapping_DN:    
                scoring_file[pred_var]=scoring_file[pred_var].map({"D":"Pathogenic", "N":"Benign"})
            #scoring_file['Aloft_pred']=scoring_file['Aloft_pred'].map({"R":"Pathogenic", "D":"Pathogenic","T":"Benign"})
            scoring_file['MutationAssessor_pred']=scoring_file['MutationAssessor_pred'].map({"H":"Pathogenic","M":"Pathogenic", "L":"Benign", "N":"Benign"})
            scoring_file['Polyphen2_HDIV_pred']=(scoring_file['Polyphen2_HDIV_score']>0.5).map({True:"Pathogenic", False:"Benign"})
            scoring_file['Polyphen2_HVAR_pred']=(scoring_file['Polyphen2_HVAR_score']>0.5).map({True:"Pathogenic", False:"Benign"})
            scoring_file['MutationTaster_pred']=(scoring_file['MutationTaster_score']<0.5).map({True:"Pathogenic", False:"Benign"})
            scoring_file['MVP_pred']=(scoring_file['MVP_score']>0.7).map({True:"Pathogenic", False:"Benign"})
            list_processed_scoring_files.append(scoring_file)
        except:
            print("Problem processing baseline scores for: "+str(protein_name))
        #try:
        #    all_baseline_scores = pd.concat([all_baseline_scores,scoring_file], axis=0)
        #except:
        #    all_baseline_scores = scoring_file
    all_baseline_scores = pd.concat(list_processed_scoring_files, axis=0)

    all_baseline_scores.to_csv(concatenated_baseline_scores_location,index=False)

classification_variables=[
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
'SIFT4G_pred',
'fathmm-MKL_coding_pred',
'fathmm-XF_coding_pred',
'LRT_pred',
'PROVEAN_pred',
#'Aloft_pred',
'MutationAssessor_pred',
'Polyphen2_HDIV_pred',
'Polyphen2_HVAR_pred',
'MVP_pred'
]

#####################################################################################
#Merge EVE scores and baseline scores

if not os.path.exists(merged_file_eve_clinvar_baseline_location):
    data = pd.read_csv(scores_location) #protein_name,mutations,evol_indices,EVE_scores,EVE_classes_100_pct_retained,uncertainty,uncertainty_deciles,uncertainty_quartiles,EVE_classes_10_pct_retained,EVE_classes_20_pct_retained,EVE_classes_30_pct_retained,EVE_classes_40_pct_retained,EVE_classes_50_pct_retained,EVE_classes_60_pct_retained,EVE_classes_70_pct_retained,EVE_classes_80_pct_retained,EVE_classes_90_pct_retained,GMM_class_25_pct_retained,GMM_class_75_pct_retained
    ClinVar_labels = pd.read_csv(labels_location) #protein_name,mutations,Starry_Coarse_Grained_Clin_Sig,ClinVar_labels
    ClinVar_labels = ClinVar_labels[['protein_name','mutations','ClinVar_labels']]
    data = data[['protein_name','mutations','EVE_scores','EVE_classes_100_pct_retained','GMM_class_75_pct_retained']]
    data.rename(columns = {'protein_name': 'pid', 'mutations': 'mutant','EVE_scores':'EVE_scores','EVE_classes_100_pct_retained':'EVE_classes_100_pct_retained','GMM_class_75_pct_retained':'EVE_classes_75_pct_retained'}, inplace = True)
    data = pd.merge(data,ClinVar_labels, how='left',left_on=['pid','mutant'],right_on=['protein_name','mutations'])
    data = data[['pid','mutant','EVE_scores','EVE_classes_100_pct_retained','EVE_classes_75_pct_retained','ClinVar_labels']]
    data = data[(data['ClinVar_labels']==0.0)|(data['ClinVar_labels']==1.0)]
    if not create_concatenation_baseline_scores:
        all_baseline_scores = pd.read_csv(concatenated_baseline_scores_location,low_memory=False)
    data = pd.merge(data,all_baseline_scores,how='left',on=['pid','mutant'])
    data.to_csv(merged_file_eve_clinvar_baseline_location,index=False)
    print("Compare ClinVar variables")
    print(data[['ClinVar_labels','clinvar_clnsig']].value_counts())
else:
    data=pd.read_csv(merged_file_eve_clinvar_baseline_location,low_memory=True)

def compute_perf_protein(data,protein_name,scoring_variables,classification_variables, use_weights=False):
    output=[protein_name, data['ClinVar_labels'].count(), data['ClinVar_labels'][data['ClinVar_labels']==1].count(), data['ClinVar_labels'][data['ClinVar_labels']==0].count()]
    for scoring_variable in scoring_variables:
        output.append(data[scoring_variable].count())
        try:
            filtered=data[[scoring_variable,'ClinVar_labels',scoring_variable+"_weights"]].dropna()
            if not use_weights:
                auc=roc_auc_score(y_score=filtered[scoring_variable], y_true=filtered['ClinVar_labels'])
            else:
                auc=roc_auc_score(y_score=filtered[scoring_variable], y_true=filtered['ClinVar_labels'], sample_weight=1.0/filtered[scoring_variable+"_weights"])
        except:
            auc=np.nan
        output.append(auc)
    for classification_variable in classification_variables:
        output.append(data[classification_variable].count())
        try:
            filtered=data[[classification_variable,'ClinVar_labels',classification_variable+"_weights"]].dropna()
            if not use_weights:
                accuracy=(filtered[classification_variable].map({"Pathogenic":1, "Benign":0})==filtered['ClinVar_labels']).astype(int).mean()
            else:
                accuracy=accuracy_score(y_pred=filtered[classification_variable].map({"Pathogenic":1, "Benign":0}), y_true=filtered['ClinVar_labels'], sample_weight=1.0/filtered[classification_variable+"_weights"])
        except:
            accuracy=np.nan
        output.append(accuracy)
    return output

def compute_perf_dataset(data, output_location, scoring_variables, classification_variables, list_proteins, use_weights=False):
    header=['pid','num_all_labels','num_P_labels','num_B_labels']+['num_'+scoring_variable+','+'AUC_'+scoring_variable for scoring_variable in scoring_variables] + ['num_'+classification_variable+','+'Accuracy_'+classification_variable for classification_variable in classification_variables]
    header=','.join(header)
    with open(output_location, "a") as log:
        log.write(header+"\n")
    for protein_name in list_proteins:
        protein_data=data[data.pid==protein_name]
        perf = compute_perf_protein(protein_data,protein_name,scoring_variables,classification_variables,use_weights)
        with open(output_location, "a") as log:
            log.write(",".join([str(x) for x in perf])+"\n")
    with open(output_location, "a") as log:
        log.write(",".join([str(x) for x in compute_perf_protein(data,"All_proteins",scoring_variables,classification_variables,use_weights)])+"\n")
    #Compute average
    all_auc_file = pd.read_csv(output_location,low_memory=False, na_values=np.nan)
    all_auc_file = all_auc_file[all_auc_file.pid != 'All_proteins']
    all_auc_file.set_index('pid',inplace=True)
    all_auc_file_average=all_auc_file.mean(axis=0, skipna=True)
    with open(output_location, "a") as log:
        log.write('Average_proteins,' + ','.join([str(x) for x in list(all_auc_file_average)])+"\n")
    #Compute weighted averages
    log_weighted_avg = 'Weigthed_average_proteins'
    reference_weight = 'num_all_labels'
    for variable in all_auc_file.columns:
        if variable.startswith('num_'):
            reference_weight = variable
            log_weighted_avg += ',-'
        else:
            temp_df = all_auc_file.dropna(axis=0, subset=[variable])
            log_weighted_avg += ',' + str(np.average(temp_df[variable], axis=0, weights=temp_df[reference_weight]))
    with open(output_location, "a") as log:
        log.write(log_weighted_avg+"\n")
    #Compute low-sample censored averages
    log_censored_avg_gt10 = 'Censored_Average_gt10'
    reference_weight = 'num_all_labels'
    for variable in all_auc_file.columns:
        if variable.startswith('num_'):
            reference_weight = variable
            log_censored_avg_gt10 += ',-'
        else:
            temp_df = all_auc_file.dropna(axis=0, subset=[variable])
            temp_df = temp_df[temp_df[reference_weight] >= 10]
            log_censored_avg_gt10 += ',' + str(np.mean(temp_df[variable], axis=0))
    with open(output_location, "a") as log:
        log.write(log_censored_avg_gt10)
        
if os.path.exists(AUC_accuracy_all_location):
    os.remove(AUC_accuracy_all_location)
if os.path.exists(AUC_accuracy_75pct_location):
    os.remove(AUC_accuracy_75pct_location)
if os.path.exists(AUC_accuracy_all_position_level_location):
    os.remove(AUC_accuracy_all_position_level_location)
if os.path.exists(AUC_accuracy_75pct_position_level_location):
    os.remove(AUC_accuracy_75pct_position_level_location)

compute_perf_dataset(data,AUC_accuracy_all_location, ['EVE_scores']+scoring_variables, ['EVE_classes_100_pct_retained']+classification_variables, list_proteins)
data_75_pct = data[data.EVE_classes_75_pct_retained!="Uncertain"]
compute_perf_dataset(data_75_pct,AUC_accuracy_75pct_location, ['EVE_scores']+scoring_variables, ['EVE_classes_75_pct_retained']+classification_variables, list_proteins)

#Compute position-level statistics
data['position'] = data['mutant'].str[1:-1].astype(int)
data_weights=data.groupby(['pid','position']).count().reset_index()
data_position_weights = pd.merge(data,data_weights,how='left',on=['pid','position'], suffixes=['','_weights'])
compute_perf_dataset(data_position_weights,AUC_accuracy_all_position_level_location, ['EVE_scores']+scoring_variables, ['EVE_classes_100_pct_retained']+classification_variables, list_proteins, use_weights=True)

data_75_pct = data[data.EVE_classes_75_pct_retained!="Uncertain"]
data_75_pct['position'] = data_75_pct['mutant'].str[1:-1].astype(int)
data_75_pct_weights=data_75_pct.groupby(['pid','position']).count().reset_index()
data_75_pct_position_weights = pd.merge(data_75_pct,data_75_pct_weights,how='left',on=['pid','position'], suffixes=['','_weights'])
compute_perf_dataset(data_75_pct_position_weights,AUC_accuracy_75pct_position_level_location, ['EVE_scores']+scoring_variables, ['EVE_classes_75_pct_retained']+classification_variables, list_proteins, use_weights=True)


