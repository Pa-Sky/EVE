import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os,sys
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve

version = 'Aug15_10b10p_v2'
plots_paper_folder = "/home/pn73/EVE/results/plots_paper/Aug15"
ouput_file_location_ROC_curves = "/home/pn73/EVE/results/ROC_PRC_curves"

all_BPU_predictions_filename = '/home/pn73/EVE/results/EVE_scores/All_singles_Jul7/all_EVE_scores_All_singles_Jul7.csv'
labels_filename = '/home/pn73/EVE/data/labels/All_3k_proteins_ClinVar_labels_Jul7.csv'
aggregated_prediction_results_filename = '/home/pn73/EVE/results/EVE_scores/All_singles_Jul7/aggregated_EVE_scores_all_singles_Jul9.csv'

AUC_accuracy_all_filename = '/home/pn73/EVE/results/AUC_Accuracy/AUC_accuracy_all_Jul9.csv'
AUC_accuracy_75pct_filename = '/home/pn73/EVE/results/AUC_Accuracy/AUC_accuracy_75pct_Jul9.csv'

list_figures_to_create = ['2bc'] #['2a','2bc','2d','supp3','supp5']

#data_Fig2a_agg_prediction_by_label = '/users/pastin/projects/proj-marks_lab_and_oatml/VAE_model/exp_results/computing_evol_index/pvae_unsupervised_aggregated_prediction_results_BPU_Jan2.csv'
#data_Fig2b_AUC_results = '/users/pastin/projects/proj-marks_lab_and_oatml/VAE_model/exp_results/AUC_analysis/ALL_AUC_Jan2.csv'
#data_Fig2cd_All_scores_files = '/users/pastin/projects/proj-marks_lab_and_oatml/VAE_model/datasets/labelled_mutations/gene_variants_cv_gnomad_ecmodels_mcap_mistic_pp2_cv2_vae_BPU_ASM_Jan2_HGMM_EM_prot-weight_0.3/0_All_scores_file_BPU.csv'
#data_SuppFig2_DMS_comparison = '/users/pastin/projects/proj-marks_lab_and_oatml/VAE_model/misc/DeepSeqV2-Vs-V1_comparison_old_DMS.csv'
#data_SuppFig3_detailed_predictions = '/users/pastin/projects/proj-marks_lab_and_oatml/VAE_model/exp_results/computing_evol_index/Average_delta_elbos_BPU_Jan2_final.csv'

def compute_accuracy(class_pred, labels):
    temp_df = pd.DataFrame({'class_pred': class_pred,'labels': labels})
    initial_num_obs = len(temp_df['labels'])
    temp_df=temp_df[temp_df['class_pred'] != 'Uncertain']
    filtered_num_obs = len(temp_df['labels'])
    temp_df['class_pred_bin'] = temp_df['class_pred'].map(lambda x: 1 if x == 'Pathogenic' else 0)
    correct_classification = (temp_df['class_pred_bin'] == temp_df['labels']).astype(int)
    accuracy = round(correct_classification.mean()*100,1)
    pct_mutations_kept = round(filtered_num_obs/float(initial_num_obs)*100,1)
    return accuracy, pct_mutations_kept

def compute_AUC_with_uncertain(scores, class_pred, labels):
    temp_df = pd.DataFrame({'class_pred': class_pred,'labels': labels, 'scores': scores})
    temp_df=temp_df[temp_df['class_pred'] != 'Uncertain']
    AUC = roc_auc_score(y_true=temp_df['labels'], y_score=temp_df['scores'])
    return AUC

def compute_pathogenic_rate_with_uncertain(class_pred, labels):
    temp_df = pd.DataFrame({'class_pred': class_pred,'labels': labels})
    temp_df=temp_df[temp_df['class_pred'] != 'Uncertain']
    rate = len(temp_df[temp_df['class_pred'] == 'Pathogenic']) / float(len(temp_df))
    return rate

list_10b10p = ['MSH2_HUMAN', 'P53_HUMAN', 'ATP7B_HUMAN', 'CO3A1_HUMAN',
                    'RB_HUMAN', 'BRCA2_HUMAN', 'MLH1_HUMAN', 'BRCA1_HUMAN_b0.05',
                    'TSC2_HUMAN', 'MYPC3_HUMAN', 'LDLR_HUMAN', 'FBN1_HUMAN',
                    'RYR2_HUMAN', 'SCN5A_HUMAN', 'RET_HUMAN', 'MYH7_HUMAN',
                    'MSH6_HUMAN', 'RYR1_HUMAN']

list_5b5p = ['TGFR2_HUMAN', 'MSH2_HUMAN', 'P53_HUMAN', #, 'MUTYH_HUMAN'
                    'ATP7B_HUMAN', 'CO3A1_HUMAN', 'APC_HUMAN', 'RB_HUMAN',
                    'PKP2_HUMAN', 'DSG2_HUMAN', 'BRCA2_HUMAN', 'MLH1_HUMAN',
                    'BRCA1_HUMAN_b0.05', 'TSC2_HUMAN', 'MYPC3_HUMAN', 'LDLR_HUMAN',
                    'TSC1_HUMAN', 'FBN1_HUMAN', 'RYR2_HUMAN', 'DESP_HUMAN',
                    'SCN5A_HUMAN', 'KCNH2_HUMAN', 'RET_HUMAN', 'MYH7_HUMAN',
                    'MSH6_HUMAN', 'RYR1_HUMAN', 'PMS2_HUMAN', 'CAC1S_HUMAN']

with open('./data/mappings/3b3p.txt') as f:
    list_3b3p_full = [x.strip().replace("'", "") for x in f.read().strip().split(",")]

with open('./data/mappings/5b5p.txt') as f:
    list_5b5p_full = [x.strip().replace("'", "") for x in f.read().strip().split(",")]

with open('./data/mappings/10b10p.txt') as f:
    list_10b10p_full = [x.strip().replace("'", "") for x in f.read().strip().split(",")]

##########################################################################################################################################
#################################Fig 2a################################
##########################################################################################################################################

all_BPU_predictions = pd.read_csv(all_BPU_predictions_filename, low_memory=False)
#protein_name,mutations,evol_indices,EVE_scores,EVE_classes_100_pct_retained,uncertainty,uncertainty_deciles,uncertainty_quartiles,EVE_classes_10_pct_retained,EVE_classes_20_pct_retained,EVE_classes_30_pct_retained,EVE_classes_40_pct_retained,EVE_classes_50_pct_retained,EVE_classes_60_pct_retained,EVE_classes_70_pct_retained,EVE_classes_80_pct_retained,EVE_classes_90_pct_retained,GMM_class_25_pct_retained,GMM_class_75_pct_retained
labels = pd.read_csv(labels_filename, low_memory=False)
#protein_name,mutations,Starry_Coarse_Grained_Clin_Sig,ClinVar_labels

print("Num mutations before merge: "+str(len(all_BPU_predictions)))
all_BPU_predictions = pd.merge(all_BPU_predictions, labels, how='inner', on=['protein_name','mutations'])
print("Num mutations after merge: "+str(len(all_BPU_predictions)))

if '2a' in list_figures_to_create:
    if os.path.exists(aggregated_prediction_results_filename):
        print("Aggregated prediction file already exists")
    else:
        with open(aggregated_prediction_results_filename, "a") as label_aggregated_stats:
            stats_header = "protein_name,mean_pred_pathogenic,std_pred_pathogenic,mean_pred_benign,std_pred_benign,mean_pred_uncertain,std_pred_uncertain"
            label_aggregated_stats.write(stats_header+"\n")

        for protein in np.unique(all_BPU_predictions.protein_name):
            try:
                BPU_predictions_protein = all_BPU_predictions[all_BPU_predictions.protein_name==protein]

                pathogenic_label_index = np.array(BPU_predictions_protein.ClinVar_labels) == 1
                benign_label_index = np.array(BPU_predictions_protein.ClinVar_labels) == 0
                uncertain_label_index = np.array(BPU_predictions_protein.ClinVar_labels) == 0.5

                mean_pred_pathogenic = BPU_predictions_protein['evol_indices'][pathogenic_label_index].mean()
                std_pred_pathogenic = BPU_predictions_protein['evol_indices'][pathogenic_label_index].std()
                
                mean_pred_benign = BPU_predictions_protein['evol_indices'][benign_label_index].mean()
                std_pred_benign = BPU_predictions_protein['evol_indices'][benign_label_index].std()

                mean_pred_uncertain = BPU_predictions_protein['evol_indices'][uncertain_label_index].mean()
                std_pred_uncertain = BPU_predictions_protein['evol_indices'][uncertain_label_index].std()
            except:
                mean_pred_pathogenic = None
                std_pred_pathogenic = None
                mean_pred_benign = None
                std_pred_benign = None
                mean_pred_uncertain = None
                std_pred_uncertain = None
            with open(aggregated_prediction_results_filename, "a") as label_aggregated_stats:
                stats_log = ",".join([ str(x) for x in [protein, mean_pred_pathogenic, std_pred_pathogenic, mean_pred_benign, std_pred_benign, mean_pred_uncertain, std_pred_uncertain] ])
                label_aggregated_stats.write(stats_log+"\n")

    aggregated_stats_df = pd.read_csv(aggregated_prediction_results_filename, low_memory=False)

    print("Number of non missing pathogenic: ")
    print( aggregated_stats_df['mean_pred_pathogenic'].notnull().value_counts() )
    print("Number of non missing benign: ")
    print( aggregated_stats_df['mean_pred_benign'].notnull().value_counts() )
    print("Number of non missing labels: ")
    print( (aggregated_stats_df['mean_pred_pathogenic'].notnull() | aggregated_stats_df['mean_pred_benign'].notnull()).value_counts() )
    print("Number of non missing uncertain: ")
    print( aggregated_stats_df['mean_pred_uncertain'].notnull().value_counts() )
    
    #Cleanups to improve data quality
    index1 = (aggregated_stats_df['mean_pred_pathogenic']=='None').copy()
    aggregated_stats_df.loc[index1,'mean_pred_pathogenic']=np.nan
    index2 = (aggregated_stats_df['mean_pred_benign']=='None').copy()
    aggregated_stats_df.loc[index2,'mean_pred_benign']=np.nan
    
    #PARAM
    #aggregated_stats_df.dropna(axis=0, subset=['mean_pred_pathogenic','mean_pred_benign'], how='any', inplace=True) #ie we only plot if we have at least 1 valid P and 1 valid B


    aggregated_stats_df['mean_pred_pathogenic']=aggregated_stats_df['mean_pred_pathogenic'].astype('float64')
    aggregated_stats_df['std_pred_pathogenic']=aggregated_stats_df['std_pred_pathogenic'].astype('float64')
    aggregated_stats_df['mean_pred_benign']=aggregated_stats_df['mean_pred_benign'].astype('float64')
    aggregated_stats_df['std_pred_benign']=aggregated_stats_df['std_pred_benign'].astype('float64')

    print(aggregated_stats_df)

    ##Default version
    num_proteins = len(aggregated_stats_df)
    x = range(num_proteins)
    y1 = aggregated_stats_df['mean_pred_pathogenic']
    e1 = aggregated_stats_df['std_pred_pathogenic']
    y2 = aggregated_stats_df['mean_pred_benign']
    e2 = aggregated_stats_df['std_pred_benign']

    fig,ax = plt.subplots(figsize=(10, 3))
    pathogenic = plt.errorbar(x, y1, yerr=e1, fmt='o', ecolor='xkcd:red', mfc='xkcd:red', alpha=0.7, mec='black')
    benign = plt.errorbar(x, y2, yerr=e2, fmt='o', ecolor='xkcd:sky blue', mfc='xkcd:sky blue', alpha=0.7, mec='black')
    plt.legend([pathogenic,benign], ["Pathogenic","Benign"], loc='upper right', prop={'size':10})
    plt.xlabel("Protein ID")
    plt.ylabel("Evolutionary index")
    fig.savefig(plots_paper_folder+os.sep+"Fig2a_Evol_index_BP_avg_across_proteins_"+version+".png", dpi=800, bbox_inches='tight')
    fig.clf()

    #Plot marginals
    #fig,ax = plt.subplots(figsize=(4,2))
    x = np.linspace(-10, 25, 2000)
    plt.hist(y1, color = 'xkcd:red', bins = 80, histtype='stepfilled', density=True, alpha=0.7)
    pdf_x = np.linspace(np.min(y1),np.max(y1),200)
    avg = np.mean(y1)
    var = np.var(y1)
    pdf_y = 1.0/np.sqrt(2*np.pi*var)*np.exp(-0.5*(pdf_x-avg)**2/var)
    plt.plot(pdf_x,pdf_y,'k--', color = 'xkcd:dark red', linewidth=2.0)
    plt.hist(y2, color = 'xkcd:sky blue', bins = 80, histtype='stepfilled', density=True, alpha=0.7)
    pdf_x = np.linspace(np.min(y2),np.max(y2),100)
    avg = np.mean(y2)
    var = np.var(y2)
    pdf_y = 1.0/np.sqrt(2*np.pi*var)*np.exp(-0.5*(pdf_x-avg)**2/var)
    plt.plot(pdf_x,pdf_y,'k--', color = 'xkcd:blue', linewidth=2.0)
    #plt.xlim(max(y1),min(y2))
    plt.xlim(25,-10)
    fig.savefig(plots_paper_folder+os.sep+"Fig2a_Evol_index_BP_avg_across_proteins_marginals_"+version+".png", dpi=800, bbox_inches='tight')
    plt.tick_params(
        axis='both',          # changes apply to both axes
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        left=False,
        labelbottom=False,
        labelleft=False)
    fig.savefig(plots_paper_folder+os.sep+"Fig2a_Evol_index_BP_avg_across_proteins_marginals_noticks_"+version+".png", dpi=800, bbox_inches='tight')
    plt.clf()

    #Save source data
    aggregated_stats_df = aggregated_stats_df[['protein_name','mean_pred_pathogenic','std_pred_pathogenic','mean_pred_benign','std_pred_benign']]
    aggregated_stats_df.rename(columns={'mean_pred_pathogenic':'Evol_index_avg_pathogenic', 
                                        'std_pred_pathogenic':'Evol_index_std_dev_pathogenic',
                                        'mean_pred_benign':'Evol_index_avg_benign',
                                        'std_pred_benign':'Evol_index_std_dev_benign',
                                        }, inplace=True)
    aggregated_stats_df.to_csv(plots_paper_folder+os.sep+'source_data/Fig2a_source_data_'+version+'.csv', index=False)


##########################################################################################################################################
#Fig 2.b & c
##########################################################################################################################################
if '2bc' in list_figures_to_create:
    fig = plt.figure(figsize=(4, 3.5))

    data_main = all_BPU_predictions[all_BPU_predictions.mutations != 'wt']
    num_BPU_mutations = len(data_main[data_main.ClinVar_labels.notnull()])
    print("Num BPU mutations")
    print(num_BPU_mutations)
    data_main = data_main[data_main.ClinVar_labels != 0.5]
    data_main = data_main[data_main.ClinVar_labels.notnull()]
    print("Num ClinVar labels :")
    num_ClinVar_labels = len(data_main)
    print(num_ClinVar_labels)
    print("Labels :")
    print(data_main.ClinVar_labels.value_counts())
    print("EVE_scores :")
    print(data_main.EVE_scores.notnull().value_counts())

    #1B1P
    #aggregated_stats_df.dropna(axis=0, subset=['Evol_index_avg_pathogenic','Evol_index_avg_benign'], how='any', inplace=True)
    #list_1b1p_full = np.unique(aggregated_stats_df.protein_name)
    #data_main = data_main[data_main.protein_name.isin(list_1b1p_full)]

    data_3b3p_full = data_main[data_main.protein_name.isin(list_3b3p_full)]
    data_5b5p_full = data_main[data_main.protein_name.isin(list_5b5p_full)]
    data_10b10p_full = data_main[data_main.protein_name.isin(list_10b10p_full)]

    #protein_name,mutations,evol_indices,EVE_scores,EVE_classes_100_pct_retained,uncertainty,uncertainty_deciles,uncertainty_quartiles,
    #EVE_classes_10_pct_retained,EVE_classes_20_pct_retained,EVE_classes_30_pct_retained,EVE_classes_40_pct_retained,EVE_classes_50_pct_retained,
    #EVE_classes_60_pct_retained,EVE_classes_70_pct_retained,EVE_classes_80_pct_retained,EVE_classes_90_pct_retained,GMM_class_25_pct_retained,GMM_class_75_pct_retained

    def compute_performance_by_uncertainty_decile(data_frame, metric="AUC"):
        performance_by_uncertainty_deciles = {}
        pathogenic_rate_by_uncertainty_deciles = {}
        for decile in range(1,11):
            #Observations dropped based on their uncertainty decile are labelled as uncertain
            classification_name = 'EVE_classes_'+str(decile*10)+"_pct_retained"
            if metric=="Accuracy":
                performance_decile = compute_accuracy(data_frame[classification_name], data_frame['ClinVar_labels'])[0]
            elif metric=="AUC":
                performance_decile = compute_AUC_with_uncertain(scores=data_frame['EVE_scores'].astype(float), class_pred=data_frame[classification_name], labels=data_frame['ClinVar_labels'])
            performance_by_uncertainty_deciles[decile] = performance_decile
            pathogenic_rate_by_uncertainty_deciles[decile] = compute_pathogenic_rate_with_uncertain(class_pred=data_frame[classification_name], labels=data_frame['ClinVar_labels'])
        
        performance_by_uncertainty_deciles_list = [performance_by_uncertainty_deciles[i] for i in range(1,11)]
        pathogenic_rate_by_uncertainty_deciles_list = [pathogenic_rate_by_uncertainty_deciles[i] for i in range(1,11)]
        return performance_by_uncertainty_deciles_list, pathogenic_rate_by_uncertainty_deciles_list 


    dict_df_full = {'data_main':data_main, 'data_3b3p_full':data_3b3p_full, 'data_5b5p_full':data_5b5p_full, 'data_10b10p_full':data_10b10p_full}
    y_AUC_full = {}
    y_accuracy_full = {}
    y_pathogenicity_rate_full = {}
    x = range(10,101,10)
    factor = np.array(x)/100 * num_BPU_mutations / num_ClinVar_labels
    print(factor)
    factor = ['','x'+str(round(factor[1])), '', 'x'+str(round(factor[3])), '', 'x'+str(round(factor[5])), '', 'x'+str(round(factor[7])), '','x'+str(round(factor[9]))]
    print(factor)
    for df in ['data_main', 'data_3b3p_full', 'data_5b5p_full', 'data_10b10p_full']:
        y_AUC_full[df], y_pathogenicity_rate_full[df] = compute_performance_by_uncertainty_decile(dict_df_full[df], "AUC")
        y_accuracy_full[df] = compute_performance_by_uncertainty_decile(dict_df_full[df], "Accuracy")[0]
        

    fig,ax = plt.subplots(figsize=(4, 3.5))
    plot_AUC_df={}
    #plot_colors_full={'data_main':'xkcd:deep blue', 'data_3b3p_full':'xkcd:blue', 'data_5b5p_full':'xkcd:sky blue', 'data_10b10p_full':'xkcd:periwinkle'}
    plot_colors_full={'data_10b10p_full':'xkcd:deep blue'}
    #for df in ['data_main', 'data_3b3p_full', 'data_5b5p_full', 'data_10b10p_full']:
    for df in ['data_10b10p_full']:
        plot_AUC_df[df] = plt.scatter(x=x, y=y_AUC_full[df], c=plot_colors_full[df], edgecolors='none')
        plt.plot(x, y_AUC_full[df], c=plot_colors_full[df])
    #plt.xlabel("% of variants retained based on uncertainty")
    ax.set_xlabel("% of variants retained based on uncertainty")
    
    plt.ylabel("AUC")
    plt.ylim(0.88, 0.99)
    #plt.legend([plot_AUC_df['data_main'],plot_AUC_df['data_3b3p_full'],plot_AUC_df['data_5b5p_full'],plot_AUC_df['data_10b10p_full']], ['All proteins', 'Proteins w/ at least 3B & 3P', 'Proteins w/ at least 5B & 5P', 'Proteins w/ at least 10B & 10P'], loc='lower left', prop={'size':8})
    plt.savefig(plots_paper_folder+os.sep+"Fig2b_AUC_Vs_uncertainty_full_"+version+".png", dpi=400, bbox_inches='tight')
    plt.clf()

    fig,ax = plt.subplots(figsize=(4, 3.5))
    plot_Accuracy_df={}
    ax2 = ax.twiny()
    #plot_colors_full={'data_main':'xkcd:deep blue', 'data_3b3p_full':'xkcd:blue', 'data_5b5p_full':'xkcd:sky blue', 'data_10b10p_full':'xkcd:periwinkle'}
    plot_colors_full={'data_10b10p_full':'xkcd:deep blue'}
    #for df in ['data_main', 'data_3b3p_full', 'data_5b5p_full', 'data_10b10p_full']:
    for df in ['data_10b10p_full']:
        plot_Accuracy_df[df] = plt.scatter(x=x, y=y_accuracy_full[df], c=plot_colors_full[df], edgecolors='none')
        plt.plot(x, y_accuracy_full[df], c=plot_colors_full[df])
    ax.vlines(x=75, ymin=85, ymax=99, colors='xkcd:red', linestyles='--')
    ax.hlines(y=90.5, xmin=8, xmax=102, colors='xkcd:red', linestyles='--')
    ax.grid(True, color='lightgrey',alpha=0.4)
    #plt.xlabel("% of variants retained based on uncertainty")
    #plt.xlim(8, 102)
    ax.set_xlim(8, 102)
    ax.set_xlabel("% of variants dropped based on uncertainty")
    ax.set_xticks(x)
    ax.set_xticklabels(['','80','','60','','40','','20','','0'])
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(x)
    ax2.set_xticklabels(factor)
    ax2.set_xlabel("Potential variant classification increase Vs ClinVar")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(85, 99)
    ax.set_yticks(np.arange(86, 99, step=2))#, fontsize=10)
    
    #Add tick for 25%
    x_ticks = np.append(ax.get_xticks(), 75)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(['','80','','60','','40','','','','0', '25'])
    ax.get_xticklabels()[-1].set_color('xkcd:red')
    
    #plt.legend([plot_Accuracy_df['data_main'],plot_Accuracy_df['data_3b3p_full'],plot_Accuracy_df['data_5b5p_full'],plot_Accuracy_df['data_10b10p_full']], ['All proteins', 'Proteins w/ at least 3B & 3P', 'Proteins w/ at least 5B & 5P', 'Proteins w/ at least 10B & 10P'], loc='lower left', prop={'size':8})
    fig.savefig(plots_paper_folder+os.sep+"Fig2c_Accuracy_Vs_uncertainty_full_"+version+".png", dpi=400, bbox_inches='tight')
    plt.clf()

    #Flipped version
    x = range(90,-1,-10)
    factor = (100-np.array(x))/100 * num_BPU_mutations / num_ClinVar_labels
    factor = ['','x'+str(round(factor[1])), '', 'x'+str(round(factor[3])), '', 'x'+str(round(factor[5])), '', 'x'+str(round(factor[7])), '','x'+str(round(factor[9]))]

    fig,ax = plt.subplots(figsize=(4, 3.5))
    plot_AUC_df={}
    #plot_colors_full={'data_main':'xkcd:deep blue', 'data_3b3p_full':'xkcd:blue', 'data_5b5p_full':'xkcd:sky blue', 'data_10b10p_full':'xkcd:periwinkle'}
    plot_colors_full={'data_10b10p_full':'xkcd:deep blue'}
    #for df in ['data_main', 'data_3b3p_full', 'data_5b5p_full', 'data_10b10p_full']:
    for df in ['data_10b10p_full']:
        plot_AUC_df[df] = plt.scatter(x=x, y=y_AUC_full[df], c=plot_colors_full[df], edgecolors='none')
        plt.plot(x, y_AUC_full[df], c=plot_colors_full[df])
    plt.xlabel("% of variants dropped based on uncertainty")
    plt.ylabel("AUC")
    plt.ylim(0.88, 0.99)
    #plt.legend([plot_AUC_df['data_main'],plot_AUC_df['data_3b3p_full'],plot_AUC_df['data_5b5p_full'],plot_AUC_df['data_10b10p_full']], ['All proteins', 'Proteins w/ at least 3B & 3P', 'Proteins w/ at least 5B & 5P', 'Proteins w/ at least 10B & 10P'], loc='lower right', prop={'size':8})
    plt.savefig(plots_paper_folder+os.sep+"Fig2b_AUC_Vs_uncertainty_full_flipped_"+version+".png", dpi=400, bbox_inches='tight')
    plt.clf()

    fig,ax = plt.subplots(figsize=(4, 3.5))
    ax2 = ax.twiny()
    plot_Accuracy_df={}
    #plot_colors_full={'data_main':'xkcd:deep blue', 'data_3b3p_full':'xkcd:blue', 'data_5b5p_full':'xkcd:sky blue', 'data_10b10p_full':'xkcd:periwinkle'}
    plot_colors_full={'data_10b10p_full':'xkcd:deep blue'}
    #for df in ['data_main', 'data_3b3p_full', 'data_5b5p_full', 'data_10b10p_full']:
    for df in ['data_10b10p_full']:
        plot_Accuracy_df[df] = plt.scatter(x=x, y=y_accuracy_full[df], c=plot_colors_full[df], edgecolors='none')
        plt.plot(x, y_accuracy_full[df], c=plot_colors_full[df])
    ax.vlines(x=25, ymin=85, ymax=99, colors='xkcd:red', linestyles='--')
    #ax.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=True)
    #ax.set_ticks_position("both")
    ax.grid(True, color='lightgrey',alpha=0.4)
    #plt.xlabel("% of variants dropped based on uncertainty")
    #plt.xlim(8, 102)
    ax.set_xlim(-2, 92)
    ax.set_xlabel("% of variants dropped based on uncertainty")
    ax.set_xticks(x)
    ax.set_xticklabels(['90','','70','','50','','30','','10',''])
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(x)
    ax2.set_xticklabels(factor)
    ax2.set_xlabel("Classification volume increase Vs ClinVar")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(85, 99)
    ax.set_yticks(np.arange(86, 99, step=2))#, fontsize=8)
    #plt.legend([plot_Accuracy_df['data_main'],plot_Accuracy_df['data_3b3p_full'],plot_Accuracy_df['data_5b5p_full'],plot_Accuracy_df['data_10b10p_full']], ['All proteins', 'Proteins w/ at least 3B & 3P', 'Proteins w/ at least 5B & 5P', 'Proteins w/ at least 10B & 10P'], loc='lower right', prop={'size':8})
    fig.savefig(plots_paper_folder+os.sep+"Fig2c_Accuracy_Vs_uncertainty_full_flipped_"+version+".png", dpi=400, bbox_inches='tight')
    plt.clf()


    #Save source data (full sets)
    y_AUC_full = pd.DataFrame(y_AUC_full)
    y_AUC_full['Percent_retained_variants_based_on_uncertainty'] = x
    y_AUC_full.rename(columns={'data_main':'AUC_all_proteins', 
                        'data_3b3p_full':'AUC_3B3P_proteins',
                        'data_5b5p_full':'AUC_5B5P_proteins',
                        'data_10b10p_full':'AUC_10B10P_proteins'
                        }, inplace=True)
    y_AUC_full.to_csv(plots_paper_folder+os.sep+'source_data/Fig2b_full_source_data_'+version+'.csv', index=False)
    y_accuracy_full = pd.DataFrame(y_accuracy_full)
    y_accuracy_full['Percent_retained_variants_based_on_uncertainty'] = x
    y_accuracy_full.rename(columns={'data_main':'Accuracy_all_proteins', 
                        'data_3b3p_full':'Accuracy_3B3P_proteins',
                        'data_5b5p_full':'Accuracy_5B5P_proteins',
                        'data_10b10p_full':'Accuracy_10B10P_proteins'
                        }, inplace=True)
    y_accuracy_full.to_csv(plots_paper_folder+os.sep+'source_data/Fig2c_full_source_data_'+version+'.csv', index=False)


##########################################################################################################################################
################################Fig 2d################################
##########################################################################################################################################
if '2d' in list_figures_to_create:
    AUC_accuracy_all = pd.read_csv(AUC_accuracy_all_filename, low_memory=False)

    AUC_accuracy_75pct = pd.read_csv(AUC_accuracy_75pct_filename, low_memory=False)
    #pid,num_all_labels,num_P_labels,num_B_labels,num_EVE_scores,AUC_EVE_scores,num_BayesDel_addAF_score,AUC_BayesDel_addAF_score,num_BayesDel_noAF_score,AUC_BayesDel_noAF_score,num_CADD_phred,AUC_CADD_phred,num_CADD_phred_hg19,AUC_CADD_phred_hg19,num_CADD_raw,AUC_CADD_raw,num_CADD_raw_hg19,AUC_CADD_raw_hg19,num_ClinPred_score,AUC_ClinPred_score,num_DANN_score,AUC_DANN_score,num_DEOGEN2_score,AUC_DEOGEN2_score,num_Eigen-PC-phred_coding,AUC_Eigen-PC-phred_coding,num_Eigen-PC-raw_coding,AUC_Eigen-PC-raw_coding,num_Eigen-phred_coding,AUC_Eigen-phred_coding,num_Eigen-raw_coding,AUC_Eigen-raw_coding,num_FATHMM_score,AUC_FATHMM_score,num_fathmm-MKL_coding_score,AUC_fathmm-MKL_coding_score,num_fathmm-XF_coding_score,AUC_fathmm-XF_coding_score,num_GenoCanyon_score,AUC_GenoCanyon_score,num_LIST-S2_score,AUC_LIST-S2_score,num_LRT_score,AUC_LRT_score,num_M-CAP_score,AUC_M-CAP_score,num_MetaLR_score,AUC_MetaLR_score,num_MetaSVM_score,AUC_MetaSVM_score,num_MPC_score,AUC_MPC_score,num_MutationAssessor_score,AUC_MutationAssessor_score,num_MutationTaster_score,AUC_MutationTaster_score,num_MutPred_score,AUC_MutPred_score,num_MVP_score,AUC_MVP_score,num_Polyphen2_HDIV_score,AUC_Polyphen2_HDIV_score,num_Polyphen2_HVAR_score,AUC_Polyphen2_HVAR_score,num_PrimateAI_score,AUC_PrimateAI_score,num_PROVEAN_score,AUC_PROVEAN_score,num_REVEL_score,AUC_REVEL_score,num_SIFT_score,AUC_SIFT_score,num_SIFT4G_score,AUC_SIFT4G_score,num_VEST4_score,AUC_VEST4_score,num_EVE_classes_75_pct_retained,Accuracy_EVE_classes_75_pct_retained,num_BayesDel_addAF_pred,Accuracy_BayesDel_addAF_pred,num_BayesDel_noAF_pred,Accuracy_BayesDel_noAF_pred,num_ClinPred_pred,Accuracy_ClinPred_pred,num_DEOGEN2_pred,Accuracy_DEOGEN2_pred,num_FATHMM_pred,Accuracy_FATHMM_pred,num_LIST-S2_pred,Accuracy_LIST-S2_pred,num_M-CAP_pred,Accuracy_M-CAP_pred,num_MetaLR_pred,Accuracy_MetaLR_pred,num_MetaSVM_pred,Accuracy_MetaSVM_pred,num_PrimateAI_pred,Accuracy_PrimateAI_pred,num_SIFT_pred,Accuracy_SIFT_pred,num_SIFT4G_pred,Accuracy_SIFT4G_pred,num_fathmm-MKL_coding_pred,Accuracy_fathmm-MKL_coding_pred,num_fathmm-XF_coding_pred,Accuracy_fathmm-XF_coding_pred,num_LRT_pred,Accuracy_LRT_pred,num_PROVEAN_pred,Accuracy_PROVEAN_pred,num_MutationAssessor_pred,Accuracy_MutationAssessor_pred,num_Polyphen2_HDIV_pred,Accuracy_Polyphen2_HDIV_pred,num_Polyphen2_HVAR_pred,Accuracy_Polyphen2_HVAR_pred,num_MVP_pred,Accuracy_MVP_pred
    AUC_accuracy_75pct = AUC_accuracy_75pct[['pid','AUC_EVE_scores']]
    AUC_accuracy_75pct = AUC_accuracy_75pct.rename(columns = {'pid':'pid', 'AUC_EVE_scores':'AUC_EVE_scores_75pct'}, inplace = False)

    data = pd.merge(AUC_accuracy_all[['pid','AUC_EVE_scores','num_all_labels']], AUC_accuracy_75pct, how='left', on='pid')

    ##########################################################################################################################################
    #Fig 2.d (10B10P)
    ##########################################################################################################################################
    data_10b_10p = data[data.pid.isin(list_10b10p)].sort_values(by='AUC_EVE_scores') #.reset_index()
    ###data_10b_10p = pd.concat([data[data.pid == "Average_proteins"], data_10b_10p], axis=0)

    fig, ax = plt.subplots(figsize=(4,5.5))
    x_all = data_10b_10p['AUC_EVE_scores'].astype(float)
    x_75 = data_10b_10p['AUC_EVE_scores_75pct'].astype(float)
    y = np.arange(len(data_10b_10p))
    y_labels = data_10b_10p['pid'].map(lambda x: x.split("_")[0] if x!="Average_proteins" else "All proteins\naverage") + " (" + data_10b_10p['num_all_labels'] + ")"
    min_AUCs = np.maximum(x_all, x_75)

    plt.hlines(y=y, xmin=0.75, xmax=min_AUCs, linestyle='dashed', color='black', linewidth=0.5, alpha=0.3)
    plt.xlim(0.75, 1.02)
    plt.xticks(np.arange(0.75, 1.02, step=0.05), fontsize=8)
    plt.yticks(y, y_labels, fontsize=8)
    plt.scatter(x=x_all, y=y, color='xkcd:sky blue', alpha=0.7, label='All variants')
    plt.scatter(x=x_75, y=y, color='xkcd:deep blue', alpha=1.0, label='Excl. 25% most uncertain')

    ###ax.get_yticklabels()[0].set_weight("bold")

    plt.legend(loc='upper left', fontsize=8)
    plt.xlabel('AUC', fontsize=8)
    #plt.ylabel('Protein', fontsize=18)
    fig.savefig(plots_paper_folder+os.sep+"Fig2d_AUC_10b10p_"+version+".png", dpi=400, bbox_inches='tight')
    fig.clf()

    ##########################################################################################################################################
    #Fig 2.d (5B5P)
    ##########################################################################################################################################
    data_5b_5p = data[data.pid.isin(list_5b5p)].sort_values(by='AUC_EVE_scores') #.reset_index()
    ###data_5b_5p = pd.concat([data[data.pid == "Average_proteins"], data_5b_5p], axis=0)

    fig, ax = plt.subplots(figsize=(4, 5.5))
    x_all = data_5b_5p['AUC_EVE_scores'].astype(float)
    x_75 = data_5b_5p['AUC_EVE_scores_75pct'].map(lambda x: 0.0 if x=='None' else x).astype(float)
    y = np.arange(len(data_5b_5p))
    y_labels = data_5b_5p['pid'].map(lambda x: x.split("_")[0] if x!="Average_proteins" else "All proteins\naverage") + " (" + data_5b_5p['num_all_labels'] + ")"
    min_AUCs = np.maximum(x_all, x_75)

    plt.hlines(y=y, xmin=0.75, xmax=min_AUCs, linestyle='dashed', color='black', linewidth=0.5, alpha=0.3)

    plt.xlim(0.75, 1.02)
    plt.xticks(np.arange(0.75, 1.02, step=0.05), fontsize=8)
    plt.yticks(y, y_labels, fontsize=8)
    plt.scatter(x=x_all, y=y, color='xkcd:sky blue', alpha=0.7, label='All variants')
    plt.scatter(x=x_75, y=y, color='xkcd:deep blue', alpha=1.0, label='Excl. 25% most uncertain')
    ###ax.get_yticklabels()[0].set_weight("bold")

    plt.legend(loc='upper left', fontsize=8)
    plt.xlabel('AUC', fontsize=8)
    #plt.ylabel('Protein')
    fig.savefig(plots_paper_folder+os.sep+"Fig2d_AUC_5b5p_"+version+".png", figsize=(5,7), dpi=400, bbox_inches='tight')
    fig.clf()

    #Save source data
    data_5b_5p.to_csv(plots_paper_folder+os.sep+"source_data/Fig2d_source_data_"+version+".csv", index=False)

##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################
#Source data for Sup Fig 3
if 'supp3' in list_figures_to_create:
    data = all_BPU_predictions[['protein_name','mutations','evol_indices','ClinVar_labels']]
    data.to_csv(plots_paper_folder+os.sep+'source_data/Supp_Fig3_source_data_'+version+'.csv', index=False)

##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################
#Create ROC & PRC curves (Supp Fig 5)

if 'supp5' in list_figures_to_create:
    if not os.path.exists(ouput_file_location_ROC_curves+os.sep+"ROC_curves_All_"+version):
        os.makedirs(ouput_file_location_ROC_curves+os.sep+"ROC_curves_All_"+version)
        os.makedirs(ouput_file_location_ROC_curves+os.sep+"ROC_curves_Both_"+version)

    if not os.path.exists(ouput_file_location_ROC_curves+os.sep+"PRC_curves_All_"+version):
        os.makedirs(ouput_file_location_ROC_curves+os.sep+"PRC_curves_All_"+version)
        os.makedirs(ouput_file_location_ROC_curves+os.sep+"PRC_curves_Both_"+version)

    #ROC Curves
    fig, ax = plt.subplots(figsize=(5, 5))
    for protein_name in np.unique(all_BPU_predictions.protein_name):
        BPU_predictions_protein = all_BPU_predictions[all_BPU_predictions.protein_name==protein_name]
        BPU_predictions_protein = BPU_predictions_protein.dropna(axis=0,subset=['ClinVar_labels'])
        BPU_predictions_protein = BPU_predictions_protein[BPU_predictions_protein.ClinVar_labels != 0.5]
        BPU_predictions_protein_75pct = BPU_predictions_protein[BPU_predictions_protein.EVE_classes_75_pct_retained != 'Uncertain']
        plt.clf()
        try:
            fpr, tpr, _ = roc_curve(BPU_predictions_protein.ClinVar_labels.ravel(), BPU_predictions_protein.EVE_scores.ravel())
            plt.plot(fpr, tpr, color='blue',lw=2, label='ROC curve')
            plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic - '+str(protein_name))
            plt.legend(loc="lower right")
            plt.savefig(ouput_file_location_ROC_curves+os.sep+"ROC_curves_All_"+version+os.sep+'ROC_curve_'+str(protein_name)+".png", dpi=400, bbox_inches='tight')
        except:
            pass
        try:
            fpr, tpr, _ = roc_curve(BPU_predictions_protein_75pct.ClinVar_labels.ravel(), BPU_predictions_protein_75pct.EVE_scores.ravel())
            plt.plot(fpr, tpr, color='navy',lw=2, linestyle='--', label='ROC curve - excl. 25% most uncertain') 
            plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic - '+str(protein_name))
            plt.legend(loc="lower right")
            plt.savefig(ouput_file_location_ROC_curves+os.sep+"ROC_curves_Both_"+version+os.sep+'ROC_curve_'+str(protein_name)+".png", dpi=400, bbox_inches='tight')
        except:
            pass
        plt.clf()

        try:    
            precision, recall, _ = precision_recall_curve(BPU_predictions_protein.ClinVar_labels.ravel(), BPU_predictions_protein.EVE_scores.ravel())
            plt.step(recall, precision, where='post',color='blue', lw=2, label='PRC curve')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.title('Precision Recall curve - '+str(protein_name))
            plt.legend(loc="lower right")
            plt.savefig(ouput_file_location_ROC_curves+os.sep+"PRC_curves_All_"+version+os.sep+'PRC_curve_'+str(protein_name)+".png", dpi=400, bbox_inches='tight')
        except:
            pass

        try:
            precision, recall, _ = precision_recall_curve(BPU_predictions_protein_75pct.ClinVar_labels.ravel(), BPU_predictions_protein_75pct.EVE_scores.ravel())
            plt.step(recall, precision, where='post',color='navy', lw=2, label='PRC curve - excl. 25% most uncertain')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.title('Precision Recall curve - '+str(protein_name))
            plt.legend(loc="lower right")
            plt.savefig(ouput_file_location_ROC_curves+os.sep+"PRC_curves_Both_"+version+os.sep+'PRC_curve_'+str(protein_name)+".png", dpi=400, bbox_inches='tight')
        except:
            pass

        plt.clf()