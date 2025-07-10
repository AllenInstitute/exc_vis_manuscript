import os 
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import f1_score
from sklearn.metrics import hamming_loss
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from scipy import stats
import json
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from matplotlib import gridspec
from matplotlib.lines import Line2D
    
from scipy import stats
import scikit_posthocs  as sp



def rfc_stratified(df,ft_cols,labels_col,title,ttype_ids,num_folds=5,plot=False):
    """
    Will run ttype prediction but with stratified kfold to properly include all ttypes in each 
    train/test split

    num_folds should always be at least 5 if we are doing an n>=5 ttype threshold. This ensures 
    a sample from every t-type is always in training and testing set
    """
    df = df.copy(deep = True)
#     df.fillna(0,inplace=True)

    n_ct_series = df.groupby(labels_col)[labels_col].transform('size').astype(str)
    n_ct_series = ' (n=' + n_ct_series + ')'
    df[labels_col] = df[labels_col] + n_ct_series.astype(str)
    


    feature_values = df[ft_cols].values
    labels = df[labels_col].values

    min_num_groupsize = min(df[labels_col].value_counts().to_dict().values())
    if min_num_groupsize < 5:
        num_folds = min_num_groupsize

    print('-----------------------------------------------------------------')
    print('RFC by {}'.format(labels_col))
    print('-----------------------------------------------------------------'+'\n')
    num_tts = len(df[labels_col].unique())
    print('There are {} unique labels in the {} cells'.format(num_tts,len(labels)))
    print('')
    print(df[labels_col].value_counts())
    print('')

    score = np.zeros(len(ft_cols))
    avg_score = []

    possible_labels = df[labels_col].unique()
    if ttype_ids:
        sorted_labels = sorted(possible_labels, key=lambda x: ttype_ids[x.split("(n=")[0].strip()])
    else:
        sorted_labels = sorted(possible_labels)
        
        
    num_correct = dict(zip(sorted_labels,np.zeros(len(possible_labels))))
    num_occurances = dict(zip(sorted_labels,np.zeros(len(possible_labels))))
    value_counts_dict = dict(zip(df[labels_col].unique(),df[labels_col].value_counts()))
    conf_mat = np.zeros([len(np.unique(labels)),len(np.unique(labels))])

    per_fold_records = {}
    fold_ct=0
    
    oob_accs = []
    train_accs = []
    test_accs = []
    # run 20 iterations of stratified 5-fold cross-validation
    for it in range(0,100//num_folds):
        #shuffle data, train and test
        skfold = StratifiedKFold(n_splits=num_folds, random_state=None, shuffle=True)
        
        for train_index, test_index in skfold.split(feature_values, labels):
            fold_ct+=1
            X_train = feature_values[train_index]
            Y_train = labels[train_index]

            X_test = feature_values[test_index]
            Y_test = labels[test_index]
            clf = RandomForestClassifier(random_state=0,
                                         n_estimators=200,
                                         min_samples_leaf=3,
                                         min_samples_split=3,
                                         oob_score=True,
                                         max_depth = None, 
                                         max_features =None,
                                         class_weight='balanced' )
            clf.fit(X_train, Y_train)
            results = clf.predict(X_test)
            
            # train and test accs
            train_score = clf.score(X_train, Y_train) 
            oob_score = clf.oob_score_
            test_score = metrics.accuracy_score(Y_test, results)
            
            train_accs.append(train_score)
            test_accs.append(test_score)
            oob_accs.append(oob_score)
            
            this_fold_f1_score = f1_score(Y_test, results, average='weighted')
            
            # getting per class accuracy scores
            for ind,res in enumerate(results):
                num_occurances[Y_test[ind]]+=1
                if res == Y_test[ind]:
                    num_correct[res]+=1           
            
            per_fold_records["fold_"+str(fold_ct)]={}
            per_fold_records["fold_"+str(fold_ct)]['truth'] = list(Y_test)
            per_fold_records["fold_"+str(fold_ct)]['prediction'] = list(results)
            per_fold_records["fold_"+str(fold_ct)]['f1_score'] = this_fold_f1_score
            
            conf_mat+=confusion_matrix(Y_test,results,labels = sorted_labels)

            mean_score = clf.score(X_test,Y_test)
            avg_score.append(mean_score)
            score+=clf.feature_importances_

    Average_performance = np.mean(np.asarray(avg_score))
    Stdev_performance = round(np.std(np.asarray(avg_score)),4)
    print('Average performance = {}% +/- {} prediction accuracy'.format(round(100*Average_performance,2), 
                                                                                100*Stdev_performance))

    score_dict = dict(zip(score,ft_cols))
    sorted_scores_dict = {}
    for enum,i in enumerate(sorted(score_dict,reverse=True)):
        sorted_scores_dict[i] = score_dict[i]
        print(score_dict[i])
    
    for ke in value_counts_dict.keys():
        class_acc = num_correct[ke]/num_occurances[ke]
#         logging.debug('{} Accuracy = {}%  (n ={})'.format(ke,round(100*class_acc,2),value_counts_dict[ke]))

#     logging.debug('-----------------------------------------------------------------'+'\n'+'\n')
    row_sums = np.sum(conf_mat,axis=1)
    percent_conf_mat = (conf_mat.T / row_sums).T
    np.nan_to_num(percent_conf_mat,0)
    
    
    
    confusion_df = pd.DataFrame(percent_conf_mat,columns=sorted_labels)
    confusion_df.set_index([sorted_labels],inplace=True)
    confusion_df = confusion_df[sorted_labels]
#     confusion_df.sort_index(inplace=True) 

    vals = confusion_df.values
    vals = vals.astype(object)
    for i in range(len(vals)):
        for j in range(len(vals)):
            if i==j:
                vals[i,j] = str(round(vals[i,j],2))
                continue
            else:
                vals[i,j] = ''
    diagonal_vals = vals.astype('str')

    fig,ax=None,None
    if plot:
        fig,ax=plt.gcf(),plt.gca()
        con = sns.heatmap(confusion_df,fmt = '', xticklabels=sorted_labels, yticklabels= sorted_labels,
                        vmin=0,vmax=1.0,annot_kws={"size":8})
    #     plt.imshow(confusion_df.values,cmap='inferno',vmax=1,vmin=0,axes=ax)
            
    #     ax.set_ylim(len(percent_conf_mat)+0.5, -0.5)
        plt.xticks(np.arange(0,len(confusion_df.columns)),
                confusion_df.columns, 
                rotation=90,
                horizontalalignment='right')
        
        plt.yticks(np.arange(0,len(confusion_df.index)),
                list(confusion_df.index), 
                rotation=0,
                horizontalalignment='right')

        ax.set_xlabel('Prediction')
        ax.set_ylabel('Truth')

        fig.set_size_inches(8,7)
        confusion_title = title + ' acc = {}, stdev = {} num labels = {}'.format(round(Average_performance,3),
                                                                            Stdev_performance,
                                                                            num_tts)
        ax.set_title(confusion_title)
    
    train_test_accs = {"test_accuracies":test_accs,"train_accuracies":train_accs,"oob_accuracies":oob_accs}
    return fig, ax, sorted_scores_dict, confusion_df, per_fold_records, train_test_accs
   
def main():
    
    with open("../data/ScriptArgs.json","r") as f:
        args = json.load(f)    
    
    metadata_file = args['fmost_metadata_file'] 
    color_file = args['color_file'] 

    ivscc_met_labels_file = args['pseq_met_labels_file']
    ivscc_raw_feature_file = args['pseq_raw_dend_features']

    fmost_dendrite_feature_file = args['fmost_slice_cropped_chamfer_correct_local_dend_feature_file'] 
    local_axon_feature_path = args['fmost_raw_local_axon_feature_file']
    long_range_axon_feature_path = args['fmost_entire_axon_features']

    with open(color_file,"r") as f:
        color_dict = json.load(f)

    with open(args['morph_feature_name_relabel'],"r") as f:
        MORPH_FEATURE_RELABEL=json.load(f)

    SNS_PAL = sns.color_palette()
    E_LABEL_COLOR = SNS_PAL[0] # blue
    M_LABEL_COLOR = SNS_PAL[1] # orange
    T_LABEL_COLOR = SNS_PAL[4] # purple
    FMOST_LOCAL_COLOR = SNS_PAL[2]
    FMOST_LONG_RANGE_COLOR  = SNS_PAL[3]
    
    
    pseq_met_labels = pd.read_csv(ivscc_met_labels_file, index_col=0)
    pseq_feature_df = pd.read_csv(ivscc_raw_feature_file, index_col=0)
    ivscc_df = pseq_met_labels.merge(pseq_feature_df, left_index=True, right_index=True)


    drop_flags = ["area","radius","surface","diameter", "axon", 'specimen_id', 'met_type', 'old_met_type_name']
    drop_feats = [c for c in ivscc_df.columns[1:] if  any([i in c for i in drop_flags])]
    # print("Dropping:\n{}".format(drop_feats))
    feature_columns = [c for c in ivscc_df.columns[1:] if not any([i in c for i in drop_flags])]
    # for f in feature_columns:
    #     print(f)
    ivscc_df['Dataset'] = ["ivscc"]*len(ivscc_df)
    ivscc_df["predicted_met_type"] = ivscc_df.met_type


    ivscc_it = ivscc_df[ivscc_df['predicted_met_type'].str.contains("IT")]
    
    fmost_feature_df = pd.read_csv(fmost_dendrite_feature_file,index_col=1)
    fmost_feature_df.index=fmost_feature_df.index.map(lambda x:x+".swc")
    fmost_metadata = pd.read_csv(metadata_file,index_col=0)
    fmost_df = fmost_metadata.merge(fmost_feature_df,left_index=True,right_index=True)#='specimen_id')

    fmost_it = fmost_df[fmost_df['predicted_met_type'].str.contains("IT")]
    
    
    # get order of features that best separate IT types
    fig, ax, it_sorted_scores_dict, confusion_df, per_fold_records, train_test_accs = rfc_stratified(ivscc_it,
                                                                                            feature_columns,
                                                                                            'predicted_met_type',
                                                                                            title="Predictings ITs",
                                                                                            ttype_ids=[],
                                                                                            num_folds=5)
    
    
    fig = plt.figure()

    all_ordered_features = list(it_sorted_scores_dict.values())[1:]
    nctoff = 30
    dotsize = 15
    n_feats = 20
    n_half = int(n_feats/2)

    # Maybe reduce the space taken up by the empty column
    gs = gridspec.GridSpec(n_half, 5, figure=fig, wspace=0.5, hspace=0.45, width_ratios=[1,1,0.05,1,1])

    # Reduced font sizes for title and label
    plt.rc('axes', titlesize=8) 
    plt.rc('axes', labelsize=6)
    plt.rc('xtick', labelsize=6)
    plt.rc('ytick', labelsize=6)

    left_feats = all_ordered_features[0:n_half]
    right_feats = all_ordered_features[n_half:n_half+n_half]

    reduced_dotsize = dotsize * 0.1

    ct=-1
    for ft in left_feats:
        ct+=1
        axl = fig.add_subplot(gs[ct,0])
        axr = fig.add_subplot(gs[ct,1])
        
        ivscc_vals = ivscc_it[ft]
        ivscc_cs = ivscc_it['predicted_met_type'].map(color_dict)
        
        fmost_vals = fmost_it[ft]
        fmost_cs = fmost_it['predicted_met_type'].map(color_dict)
        
        axl.scatter(ivscc_vals,ivscc_it['soma_aligned_dist_from_pia'],c=ivscc_cs,s=reduced_dotsize)
        axr.scatter(fmost_vals,fmost_it['soma_aligned_dist_from_pia'],c=fmost_cs,s=reduced_dotsize)
    #     axl.set_ylim(0,1000)
    #     axr.set_ylim(0,1000)
        
        
        axl.invert_yaxis()    
        axr.invert_yaxis()    
        axr.set_yticklabels([])
        if ct!=0:
            axl.set_yticklabels([])
        else:
            axl.set_ylabel("Soma Depth (um)")
            axl.set_title("Patch-Seq")
            axr.set_title("WNM")
            
        ft = MORPH_FEATURE_RELABEL[ft]
    #     if len(ft)<nctoff:
    #         ft = "".join([" "]*(nctoff-len(ft)))+ft
        axl.set_xlabel(ft,ha='center',color=M_LABEL_COLOR)
        axl.xaxis.set_label_coords(1.25, -0.27)

    #     axl.set_xlabel(ft,ha='left',color=M_LABEL_COLOR)
    #     axl.xaxis.set_label_coords(0.5, -0.25)
        axl.sharex(axr)
        
    ct=-1
    for ft in right_feats:
        ct+=1
        axl = fig.add_subplot(gs[ct,3])
        axr = fig.add_subplot(gs[ct,4])
        
        ivscc_vals = ivscc_it[ft]
        ivscc_cs = ivscc_it['predicted_met_type'].map(color_dict)
        
        fmost_vals = fmost_it[ft]
        fmost_cs = fmost_it['predicted_met_type'].map(color_dict)
        
        axl.scatter(ivscc_vals,ivscc_it['soma_aligned_dist_from_pia'],c=ivscc_cs,s=reduced_dotsize)
        axr.scatter(fmost_vals,fmost_it['soma_aligned_dist_from_pia'],c=fmost_cs,s=reduced_dotsize)
    #     axl.set_ylim(0,1000)
    #     axr.set_ylim(0,1000)
        axl.invert_yaxis()    
        axr.invert_yaxis()    
        axr.set_yticklabels([])
        if ct!=0:
            axl.set_yticklabels([])
        else:
            axl.set_ylabel("Soma Depth (um)")
            axl.set_title("Patch-Seq")
            axr.set_title("WNM")
            
            legend_handles = []
            
            these_mets = sorted(set(fmost_it['predicted_met_type'].tolist()+ivscc_it['predicted_met_type'].tolist()))
            for label in these_mets:
                hand = Line2D([0], [0], marker='o', color='w', label=label,
                            markersize=5, markerfacecolor=color_dict[label]) 
                legend_handles.append(hand)
            # Add custom legend
            axr.legend(handles=legend_handles, title='MET-type', loc='upper left',bbox_to_anchor=[1,1],
                    fontsize=5, title_fontsize=5)
            
            
            
        ft = MORPH_FEATURE_RELABEL[ft]
    #     if len(ft)<nctoff:
    #         ft = "".join([" "]*(nctoff-len(ft)))+ft
        axl.set_xlabel(ft,ha='center',color=M_LABEL_COLOR)
        axl.xaxis.set_label_coords(1.25, -0.27)
        axl.sharex(axr)
        
    plt.tight_layout() # Add this after all plotting is done
    fig.set_size_inches(7, 11)
    fig.savefig("plot_SuppFig_SomaDepth_By_Feature_PSEQ_IT_Dend.pdf",dpi=600,bbox_inches='tight')
    plt.close()
    
    #
    #
    #
    #
    #    Non- IT PSEQ integration plot
    #
    #
    #
    
    ivscc_non_it = ivscc_df[~ivscc_df['predicted_met_type'].str.contains("IT")]
    fmost_non_it = fmost_df[~fmost_df['predicted_met_type'].str.contains("IT")]
    
    fig, ax, sorted_scores_dict_nonit, confusion_df, per_fold_records, train_test_accs = rfc_stratified(ivscc_non_it,
                                                                                            feature_columns,
                                                                                            'predicted_met_type',
                                                                                            title="Predictings Non-ITs",
                                                                                            ttype_ids=[],
                                                                                            num_folds=5)
    
    fig = plt.figure()

    all_ordered_features = list(sorted_scores_dict_nonit.values())[1:]

    n_feats = 20
    n_half = int(n_feats/2)

    # Maybe reduce the space taken up by the empty column
    gs = gridspec.GridSpec(n_half, 5, figure=fig, wspace=0.5, hspace=0.5, width_ratios=[1,1,0.02,1,1])

    # Reduced font sizes for title and label
    plt.rc('axes', titlesize=8) 
    plt.rc('axes', labelsize=6)
    plt.rc('xtick', labelsize=6)
    plt.rc('ytick', labelsize=6)

    left_feats = all_ordered_features[0:n_half]
    right_feats = all_ordered_features[n_half:n_half+n_half]


    ct=-1
    for ft in left_feats:
        ct+=1
        axl = fig.add_subplot(gs[ct,0])
        axr = fig.add_subplot(gs[ct,1])
        
        ivscc_vals = ivscc_non_it[ft]
        ivscc_cs = ivscc_non_it['predicted_met_type'].map(color_dict)
        
        fmost_vals = fmost_non_it[ft]
        fmost_cs = fmost_non_it['predicted_met_type'].map(color_dict)
        
        axl.scatter(ivscc_vals,ivscc_non_it['soma_aligned_dist_from_pia'],c=ivscc_cs,s=reduced_dotsize)
        axr.scatter(fmost_vals,fmost_non_it['soma_aligned_dist_from_pia'],c=fmost_cs,s=reduced_dotsize)
    #     axl.set_ylim(0,1000)
    #     axr.set_ylim(0,1000)
        
        
        axl.invert_yaxis()    
        axr.invert_yaxis()    
        axr.set_yticklabels([])
        if ct!=0:
            axl.set_yticklabels([])
        else:
            axl.set_ylabel("Soma Depth (um)")
            axl.set_title("Patch-Seq")
            axr.set_title("WNM")
            
        ft = MORPH_FEATURE_RELABEL[ft]
    #     if len(ft)<nctoff:
    #         ft = "".join([" "]*(nctoff-len(ft)))+ft
        axl.set_xlabel(ft,ha='center',color=M_LABEL_COLOR)
        axl.xaxis.set_label_coords(1.25, -0.3)

    #     axl.set_xlabel(ft,ha='left',color=M_LABEL_COLOR)
    #     axl.xaxis.set_label_coords(0.5, -0.25)
        axl.sharex(axr)
        
    ct=-1
    for ft in right_feats:
        ct+=1
        axl = fig.add_subplot(gs[ct,3])
        axr = fig.add_subplot(gs[ct,4])
        
        ivscc_vals = ivscc_non_it[ft]
        ivscc_cs = ivscc_non_it['predicted_met_type'].map(color_dict)
        
        fmost_vals = fmost_non_it[ft]
        fmost_cs = fmost_non_it['predicted_met_type'].map(color_dict)
        
        axl.scatter(ivscc_vals,ivscc_non_it['soma_aligned_dist_from_pia'],c=ivscc_cs,s=reduced_dotsize)
        axr.scatter(fmost_vals,fmost_non_it['soma_aligned_dist_from_pia'],c=fmost_cs,s=reduced_dotsize)

        
        axl.invert_yaxis()    
        axr.invert_yaxis()    
        axr.set_yticklabels([])
        if ct!=0:
            axl.set_yticklabels([])
        else:
            axl.set_ylabel("Soma Depth (um)")
            axl.set_title("Patch-Seq")
            axr.set_title("WNM")
            
            legend_handles = []
            these_mets = sorted(set(fmost_non_it['predicted_met_type'].tolist()+ivscc_non_it['predicted_met_type'].tolist()))
            for label in these_mets:
                actual_label = label
                if label == "L6b":
                    actual_label = "L6b (WNM)"
                hand = Line2D([0], [0], marker='o', color='w', label=actual_label,
                            markersize=5, markerfacecolor=color_dict[label]) 
                legend_handles.append(hand)
            # Add custom legend
            axr.legend(handles=legend_handles, title='MET-type', loc='upper left',bbox_to_anchor=[1,1],
                    fontsize=5, title_fontsize=5)
            
            
            
            
            
        ft = MORPH_FEATURE_RELABEL[ft]

        axl.set_xlabel(ft,ha='center',color=M_LABEL_COLOR)
        axl.xaxis.set_label_coords(1.25, -0.3)
        axl.sharex(axr)
        
    plt.tight_layout() # Add this after all plotting is done
    fig.set_size_inches(7, 11)
    fig.savefig("plot_SuppFig_SomaDepth_By_Feature_PSEQ_NonIT_Dend.pdf",dpi=600,bbox_inches='tight')
    plt.close()

    #
    #
    #
    #    WNM only axon features
    #
    #
    #
    local_axon_feat_df = pd.read_csv(local_axon_feature_path,index_col=0)
    axon_features = [c for c in local_axon_feat_df if "axon" in c]

    projection_feats = pd.read_csv(long_range_axon_feature_path,index_col=0)
    long_range_axon_features =  projection_feats.columns.tolist()

    local_axon_feat_df.index=local_axon_feat_df.index.map(lambda x:x+'.swc')

    axon_df = local_axon_feat_df.merge(fmost_metadata,left_index=True,right_index=True)
    axon_df = axon_df.merge(projection_feats,left_index=True,right_index=True)

    all_axon_feats = axon_features + long_range_axon_features
    # all_axon_feats.remove("complete_axon_total_projection_length")

    keeps = [k for k,v in axon_df.predicted_met_type.value_counts().to_dict().items() if v>=2]
    axon_df_thresh = axon_df[axon_df['predicted_met_type'].isin(keeps)]

    fig, ax, ax_sorted_scores_dict_nonit, ax_confusion_df, ax_per_fold_records, ax_train_test_accs = rfc_stratified(axon_df_thresh,
                                                                                                all_axon_feats,
                                                                                                'predicted_met_type',
                                                                                                title="Predictings Non-ITs",
                                                                                                ttype_ids=[],
                                                                                                num_folds=5)
    
    # updated_names = {'complete_axon_total_length':'complete axon length',
    # 'complete_axon_max_euclidean_distance':'complete axon max euclid. dist.',
    # 'complete_axon_num_branches':'complete axon num. branches',
    # 'complete_axon_num_tips':'complete axon num. tips',
    # 'complete_axon_max_branch_order':'complete axon max branch order',
    # 'complete_axon_max_path_distance':'complete axon max path dist.',
    # 'complete_axon_mean_contraction':'complete axon mean contraction',
    # 'axon_width':'complete axon width',
    # 'axon_depth':'complete axon depth',
    # 'axon_height':'complete axon height',
    # 'complete_axon_total_number_of_targets':'num. targets',
    # 'complete_axon_total_projection_length':'total projection length',
    # 'complete_axon_VIS_length':'VIS axon length',
    # 'complete_axon_ipsi_VIS_length':'ips. VIS axon length',
    # 'complete_axon_length_in_soma_structure':'axon length in soma structure',
    # 'complete_axon_number_of_VIS_targets':'num. VIS targets',
    # 'complete_axon_number_of_contra_VIS_targets':'num. contra. VIS targets',
    # 'fraction_of_complete_axon_in_soma_structure':'fract. axon in soma structure'}

    # for k,v in updated_names.items():
    #     MORPH_FEATURE_RELABEL[k]=v
        
    
    fig = plt.figure()
    n_cols = 5
    n_rows = 9
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, wspace=0.2,hspace=0.45, width_ratios=[1,1,1,1,1])


    # Reduced font sizes for title and label
    plt.rc('axes', titlesize=6) 
    plt.rc('axes', labelsize=5)
    plt.rc('xtick', labelsize=5)
    plt.rc('ytick', labelsize=5)


    sorted_ax_feats = [v for v in ax_sorted_scores_dict_nonit.values() if v not in projection_feats] + [v for v in ax_sorted_scores_dict_nonit.values() if v in projection_feats]
    reduced_dotsize = dotsize * 0.15 # Reduce the dotsize, adjust as needed

    col_cter=-1
    row_cter=0
    title_y = 0.89
    for ft in sorted_ax_feats:
        ct+=1
        col_cter+=1
        if col_cter==n_cols:
            col_cter=0
            row_cter+=1
        axe = fig.add_subplot(gs[row_cter, col_cter])
        
        fmost_vals = axon_df[ft]
        fmost_cs = axon_df['predicted_met_type'].map(color_dict)
        
        axe.scatter(fmost_vals,axon_df['soma_aligned_dist_from_pia'],c=fmost_cs,s=reduced_dotsize)
        
        
        if (row_cter==0) & (col_cter == (n_cols-1)):
            legend_handles = []
            for label in sorted(axon_df['predicted_met_type'].unique()):
                this_df = axon_df[axon_df['predicted_met_type']==label]
                ncells = len(this_df)
                hand = Line2D([0], [0], marker='o', color='w', label="{} (n={})".format(label,ncells),
                            markersize=5, markerfacecolor=color_dict[label]) 
                legend_handles.append(hand)


            # Add custom legend
            axe.legend(handles=legend_handles, title='MET-type', loc='upper left',bbox_to_anchor=[1,1],
                    fontsize=5, title_fontsize=5)
            
            
        axe.invert_yaxis()    
        axr.set_yticklabels([])
        if row_cter==0 and col_cter==0:
            
            axe.set_ylabel("Soma Depth (um)")
        else:
            axe.set_yticklabels([])
            
        col = FMOST_LOCAL_COLOR
        if ft in projection_feats:
            col = FMOST_LONG_RANGE_COLOR
            
        ft = MORPH_FEATURE_RELABEL[ft]
        
        if len(ft)>25 and col == FMOST_LONG_RANGE_COLOR:
            print(ft)
            word_split = ft.split(" ")
            first_bit = " ".join(word_split[0:len(word_split)//2])
            second_bit = " ".join(word_split[len(word_split)//2 :])

            ft = first_bit +"\n"+second_bit
            
        axe.set_title(ft,color=col,y=title_y)
        axe.spines['top'].set_visible(False)
        axe.spines['right'].set_visible(False)
        
    plt.tight_layout() 
    fig.set_size_inches(7.5,11)
    
    fig.savefig("plot_SuppFig_SomaDepth_By_Feature_WNM_Axon.pdf",dpi=600,bbox_inches='tight')
    
    plt.close()
    
    #
    #
    #
    #  Sig diff axon features
    #
    #


    it_axon_df = axon_df[axon_df['predicted_met_type'].str.contains("IT")]
    it_types = sorted(it_axon_df.predicted_met_type.unique())

    non_it_axon_df = axon_df[~axon_df['predicted_met_type'].str.contains("IT")]
    non_it_types = sorted(non_it_axon_df.predicted_met_type.unique())

    pval_thresh=0.05
    non_it_axon_df.groupby("predicted_met_type")['complete_axon_total_number_of_targets'].mean()
    master_sig_results = []
    for ft in all_axon_feats:
        True
        
        # ITs
        a = [it_axon_df[it_axon_df['predicted_met_type']==mt][ft].values.tolist() for mt in it_types]
        it_sig_results = []
        if not all([set(i)==set(a[0]) for i in a]):
            ks, kp = stats.kruskal(*a)
            if kp<0.05:

                dunn_df = sp.posthoc_dunn(a,p_adjust ='fdr_bh')
                dunn_df.index=it_types
                dunn_df.columns=it_types
                sig_diff_dunn = dunn_df[dunn_df<pval_thresh]
                sig_diff_dunn = sig_diff_dunn.mask(np.triu(np.ones(sig_diff_dunn.shape, dtype=np.bool_)))

                it_sig_results = list(sig_diff_dunn[sig_diff_dunn.notnull()].stack().index)
        
        if it_sig_results:
            for pair in it_sig_results:
                mt1,mt2 = pair[0],pair[1]
                mt1_mean = it_axon_df[it_axon_df['predicted_met_type']==mt1][ft].mean()
                mt1_std = it_axon_df[it_axon_df['predicted_met_type']==mt1][ft].std()
                
                mt2_mean = it_axon_df[it_axon_df['predicted_met_type']==mt2][ft].mean()
                mt2_std = it_axon_df[it_axon_df['predicted_met_type']==mt2][ft].std()

                other_mets = [m for m in it_types if m not in [mt1,mt2]]
                other_mets_data= {}
                for om in other_mets:
                    om_mean = it_axon_df[it_axon_df['predicted_met_type']==om][ft].mean()
                    om_std = it_axon_df[it_axon_df['predicted_met_type']==om][ft].std()
                    other_mets_data[om]=(om_mean,om_std)
                
                p_val_adj = sig_diff_dunn.loc[mt1,mt2]
                records = {
                    "met_type_1":mt1,
                    "met_type_2":mt2,
                    "feature":MORPH_FEATURE_RELABEL[ft],
                    "p_value_adjusted":p_val_adj,
                    "met_type_1_mean":mt1_mean,
                    "met_type_1_std":mt1_std,
                    "met_type_2_mean":mt2_mean,
                    "met_type_2_std":mt2_std,
                    "other_mets_data":other_mets_data,
                    
                }
                master_sig_results.append(records)
                
                
        
        # NON ITs       
        a = [non_it_axon_df[non_it_axon_df['predicted_met_type']==mt][ft].values.tolist() for mt in non_it_types]
        non_it_sig_results = []
        if not all([set(i)==set(a[0]) for i in a]):
            ks, kp = stats.kruskal(*a)
            if kp<0.05:

                dunn_df = sp.posthoc_dunn(a,p_adjust ='fdr_bh')
                dunn_df.index=non_it_types
                dunn_df.columns=non_it_types
                sig_diff_dunn = dunn_df[dunn_df<pval_thresh]
                sig_diff_dunn = sig_diff_dunn.mask(np.triu(np.ones(sig_diff_dunn.shape, dtype=np.bool_)))

                non_it_sig_results = list(sig_diff_dunn[sig_diff_dunn.notnull()].stack().index)

                
                
                
        if non_it_sig_results:
            for pair in non_it_sig_results:
                mt1,mt2 = pair[0],pair[1]
                mt1_mean = non_it_axon_df[non_it_axon_df['predicted_met_type']==mt1][ft].mean()
                mt1_std = non_it_axon_df[non_it_axon_df['predicted_met_type']==mt1][ft].std()
                
                mt2_mean = non_it_axon_df[non_it_axon_df['predicted_met_type']==mt2][ft].mean()
                mt2_std = non_it_axon_df[non_it_axon_df['predicted_met_type']==mt2][ft].std()

                other_mets = [m for m in non_it_types if m not in [mt1,mt2]]
                other_mets_data= {}
                for om in other_mets:
                    om_mean = non_it_axon_df[non_it_axon_df['predicted_met_type']==om][ft].mean()
                    om_std = non_it_axon_df[non_it_axon_df['predicted_met_type']==om][ft].std()
                    other_mets_data[om] = (om_mean,om_std)
                
                p_val_adj = sig_diff_dunn.loc[mt1,mt2]
                records = {
                    "met_type_1":mt1,
                    "met_type_2":mt2,
                    "feature":MORPH_FEATURE_RELABEL[ft],
                    "p_value_adjusted":p_val_adj,
                    "met_type_1_mean":mt1_mean,
                    "met_type_1_std":mt1_std,
                    "met_type_2_mean":mt2_mean,
                    "met_type_2_std":mt2_std,
                    "other_mets_data":other_mets_data,
                    
                }
                master_sig_results.append(records)
                
                
    sig_diff_resdf = pd.DataFrame.from_records(master_sig_results)
    sig_diff_resdf = sig_diff_resdf.sort_values(by=['met_type_1','p_value_adjusted'],ascending=[True,True])
    sig_diff_resdf.set_index('met_type_1').to_csv("../derived_data/wnm_AllAxonFeatures_SigDiffsByMET.csv")

    #
    #
    #
    # same as above but VISp only
    #
    #
    #
    visp_axon_df = axon_df[axon_df['ccf_soma_location_nolayer']=='VISp']

    it_axon_df = visp_axon_df[visp_axon_df['predicted_met_type'].str.contains("IT")]
    it_types = sorted(it_axon_df.predicted_met_type.unique())

    non_it_axon_df = visp_axon_df[~visp_axon_df['predicted_met_type'].str.contains("IT")]
    non_it_types = sorted(non_it_axon_df.predicted_met_type.unique())

    pval_thresh=0.05

        
    master_sig_results = []
    for ft in all_axon_feats:
        True
        
        # ITs
        a = [it_axon_df[it_axon_df['predicted_met_type']==mt][ft].values.tolist() for mt in it_types]
        it_sig_results = []
        if not all([set(i)==set(a[0]) for i in a]):
            ks, kp = stats.kruskal(*a)
            if kp<0.05:

                dunn_df = sp.posthoc_dunn(a,p_adjust ='fdr_bh')
                dunn_df.index=it_types
                dunn_df.columns=it_types
                sig_diff_dunn = dunn_df[dunn_df<pval_thresh]
                sig_diff_dunn = sig_diff_dunn.mask(np.triu(np.ones(sig_diff_dunn.shape, dtype=np.bool_)))

                it_sig_results = list(sig_diff_dunn[sig_diff_dunn.notnull()].stack().index)
        
        if it_sig_results:
            for pair in it_sig_results:
                mt1,mt2 = pair[0],pair[1]
                mt1_mean = it_axon_df[it_axon_df['predicted_met_type']==mt1][ft].mean()
                mt1_std = it_axon_df[it_axon_df['predicted_met_type']==mt1][ft].std()
                
                mt2_mean = it_axon_df[it_axon_df['predicted_met_type']==mt2][ft].mean()
                mt2_std = it_axon_df[it_axon_df['predicted_met_type']==mt2][ft].std()

                other_met_types = [i for i in it_types if i not in [mt1,mt2]]
                other_met_type_data = {}
                for o_mt in other_met_types:
                    o_mean = it_axon_df[it_axon_df['predicted_met_type']==o_mt][ft].mean()
                    o_std = it_axon_df[it_axon_df['predicted_met_type']==o_mt][ft].std()
                    other_met_type_data[o_mt] = (o_mean,o_std)

                p_val_adj = sig_diff_dunn.loc[mt1,mt2]
                records = {
                    "met_type_1":mt1,
                    "met_type_2":mt2,
                    "feature":MORPH_FEATURE_RELABEL[ft],
                    "p_value_adjusted":p_val_adj,
                    "met_type_1_mean":mt1_mean,
                    "met_type_1_std":mt1_std,
                    "met_type_2_mean":mt2_mean,
                    "met_type_2_std":mt2_std,
                    "other_met_type_data":other_met_type_data,
                }
                
                master_sig_results.append(records)
                
                
        
        # NON ITs       
        a = [non_it_axon_df[non_it_axon_df['predicted_met_type']==mt][ft].values.tolist() for mt in non_it_types]
        non_it_sig_results = []
        if not all([set(i)==set(a[0]) for i in a]):
            ks, kp = stats.kruskal(*a)
            if kp<0.05:

                dunn_df = sp.posthoc_dunn(a,p_adjust ='fdr_bh')
                dunn_df.index=non_it_types
                dunn_df.columns=non_it_types
                sig_diff_dunn = dunn_df[dunn_df<pval_thresh]
                sig_diff_dunn = sig_diff_dunn.mask(np.triu(np.ones(sig_diff_dunn.shape, dtype=np.bool_)))

                non_it_sig_results = list(sig_diff_dunn[sig_diff_dunn.notnull()].stack().index)

                
                
                
        if non_it_sig_results:
            for pair in non_it_sig_results:
                mt1,mt2 = pair[0],pair[1]
                mt1_mean = non_it_axon_df[non_it_axon_df['predicted_met_type']==mt1][ft].mean()
                mt1_std = non_it_axon_df[non_it_axon_df['predicted_met_type']==mt1][ft].std()
                
                mt2_mean = non_it_axon_df[non_it_axon_df['predicted_met_type']==mt2][ft].mean()
                mt2_std = non_it_axon_df[non_it_axon_df['predicted_met_type']==mt2][ft].std()

                other_met_types = [i for i in non_it_types if i not in [mt1,mt2]]
                other_met_type_data = {}
                for o_mt in other_met_types:
                    o_mean = it_axon_df[it_axon_df['predicted_met_type']==o_mt][ft].mean()
                    o_std = it_axon_df[it_axon_df['predicted_met_type']==o_mt][ft].std()
                    other_met_type_data[o_mt] = (o_mean,o_std)

                p_val_adj = sig_diff_dunn.loc[mt1,mt2]
                records = {
                    "met_type_1":mt1,
                    "met_type_2":mt2,
                    "feature":MORPH_FEATURE_RELABEL[ft],
                    "p_value_adjusted":p_val_adj,
                    "met_type_1_mean":mt1_mean,
                    "met_type_1_std":mt1_std,
                    "met_type_2_mean":mt2_mean,
                    "met_type_2_std":mt2_std,
                    "other_met_type_data":other_met_type_data
                }
                master_sig_results.append(records)
                
      
    visp_sig_diff_resdf = pd.DataFrame.from_records(master_sig_results)

    visp_sig_diff_resdf = visp_sig_diff_resdf.sort_values(by=['met_type_1','p_value_adjusted'],ascending=[True,True])
    visp_sig_diff_resdf.set_index('met_type_1').to_csv("../derived_data/wnm_AllAxonFeatures_SigDiffsByMET_VISpOnly.csv")

if __name__ == "__main__":
    main()