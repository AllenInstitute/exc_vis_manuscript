import os
import json
import pandas as pd


from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from collections import defaultdict
import json
import os
import time

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from scipy import stats
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import itertools
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.class_weight import compute_sample_weight

from copy import copy
from sklearn.base import clone

def rfc_stratified(df, ft_cols,labels_col,title, ClassifierObj, clf_params, undersample_dict = {}, num_folds=5,num_outter_its=1,plot=False):
    """
    Will run  labels_colprediction  with stratified kfold to properly include all ttypes in each train/test split

    num folds should  equal 5 if we are doing an n>=5 threshold
    undersample_dict = dicitonary where keys are met-labels and values are the fraction of datapoints to drop
    from that group
    
    """
    df = copy(df)
    shape_0=len(df)
    df.dropna(inplace=True)
    shape_1=len(df)
    
    if shape_1 != shape_0:
        print("Dropped {} Nan Containing Cells".format(shape_0-shape_1))
        
    n_ct_series = df.groupby(labels_col)[labels_col].transform('size').astype(str)
    n_ct_series = ' (n=' + n_ct_series + ')'
#     df[labels_col] = df[labels_col] + n_ct_series.astype(str)
    


    feature_values = df[ft_cols].values
    labels = df[labels_col].values

    min_num_groupsize = min(df[labels_col].value_counts().to_dict().values())
    if min_num_groupsize < 5:
        num_folds = min_num_groupsize
        
    if num_folds < 2:
        num_folds = 2
    print('-----------------------------------------------------------------')
    print('RFC by {}'.format(labels_col))
    print('-----------------------------------------------------------------'+'\n')

    print('There are {} unique labels in the {} cells'.format(len(df[labels_col].unique()),len(labels)))
    print('')
    print(df[labels_col].value_counts())
    print('')

    score = np.zeros(len(ft_cols))
    avg_score = []

    possible_labels = df[labels_col].unique()
    num_correct = dict(zip(possible_labels,np.zeros(len(possible_labels))))
    num_occurances = dict(zip(possible_labels,np.zeros(len(possible_labels))))
    value_counts_dict = dict(zip(df[labels_col].unique(),df[labels_col].value_counts()))
    conf_mat = np.zeros([len(np.unique(labels)),len(np.unique(labels))])

    for it in range(num_outter_its):# range(0,100//num_folds):
        
        #shuffle data, train and test
        skfold = StratifiedKFold(n_splits=num_folds, random_state=None, shuffle=True)

        feature_values = df[ft_cols].values
        labels = df[labels_col].values
        
        if undersample_dict != {}:
            
            train_df = pd.DataFrame(np.hstack((feature_values, labels.reshape(-1,1))),columns=ft_cols+[labels_col])
            for k,pct_to_erod in undersample_dict.items():
                n_drop_from_training = int( len(train_df[train_df[labels_col]==k])*pct_to_erod)
                tr_drop_index = np.random.choice(train_df[train_df[labels_col]==k].index, n_drop_from_training, replace=False, )
                train_df = train_df.loc[~train_df.index.isin(tr_drop_index)]

            feature_values = train_df[ft_cols].values
            labels = train_df[labels_col].values
            
            # print(train_df[labels_col].value_counts())
        
        for train_index, test_index in skfold.split(feature_values, labels):
            X_train = feature_values[train_index]
            Y_train = labels[train_index]

            X_test = feature_values[test_index]
            Y_test = labels[test_index]
            
            if clf_params is None:
                clf = RandomForestClassifier(random_state=0,n_estimators=250,min_samples_leaf=3,
                                             min_samples_split=3,oob_score=True,max_depth = None, class_weight='balanced' )
            else:
                clf = ClassifierObj(**clf_params)
                
            clf.fit(X_train, Y_train)

            results = clf.predict(X_test)

            # getting per class accuracy scores
            for ind,res in enumerate(results):
                num_occurances[Y_test[ind]]+=1
                if res == Y_test[ind]:
                    num_correct[res]+=1           

            conf_mat+=confusion_matrix(Y_test,results,labels = possible_labels)

            mean_score = clf.score(X_test,Y_test)
            avg_score.append(mean_score)
            if hasattr(clf,"feature_importances_"):
                score+=clf.feature_importances_


    Average_performance = np.mean(np.asarray(avg_score))
    Std_performance = np.std(np.asarray(avg_score))
    score_dict = dict(zip(score,ft_cols))
    sorted_scores_dict = {}
    for enum,i in enumerate(sorted(score_dict,reverse=True)):
        sorted_scores_dict[i] = score_dict[i]
        # print(score_dict[i])
    
    for ke in value_counts_dict.keys():
        class_acc = num_correct[ke]/num_occurances[ke]

    row_sums = np.sum(conf_mat,axis=1)
    percent_conf_mat = (conf_mat.T / row_sums).T
    np.nan_to_num(percent_conf_mat,0)

    
    
    use_custom_sort = False
    
    if use_custom_sort:
        subclass_order = ["IT","PT","NP","CT","L6b"]
        sorted_labels = sorted(possible_labels, key=lambda x: (subclass_order.index(x.split("-")[0]),x.split("-")[-1]))
    else:
        sorted_labels = sorted(possible_labels)
    
    confusion_df = pd.DataFrame(percent_conf_mat,columns=possible_labels)
    confusion_df.set_index([possible_labels],inplace=True)
    confusion_df = confusion_df[sorted_labels]
    
    if use_custom_sort:
        new_confusion_df = pd.DataFrame()
        for lab in sorted_labels:
            this_lab = confusion_df.loc[lab]
            new_confusion_df= new_confusion_df.append(this_lab)
        confusion_df = new_confusion_df[sorted_labels]
        
    else:
        confusion_df.sort_index(inplace=True) 

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
        fig = plt.gcf()
        ax = plt.gca()

        con = sns.heatmap(confusion_df,
                        annot=diagonal_vals,
                        fmt = '', 
                        xticklabels=sorted_labels,
                        yticklabels= sorted_labels,
                        vmin=0,vmax=1.0,
                        annot_kws={"size":8})

        ax.set_xticklabels(ax.get_xticklabels(), rotation=90,horizontalalignment='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0,horizontalalignment='right')

        con.set_xlabel('Prediction')
        con.set_ylabel('Truth')

        fig.set_size_inches(8,7)
        confusion_title = title + ' acc = {} +/- {}'.format(round(Average_performance,3),round(Std_performance,3))
        ax.set_title(confusion_title)
    return fig,ax, sorted_scores_dict, confusion_df


def main():
        
    use_precalculated_optimal_params = True
    with open("../data/ScriptArgs.json","r") as f:
        args = json.load(f)


    metadata_file = args['fmost_metadata_file'] 
    local_axon_feature_path = args['fmost_raw_local_axon_feature_file']
    fmost_full_axon_feature_file = args["fmost_entire_axon_features"]
        
    meta_df = pd.read_csv(metadata_file,index_col=0)

    fmost_local_axon_df = pd.read_csv(local_axon_feature_path,index_col=0)
    fmost_local_axon_df.index=fmost_local_axon_df.index.map(lambda x:x+'.swc')

    keep_axon_feat_cols = [c for c in fmost_local_axon_df.columns.tolist() if "axon" in c]
    fmost_local_axon_df = fmost_local_axon_df[keep_axon_feat_cols]

    fmost_local_axon_df = fmost_local_axon_df.rename(columns = {c:c.replace("apical_","") for c in keep_axon_feat_cols})

    fmost_local_axon_feats = fmost_local_axon_df.columns.tolist()

    merged_df = fmost_local_axon_df.merge(meta_df,left_index=True,right_index=True)

    # # merge with dend features
    # fmost_all_features_merged = fmost_merged_df.merge(fmost_axon_feats,left_on='specimen_id',right_index=True)
    # fmost_all_features_merged = fmost_all_features_merged.drop(columns='Dataset')



    # # get whole axon features
    full_axon_df = pd.read_csv(fmost_full_axon_feature_file,index_col=0)
    full_axon_no_projection_feats = full_axon_df.columns.tolist()[:10]
    print("Complete axon morpho features, no projection features:")
    for f in full_axon_no_projection_feats:
        print(f)
    full_axon_df = full_axon_df[full_axon_no_projection_feats]

    fmost_all_axon_features = fmost_local_axon_feats + full_axon_no_projection_feats

    merged_df = full_axon_df.merge(merged_df,left_index=True,right_index=True)
    this_df = merged_df.copy()
    feature_columns = fmost_all_axon_features
    prediction_columns = 'predicted_met_type'
    keepers = [k for k,v in this_df[prediction_columns].value_counts().to_dict().items() if v>4]
    this_df = this_df[this_df[prediction_columns].isin(keepers)]
    this_df = this_df.sample(frac=1)

    if use_precalculated_optimal_params:
        init_best_axon_params = {'n_estimators': 25,
        'min_samples_split': 5,
        'min_samples_leaf': 5,
        'max_features': 'sqrt',
        'max_depth': 30,
        'class_weight': 'balanced',
        'bootstrap': True}
    else:

        X = this_df[feature_columns].fillna(0)
        Y = this_df[prediction_columns]


        # xmean = X.mean()
        # xstd = X.std()
        # X = (X-X.mean())/X.std()

        X_train, X_test, y_train, y_test = train_test_split(X,Y,stratify=Y, test_size=0.2)


        #Random Forest
        rfc_n_estimators = [25, 75, 100, 150]
        rfc_max_features = ['sqrt', None]
        rfc_max_depth = [10, 20, 30, 50]
        rfc_min_samples_split = [2, 3, 5]
        rfc_min_samples_leaf = [1, 2, 5,]
        rfc_bootstrap = [True]
        rfc_class_weight= ['balanced']

        rfc_random_grid = {
            'n_estimators': rfc_n_estimators,
            'max_features': rfc_max_features,
            'max_depth': rfc_max_depth,
            'min_samples_split': rfc_min_samples_split,
            'min_samples_leaf': rfc_min_samples_leaf,
            'bootstrap': rfc_bootstrap,
            'class_weight':rfc_class_weight,
        }

        meta_config_dict = {
            
            
            "RandomForest": {    
                "sklearn_classifier":RandomForestClassifier(),
                "parameter_grid":rfc_random_grid,
            },

        }




        gs_records = {}
        for name, config_dict in meta_config_dict.items():   
            print(name)
            print()
            grid_params = config_dict['parameter_grid']
            this_estimator = config_dict['sklearn_classifier']
            this_gs = RandomizedSearchCV(estimator = this_estimator, 
                                param_distributions = grid_params, 
                                n_iter = 500, 
                                cv = 5, 
                                verbose=10,
                                random_state=42, 
                                return_train_score=True,
                                n_jobs = -1)

            sample_weights = compute_sample_weight(
                    class_weight='balanced',
                    y=y_train,
                )

            

            this_gs.fit(X_train,y_train, sample_weight=sample_weights)
            
            gs_records[name]=this_gs
            

            best_idx = this_gs.best_index_

            best_test_score_mean = this_gs.cv_results_['mean_test_score'][best_idx]
            best_test_score_std = this_gs.cv_results_['std_test_score'][best_idx]

            best_train_score_mean = this_gs.cv_results_['mean_train_score'][best_idx]
            best_train_score_std = this_gs.cv_results_['std_train_score'][best_idx]

            best_clf_validation_acc = this_gs.best_estimator_.score(X_test, y_test)
            
            print("Best Classifier CV Results:\n")
            print("Train Acc: {}+/-{}".format(best_train_score_mean,best_train_score_std))
            print("Test Acc: {}+/-{}".format(best_test_score_mean,best_test_score_std))
            print("Validation Acc: {}".format(best_clf_validation_acc))
            print('\n\n')


            
            gs_df = pd.DataFrame(this_gs.cv_results_)

            gs_df = gs_df[~gs_df['mean_test_score'].isnull()]
            gs_df = gs_df.sort_values(by='mean_test_score')
            plt.plot(range(len(gs_df)), gs_df['mean_train_score'].values,label='train')
            plt.plot(range(len(gs_df)),gs_df['mean_test_score'].values,label='test')
            plt.axhline(best_clf_validation_acc,c='green',label='Best Clf Validation')
            plt.title(name)
            plt.ylabel("Accuracy")
            plt.legend()
            plt.xlabel("GridSearch Epoch")
            # plt.show()
            plt.clf()
            
            
        init_best_axon_params = gs_records['RandomForest'].best_params_




    axon_init_cv_results = rfc_stratified(df=this_df, 
                ft_cols=feature_columns,
                labels_col=prediction_columns,
                title='fMOST All Axon Features Predicted pMET', 
                clf_params=init_best_axon_params, 
                ClassifierObj=RandomForestClassifier,
                undersample_dict = {"L6 CT-2":0.32}, 
                num_folds=5, num_outter_its=100)

    conf_mat = axon_init_cv_results[3]
    sorted_mets = sorted(conf_mat.index, key = lambda x: ([m in x for m in ['IT','L6b','ET','NP','CT']].index(True),x))
    conf_mat = conf_mat.loc[sorted_mets][sorted_mets]


    vals = conf_mat.values
    vals = vals.astype(object)
    for i in range(len(vals)):
        for j in range(len(vals)):
            if i==j:
                vals[i,j] = str(round(vals[i,j],2))
                continue
            else:
                vals[i,j] = ''
    diagonal_vals = vals.astype('str')
    fig,ax=plt.gcf(),plt.gca()
    con = sns.heatmap(conf_mat,
                    annot=diagonal_vals,
                    fmt = '', 
                    xticklabels=sorted_mets,
                    yticklabels= sorted_mets,
                    vmin=0,vmax=1.0,
                    annot_kws={"size":8})

    ax.set_xticklabels(ax.get_xticklabels(), rotation=90,horizontalalignment='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0,horizontalalignment='right')
    ax.set_title("Axon Morphology Predicts Predicted MET-Type")
    con.set_xlabel('Prediction')
    con.set_ylabel('Truth')
    plt.close()
    
    conf_mat.to_csv("../derived_data/wnm_local_axon_predicts_predicted_met_type_confmat.csv")
    

if __name__ == "__main__":
    main()