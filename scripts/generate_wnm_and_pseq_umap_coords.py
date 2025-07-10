import pandas as pd
import umap
import json
import os

def main():
    
    with open("../data/ScriptArgs.json","r") as f:
        args = json.load(f)
        
    fmost_dendrite_feature_file = args['fmost_slice_cropped_chamfer_correct_local_dend_feature_file']  
    metadata_file = args['fmost_metadata_file'] 
    local_axon_feature_path = args['fmost_raw_local_axon_feature_file']
    
    ivscc_met_labels_file = args['pseq_met_labels_file']
    ivscc_raw_feature_file = args['pseq_raw_dend_features']
    pseq_met_labels = pd.read_csv(ivscc_met_labels_file, index_col=0)
    pseq_feature_df = pd.read_csv(ivscc_raw_feature_file, index_col=0)
    ivscc_df = pseq_met_labels.merge(pseq_feature_df, left_index=True, right_index=True)
    
    
    outdir = "./"
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    

    pseq_morph_umap_coords = pd.read_csv(args['patch_seq_morph_umap_locations'],index_col=0)
    pseq_morph_umap_coords = pseq_morph_umap_coords.merge(ivscc_df,left_index=True,right_index=True)
    pseq_morph_umap_met_centers = pseq_morph_umap_coords.groupby('met_type')['x','y'].mean()
    met_centers = {i:[r.x,r.y] for i,r in pseq_morph_umap_met_centers.iterrows()}

    drop_flags = ["area","radius","surface","diameter", "axon", 'specimen_id', 'met_type', 'old_met_type_name']
    drop_feats = [c for c in ivscc_df.columns[1:] if  any([i in c for i in drop_flags])]
    print("Dropping:\n{}".format(drop_feats))   
    feature_columns = [c for c in ivscc_df.columns[1:] if not any([i in c for i in drop_flags])]
    for f in feature_columns:
        print(f)
    ivscc_df['dataset'] = ["ivscc"]*len(ivscc_df)
    ivscc_df["predicted_met_type"] = ivscc_df.met_type


    # load fmost data
    fmost_features = pd.read_csv(fmost_dendrite_feature_file,index_col=0)
    fmost_features['swc_path']=fmost_features['swc_path']+'.swc'

    fmost_labels = pd.read_csv(metadata_file,index_col=0)

    fmost_merged_df = fmost_labels.merge(fmost_features,left_index=True,right_on='swc_path')
    fmost_merged_df['specimen_id']=fmost_merged_df['swc_path']

    fmost_merged_df = fmost_merged_df[['specimen_id','predicted_met_type']+feature_columns].copy()
    fmost_merged_df['dataset']= 'fMOST'

    ivscc_df['specimen_id'] = ivscc_df.index.astype(str)

    ivscc_df_oi = ivscc_df[['specimen_id', 'dataset', 'predicted_met_type']+feature_columns]

    stacked_df = fmost_merged_df.append(ivscc_df_oi)
    stacked_df["seed_x"] = stacked_df['predicted_met_type'].apply(lambda x:met_centers[x][0])
    stacked_df["seed_y"] = stacked_df['predicted_met_type'].apply(lambda x:met_centers[x][1])


    X = (stacked_df[feature_columns]-stacked_df[feature_columns].mean()) / stacked_df[feature_columns].std()

    reducer = umap.UMAP(init =stacked_df[['seed_x','seed_y']].values )
    embedding = reducer.fit_transform(X)

    stacked_df['combined_umap_embedding_1'] = embedding[:,0]
    stacked_df['combined_umap_embedding_2'] = embedding[:,1]

    stacked_keep_cols = [
        'specimen_id',
        'dataset',
        'combined_umap_embedding_1',
        'combined_umap_embedding_2',
    ]
    stacked_df[stacked_keep_cols].set_index('specimen_id').to_csv("../derived_data/pseq_and_wnm_combined_morph_umap.csv")
    
    
    #
    #
    # generate wnm only
    #
    #
    
    fmost_features = pd.read_csv(fmost_dendrite_feature_file,index_col=0)
    fmost_features['swc_path']=fmost_features['swc_path']+'.swc'

    fmost_labels = pd.read_csv(metadata_file,index_col=0)

    fmost_merged_df = fmost_labels.merge(fmost_features,left_index=True,right_on='swc_path')
    fmost_merged_df['specimen_id']=fmost_merged_df['swc_path']

    fmost_merged_df = fmost_merged_df[['specimen_id','predicted_met_type']+feature_columns].copy()
    fmost_merged_df['dataset']= 'fMOST'



    fmost_axon_feats = pd.read_csv(local_axon_feature_path,index_col=0)
    fmost_axon_feats.index=fmost_axon_feats.index.map(lambda x:x+'.swc')

    keep_axon_feat_cols = [c for c in fmost_axon_feats.columns.tolist() if "axon" in c]
    fmost_axon_feats = fmost_axon_feats[keep_axon_feat_cols]

    fmost_axon_feats = fmost_axon_feats.rename(columns = {c:c.replace("apical_","") for c in keep_axon_feat_cols})

    fmost_axon_feature_columns = fmost_axon_feats.columns.tolist()



    fmost_all_features_merged = fmost_merged_df.merge(fmost_axon_feats,left_on='specimen_id',right_index=True)

    fmost_all_features_merged = fmost_all_features_merged.drop(columns='dataset')



    full_axon_feats = pd.read_csv(args["fmost_entire_axon_features"],index_col=0)
    projection_related_full_axon_features = ['complete_axon_total_number_of_targets',
    'complete_axon_total_projection_length',
    'complete_axon_VIS_length',
    'complete_axon_ipsi_VIS_length',
    'complete_axon_length_in_soma_structure',
    'complete_axon_number_of_VIS_targets',
    'complete_axon_number_of_contra_VIS_targets',
    'fraction_of_complete_axon_in_soma_structure']

    full_axon_no_projection_feats = [c for c in full_axon_feats.columns.tolist() if c not in projection_related_full_axon_features]
    full_axon_feats = full_axon_feats[full_axon_no_projection_feats]



    fmost_all_features_merged = fmost_all_features_merged.merge(full_axon_feats,left_on='specimen_id',right_index=True)
    fmost_all_features_merged = fmost_all_features_merged.set_index('specimen_id')


    fmost_all_features_merged["seed_x"] = fmost_all_features_merged['predicted_met_type'].apply(lambda x:met_centers[x][0])
    fmost_all_features_merged["seed_y"] = fmost_all_features_merged['predicted_met_type'].apply(lambda x:met_centers[x][1])


    all_feature_columns = feature_columns + fmost_axon_feature_columns + full_axon_no_projection_feats #fmost_all_features_merged.columns.tolist()[2:-2]


    X_all = fmost_all_features_merged[all_feature_columns]
    X_all = (X_all-X_all.mean()) / X_all.std()

    X_ax = fmost_all_features_merged[fmost_axon_feature_columns+full_axon_no_projection_feats]
    X_ax = (X_ax-X_ax.mean()) / X_ax.std()
    
    X_local_ax = fmost_all_features_merged[fmost_axon_feature_columns]
    X_local_ax = (X_local_ax-X_local_ax.mean()) / X_local_ax.std()

    reducer = umap.UMAP(init =fmost_all_features_merged[['seed_x','seed_y']].values )
    embedding_all = reducer.fit_transform(X_all)

    reducer = umap.UMAP(init =fmost_all_features_merged[['seed_x','seed_y']].values )
    embedding_axon = reducer.fit_transform(X_ax)

    reducer = umap.UMAP(init =fmost_all_features_merged[['seed_x','seed_y']].values )
    embedding_local_axon = reducer.fit_transform(X_local_ax)

    fmost_all_features_merged['local_and_complete_axon_and_dendrite_umap_embedding_1'] = embedding_all[:,0]
    fmost_all_features_merged['local_and_complete_axon_and_dendrite_umap_embedding_2'] = embedding_all[:,1]

    fmost_all_features_merged['local_and_complete_axon_only_umap_embedding_1'] = embedding_axon[:,0]
    fmost_all_features_merged['local_and_complete_axon_only_umap_embedding_2'] = embedding_axon[:,1]

    fmost_all_features_merged['local_axon_only_umap_embedding_1'] = embedding_local_axon[:,0]
    fmost_all_features_merged['local_axon_only_umap_embedding_2'] = embedding_local_axon[:,1]



    keep_cols = [
        'local_and_complete_axon_and_dendrite_umap_embedding_1',
        'local_and_complete_axon_and_dendrite_umap_embedding_2',
        'local_and_complete_axon_only_umap_embedding_1',
        'local_and_complete_axon_only_umap_embedding_2',
        'local_axon_only_umap_embedding_1',
        'local_axon_only_umap_embedding_2',
    ]
    
    fmost_all_features_merged[keep_cols].to_csv("../derived_data/wnm_only_umap_embeddings.csv")


if __name__=="__main__":
    main()