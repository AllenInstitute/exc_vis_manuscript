#!/usr/bin/env python

import numpy as np
import pandas as pd
import feather
import umap
import argschema as ags
import mouse_met_figs.utils as utils
import warnings
warnings.filterwarnings('ignore')

class CoEmbeddedTranscriptomicUmapParameters(ags.ArgSchema):
    tx_data_file_1 = ags.fields.InputFile(
        description="feather file with transcriptomic data")
    tx_data_file_2 = ags.fields.InputFile(
        description="feather file with transcriptomic data")
    anno_file_1 = ags.fields.InputFile(
        description="feather file with reference transcriptomic annotations (optional)",
        required=False,
        allow_none=True,
        default=None)
    anno_file_2 = ags.fields.InputFile(
        description="feather file with reference transcriptomic annotations (optional)",
        required=False,
        allow_none=True,
        default=None)
    facs_anno_file = ags.fields.InputFile(
        description="feather file with reference FACS transcriptomic annotations")
    cluster_label_1 = ags.fields.String(
        description="column in anno_file to use for transcriptomic label")
    cluster_label_2 = ags.fields.String(
        description="column in anno_file to use for transcriptomic label")
    genes_file = ags.fields.InputFile(
        description="csv with genes to use and optional beta scores")
    specimens_file_1 = ags.fields.InputFile(
        description="txt file with specimens to use (optional)",
        required=False,
        allow_none=True,
        default=None)
    specimens_file_2 = ags.fields.InputFile(
        description="txt file with specimens to use (optional)",
        required=False,
        allow_none=True,
        default=None)
    beta_threshold = ags.fields.Float(
        default=0.425,
        allow_none=True,
        description="minimum beta score for gene inclusion")
    log_data = ags.fields.Boolean(
        default=True,
        description="indicates whether to take log2(CPM + 1)")
    n_pcs = ags.fields.Integer(
        default=None,
        allow_none=True,
        description="number of principal components for PCA (or None if no PCA before UMAP)")
    scale_features = ags.fields.Boolean(
        default=False,
        description="indicates whether to apply StandardScaler to features before PCA. Defaults to False on Zizhen's suggestion that some genes should be allowed to vary more.")
    corr_threshold = ags.fields.Float(
        default=None,
        allow_none=True,
        description="maximum Pearson PC to dataset label correlation coefficient for PC inclusion in final embedding")
    min_dist = ags.fields.Float(
        default=0.1,
        allow_none=True,
        description="min_dist parameter for UMAP")
    n_neighbors = ags.fields.Integer(
        default=15,
        allow_none=True,
        description="n_neighbors parameter for UMAP")
    required_tree_calls_1 = ags.fields.List(ags.fields.String,
        default=['Core','I1','I2','I3'],
        allow_none=True,
        description="tree call categories to restrict data (optional)",
        cli_as_single_argument=True)
    required_tree_calls_2 = ags.fields.List(ags.fields.String,
        default=None,
        allow_none=True,
        description="tree call categories to restrict data (optional)",
        cli_as_single_argument=True)
    required_subclasses_1 = ags.fields.List(ags.fields.String, 
        default=['L2/3 IT','L4','L5 IT','L6 IT','L5 PT','NP','L6 CT','L6b'],
        allow_none=True,
        description="transcriptomic subclasses to restrict data (optional)",
        cli_as_single_argument=True)
    required_subclasses_2 = ags.fields.List(ags.fields.String,
        default=['L2/3 IT','L4','L5 IT','L6 IT','L5 PT','NP','L6 CT','L6b'],
        allow_none=True,
        description="transcriptomic subclasses to restrict data (optional)",
        cli_as_single_argument=True)
    subsample_ratio = ags.fields.Float(
        description="ratio for subsampling larger (FACS) data in figure (ex. 1.5x Patch-seq data)",
        allow_none=True,
        default=None)
    output_file = ags.fields.OutputFile(description="CSV with UMAP coordinates")

def get_technical_bias_pcs(scores_df, dataset_1_ix, dataset_2_ix, n_pcs, corr_threshold):
    """[summary]
    Parameters
    ----------
        scores_df (Pandas DataFrame): combined PC scores from datasets 1 and 2
        dataset_1_ix (list): cell specimen IDs for dataset 1 (ex Patch-seq)
        dataset_2_ix (list): cell specimen IDs for dataset 2 (ex FACS)
        corr_threshold (float): maximum Pearson PC to dataset label correlation coefficient for PC inclusion in final embedding     
    
    Returns
    -------
        technical_bias_pcs (list): PCs to remove from scores_df
        correlations_series (pandas Series): PC correlations with dataset
    """

    from scipy import stats

    # Binarize label        
    patchseq_label =  list(map(lambda x: 0 if x in dataset_2_ix else (1 if x in dataset_1_ix else None), scores_df.index))
    correlations = list(map(lambda x: stats.pearsonr(scores_df.loc[:, x].values, patchseq_label)[0], np.arange(0,n_pcs,1)))

    # Calculate Pearson correlation coefficient
    correlations_series = pd.Series(data=correlations, index=scores_df.columns)
    correlations_series = correlations_series.abs().sort_values(ascending=False)
    
    # Create a list of PCs to drop, in order
    technical_bias_pcs = correlations_series[correlations_series >= corr_threshold].index.values.tolist()

    return technical_bias_pcs, correlations_series

def main(tx_data_file_1, tx_data_file_2, anno_file_1, anno_file_2, facs_anno_file, cluster_label_1, cluster_label_2, genes_file, specimens_file_1, specimens_file_2, beta_threshold, log_data, n_pcs, scale_features, corr_threshold, min_dist, n_neighbors, required_tree_calls_1, required_tree_calls_2, required_subclasses_1, required_subclasses_2, subsample_ratio, output_file, **kwargs):
    
    # Load gene info
    genes_df = pd.read_csv(genes_file, index_col=0)
    if "BetaScore" in genes_df.columns:
        genes_df = genes_df.loc[genes_df["BetaScore"] > beta_threshold, :]
    ngenes = genes_df.shape[0]
    print("Using {} genes".format(ngenes))
    tx_data_columns = ["sample_id"] + genes_df.iloc[:,0].tolist()

    # Create t-type to subclass dict
    facs_anno_df = feather.read_dataframe(facs_anno_file)
    subclass_dict = utils.dict_from_facs(facs_anno_df, key_column="cluster_label", value_column="subclass_label")

    # Get data for each dataset
    datasets = []
    for i, (tx_data_file, anno_file, cluster_label, specimens_file, 
                required_tree_calls, required_subclasses) in enumerate(zip(
                (tx_data_file_1, tx_data_file_2), 
                (anno_file_1, anno_file_2), 
                (cluster_label_1, cluster_label_2), 
                (specimens_file_1, specimens_file_2),
                (required_tree_calls_1, required_tree_calls_2), 
                (required_subclasses_1, required_subclasses_2))):

        print(F"Dataset {i}: {required_subclasses} subclasses and {required_tree_calls}")

        # Load transcriptomic gene data for dataset
        tx_data = pd.read_feather(tx_data_file, columns=tx_data_columns)
        tx_data.set_index("sample_id", inplace=True)
        if log_data == True:
            tx_data_raw = tx_data.copy()
            tx_data = tx_data.applymap(lambda x: np.log2(x+1))

        # Load annotation data for dataset
        tx_anno_df = feather.read_dataframe(anno_file)
        tx_anno_df.loc[:,"subclass"] = tx_anno_df[cluster_label].apply(lambda x: subclass_dict[x] if x in subclass_dict.keys() else "None")

        # Filter data if necessary with tree call, subclass and specimen list masks
        tree_mask, subclass_mask, specimens_mask = utils.get_tx_masks(tx_anno_df, specimens_file, required_tree_calls, required_subclasses)
        tx_mask = tree_mask & subclass_mask & specimens_mask
        print(F"Dataset {i} transcriptomic mask sum: {sum(tx_mask)}")
        data = tx_data.loc[tx_mask, :]
        datasets.append(data)

    # Combine datasets
    combined_df = pd.concat(datasets, axis=0)

    # Reduce dimensionality if n_pcs is specified
    # Save PCA results
    if n_pcs is not None:
        print(F"Running PCA to reduce dimensionality to {n_pcs}")
        scores, explained_variances, loadings = utils.reduce_tx_dims(n_pcs, combined_df, scale_features=scale_features)
        scores_df = pd.DataFrame(scores, index=combined_df.index)
        
        # Remove PCs based on correlation with binarized dataset label
        if corr_threshold is not None:
            technical_bias_pcs, correlations = get_technical_bias_pcs(scores_df, datasets[0].index, datasets[1].index, n_pcs, corr_threshold)
            print("Pearson correlation with Patch-seq label:\n"+F"{correlations}"+"\n")
            print("Technical bias PCs:\n"+F"{technical_bias_pcs}"+"\n")
            for i, pc in enumerate(technical_bias_pcs):
                print(F"Removing PC {pc} with {correlations.loc[pc]} correlation coefficent")
                scores_df.drop(columns=pc, inplace=True)
                print(F"Remaining columns: {scores_df.columns}")
        
        final_df = scores_df
    else:
        final_df = combined_df

    # Perform UMAP embedding on final_data 
    print(F"Using data with shape {final_df.shape}, min_dist={min_dist} and n_neighbors={n_neighbors} for UMAP embedding")
    embedding = umap.UMAP(random_state=42, min_dist=min_dist, n_neighbors=n_neighbors).fit_transform(final_df.values)
    out_df = pd.DataFrame(
        {
            "x": embedding[:, 0],
            "y": embedding[:, 1]
        },
        index=final_df.index,
        )
    out_df.to_csv(output_file)

if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=CoEmbeddedTranscriptomicUmapParameters)
    main(**module.args)