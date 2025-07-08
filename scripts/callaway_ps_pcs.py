import numpy as np
import pandas as pd
import argschema as ags
from sklearn.decomposition import PCA


class CallawayPcParameters(ags.ArgSchema):
    call_anno_file = ags.fields.InputFile(
        description='excel file with kim et al. 2020 annotations',
    )
    call_logcpm_data_file = ags.fields.InputFile(
        description="log-transformed CPM data for Kim et al. 2020 data set",
    )
    call_mapping_file = ags.fields.InputFile(
        description="cell type mappings of kim et al. 2020 data",
    )
    call_edger_file = ags.fields.InputFile(
        description="excel file with Zinbwave-EdgeR DE genes",
    )
    call_deseq2_file = ags.fields.InputFile(
        description="excel file with Zinbwave-EdgeR DE genes",
    )
    ps_tx_anno_file = ags.fields.InputFile(
        description="Feather file with Patch-seq transcriptomic annotations",
    )
    ps_tx_data_file = ags.fields.InputFile(
        description="Feather file with Patch-seq transcriptomic data",
    )
    inf_met_type_file = ags.fields.InputFile(
        description="csv file with inferred met type text labels",
    )
    met_types_to_include = ags.fields.List(
        ags.fields.String,
        default=["L2/3 IT"],
    )
    output_file = ags.fields.OutputFile(
    )


def read_xlsx_with_shifted_column_names(fp):
    ### Looks like the Callaway supplementary xlsx files were saved with column names shifted over by one
    df = pd.read_excel(fp, index_col=0)
    df.index.names = ['index']
    df.columns = ["GeneID"] + [df.columns[i] for i in range(len(df.columns)-1)]
    return df


def rename_callaway_gene(gene):
    gene_name_dict = {
        '2019-03-01 00:00:00':'March1',
        '2019-03-02 00:00:00':'March2',
        '2019-03-04 00:00:00':'March4',
        '2019-03-05 00:00:00':'March5',
        '2019-03-06 00:00:00':'March6',
        '2019-03-07 00:00:00':'March7',
        '2019-03-08 00:00:00':'March8',
        '2019-03-10 00:00:00':'March10',
        '2019-03-11 00:00:00':'March11',
        '2019-09-02 00:00:00':"Sep2",
        '2019-09-03 00:00:00':"Sep3",
        '2019-09-04 00:00:00':"Sep4",
        '2019-09-05 00:00:00':"Sep5",
        '2019-09-06 00:00:00':"Sep6",
        '2019-09-07 00:00:00':"Sep7",
        '2019-09-08 00:00:00':"Sep8",
        '2019-09-09 00:00:00':"Sep9",
        '2019-09-11 00:00:00':"Sep11",
        '2019-09-15 00:00:00':"Sep15"
    }

    if gene in gene_name_dict.keys():
        gene_new = gene_name_dict[gene]
    elif "RIK" in gene:
        if gene == "GRIK1":
            gene_new = "Grik1"
        else:
            gene_new = gene.upper()
            gene_new = gene_new.replace("RIK", "Rik")
    else:
        gene_new = gene.capitalize()
    if gene_new == 'H2-ke6':
        gene_new = 'H2-Ke6'
    if gene_new == 'D17wsu92e':
        gene_new = 'D17Wsu92e'

    return gene_new


def diff_gene_mask(df, logfc_thresh, adjp_thresh):
    """

    Parameters
    ----------
        df ([type]): [description]
        logfc_thresh ([type]): minimum log-fold change for inclusion of differentially expressed gene
        adjp_thresh ([type]): minimum adjusted p-value for inclusion of differentially expressed gene

    Returns
    -------
        [type]: [description]
    """
    mask1 = (df["log2FoldChange"] < -logfc_thresh) & (df["padj"] < adjp_thresh)
    mask2 = (df["log2FoldChange"] > logfc_thresh) & (df["padj"] < adjp_thresh)

    return df[mask1], df[mask2]


def main(args):
    # Load data


    print("loading patch-seq tx data")
    tx_anno_df = pd.read_feather(args['ps_tx_anno_file'])
    tx_anno_df["spec_id_label"] = pd.to_numeric(tx_anno_df["spec_id_label"])
    tx_anno_df.set_index("spec_id_label", inplace=True)
    tx_data_df = pd.read_feather(args['ps_tx_data_file']).set_index("sample_id")
    ps_gene_list = tx_data_df.columns

    inf_met_type_df = pd.read_csv(args['inf_met_type_file'], index_col=0)

    # Process Callaway data

    # Load Callaway annotations
    call_anno_df = pd.read_excel(args['call_anno_file'], index_col="Sample")
    call_anno_df["Projection_AIname"] = call_anno_df["Projection"].apply(lambda x: F"VIS{x.lower()}")
    call_anno_df["FinalType"] = call_anno_df["FinalType"].apply(lambda x :x.replace("_"," "))

    # Load Callaway mappings to VISp taxonomy
    call_map_df = pd.read_csv(args['call_mapping_file'], index_col=0)
    call_map_df.loc[:,"simple_v1_first_cl"] = call_map_df["v1_first_cl"].str.replace(" VISp ", " ")

    joined_call_anno_df = call_anno_df.join(call_map_df[["simple_v1_first_cl", "v1_first_cl", "v1_call"]])
    joined_call_anno_df["annotatedType"] = joined_call_anno_df.apply(
        lambda x: x['FinalType'].split(' ')[0]
        if x['Type'] == x['FinalType']
        else F"{x['FinalType'].split(' ')[0]} (reassigned from {x['Type'].split(' ')[0]})",
        axis=1)
    joined_call_anno_df["annotatedType"].replace(
        'L56 (reassigned from L5)','L56',
        inplace=True)
    joined_call_anno_df["annotatedMapping"] = joined_call_anno_df.apply(
        lambda x: F"{x['v1_first_cl']} ({x['v1_call']})", axis=1)

    # Other parameters and constants
    Callaway_seq_runs = ['R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R12']
    layer = "L23"
    call_filter = ['AL','PM']
    drop_groups = [
        'L23 (reassigned from L5)',
        'L56 (reassigned from L23)',
        'L56',
        'Glia (reassigned from L23)',
        'Glia (reassigned from L5)',
        'L4 (reassigned from L5)'
        #'L4 (reassigned from L23)'
    ]

    filt_call_anno_df = joined_call_anno_df[
        (~joined_call_anno_df["annotatedType"].isin(drop_groups)) &
        (joined_call_anno_df["v1_call"] != "PoorQ") &
        (joined_call_anno_df["SeqRun"].isin(Callaway_seq_runs))
    ]

    filt_call_ids = filt_call_anno_df.index.tolist()
    print(F"{len(filt_call_ids)}")
    # Also drop one cell that maps I2 to a L5 type (and had been re-assigned by
    # Callaway lab as a L4. Seems a bit fishy)
    remove_cell_ids = ['T41_T35.18_S212_L006_2306839.0']
    filt_call_ids = [i for i in filt_call_ids if i not in remove_cell_ids]
    filt_call_anno_df = filt_call_anno_df.loc[filt_call_ids,:]
    print(F"Removing {len(remove_cell_ids)} extra cell(s): {remove_cell_ids}")
    print(F"Keeping {len(filt_call_ids)} {layer} Callaway cells from original {joined_call_anno_df.shape[0]}")

    # Check balance of Callaway dataset
    print(F"\nHow balanced is the Callaway AL/PM data?"),
    print(filt_call_anno_df["Projection_AIname"].value_counts(normalize=True).round(2))


    # Load Callaway gene data
    print("loading kim et al. 2020 tx data")
    call_log_cpm_data_df = pd.read_csv(args['call_logcpm_data_file'], index_col=0)
    call_log_cpm_data_df = call_log_cpm_data_df.T
    call_log_cpm_data_df.columns = list(map(lambda x: rename_callaway_gene(x), call_log_cpm_data_df.columns))
    call_log_cpm_data_df = call_log_cpm_data_df.loc[filt_call_ids,:]
    print(F"Callaway data shape: {call_log_cpm_data_df.shape}")

    # Callaway gene parameters
    logfc_threshold = 1.
    adjp_threshold = 0.05

    # Callaway genes have a different naming convention.
    # Rename genes and see how many are the same in our dataset
    # Load in the files
    edger_df = read_xlsx_with_shifted_column_names(args['call_edger_file'])
    edger_df.set_index("Symbol", drop=False, inplace=True)
    edger_df.rename(columns={"logFC":"log2FoldChange", "padjFilter":"padj"}, inplace=True)

    deseq2_df = read_xlsx_with_shifted_column_names(args['call_deseq2_file'])
    deseq2_df.set_index("Symbol", drop=False, inplace=True)

    # Rename genes to match Patch-seq and FACS
    edger_df.index = list(map(lambda x: rename_callaway_gene(str(x)), edger_df.index))
    deseq2_df.index = list(map(lambda x: rename_callaway_gene(str(x)), deseq2_df.index))

    # Get sets of differentially expressed genes from each method
    d1_EdgeR, d2_EdgeR = diff_gene_mask(df=edger_df, logfc_thresh=logfc_threshold, adjp_thresh=adjp_threshold)
    print(F"{d1_EdgeR.shape[0]} + {d2_EdgeR.shape[0]} = {d1_EdgeR.shape[0] + d2_EdgeR.shape[0]} most differentially expressed genes in AL/PM projecting {layer} cells by Zinbwave-EdgeR")

    d1_DESeq2, d2_DESeq2 = diff_gene_mask(df=deseq2_df, logfc_thresh=logfc_threshold, adjp_thresh=adjp_threshold)
    print(F"{d1_DESeq2.shape[0]} + {d2_DESeq2.shape[0]} = {d1_DESeq2.shape[0] + d2_DESeq2.shape[0]} most differentially expressed genes in AL/PM projecting {layer} cells by Zinbwave-DESeq2")

    attr = list(set(d1_EdgeR.index.tolist() + d2_EdgeR.index.tolist() + d1_DESeq2.index.tolist() + d2_DESeq2.index.tolist()))
    print(F"Total set of {len(attr)} differentially expressed genes")

    features = [c for c in attr if c in ps_gene_list]
    missing_attr = [c for c in attr if c not in ps_gene_list]
    print(F"{len(features)} out of {len(attr)} DE genes matched in Patch-seq data.\nNo match found for {len(missing_attr)} genes:\n")
    print(missing_attr)

    # Perform PCA on projection DE genes
    X = call_log_cpm_data_df.loc[filt_call_ids, features]
    y = filt_call_anno_df.loc[filt_call_ids,"Projection"].tolist()
    n_pcs = 20

    print(F"Running PCA to reduce full data of shape {X.shape} to {n_pcs} PCs")
    pca_full = PCA(n_components=n_pcs, random_state=0)
    transformed = pca_full.fit_transform(X)
    X_transformed = pd.DataFrame(transformed, index=filt_call_ids)
    X_transformed.columns = list(map(lambda x: F"PC{x+1}",X_transformed.columns))
    print(F"Number of features after PCA: {X_transformed.shape[1]}")
    print(F"Total explained variance for {n_pcs} components: {sum(pca_full.explained_variance_ratio_).round(2)}")


    # Select only IT-MET-1 cells
    ps_ids_to_use = inf_met_type_df.loc[
        inf_met_type_df['inferred_met_type'].isin(args['met_types_to_include']),
        :].index.values
    print(f"applying PCs to {len(ps_ids_to_use)} patch-seq cells")
    ps_samples_to_use = tx_anno_df.loc[ps_ids_to_use, "sample_id"].values
    tx_data_df = tx_data_df.loc[ps_samples_to_use, :]

    # log-transform patch-seq data
    tx_data_df = np.log2(tx_data_df + 1)
    X_ps = tx_data_df.loc[:, features]
    X_ps_transformed = pd.DataFrame(
            data=pca_full.transform(X_ps),
            index=ps_ids_to_use,
        )
    X_ps_transformed.columns = [f"PC{x + 1}" for x in X_ps_transformed.columns]

    print("saving results")
    X_ps_transformed.to_csv(args['output_file'])


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=CallawayPcParameters)
    main(module.args)


