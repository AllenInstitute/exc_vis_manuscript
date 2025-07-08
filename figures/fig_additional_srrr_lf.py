import os
import h5py
import json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import argschema as ags
import latent_factor_plot as lfp
import fig_sparse_rrr


class FigAdditionalSparseRrrLatentFactorsParameters(ags.ArgSchema):
    ps_tx_anno_file = ags.fields.InputFile(
        description="Feather file with Patch-seq transcriptomic annotations",
    )
    ps_tx_data_file = ags.fields.InputFile(
        description="Feather file with Patch-seq transcriptomic data",
    )
    inferred_met_type_file = ags.fields.InputFile()
    sparse_rrr_fit_file = ags.fields.InputFile()
    sparse_rrr_feature_file = ags.fields.InputFile()
    sparse_rrr_parameters_file = ags.fields.InputFile()
    morph_file = ags.fields.InputFile(
        description="csv file with unnormalized morph features",
    )
    ephys_file = ags.fields.InputFile(
        description="csv file with sPCA ephys features",
    )
    ephys_features_file = ags.fields.InputFile(
        description="csv file with traditional ephys features",
    )
    spca_feature_info_file = ags.fields.InputFile(
        description="json file with spca component information"
    )
    output_file = ags.fields.OutputFile()

MET_TYPE_COLORS = {
    'L2/3 IT': '#7AE6AB',
    'L4 IT': '#00979D',
    'L4/L5 IT': '#00DDC5',
    'L5 IT-1': '#00A809',
    'L5 IT-2': '#00FF00',
    'L5 IT-3 Pld5': '#26BF64',
    'L6 IT-1': '#C2E32C',
    'L6 IT-2': '#96E32C',
    'L6 IT-3': '#A19922',
    'L5/L6 IT Car3': '#5100FF',
    'L5 ET-1 Chrna6': '#0000FF',
    'L5 ET-2': '#22737F',
    'L5 ET-3': '#29E043',
    'L5 NP': '#73CA95',
    'L6 CT-1': '#74CAFF',
    'L6 CT-2': '#578EBF',
    'L6b': '#2B7880',
}



def main(args):
    ps_anno_df = pd.read_feather(args['ps_tx_anno_file'])
    ps_anno_df["spec_id_label"] = pd.to_numeric(ps_anno_df["spec_id_label"])
    ps_anno_df.set_index("spec_id_label", inplace=True)

    # figure out which genes we'll eventually be using and only load those
    h5f = h5py.File(args["sparse_rrr_fit_file"], "r")
    all_genes_list = []
    for k in h5f.keys():
        for m in h5f[k].keys():
            genes = h5f[k][m]["genes"][:]
            genes = [s.decode() for s in genes]
            all_genes_list.append(genes)
    all_genes = np.unique(np.concatenate(all_genes_list))
    print(f"Loading {len(all_genes)} genes")
    col_to_load = ["sample_id"] + all_genes.tolist()
    ps_data_df = pd.read_feather(
        args['ps_tx_data_file'], columns=col_to_load).set_index("sample_id")
    ps_data_df = np.log1p(ps_data_df)

    inf_met_df = pd.read_csv(args["inferred_met_type_file"], index_col=0)

    ephys_df = pd.read_csv(args["ephys_file"], index_col=0)
    morph_df = pd.read_csv(args["morph_file"], index_col=0)
    feature_df_dict = {
        "ephys": ephys_df,
        "morph": morph_df,
    }

    prefix_dict = {
        "ephys": "E",
        "morph": "M",
    }

    with open(args["sparse_rrr_feature_file"], "r") as f:
        sp_rrr_features_info = json.load(f)

    with open(args["sparse_rrr_parameters_file"], "r") as f:
        sp_rrr_parameters_info = json.load(f)

    with open(args["spca_feature_info_file"], "r") as f:
        spca_info = json.load(f)
    ind_counter = 0
    spca_info_dict = {}
    for si in spca_info:
        for i in si['indices']:
            spca_info_dict[si['key'] + "_" + str(i)] = ind_counter
            ind_counter += 1

    filepart_corr_radius = {
        "L5-ET": 3.5,
    }

    fig = plt.figure(figsize=(7.5, 9))
    gs = gridspec.GridSpec(
        4, 2,
        hspace=0.35,
        wspace=0.4,
    )
    plot_info_list = [
        ("L4-L5-IT", "ephys", (2, 3), gs[0, 0]),
        ("L4-L5-IT", "ephys", (2, 4), gs[0, 1]),
        ("L6-IT", "ephys", (2, 3), gs[1, 0]),
        ("L5-ET", "ephys", (2, 3), gs[1, 1]),
        ("L5-ET", "ephys", (4, 5), gs[2, 0]),
        ("L6-CT", "ephys", (0, 2), gs[2, 1]),
        ("L6-CT", "morph", (0, 2), gs[3, 0]),
        ("L6b", "ephys", (0, 2), gs[3, 1]),
    ]

    for info in plot_info_list:
        sc, modality, indices, gs_sub = info
        print(sc, modality, indices)
        specimen_ids, genes, w, v = fig_sparse_rrr.get_sp_rrr_fit_info(
            sc, modality, h5f
        )
        cell_colors = [MET_TYPE_COLORS[t] for t in inf_met_df.loc[specimen_ids, "inferred_met_type"]]
        feature_data = fig_sparse_rrr.select_and_normalize_feature_data(
            specimen_ids, sc, modality,
            sp_rrr_features_info, feature_df_dict[modality])
        sample_ids = ps_anno_df.loc[specimen_ids, "sample_id"]
        gene_data = ps_data_df.loc[sample_ids, genes]
        if sc in filepart_corr_radius:
            corr_radius = filepart_corr_radius[sc]
        else:
            corr_radius = 2.7

        lfp.plot_side_by_side_latent_factors(
            w, v,
            gene_data, feature_data,
            gs_sub, prefix_dict[modality],
            corr_radius=corr_radius,
            n_features_to_plot=3, n_genes_to_plot=3,
            x_index=indices[0], y_index=indices[1],
            scatter_colors=cell_colors)

    plt.savefig(args["output_file"], dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=FigAdditionalSparseRrrLatentFactorsParameters)
    main(module.args)

