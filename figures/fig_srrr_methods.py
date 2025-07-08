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
from scipy.stats import spearmanr


class FigSparseRrrMethodsParameters(ags.ArgSchema):
    ps_tx_anno_file = ags.fields.InputFile(
        description="Feather file with Patch-seq transcriptomic annotations",
    )
    ps_tx_data_file = ags.fields.InputFile(
        description="Feather file with Patch-seq transcriptomic data",
    )
    inferred_met_type_file = ags.fields.InputFile()
    sparse_rrr_cv_fit_dir = ags.fields.InputDir()
    sparse_rrr_fit_file = ags.fields.InputFile()
    sparse_rrr_feature_file = ags.fields.InputFile()
    sparse_rrr_parameters_file = ags.fields.InputFile()
    output_file = ags.fields.OutputFile()

SUBCLASS_DISPLAY_NAMES = {
    "L23-IT": "L2/3 IT",
    "L4-L5-IT": "L4 & L5 IT",
    "L6-IT": "L6 IT",
    "L5L6-IT-Car3": "L5/L6 IT Car3",
    "L5-ET": "L5 ET",
    "L5-NP": "L5 NP",
    "L6-CT": "L6 CT",
    "L6b": "L6b",
}


def plot_cross_modal_heatmap(ax, sc, h5f, ps_anno_df, ps_tx_data_file):
    gene_lf = {}
    specimen_ids = h5f[sc]["morph"]["specimen_ids"][:]
    for modality in ("ephys", "morph"):
        genes = h5f[sc][modality]["genes"][:]
        genes = np.array([s.decode() for s in genes])
        ps_data_df = pd.read_feather(
            ps_tx_data_file,
            columns=["sample_id"] + genes.tolist())
        ps_data_df.set_index("sample_id", inplace=True)
        sample_ids = ps_anno_df.loc[specimen_ids, "sample_id"]
        gene_data = ps_data_df.loc[sample_ids, genes]
        gene_data = np.log1p(gene_data)

        w = h5f[sc][modality]["w"][:].T

        gene_lf[modality] = gene_data.values @ w
    sp_corrs = spearmanr(gene_lf["ephys"], gene_lf["morph"]).statistic
    if np.isscalar(sp_corrs):
        sp_corrs = np.array([[sp_corrs]])
    else:
        sp_corrs = sp_corrs[:gene_lf["ephys"].shape[1], gene_lf["ephys"].shape[1]:]
        sp_corrs = np.reshape(sp_corrs, (gene_lf["ephys"].shape[1], gene_lf["morph"].shape[1]))
    sns.heatmap(np.abs(sp_corrs).T, square=True, vmin=0, vmax=1, cmap="Reds",
       xticklabels=np.arange(gene_lf["ephys"].shape[1]) + 1,
       yticklabels=np.arange(gene_lf["morph"].shape[1]) + 1,
       ax=ax,
       cbar=False,
    )
    ax.set_ylabel("M-LF", fontsize=6)
    ax.set_xlabel("E-LF", fontsize=6)
    ax.tick_params("both", labelsize=6)
    ax.tick_params("y", rotation=0)
    ax.set_title(SUBCLASS_DISPLAY_NAMES[sc], fontsize=7)


def main(args):
    ps_anno_df = pd.read_feather(args['ps_tx_anno_file'])
    ps_anno_df["spec_id_label"] = pd.to_numeric(ps_anno_df["spec_id_label"])
    ps_anno_df.set_index("spec_id_label", inplace=True)

    example_sc = "L4-L5-IT"
    h5f = h5py.File(os.path.join(args["sparse_rrr_cv_fit_dir"], f"sparse_rrr_cv_{example_sc}.h5"), "r")
    modality = "ephys"

    # Fill in NaNs (dealing with rhdf5 implementation)
    lambdas = h5f[modality]['effect_of_alpha/lambda'][:]
    alphas = h5f[modality]['effect_of_alpha/alpha'][:]
    r2_relaxed = np.nanmean(h5f[modality]['effect_of_alpha/r2_relaxed'][:], axis=(2, 3))


    fig = plt.figure(figsize=(7.5, 10))
    gs = gridspec.GridSpec(
        6, 1,
        height_ratios=(1.75, 0.2, 1.5, 0.2, 0.75, 0.75),
        hspace=0.5,
        wspace=0.4,
    )

    gs_hyper = gridspec.GridSpecFromSubplotSpec(
        1, 2,
        subplot_spec=gs[0],
    )

    ax_alpha = plt.subplot(gs_hyper[0])
    pal = sns.color_palette(n_colors=len(alphas))
    for i, a in enumerate(alphas):
        print(i, a)
        ax_alpha.plot(lambdas[:, i], r2_relaxed[:, i], color=pal[i], label=f"alpha = {a}")
    ax_alpha.set_xscale('log')
    ax_alpha.set_ylim(-0.2, 0.2)
    ax_alpha.legend(fontsize=6, frameon=False, ncols=2, loc="best")
    ax_alpha.set_xlabel("lambda", fontsize=7)
    ax_alpha.set_ylabel(r"CV-$R^2$", fontsize=7)
    ax_alpha.tick_params("both", labelsize=6)
    sns.despine(ax=ax_alpha)

    ax_rank = plt.subplot(gs_hyper[1])
    ranks = h5f[modality]['effect_of_rank']['rank'][:]
    n_ranks = len(ranks)
    pal = sns.color_palette('cubehelix', n_colors=n_ranks)
    for i, r in enumerate(ranks):
        r_key = f"rank_{r}"
        print(r_key)
        lambdas = h5f[modality]["effect_of_rank"][r_key]["lambda"][:]
        r2_relaxed = np.nanmean(h5f[modality]["effect_of_rank"][r_key]['r2_relaxed'][:], axis=(1, 2, 3))
        ax_rank.plot(lambdas, r2_relaxed, color=pal[i], label=f"rank = {r}")
    ax_rank.set_xscale('log')
    ax_rank.set_ylim(-0.2, 0.2)
    ax_rank.legend(fontsize=6, frameon=False, ncols=2, loc="best")
    ax_rank.set_xlabel("lambda", fontsize=7)
    ax_rank.set_ylabel(r"CV-$R^2$", fontsize=7)
    ax_rank.tick_params("both", labelsize=6)
    sns.despine(ax=ax_rank)

    h5f.close()

    # Parameters table
    ax_dummy = plt.subplot(gs[2])
    col_labels = [
        "subclass",
        "modality",
        "n cells",
        "n features",
        "rank",
        "alpha",
        "lambda",
        "n genes",
    ]


    with open(args["sparse_rrr_feature_file"], "r") as f:
        sp_rrr_features_info = json.load(f)
    with open(args["sparse_rrr_parameters_file"], "r") as f:
        sp_rrr_parameters_info = json.load(f)
    h5f = h5py.File(args["sparse_rrr_fit_file"], "r")

    cell_list = []
    n_ephys_lf = {}
    for sc in SUBCLASS_DISPLAY_NAMES.keys():
        for modality in sp_rrr_parameters_info[sc].keys():
            n_cells = len(h5f[sc][modality]['specimen_ids'])
            n_features = len(sp_rrr_features_info[sc][modality])
            rank = sp_rrr_parameters_info[sc][modality]["sparse_rrr"]["rank"]
            alpha = sp_rrr_parameters_info[sc][modality]["sparse_rrr"]["alpha"]
            lambda_val = sp_rrr_parameters_info[sc][modality]["sparse_rrr"]["lambda"]
            n_genes = len(h5f[sc][modality]['genes'])

            if modality == "morph":
                sc_cell = ""
            else:
                n_ephys_lf[sc] = rank
                sc_cell = SUBCLASS_DISPLAY_NAMES[sc]
            row_list = [sc_cell, modality, str(n_cells), str(n_features), str(rank), str(alpha),
                f"{lambda_val:.2f}", str(n_genes)]
            cell_list.append(row_list)

    tab = ax_dummy.table(
        cellText=cell_list,
        colLabels=col_labels,
        loc="center"
    )

    cell_dict = tab.get_celld()
    for j in range(len(col_labels)):
        cell_dict[(0, j)].get_text().set_fontweight("bold")
        cell_dict[(0, j)].visible_edges = "B"
        for i in range(1, len(cell_list) + 1):
            cell_dict[(i, j)].visible_edges = "open"
    tab.auto_set_font_size(False)
    tab.set_fontsize(7)
    ax_dummy.set_axis_off()

    sc_to_plot = list(SUBCLASS_DISPLAY_NAMES.keys())
    sc_top = sc_to_plot[:4]
    sc_bottom = sc_to_plot[4:]

    print(n_ephys_lf)

    gs_crossmodal_top = gridspec.GridSpecFromSubplotSpec(
        1, 4,
        width_ratios=[n_ephys_lf[sc] for sc in sc_top],
        subplot_spec=gs[4],
        wspace=0.5,
    )
    gs_crossmodal_bottom = gridspec.GridSpecFromSubplotSpec(
        1, 4,
        width_ratios=[n_ephys_lf[sc] for sc in sc_bottom],
        subplot_spec=gs[5],
        wspace=0.5,
    )
    for i, sc in enumerate(sc_top):
        print(sc)
        ax = plt.subplot(gs_crossmodal_top[i])
        plot_cross_modal_heatmap(ax, sc, h5f, ps_anno_df, args["ps_tx_data_file"])

    for i, sc in enumerate(sc_bottom):
        print(sc)
        ax = plt.subplot(gs_crossmodal_bottom[i])
        plot_cross_modal_heatmap(ax, sc, h5f, ps_anno_df, args["ps_tx_data_file"])

    norm = plt.Normalize(0, 1)
    ax_cbar = ax.inset_axes([1.05, -0.5, 0.1, 2], transform=ax.transAxes)
    cbar = plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.colormaps["Reds"]),
             cax=ax_cbar, orientation='vertical')
    cbar.set_label("Spearman $r$\n(abs. value)", size=6)
    cbar.outline.set_visible(False)
    ax_cbar.tick_params(axis='y', labelsize=6, length=3)

    plt.savefig(args["output_file"], dpi=300, bbox_inches="tight")

if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=FigSparseRrrMethodsParameters)
    main(module.args)
