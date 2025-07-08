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


class FigSparseRrrLfVsTxPcsParameters(ags.ArgSchema):
    ps_tx_anno_file = ags.fields.InputFile(
        description="Feather file with Patch-seq transcriptomic annotations",
    )
    ps_tx_data_file = ags.fields.InputFile(
        description="Feather file with Patch-seq transcriptomic data",
    )
    inferred_met_type_file = ags.fields.InputFile()
    tx_pc_dir = ags.fields.InputDir()
    sparse_rrr_cv_fit_dir = ags.fields.InputDir()
    sparse_rrr_fit_file = ags.fields.InputFile()
    sparse_rrr_feature_file = ags.fields.InputFile()
    sparse_rrr_parameters_file = ags.fields.InputFile()
    output_file = ags.fields.OutputFile()


SUBCLASS_DISPLAY_NAMES = {
    "L23-IT": "L2/3 IT",
    "L4-L5-IT": "L4 & L5 IT",
    "L6-IT": "L6 IT",
    "L5-ET": "L5 ET",
    "L6-CT": "L6 CT",
    "L6b": "L6b",
}


def plot_ephys_morph_to_tx_corrs(subplot_spec, sc, tx_pc_df, corr_dict):
    gs = gridspec.GridSpecFromSubplotSpec(
        2, 1,
        height_ratios=(corr_dict["ephys"].shape[0], corr_dict["morph"].shape[0]),
        hspace=0.3,
        subplot_spec=subplot_spec,
    )

    ax_ephys = plt.subplot(gs[0])
    sns.heatmap(np.abs(corr_dict["ephys"]), square=True, vmin=0, vmax=1, cmap="Reds",
       xticklabels=np.arange(corr_dict["ephys"].shape[1]) + 1,
       yticklabels=[f"E-LF-{i}" for i in np.arange(corr_dict["ephys"].shape[0]) + 1],
       ax=ax_ephys,
       cbar=False,
    )
    ax_ephys.tick_params("both", labelsize=6)
    ax_ephys.tick_params("y", rotation=0)
    ax_ephys.set_title(SUBCLASS_DISPLAY_NAMES[sc], fontsize=8)

    ax_morph = plt.subplot(gs[1])
    sns.heatmap(np.abs(corr_dict["morph"]), square=True, vmin=0, vmax=1, cmap="Reds",
       xticklabels=np.arange(corr_dict["morph"].shape[1]) + 1,
       yticklabels=[f"M-LF-{i}" for i in np.arange(corr_dict["morph"].shape[0]) + 1],
       ax=ax_morph,
       cbar=False,
    )

    ax_morph.tick_params("both", labelsize=6)
    ax_morph.tick_params("y", rotation=0)
    ax_morph.set_xlabel("Tx PCs", fontsize=6)

    return ax_ephys, ax_morph

def main(args):
    ps_anno_df = pd.read_feather(args['ps_tx_anno_file'])
    ps_anno_df["spec_id_label"] = pd.to_numeric(ps_anno_df["spec_id_label"])
    ps_anno_df.set_index("spec_id_label", inplace=True)


    # figure out which genes we'll eventually be using and only load those
    h5f = h5py.File(args["sparse_rrr_fit_file"], "r")

    with open(args["sparse_rrr_parameters_file"], "r") as f:
        sp_rrr_parameters_info = json.load(f)

    sc_list = list(SUBCLASS_DISPLAY_NAMES.keys())
    sc_top = sc_list[:3]
    sc_bottom = sc_list[3:]

    n_tx_pcs = {}
    corr_dict = {}
    tx_pc_dict = {}
    for sc in sc_list:
        tx_pc_df = pd.read_csv(
            os.path.join(args["tx_pc_dir"], f"{sc}_ps_transformed_pcs.csv"),
            index_col=0,
        )
        tx_pc_dict[sc] = tx_pc_df
        n_tx_pcs[sc] = tx_pc_df.shape[1]
        corr_dict[sc] = {}

        for modality in ("ephys", "morph"):
            specimen_ids = h5f[sc][modality]["specimen_ids"][:]
            genes = h5f[sc][modality]["genes"][:]
            genes = np.array([s.decode() for s in genes])
            ps_data_df = pd.read_feather(
                args["ps_tx_data_file"],
                columns=["sample_id"] + genes.tolist())
            ps_data_df.set_index("sample_id", inplace=True)
            sample_ids = ps_anno_df.loc[specimen_ids, "sample_id"]
            gene_data = ps_data_df.loc[sample_ids, genes]
            gene_data = np.log1p(gene_data)

            w = h5f[sc][modality]["w"][:].T

            gene_lf = gene_data.values @ w

            tx_pc_vals = tx_pc_df.loc[specimen_ids, :].values
            sp_corrs = spearmanr(tx_pc_vals, gene_lf).statistic
            sp_corrs = sp_corrs[tx_pc_vals.shape[1]:, :tx_pc_vals.shape[1]]
            print(sc, modality, n_tx_pcs[sc], sp_corrs.shape)
            corr_dict[sc][modality] = sp_corrs

    fig = plt.figure(figsize=(7.5, 9))
    gs = gridspec.GridSpec(
        3, 1,
        height_ratios=(3, 3, 1),
        hspace=0.5,
    )

    gs_top = gridspec.GridSpecFromSubplotSpec(
        1, 3,
        subplot_spec=gs[0],
        wspace=0.6,
        width_ratios=[n_tx_pcs[sc] for sc in sc_top],
    )
    for i, sc in enumerate(sc_top):
        print(sc)
        ax_ephys, ax_morph = plot_ephys_morph_to_tx_corrs(
            gs_top[i], sc, tx_pc_dict[sc], corr_dict[sc])

    gs_bottom = gridspec.GridSpecFromSubplotSpec(
        1, 3,
        subplot_spec=gs[1],
        wspace=0.6,
        width_ratios=[n_tx_pcs[sc] for sc in sc_bottom],
    )
    for i, sc in enumerate(sc_bottom):
        print(sc)
        ax_ephys, ax_morph = plot_ephys_morph_to_tx_corrs(
            gs_bottom[i], sc, tx_pc_dict[sc], corr_dict[sc])

    norm = plt.Normalize(0, 1)
    ax_cbar = ax_ephys.inset_axes([1.05, -0.25, 0.1, 1.5], transform=ax_ephys.transAxes)
    cbar = plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.colormaps["Reds"]),
             cax=ax_cbar, orientation='vertical')
    cbar.set_label("Spearman $r$\n(abs. value)", size=7)
    cbar.outline.set_visible(False)
    ax_cbar.tick_params(axis='y', labelsize=6, length=3)

    gs_r2 = gridspec.GridSpecFromSubplotSpec(
        1, 2,
        subplot_spec=gs[2],
        wspace=0.4,
    )
    values_for_df = []
    for sc in sc_list:
        for modality in ("ephys", "morph"):
            values_for_df.append((
                sc,
                modality,
                sp_rrr_parameters_info[sc][modality]["sparse_rrr"]["r2_relaxed"],
                sp_rrr_parameters_info[sc][modality]["tx_pc_elasticnet"]["r2"],
            ))
    r2_df = pd.DataFrame(values_for_df, columns=("subclass", "modality", "srrr_r2", "tx_pc_r2"))

    pal = sns.color_palette(n_colors=2)
    x = np.arange(len(sc_list))
    width=0.25

    ax_ephys = plt.subplot(gs_r2[0])
    r2_df_e = r2_df.loc[r2_df["modality"] == "ephys", :]
    ax_ephys.bar(x, r2_df_e["srrr_r2"], width, facecolor=pal[0], edgecolor=pal[0])
    ax_ephys.bar(x + width, r2_df_e["tx_pc_r2"], width, facecolor="white", edgecolor=pal[0])
    ax_ephys.set_title("electrophysiology", fontsize=8, loc="left")
    ax_ephys.set_ylabel(r"CV-$R^2$", fontsize=7)
    ax_ephys.set_xticks(x + width / 2)
    ax_ephys.set_xticklabels([SUBCLASS_DISPLAY_NAMES[sc] for sc in sc_list])
    ax_ephys.tick_params("both", labelsize=6)
    sns.despine(ax=ax_ephys)

    ax_morph = plt.subplot(gs_r2[1])
    r2_df_m = r2_df.loc[r2_df["modality"] == "morph", :]
    ax_morph.bar(x, r2_df_m["srrr_r2"], width, facecolor=pal[1], edgecolor=pal[1])
    ax_morph.bar(x + width, r2_df_m["tx_pc_r2"], width, facecolor="white", edgecolor=pal[1])
    ax_morph.set_title("morphology", fontsize=8, loc="left")
    ax_morph.set_ylabel(r"CV-$R^2$", fontsize=7)
    ax_morph.set_xticks(x + width / 2)
    ax_morph.set_xticklabels([SUBCLASS_DISPLAY_NAMES[sc] for sc in sc_list])
    ax_morph.tick_params("both", labelsize=6)
    sns.despine(ax=ax_morph)

    plt.savefig(args["output_file"], dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=FigSparseRrrLfVsTxPcsParameters)
    main(module.args)

