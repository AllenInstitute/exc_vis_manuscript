import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import seaborn as sns
import argschema as ags
from matplotlib.collections import PolyCollection
from scipy.stats import spearmanr
import mouse_met_figs.simple_sankey as sankey


class FigRespGeneParameters(ags.ArgSchema):
    ref_tx_anno_file = ags.fields.InputFile()
    ref_tx_data_file = ags.fields.InputFile()
    ps_tx_anno_file = ags.fields.InputFile(
        description="Feather file with Patch-seq transcriptomic annotations",
    )
    ps_tx_data_file = ags.fields.InputFile(
        description="Feather file with Patch-seq transcriptomic data",
    )
    resp_umap_file = ags.fields.InputFile(
        default="../derived_data/ref_exc_umap_coordinates.csv",
    )
    common_umap_file = ags.fields.InputFile()
    resp_pc_weight_file = ags.fields.InputFile()
    resp_pc_means_file = ags.fields.InputFile()
    genes_to_plot = ags.fields.List(
        ags.fields.String,
        default=["Tsnax", "Nrn1", "Nhp2l1"],
        cli_as_single_argument=True,
    )
    inf_met_type_file = ags.fields.InputFile(
        description="csv file with inferred met type text labels",
    )
    ref_pc_dir = ags.fields.InputDir()
    ps_transformed_pc_dir = ags.fields.InputDir(
        description="directory with transformed patch-seq tx pca files",
    )
    lfc_file = ags.fields.InputFile()
    recluster_file = ags.fields.InputFile()
    output_file = ags.fields.OutputFile()

MET_TO_SUBCLASS_MAP = {
	"L2/3 IT": "L2/3 IT",
	"L4 IT": "L4 & L5 IT",
	"L4/L5 IT": "L4 & L5 IT",
	"L5 IT-1": "L4 & L5 IT",
	"L5 IT-2": "L4 & L5 IT",
	"L5 IT-3 Pld5": "L4 & L5 IT",
	"L6 IT-1": "L6 IT",
	"L6 IT-2": "L6 IT",
	"L6 IT-3": "L6 IT",
	"L5/L6 IT Car3": "L6 IT Car3",
	"L5 ET-1 Chrna6": "L5 ET",
	"L5 ET-2": "L5 ET",
	"L5 ET-3": "L5 ET",
	"L5 NP": "L5 NP",
	"L6 CT-1": "L6 CT",
	"L6 CT-2": "L6 CT",
	"L6b": "L6b",
}

REF_SC_TO_SUBCLASS_MAP = {
    "L2/3 IT": "L2/3 IT",
    "L4": "L4 & L5 IT",
    "L5 IT": "L4 & L5 IT",
    "L6 IT": "L6 IT",
    "L5 PT": "L5 ET",
    "NP": "L5 NP",
    "L6 CT": "L6 CT",
    "L6b": "L6b",
}

SUBCLASS_COLORS = {
    'L2/3 IT': '#94D9A1',
    'L4 & L5 IT': '#008A61',
    'L6 IT': '#A19922',
    'L6 IT Car3': '#5100FF',
    'L5 ET': '#0D5B78',
    'L5 NP': '#3E9E64',
    'L6 CT': '#2D8CB8',
    'L6b': '#25596D',
}

SUBCLASS_ORDER = [
    'L2/3 IT',
    'L4 & L5 IT',
    'L6 IT',
    'L6 IT Car3',
    'L5 ET',
    'L5 NP',
    'L6 CT',
    'L6b',
]

TTYPE_TO_SUBCLASS = {
    'L2/3 IT VISp Agmat': "L2/3 IT",
    'L2/3 IT VISp Rrad': "L2/3 IT",
    'L2/3 IT VISp Adamts2': "L2/3 IT",
    'L4 IT VISp Rspo1': "L4 & L5 IT",
    'L5 IT VISp Hsd11b1 Endou': "L4 & L5 IT",
    'L5 IT VISp Whrn Tox2': "L4 & L5 IT",
    'L5 IT VISp Col6a1 Fezf2': "L4 & L5 IT",
    'L5 IT VISp Batf3': "L4 & L5 IT",
    'L5 IT VISp Col27a1': "L4 & L5 IT",
    'L6 IT VISp Col18a1': "L6 IT",
    'L6 IT VISp Car3': "L6 IT Car3",
    'L6 IT VISp Penk Col27a1': "L6 IT",
    'L6 IT VISp Penk Fst': "L6 IT",
    'L6 IT VISp Col23a1 Adamts2': "L6 IT",
    'L5 PT VISp Chrna6': "L5 ET",
    'L5 PT VISp C1ql2 Cdh13': "L5 ET",
    'L5 PT VISp C1ql2 Ptgfr': "L5 ET",
    'L5 PT VISp Krt80': "L5 ET",
    'L5 PT VISp Lgr5': "L5 ET",
    'L5 NP VISp Trhr Met': "L5 NP",
    'L5 NP VISp Trhr Cpne7': "L5 NP",
    'L6 CT VISp Ctxn3 Brinp3': "L6 CT",
    'L6 CT VISp Ctxn3 Sla': "L6 CT",
    'L6 CT VISp Nxph2 Wls': "L6 CT",
    'L6 CT VISp Krt80 Sla': "L6 CT",
    'L6 CT VISp Gpr139': "L6 CT",
    'L6b VISp Mup5': "L6b",
    'L6b Col8a1 Rprm': "L6b",
    'L6b VISp Col8a1 Rxfp1': "L6b",
    'L6b VISp Crh': "L6b",
    'L6b P2ry12': "L6b",
}


def main(args):

    ps_anno_df = pd.read_feather(args['ps_tx_anno_file'])
    ps_anno_df["spec_id_label"] = pd.to_numeric(ps_anno_df["spec_id_label"])
    ps_anno_df.set_index("spec_id_label", inplace=True)

    ref_anno_df = pd.read_feather(args['ref_tx_anno_file']).set_index("sample_id")
    ttype_colors = dict(zip(ref_anno_df['cluster_label'], ref_anno_df['cluster_color']))
    ttype_ids = dict(zip(ref_anno_df['cluster_label'], ref_anno_df['cluster_id']))

    inf_met_type_df = pd.read_csv(args['inf_met_type_file'], index_col=0)
    inf_met_type_df["inferred_met_type"] = [t.replace("PT-", "ET-") if type(t) is str else t for t in inf_met_type_df["inferred_met_type"]]

    ref_umap_df = pd.read_csv(args["resp_umap_file"], index_col=0)
    common_umap_df = pd.read_csv(args["common_umap_file"], index_col=0)
    resp_pc_weight_df = pd.read_csv(args["resp_pc_weight_file"], index_col=0)
    resp_pc_mean_df = pd.read_csv(args["resp_pc_means_file"], index_col=0)

    recluster_df = pd.read_csv(args['recluster_file'], index_col=0)

    lfc_df = pd.read_csv(args['lfc_file'], index_col=0)



    # Load gene data
    genes_to_load = resp_pc_weight_df.index.tolist()
    genes_to_plot = args['genes_to_plot']
    for g in genes_to_plot:
        if g not in genes_to_load:
            genes_to_load.append(g)

    ps_data_df = pd.read_feather(args['ps_tx_data_file'],
        columns=['sample_id'] + genes_to_load).set_index("sample_id")
    ref_data_df = pd.read_feather(args['ref_tx_data_file'],
        columns=['sample_id'] + genes_to_load).set_index("sample_id")

    # log transform
    ps_data_df = (1 + ps_data_df).apply(np.log2)
    ref_data_df = (1 + ref_data_df).apply(np.log2)

    print(ref_data_df.head())

    resp_pc_vals_patchseq = (
        (ps_data_df.loc[:, resp_pc_weight_df.index].values - resp_pc_mean_df['mean'].values) @
        resp_pc_weight_df["0"].values.reshape(-1, 1)
    )
    ps_spec_ids = ps_anno_df.reset_index().set_index("sample_id").loc[ps_data_df.index, "spec_id_label"].values
    resp_pc_patchseq_df = pd.Series(np.squeeze(resp_pc_vals_patchseq), index=ps_spec_ids)

    # Make data frame for violin plots

    ps_specimen_ids = inf_met_type_df.index.values
    ps_sample_ids = ps_anno_df.loc[ps_specimen_ids, "sample_id"].values
    ref_sample_ids = ref_umap_df.index.values

    ps_data_df = ps_data_df.loc[ps_sample_ids, :]
    ps_data_df['dataset'] = 'patch-seq'
    ps_subclass = inf_met_type_df.loc[ps_specimen_ids, "inferred_met_type"].map(MET_TO_SUBCLASS_MAP)
    ps_data_df['subclass'] = ps_subclass.values

    ref_data_df = ref_data_df.loc[ref_sample_ids, :]
    ref_data_df['dataset'] = 'facs'
    ref_subclass = ref_anno_df.loc[ref_data_df.index, "subclass_label"].map(REF_SC_TO_SUBCLASS_MAP)
    # fix Car3
    ref_subclass[ref_anno_df['cluster_label'] == 'L6 IT VISp Car3'] = "L6 IT Car3"
    ref_data_df['subclass'] = ref_subclass.values

    violin_df = pd.concat([ps_data_df, ref_data_df])



    fig = plt.figure(figsize=(7, 9))

    gs_rows = gridspec.GridSpec(6, 1,
        height_ratios=(1, 0.25, 0.02, 0.5, 0.02, 1.5),
        hspace=0.5,
    )

    gs_plots = gridspec.GridSpecFromSubplotSpec(4, 5, subplot_spec=gs_rows[0],
        width_ratios=(1, 1, 0.05, 0.3, 1), wspace=0.5, hspace=0.3)

    for sc in SUBCLASS_ORDER:
        mask = ref_data_df.loc[ref_umap_df.index, "subclass"] == sc
        print(sc)
        print(ref_umap_df.loc[mask, ["x", "y"]].mean(axis=0))

    umap_sc_ax = plt.subplot(gs_plots[:, 0])
    common_ids = common_umap_df.index.intersection(ref_umap_df.index)
    umap_subclass_colors = ref_data_df.loc[common_ids, "subclass"].map(SUBCLASS_COLORS).values
    umap_sc_ax.scatter(
        common_umap_df.loc[common_ids, 'x'][~pd.isnull(umap_subclass_colors)],
        common_umap_df.loc[common_ids, 'y'][~pd.isnull(umap_subclass_colors)],
        s=2,
        edgecolors='none',
        c=umap_subclass_colors[~pd.isnull(umap_subclass_colors)],
    )
    umap_sc_ax.set_aspect('equal')
    sns.despine(ax=umap_sc_ax, left=True, bottom=True)
    umap_sc_ax.set_xticks([])
    umap_sc_ax.set_yticks([])
    umap_sc_ax.annotate("",
        xy=(-12, -5), xycoords='data',
        xytext=(-5, -12), textcoords='data',
        arrowprops=dict(arrowstyle="<|-|>", connectionstyle="angle,angleA=180,angleB=-90,rad=0", color='k', shrinkA=0, shrinkB=0),
        annotation_clip=False,
        )
    umap_sc_ax.text(-14, -8.5, "t-UMAP-2", ha='center', va='center', fontsize=5, rotation=90)
    umap_sc_ax.text(-8.5, -14, "t-UMAP-1", ha='center', va='center', fontsize=5)
    umap_sc_ax.set_title("reference FACS data", fontsize=7)

    annot_coords = {
        'L2/3 IT': (4, -13), #
        'L4 & L5 IT': (-8, 1.5), #
        'L6 IT': (-7, 7), #
        'L6 IT Car3': (-13, 14), #
        'L5 ET': (9, 8), #
        'L5 NP': (21, 14), #
        'L6 CT': (21, -1.5), #
        'L6b': (15, -11), #
    }
    for k, coords in annot_coords.items():
        umap_sc_ax.text(coords[0], coords[1],
            k, fontsize=6, color=SUBCLASS_COLORS[k], va='center', ha='center')

    umap_pc_ax = plt.subplot(gs_plots[:, 1])
    cbar_ax = plt.subplot(gs_plots[1:3, 2])
    sc = umap_pc_ax.scatter(
        common_umap_df.loc[common_ids, 'x'][~pd.isnull(umap_subclass_colors)],
        common_umap_df.loc[common_ids, 'y'][~pd.isnull(umap_subclass_colors)],
        s=2,
        edgecolors='none',
        c=ref_umap_df.loc[common_ids, 'resp_pc_1'].values,
        cmap='inferno',
    )
    umap_pc_ax.set_aspect('equal')
    sns.despine(ax=umap_pc_ax, left=True, bottom=True)
    umap_pc_ax.set_xticks([])
    umap_pc_ax.set_yticks([])
    cb = fig.colorbar(sc, cax=cbar_ax)
    cbar_ax.tick_params(axis='y', labelsize=6)
    cb.outline.set_visible(False)

    umap_pc_ax.annotate("",
        xy=(-12, -5), xycoords='data',
        xytext=(-5, -12), textcoords='data',
        arrowprops=dict(arrowstyle="<|-|>", connectionstyle="angle,angleA=180,angleB=-90,rad=0", color='k', shrinkA=0, shrinkB=0),
        annotation_clip=False,
        )
    umap_pc_ax.text(-14, -8.5, "t-UMAP-2", ha='center', va='center', fontsize=5, rotation=90)
    umap_pc_ax.text(-8.5, -14, "t-UMAP-1", ha='center', va='center', fontsize=5)

    umap_pc_ax.set_title("response gene PC-1", fontsize=7)

    # L2/3 IT
    filepart = "L23-IT"
    ref_pc_dir = args['ref_pc_dir']
    ps_transformed_pc_dir = args['ps_transformed_pc_dir']
    ref_pc_transformed = pd.read_csv(os.path.join(ref_pc_dir, f"{filepart}_tx_pca_transformed.csv"), index_col=0)
    pc_ps_transformed_df = pd.read_csv(os.path.join(ps_transformed_pc_dir, f"{filepart}_ps_transformed_pcs.csv"), index_col=0)

    pc1_scatter_ax = plt.subplot(gs_plots[:2, 4])
    pc2_scatter_ax = plt.subplot(gs_plots[2:, 4])
    for ax, pc_name in zip((pc1_scatter_ax, pc2_scatter_ax), ("PC1", "PC2")):
        sns.regplot(
            x=ref_umap_df.loc[ref_pc_transformed.index, "resp_pc_1"],
            y=ref_pc_transformed[pc_name],
            ci=None,
            scatter_kws=dict(
                edgecolors='white',
                s=4,
            ),
            line_kws=dict(color='black', lw=1),
            ax=ax,
        )
        ax.scatter(
            ref_umap_df.loc[ref_pc_transformed.index, "resp_pc_1"],
            ref_pc_transformed[pc_name],
            c=ref_anno_df.loc[ref_pc_transformed.index, "cluster_color"].values,
            s=4,
            edgecolors='white',
            lw=0.25,
        )
        ax.tick_params(axis='both', labelsize=6)

        if pc_name == "PC1":
            ax.set_xlabel("")
            ax.set_xticklabels([])
            ttypes = [
                'L2/3 IT VISp Agmat',
                'L2/3 IT VISp Rrad',
                'L2/3 IT VISp Adamts2',
            ]
            for i, ttype in enumerate(ttypes):
                ax.text(0.6, 0.25 - 0.1 * i, ttype, color=ttype_colors[ttype],
                    transform=ax.transAxes,
                    fontsize=6, va='baseline', ha='left')
        else:
            ax.set_xlabel("response gene PC-1", fontsize=6)
        ax.set_ylabel(f"L2/3 IT Tx {pc_name[:2] + '-' + pc_name[-1]}\n(FACS data)", fontsize=6)
        sns.despine(ax=ax)

        result = spearmanr(ref_umap_df.loc[ref_pc_transformed.index, "resp_pc_1"], ref_pc_transformed[pc_name])
        print("spearman r L2/3 it", pc_name)
        print(result)
        ax.text(
            0.95, 0.7,
            r"$r_\mathrm{s}$" + f" = {result[0]:.2f}",
            transform=ax.transAxes,
            fontsize=6,
        )

    # Correlations with response gene PC
    fileparts = {
        "L2/3 IT": "L23-IT",
        "L4 & L5 IT": "L4-L5-IT",
        "L6 IT": "L6-IT",
        "L6 IT Car3": "L5L6-IT-Car3",
        "L5 ET": "L5-ET",
        "L5 NP": "L5-NP",
        "L6 CT": "L6-CT",
        "L6b": "L6b",
    }

    corrs_for_plot = []
    labels = []
    label_pos = []
    sc_pos = []
    highest = 0
    for k, fp in fileparts.items():
        ref_pc_transformed = pd.read_csv(os.path.join(ref_pc_dir, f"{fp}_tx_pca_transformed.csv"), index_col=0)
        corr, pval = spearmanr(ref_umap_df.loc[ref_pc_transformed.index, "resp_pc_1"], ref_pc_transformed)
        corr_to_add = np.concatenate([corr[0, 1:], [np.nan]])
        corrs_for_plot.append(corr_to_add)

        labels += [str(i + 1) for i in range(ref_pc_transformed.shape[1])] + [""]
        this_label_pos = np.arange(ref_pc_transformed.shape[1] + 1) + highest
        print(this_label_pos, np.mean(this_label_pos))
        this_sc_pos = np.mean(this_label_pos)
        label_pos.append(this_label_pos)
        sc_pos.append(this_sc_pos)
        highest = this_label_pos[-1] + 1

    plot_corr = np.hstack(corrs_for_plot)
    gs_heatmap = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_rows[1],
        width_ratios=(1, 0.01), wspace=0.05
    )
    ax = plt.subplot(gs_heatmap[0])
    cbar_ax = plt.subplot(gs_heatmap[1])
    sns.heatmap(
        plot_corr.reshape(1, -1),
        vmin=-1,
        vmax=1,
        cmap='RdYlBu_r',
        square=True,
        xticklabels=labels,
        yticklabels=False,
        ax=ax,
        cbar_ax=cbar_ax,
        cbar_kws=dict(orientation="vertical")
    )
    ax.tick_params(axis='x', labelsize=5)
    for i, l in enumerate(labels):
        if l == "":
            ax.xaxis.get_major_ticks()[i].tick1line.set_markersize(0)

    ax.set_title("Correlations between Tx-PCs and response gene PC-1", loc='left', fontsize=6)
    for pos, sc_l in zip(sc_pos, fileparts.keys()):
        print(sc_l, pos)
        ax.text(pos, -2,
            sc_l,
            fontsize=6,
            va='center',
            ha='center',
            transform=ax.get_xaxis_transform())
    cbar_ax.tick_params(axis='y', labelsize=5)
    cbar_ax.set_ylabel("Spearman's r", fontsize=6)

    # Reclustering river plot
    recluster_df['orig'] = ref_anno_df.loc[recluster_df.index, "cluster_label"].values
    river_ax = plt.subplot(gs_rows[3])
    for v in recluster_df["x"].unique():
        ttype_colors[str(v)] = 'gray'

    sankey.sankey(
        recluster_df["x"].values.astype(str),
        recluster_df["orig"].values,
        rightLabels=sorted(recluster_df["orig"].unique().tolist(), key=lambda x: ttype_ids[x]),
        show_left_label=False,
        ax=river_ax,
        rearrange=True,
        orientation="horizontal",
        fontsize=4,
        rightColor=True,
        colorDict=ttype_colors,
        aspect=5,
        text_offset=100,
    )

    river_ax.text(
        0.05, 0.5, "reference FACS data\noriginal clusters",
        va='baseline', ha='right', fontsize=6,
        transform=river_ax.transAxes,
    )
    river_ax.text(
        0.05, 0.1, "reclustered after removing\nresponse gene signal",
        va='top', ha='right', fontsize=6,
        transform=river_ax.transAxes,
    )
    river_ax.annotate(
        "merged",
        xy=(0.135, 0.15), xycoords='axes fraction',
        xytext=(0.135, -0.1), textcoords='axes fraction',
        arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3", linewidth=0.5, shrinkA=0, shrinkB=1, color='black'),
        ha='center',
        fontsize=5,
    )
    river_ax.annotate(
        "merged",
        xy=(0.57, 0.15), xycoords='axes fraction',
        xytext=(0.57, -0.1), textcoords='axes fraction',
        arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3", linewidth=0.5, shrinkA=0, shrinkB=1, color='black'),
        ha='center',
        fontsize=5,
    )

    y_lim = river_ax.get_ylim()
    y_delt = y_lim[1] - y_lim[0]
    river_ax.set_ylim(y_lim[0] - 0.5 * y_delt, y_lim[1] + 1.75 * y_delt)

    # DE responses genes - FACS vs Patch-seq
    n_per_dir = 4
    gs_de = gridspec.GridSpecFromSubplotSpec(n_per_dir * 2, 2, subplot_spec=gs_rows[5],
        width_ratios=(2, 3), wspace=0.3, hspace=0.4)

    ref_resp_sc_means = (
        ref_data_df
        .drop(columns=["dataset"])
        .groupby("subclass")
        .mean()
        .reset_index()
        .melt(id_vars=["subclass"], var_name="gene")
        .set_index(["subclass", "gene"])
    )
    ps_resp_sc_means = (
        ps_data_df
        .drop(columns=["dataset"])
        .groupby("subclass")
        .mean()
        .reset_index()
        .melt(id_vars=["subclass"], var_name="gene")
        .set_index(["subclass", "gene"])
    )
    # take top & bottom by cross-group mean LFC
    lfc_sort_mean = lfc_df.mean(axis=1).sort_values()
    print(lfc_sort_mean)
    genes_to_plot = lfc_sort_mean.index[-n_per_dir:].tolist()[::-1] + lfc_sort_mean.index[:n_per_dir].tolist()
    up_genes = lfc_sort_mean[lfc_sort_mean > 0].index.tolist()[::-1]
    down_genes = lfc_sort_mean[lfc_sort_mean < 0].index.tolist()

    up_ax = plt.subplot(gs_de[:n_per_dir, 0])
    down_ax = plt.subplot(gs_de[n_per_dir:, 0])

    up_ax.scatter(
        ref_resp_sc_means['value'],
        ps_resp_sc_means.loc[ref_resp_sc_means.index, 'value'],
        color='#cccccc',
        s=2,
        edgecolors='none',
    )
    pal = sns.color_palette('tab20c', n_colors=len(up_genes))
    for i, g in enumerate(up_genes):
        up_ax.scatter(
            ref_resp_sc_means['value'][ref_resp_sc_means.reset_index()['gene'].values == g],
            ps_resp_sc_means.loc[ref_resp_sc_means.index, 'value'][ref_resp_sc_means.reset_index()['gene'].values == g],
            color=pal[i],
            s=4,
            edgecolors='white',
            linewidth=0.25,
            label=g,
        )
    up_ax.set_aspect('equal')
    up_ax.set(anchor="W")
    up_ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=4, ncols=1, frameon=False)
    down_ax.scatter(
        ref_resp_sc_means['value'],
        ps_resp_sc_means.loc[ref_resp_sc_means.index, 'value'],
        color='#cccccc',
        s=2,
        edgecolors='none',
    )
    up_ax.tick_params(axis='y', labelsize=6)
    up_ax.set_yticks([0, 5, 10])
    up_ax.set_xticks([0, 5, 10])
    up_ax.set_xticklabels([])
    up_ax.set_ylabel("Patch-seq\nmean " + r"$\log_2$(CPM + 1)", fontsize=6)
    up_ax.text(0.65, 0.2, "lower in\nPatch-seq", transform=up_ax.transAxes, fontsize=6, va='center', ha='left')
    up_ax.set_title("average expression of\nresponse genes by subclass", fontsize=6)
    sns.despine(ax=up_ax)

    pal = sns.color_palette('tab20b', n_colors=len(down_genes))
    for i, g in enumerate(down_genes):
        down_ax.scatter(
            ref_resp_sc_means['value'][ref_resp_sc_means.reset_index()['gene'].values == g],
            ps_resp_sc_means.loc[ref_resp_sc_means.index, 'value'][ref_resp_sc_means.reset_index()['gene'].values == g],
            color=pal[i],
            s=4,
            edgecolors='white',
            linewidth=0.25,
            label=g,
        )
    down_ax.set_aspect('equal')
    down_ax.set(anchor="W")
    down_ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=4, ncols=1, frameon=False)
    down_ax.tick_params(axis='both', labelsize=6)
    down_ax.set_ylabel("Patch-seq\nmean " + r"$\log_2$(CPM + 1)", fontsize=6)
    down_ax.set_xlabel("FACS\nmean " + r"$\log_2$(CPM + 1)", fontsize=6)
    down_ax.set_yticks([0, 5, 10])
    down_ax.set_xticks([0, 5, 10])
    down_ax.text(0.05, 0.85, "higher in\nPatch-seq", transform=down_ax.transAxes, fontsize=6, va='center', ha='left')
    sns.despine(ax=down_ax)


    for i, gene in enumerate(genes_to_plot):
        print(gene, i)
        ax = plt.subplot(gs_de[i, 1])

        sns.violinplot(
            data=violin_df,
            x="subclass",
            y=gene,
            order=SUBCLASS_ORDER,
            cut=0,
            hue="dataset",
            hue_order=['facs', 'patch-seq'],
            split=True,
            density_norm='width',
            linewidth=0.5,
            inner=None,
            ax=ax,
        )
        sns.pointplot(
            data=violin_df,
            x="subclass",
            y=gene,
            order=SUBCLASS_ORDER,
            estimator='median',
            errorbar=None,
            hue="dataset",
            hue_order=['facs', 'patch-seq'],
            palette={'facs': 'k', 'patch-seq': 'k'},
            linestyle='none',
            dodge=0.2,
            markersize=1,
            ax=ax,
        )

        # Recolor violins
        for ind, violin in enumerate(ax.findobj(PolyCollection)):
            rgb = colors.to_rgb(SUBCLASS_COLORS[SUBCLASS_ORDER[ind // 2]])
            if ind % 2 != 0:
                rgb = 0.5 + 0.5 * np.array(rgb)  # make whiter
            violin.set_facecolor(rgb)

        ax.get_legend().remove()

        sns.despine(ax=ax)
        if i < len(genes_to_plot) - 1:
            ax.set_xticklabels([])
        else:
            ax.tick_params(axis='x', labelsize=6)
        ax.tick_params(axis='y', labelsize=6)
        ax.set_ylabel(gene, fontsize=6, rotation=0, va='center', ha='right', style='italic')
        ax.set_xlabel("")

        if i == 0:
            # annotate left/right
            ax.annotate("FACS",
                    xy=(-0.1, 8), xycoords='data',
                    xytext=(0.6, 15), textcoords='data',
                    arrowprops=dict(arrowstyle="-", connectionstyle="angle,angleA=180,angleB=-90,rad=2", linewidth=0.5),
                    fontsize=5,
                    annotation_clip=False)
            ax.annotate("Patch-seq",
                    xy=(0.1, 8), xycoords='data',
                    xytext=(0.6, 11), textcoords='data',
                    arrowprops=dict(arrowstyle="-", connectionstyle="angle,angleA=180,angleB=-90,rad=2", linewidth=0.5),
                    fontsize=5,
                    annotation_clip=False)
            ax.set_title(r"values in $\log_2$(CPM + 1)", loc='right', fontsize=6)

    # Panel labels
    ax.text(
        gs_plots.get_grid_positions(fig)[2][0] - 0.04,
        gs_plots.get_grid_positions(fig)[1][0],
        "a", transform=fig.transFigure,
        fontsize=16, fontweight="bold", va="baseline", ha="right")
    ax.text(
        gs_plots.get_grid_positions(fig)[2][4] - 0.04,
        gs_plots.get_grid_positions(fig)[1][0],
        "b", transform=fig.transFigure,
        fontsize=16, fontweight="bold", va="baseline", ha="right")
    ax.text(
        gs_heatmap.get_grid_positions(fig)[2][0] - 0.04,
        gs_heatmap.get_grid_positions(fig)[1][0],
        "c", transform=fig.transFigure,
        fontsize=16, fontweight="bold", va="baseline", ha="right")
    ax.text(
        gs_rows.get_grid_positions(fig)[2][0] - 0.04,
        gs_rows.get_grid_positions(fig)[1][3],
        "d", transform=fig.transFigure,
        fontsize=16, fontweight="bold", va="baseline", ha="right")
    ax.text(
        gs_de.get_grid_positions(fig)[2][0] - 0.04,
        gs_de.get_grid_positions(fig)[1][0],
        "e", transform=fig.transFigure,
        fontsize=16, fontweight="bold", va="baseline", ha="right")
    ax.text(
        gs_de.get_grid_positions(fig)[2][1] - 0.04,
        gs_de.get_grid_positions(fig)[1][0],
        "f", transform=fig.transFigure,
        fontsize=16, fontweight="bold", va="baseline", ha="right")


    plt.savefig(args['output_file'], bbox_inches='tight', dpi=300)



if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=FigRespGeneParameters)
    main(module.args)
