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
from matplotlib.patches import Patch
from adjustText import adjust_text


class FigTxPcSupplementParameters(ags.ArgSchema):
    ref_tx_anno_file = ags.fields.InputFile()
    ps_tx_anno_file = ags.fields.InputFile(
        description="Feather file with Patch-seq transcriptomic annotations",
    )
    ps_tx_data_file = ags.fields.InputFile(
        description="Feather file with Patch-seq transcriptomic data",
    )
    ref_tx_pc_dir = ags.fields.InputDir()
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


def main(args):
    # Load annotations
    tx_anno_df = pd.read_feather(args['ref_tx_anno_file'])
    tx_anno_df["cluster_label"] = [t.replace("PT", "ET") if type(t) is str else t for t in tx_anno_df["cluster_label"]]
    tx_anno_df.set_index("sample_id", inplace=True)
    ttype_colors = dict(zip(tx_anno_df["cluster_label"], tx_anno_df["cluster_color"]))
    ttype_ids = dict(zip(tx_anno_df["cluster_label"], tx_anno_df["cluster_id"]))

    fig = plt.figure(figsize=(7.5, 10))
    gs = gridspec.GridSpec(
        8, 5,
        wspace=0.4,
        hspace=0.4,
        width_ratios=(2, 1, 1, 1, 1),
    )
    n_genes_per_pc = 4

    sc_index = 0
    for sc, sc_name in SUBCLASS_DISPLAY_NAMES.items():
        print(sc)
        ref_tx_pc_df = pd.read_csv(os.path.join(args["ref_tx_pc_dir"], f"{sc}_tx_pca_transformed.csv"), index_col=0)
        clusters = tx_anno_df.loc[ref_tx_pc_df.index, "cluster_label"]
        cluster_colors = [ttype_colors[cl] for cl in clusters]

        print(ref_tx_pc_df.head())
        ax = plt.subplot(gs[sc_index, 1])
        ax.scatter(
            ref_tx_pc_df["PC1"],
            ref_tx_pc_df["PC2"],
            c=cluster_colors,
            s=1,
        )
        ax.set_aspect("equal", adjustable="datalim")
        sns.despine(ax=ax)
        ax.set(xticks=[], yticks=[])
        ax.set_xlabel("Tx PC-1", fontsize=6)
        ax.set_ylabel("Tx PC-2", fontsize=6)

        unique_clusters = np.unique(clusters)
        unique_clusters_ordered = sorted(unique_clusters, key=lambda x: ttype_ids[x])

        legend_elements = []
        for ttype in unique_clusters_ordered:
            n_ttype = np.sum(clusters == ttype)
            legend_elements.append(
                Patch(facecolor=ttype_colors[ttype], label=f"{ttype} ({n_ttype})")
            )
        ax.legend(handles=legend_elements, loc="center left", fontsize=6,
            bbox_to_anchor=(-2.5, 0.5), frameon=False)


        n_weight_plots = int(np.ceil(ref_tx_pc_df.shape[1] / 2))
        gene_weight_df = pd.read_csv(os.path.join(args["ref_tx_pc_dir"], f"{sc}_tx_pca_weights.csv"), index_col=0)
        print("n weight plots", n_weight_plots)
        for weight_plot_index in range(n_weight_plots):
            if (weight_plot_index == n_weight_plots - 1) and ref_tx_pc_df.shape[1] % 2 == 1:
                x_ind = f"PC{2 * weight_plot_index - 1}"
                y_ind = f"PC{2 * weight_plot_index + 1}"
            else:
                x_ind = f"PC{2 * weight_plot_index + 1}"
                y_ind = f"PC{2 * weight_plot_index + 2}"
            print(weight_plot_index, x_ind, y_ind)

            x_order = np.argsort(np.abs(gene_weight_df[x_ind]))
            top_x_genes = gene_weight_df.index[x_order[-n_genes_per_pc:]]
            y_order = np.argsort(np.abs(gene_weight_df[y_ind]))
            top_y_genes = gene_weight_df.index[y_order[-n_genes_per_pc:]]

            combo_genes = np.unique(top_x_genes.tolist() + top_y_genes.tolist())

            ax = plt.subplot(gs[sc_index, 2 + weight_plot_index])
            ax.scatter(
                gene_weight_df[x_ind],
                gene_weight_df[y_ind],
                c="gray",
                s=1,
            )
            ax.scatter(
                gene_weight_df.loc[combo_genes, x_ind],
                gene_weight_df.loc[combo_genes, y_ind],
                c="firebrick",
                s=1,
            )
            for g in combo_genes:
                ax.annotate(
                    g,
                    xy=(gene_weight_df.loc[g, x_ind], gene_weight_df.loc[g, y_ind]),
                    xytext=(1, 1),
                    textcoords="offset points",
                    fontsize=5,
                )
            ax.set_aspect("equal", adjustable="datalim")
            sns.despine(ax=ax)
            ax.set(xticks=[], yticks=[])
            ax.set_xlabel("Tx " + x_ind, fontsize=6)
            ax.set_ylabel("Tx " + y_ind, fontsize=6)

        sc_index += 1

    plt.savefig(args['output_file'], bbox_inches='tight', dpi=300)

if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=FigTxPcSupplementParameters)
    main(module.args)
