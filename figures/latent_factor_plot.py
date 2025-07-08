import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from adjustText import adjust_text

def plot_side_by_side_latent_factors(w, v, gene_data, feature_data, gs, ax_label_prefix,
    corr_radius=1, x_index=0, y_index=1,
    n_genes_to_plot=10, n_features_to_plot=10, scatter_colors=None,
    fixed_size=True, axis_arrow_length=0.6):

    if scatter_colors is None:
        scatter_colors = "k"

    if w.shape[1] < 2:
        print("Too few ranks of latent space to plot")
        return

    gene_lf = gene_data.values @ w
    feature_lf = feature_data.values @ v

    gene_lf = gene_lf / gene_lf.std(axis=0)
    feature_lf = feature_lf / feature_lf.std(axis=0)

    gs_lf = gridspec.GridSpecFromSubplotSpec(
        1, 2,
        subplot_spec=gs
    )
    ax_l = plt.subplot(gs_lf[0])
    ax_r = plt.subplot(gs_lf[1])

    ax_l.scatter(
        gene_lf[:, x_index],
        gene_lf[:, y_index],
        c=scatter_colors,
        s=2,
        edgecolors="white",
        linewidths=0.25,
    )
    ax_r.scatter(
        feature_lf[:, x_index],
        feature_lf[:, y_index],
        c=scatter_colors,
        s=2,
        edgecolors="white",
        linewidths=0.25,
    )

    # select non-zero gene weights
    nz_gene_mask = np.abs(w).sum(axis=1) > 0
    gene_corrs = latent_factor_correlations(gene_data.loc[:, nz_gene_mask],
        gene_lf)
    genes_to_plot = []
    for ind in (x_index, y_index):
        gene_order = np.argsort(np.abs(gene_corrs[ind, :]))
        genes_to_plot += gene_order[-n_genes_to_plot:].tolist()
    genes_to_plot = np.unique(genes_to_plot).tolist()

    label_bbox_props = dict(
        boxstyle="square, pad=0", facecolor='white', alpha=0.4, edgecolor="none")

    text_list_l = []
    line_list_l = []
    for ind, g in zip(genes_to_plot, gene_data.columns[nz_gene_mask].values[genes_to_plot]):
        lines = ax_l.plot(
            [0, corr_radius * gene_corrs[x_index, ind]],
            [0, corr_radius * gene_corrs[y_index, ind]],
            linewidth=0.5,
            color="#444444",
        )
        line_list_l += lines
        if gene_corrs[x_index, ind] < -0.2:
            ha = "right"
        elif gene_corrs[x_index, ind] > 0.2:
            ha = "left"
        else:
            ha = "center"
        t = ax_l.text(
            corr_radius * gene_corrs[x_index, ind] * 1.2,
            corr_radius * gene_corrs[y_index, ind] * 1.2,
            g,
            ha=ha,
            va="center",
            color="#000000",
            fontsize=5,
            zorder=100,
        )
        t.set_bbox(label_bbox_props)
        text_list_l.append(t)

    feature_corrs = latent_factor_correlations(feature_data, feature_lf)
    features_to_plot = []
    for ind in (x_index, y_index):
        feature_order = np.argsort(np.abs(feature_corrs[ind, :]))
        features_to_plot += feature_order[-n_features_to_plot:].tolist()
    features_to_plot = np.unique(features_to_plot).tolist()
    text_list_r = []
    line_list_r = []
    for ind, g in zip(features_to_plot, feature_data.columns.values[features_to_plot]):
        lines = ax_r.plot(
            [0, corr_radius * feature_corrs[x_index, ind]],
            [0, corr_radius * feature_corrs[y_index, ind]],
            linewidth=0.5,
            color="#444444",
        )
        line_list_r += lines
        if feature_corrs[x_index, ind] < -0.2:
            ha = "right"
        elif feature_corrs[x_index, ind] > 0.2:
            ha = "left"
        else:
            ha = "center"
        t = ax_r.text(
            corr_radius * feature_corrs[x_index, ind] * 1.2,
            corr_radius * feature_corrs[y_index, ind] * 1.2,
            g,
            ha=ha,
            va="center",
            color="#000000",
            fontsize=5,
            zorder=100,
        )
        t.set_bbox(label_bbox_props)
        text_list_r.append(t)

    corr_circle_l = plt.Circle((0, 0), corr_radius, edgecolor="#999999", fill=False, linestyle="dotted")
    corr_circle_r = plt.Circle((0, 0), corr_radius, edgecolor="#999999", fill=False, linestyle="dotted")
    ax_l.add_patch(corr_circle_l)
    ax_r.add_patch(corr_circle_r)
    for ax, text_list, line_list in zip((ax_l, ax_r), (text_list_l, text_list_r), (line_list_l, line_list_r)):
        ax.set_aspect("equal")
        sns.despine(ax=ax, bottom=True, left=True)
        ax.set_xticks([])
        ax.set_yticks([])
        if fixed_size:
            ax.set_xlim(-1.1 * corr_radius, 1.1 * corr_radius)
            ax.set_ylim(-1.1 * corr_radius, 1.1 * corr_radius)
        # x-axis label
        a = ax.arrow(
            -corr_radius, -corr_radius,
            axis_arrow_length, 0,
            color="k",
            head_width=axis_arrow_length * 0.2,
            head_length=axis_arrow_length * 0.2,
        )
        line_list.append(a)
        t = ax.text(
            -corr_radius + axis_arrow_length / 2,
            -corr_radius - axis_arrow_length / 2 - 0.1,
            f"{ax_label_prefix}-LF-{x_index + 1}",
            fontsize=5,
            ha="center",
            va="top",
        )
        line_list.append(t)
        # y-axis label
        a = ax.arrow(
            -corr_radius, -corr_radius,
            0, axis_arrow_length,
            color="k",
            head_width=axis_arrow_length * 0.2,
            head_length=axis_arrow_length * 0.2,
        )
        line_list.append(a)
        t = ax.text(
            -corr_radius - axis_arrow_length / 2 - 0.1,
            -corr_radius + axis_arrow_length / 2,
            f"{ax_label_prefix}-LF-{y_index + 1}",
            fontsize=5,
            ha="right",
            va="center",
            rotation=0,
        )
        line_list.append(t)

    adjust_text(text_list_l, objects=line_list_l, ax=ax_l, ensure_inside_axes=False)
    adjust_text(text_list_r, objects=line_list_r, ax=ax_r, ensure_inside_axes=False)


def plot_single_rank_latent_factor(w, v, gene_data, feature_data, gs, ax_label_prefix,
    corr_radius=1, axis_arrow_length=0.4,
    n_genes_to_plot=10, n_features_to_plot=10, scatter_colors=None):

    if scatter_colors is None:
        scatter_colors = "k"

    gene_lf = gene_data.values @ w
    feature_lf = feature_data.values @ v

    gene_lf = gene_lf / gene_lf.std(axis=0)
    feature_lf = feature_lf / feature_lf.std(axis=0)


    gs_lf = gridspec.GridSpecFromSubplotSpec(
        2, 2,
        height_ratios=(1, 1.5),
        wspace=0.4,
        subplot_spec=gs
    )
    ax_l = plt.subplot(gs_lf[0, 0])
    ax_r = plt.subplot(gs_lf[0, 1])

    sns.stripplot(
        x=gene_lf[:, 0],
        y=["category"] * gene_lf.shape[0],
        c=scatter_colors,
        s=2,
        edgecolor="white",
        linewidth=0.25,
        ax=ax_l
    )
    sns.stripplot(
        x=feature_lf[:, 0],
        y=["category"] * gene_lf.shape[0],
        c=scatter_colors,
        s=2,
        edgecolor="white",
        linewidth=0.25,
        ax=ax_r
    )
    print(ax_l.get_ylim())
    max_l_coord = max(np.abs(ax_l.get_xlim()))
    max_r_coord = max(np.abs(ax_r.get_xlim()))
    max_coord = max(max_l_coord, max_r_coord)
    for ax in (ax_l, ax_r):
        ax.set_xlim(-max_coord, max_coord)
        sns.despine(ax=ax, left=True, bottom=True)
        ax.set_xticks([])
        ax.set_yticks([])
            # x-axis label
        ax.arrow(
            -axis_arrow_length / 2, 0.25,
            axis_arrow_length, 0,
            color="k",
            head_width=axis_arrow_length * 0.15,
            head_length=axis_arrow_length * 0.2,
        )
        ax.text(
            0,
            0.25 + axis_arrow_length / 2,
            f"{ax_label_prefix}-LF-1",
            fontsize=5,
            ha="center",
            va="top",
        )
        ax.set_ylim(0.5, -0.5)
    print(ax_l.get_ylim())


    label_offset = 0.1 * corr_radius
    # select non-zero gene weights
    nz_gene_mask = np.abs(w).sum(axis=1) > 0
    gene_corrs = latent_factor_correlations(gene_data.loc[:, nz_gene_mask],
        gene_lf)
    gene_order = np.argsort(np.squeeze(np.abs(gene_corrs)))
    genes_to_plot = gene_order[-n_genes_to_plot:]

    ax_l = plt.subplot(gs_lf[1, 0])
    ax_r = plt.subplot(gs_lf[1, 1])

    counter = 0
    for ind, g in zip(genes_to_plot, gene_data.columns[nz_gene_mask].values[genes_to_plot]):
        ax_l.plot(
            [0, corr_radius * gene_corrs[0, ind]],
            np.array([0, 0]) - counter,
            linewidth=1,
            color="#444444",
        )
        if np.sign(gene_corrs[0, ind]) > 0:
            ha = "left"
        else:
            ha = "right"
        ax_l.text(
            corr_radius * gene_corrs[0, ind] + label_offset * np.sign(gene_corrs[0, ind]),
            -counter,
            g,
            ha=ha,
            va="center",
            color="#000000",
            fontsize=5,
            zorder=100,
        )
        counter += 1
    ax_l.set_ylim(-counter + n_genes_to_plot + 0.5, -counter + 0.5, )


    feature_corrs = latent_factor_correlations(feature_data, feature_lf)
    feature_order = np.argsort(np.abs(np.squeeze(feature_corrs)))
    features_to_plot = feature_order[-n_features_to_plot:]

    counter = 0
    for ind, g in zip(features_to_plot, feature_data.columns.values[features_to_plot]):
        ax_r.plot(
            [0, corr_radius * feature_corrs[0, ind]],
            np.array([0, 0]) - counter,
            linewidth=1,
            color="#444444",
        )
        if np.sign(feature_corrs[0, ind]) > 0:
            ha = "left"
        else:
            ha = "right"
        ax_r.text(
            corr_radius * feature_corrs[0, ind] + label_offset * np.sign(feature_corrs[0, ind]),
            -counter,
            g,
            ha=ha,
            va="center",
            color="#444444",
            fontsize=5,
            zorder=100,
        )
        counter += 1
    ax_r.set_ylim(-counter + n_features_to_plot + 0.5, -counter + 0.5, )


    for ax in (ax_l, ax_r):
        ax.set_aspect(0.3, adjustable="box")
        ax.set_xlim(-1, 1)
        sns.despine(ax=ax, left=True, bottom=False)
        ax.set_xticks([-1, 0, 1])
        ax.set_xticklabels([-1, 0, 1], size=6)
        ax.set_yticks([])
        ax.axvline(0, color="#aaaaaa", linewidth=0.5)
        ax.set_xlabel("correlation", fontsize=6)



def latent_factor_correlations(data, latent_factors):
    corrs = np.corrcoef(data, latent_factors, rowvar=False)

    return corrs[data.shape[1]:, :data.shape[1]]
