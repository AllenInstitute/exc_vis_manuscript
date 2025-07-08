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
import allensdk.core.swc as swc
from fig_sparse_rrr import plot_morph_lineup_with_lf
from morph_plotting import basic_morph_plot
from ccf_streamlines.projection import BoundaryFinder
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
from PIL import Image
import ccf_streamlines.morphology as ccfmorph

class FigWnmLfParameters(ags.ArgSchema):
    full_swc_dir = ags.fields.InputDir()
    ccf_swc_dir = ags.fields.InputDir()
    local_axon_feature_file = ags.fields.InputFile()
    complete_axon_feature_file = ags.fields.InputFile()
    axon_pc_weight_file = ags.fields.InputFile()
    full_morph_lf_file = ags.fields.InputFile()
    full_morph_meta_file = ags.fields.InputFile()
    layer_depths_file = ags.fields.InputFile(
        description="json file with distances from top of layer to pia",
    )
    srrr_fits_no_diam_file = ags.fields.InputFile()
    srrr_features_file = ags.fields.InputFile()
    projected_atlas_file = ags.fields.InputFile()
    atlas_labels_file = ags.fields.InputFile()
    l23_flat_morph_file = ags.fields.InputFile()
    l6b_flat_morph_file = ags.fields.InputFile()
    neuroglancer_pngs_dir = ags.fields.InputDir()
    output_file = ags.fields.OutputFile(
        default="fig_wnm_lf.pdf"
    )

MET_PRED_TYPES = {
    "L23-IT": ("L2/3 IT",),
    "L4-L5-IT": ("L4 IT", "L4/L5 IT", "L5 IT-1", "L5 IT-2", "L5 IT-3 Pld5"),
    "L6-IT": ("L6 IT-1", "L6 IT-2", "L6 IT-3"),
    "L5-ET": ("L5 ET-1 Chrna6", "L5 ET-2", "L5 ET-3"),
    "L6-CT": ("L6 CT-1", "L6 CT-2",),
    "L6b": ("L6b",),
}


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

FEATURE_RELABEL = {
    "axon_bias_x": "local axon horiz. bias",
    "axon_bias_y": "local axon vert. bias",
    "axon_calculate_number_of_stems": "num. axon stems",
    "axon_emd_with_apical_dendrite": "local axon / apic. dend EMD",
    "axon_extent_x": "local axon width",
    "axon_extent_y": "local axon height",
    "axon_max_branch_order": "local axon max. branch order",
    "axon_max_euclidean_distance": "local axon max. euclid. dist.",
    "axon_max_path_distance": "local axon max path dist.",
    "axon_mean_contraction": "local axon contract.",
    "axon_num_branches": "local axon num. branches",
    "axon_frac_above_apical_dendrite": "local axon frac. above apic. dend.",
    "axon_frac_below_apical_dendrite": "local axon frac. below apic. dend.",
    "axon_frac_intersect_apical_dendrite": "local axon frac. overlap apic. dend.",
    "axon_soma_percentile_x": "local axon soma percentile (horiz.)",
    "axon_soma_percentile_y": "local axon soma percentile (vert.)",
    "axon_total_length": "local axon length",
    "axon_depth_pc_0": "local axon depth PC-1",
    "axon_depth_pc_1": "local axon depth PC-2",
    "axon_depth_pc_2": "local axon depth PC-3",
    "axon_depth_pc_3": "local axon depth PC-4",
    "axon_depth_pc_4": "local axon depth PC-5",
    "axon_depth_pc_5": "local axon depth PC-6",
    "axon_depth_pc_6": "local axon depth PC-7",
    "complete_axon_total_length": "complete axon length",
    "complete_axon_max_euclidean_distance": "complete axon max. euclid. distance",
    "complete_axon_num_branches": "complete axon num. branches",
    "complete_axon_num_tips": "complete axon num. tips",
    "complete_axon_max_branch_order": "complete axon max. branch order",
    "complete_axon_max_path_distance": "complete axon max. path distance",
    "complete_axon_mean_contraction": "complete axon mean contraction",
    "axon_width": "complete axon width",
    "axon_depth": "complete axon depth",
    "axon_height": "complete axon height",
    "complete_axon_total_number_of_targets": "num. targets",
    "complete_axon_total_projection_length": "total projection length",
    "complete_axon_VIS_length": "axon length in visual areas",
    "complete_axon_ipsi_VIS_length": "axon length in ipsi. visual areas",
    "complete_axon_length_in_soma_structure": "axon length in soma region",
    "complete_axon_number_of_VIS_targets": "num. visual area targets",
    "complete_axon_number_of_contra_VIS_targets": "num. contra. visual area targets",
    "fraction_of_complete_axon_in_soma_structure": "frac. axon in soma region",
}


def plot_axon_pc_weights(ax, loadings, depths, layer_edges, xlabel="", xlim=(-0.2, 0.2)):
    ax.fill_between(loadings, depths, lw=0.75, zorder=20)
    ax.axvline(0, linestyle="solid", color="black", lw=0.5)
    ax.set_xlim(xlim)
    path_color = "#cccccc"
    for e in layer_edges:
        ax.axhline(-e, zorder=-5, color=path_color, linewidth=0.75)
    ax.set_xlabel(xlabel, fontsize=6)
    ax.tick_params(axis='x', labelsize=6)
    ax.tick_params(axis="y", left=False, labelleft=False)
    sns.despine(ax=ax, left=True)


def sp_rrr_fit_info(filepart, modality, h5f, select_features, excluded_features=None):
    specimen_ids = h5f[filepart][modality]["specimen_ids"][:]
    features = select_features[filepart][modality]

    feat_drop_mask = np.array([f not in excluded_features for f in features])
    features = [f for f in features if f not in excluded_features]

    w = h5f[filepart][modality]["w"][:].T
    v = h5f[filepart][modality]["v"][:].T

    genes = h5f[filepart][modality]["genes"][:]
    genes = np.array([s.decode() for s in genes])

    return {
        "specimen_ids": specimen_ids,
        "features": features,
        "w": w,
        "v": v,
        "genes": genes,
    }


def flatmap_morph_plot(morph, flatmap_coords, ax, alpha=1.0, scale_factor=1, zorder=1, color="black", x_ind=0, y_ind=1):
    lines_x = []
    lines_y = []
    morph_vals = morph.values
    for i in range(morph_vals.shape[0]):
        if np.isnan(flatmap_coords[i, x_ind]) or np.isnan(flatmap_coords[i, y_ind]):
            continue
        parent_id = morph_vals[i, 6]
        if parent_id == -1:
            continue
        p_ind = np.flatnonzero(morph_vals[:, x_ind] == parent_id)[0]
        if np.isnan(flatmap_coords[p_ind, x_ind]) or np.isnan(flatmap_coords[p_ind, y_ind]):
            continue
        lines_x += [flatmap_coords[p_ind, x_ind] / scale_factor, flatmap_coords[i, x_ind] / scale_factor, None]
        lines_y += [flatmap_coords[p_ind, y_ind] / scale_factor, flatmap_coords[i, y_ind] / scale_factor, None]
    ax.plot(lines_x, lines_y, linewidth=0.25, alpha=alpha, zorder=zorder, color=color)
    return ax



def plot_flat_morphs(subplot_spec, specimen_ids, lf_values, full_swc_dir, h5f, boundaries,
        n_to_plot, boundary_color="#cccccc", vmin=None, vmax=None, cbar_label="",
        cbar_width=0.05, select_indices=None, selected_areas=None, cbar_label_pos="default"):
    morph_cmap = matplotlib.colormaps["magma"]
    if vmin is None:
        vmin = lf_values.min()
    if vmax is None:
        vmax = lf_values.max()
    norm = plt.Normalize(vmin, vmax)

    if len(specimen_ids) != len(lf_values):
        raise ValueError("specimen_ids and lf_values must be the same length")

    if select_indices is not None:
        indices = select_indices
    elif n_to_plot >= len(specimen_ids):
        indices = np.arange(len(specimen_ids)).astype(int)
    else:
        indices = np.linspace(0, len(specimen_ids) - 1, num=n_to_plot).astype(int)
    print(indices)

    gs_flat = gridspec.GridSpecFromSubplotSpec(
        1, len(indices),
        subplot_spec=subplot_spec
    )

    sorted_by_val = np.argsort(lf_values)
    for i, idx in enumerate(indices):
        ax = plt.subplot(gs_flat[i])

        # Region borders
        for bounds in boundaries:
            for k, boundary_coords in bounds.items():
                if selected_areas is not None and k not in selected_areas:
                    continue
                edgecolor = 'gray'
                lw = 0.25
                ax.fill(*boundary_coords.T, color="white", edgecolor=edgecolor,
                    lw=lw, zorder=1)

        # Morphology
        spec_id = specimen_ids[sorted_by_val][idx]
        flatmap_coords = h5f[spec_id][:]
        swc_path = os.path.join(full_swc_dir, f"{spec_id}.swc")
        morph = ccfmorph.load_swc_as_dataframe(swc_path)
        color = morph_cmap(norm(lf_values[sorted_by_val][idx]))
        flatmap_morph_plot(morph, flatmap_coords, ax, color=color, zorder=10)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set(xticks=[], yticks=[])
        sns.despine(ax=ax, left=True, bottom=True)

    ax_cbar = ax.inset_axes([1.05, 0, 0.05, 1], transform=ax.transAxes)
    cbar = plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=morph_cmap),
             cax=ax_cbar, orientation='vertical')
    cbar.set_label(cbar_label, size=6)
    cbar.outline.set_visible(False)
    if cbar_label_pos == "below":
        ax_cbar.set_ylabel(cbar_label, size=6, rotation=0, pos="bottom", ha="right")
    ax_cbar.tick_params(axis='y', labelsize=6, length=3)

    print(specimen_ids[sorted_by_val][indices])
    return specimen_ids[sorted_by_val][indices]

def main(args):
    full_morph_lf_df = pd.read_csv(args["full_morph_lf_file"], index_col=0)
    full_morph_lf_df.index = [i[:-4] for i in full_morph_lf_df.index]

    full_morph_meta_df = pd.read_csv(args["full_morph_meta_file"], index_col=0)
    full_morph_meta_df.index = [i[:-4] for i in full_morph_meta_df.index]

    full_morph_lf_df = full_morph_lf_df.merge(full_morph_meta_df, left_index=True, right_index=True)

    local_axon_df = pd.read_csv(args["local_axon_feature_file"], index_col=0)
    complete_axon_df = pd.read_csv(
        args["complete_axon_feature_file"],
        index_col=0)
    complete_axon_df.index = [i[:-4] for i in complete_axon_df.index]


    axon_pc_weights_df = pd.read_csv(args["axon_pc_weight_file"], header=None)
    axon_pc_depths = -np.arange(axon_pc_weights_df.shape[1]) * 5.

    with open(args["layer_depths_file"], "r") as f:
        layer_info = json.load(f)
    layer_edges = [0] + list(layer_info.values())

    h5f_no_diam = h5py.File(args["srrr_fits_no_diam_file"], "r")
    with open(args["srrr_features_file"], "r") as f:
        select_features = json.load(f)
    excluded_features = [
        "apical_dendrite_total_surface_area",
        "basal_dendrite_total_surface_area",
        "apical_dendrite_mean_diameter",
        "basal_dendrite_mean_diameter",
    ]

    bf_boundary_finder = BoundaryFinder(
        projected_atlas_file=args["projected_atlas_file"],
        labels_file=args["atlas_labels_file"],
    )
    bounds_ipsi = bf_boundary_finder.region_boundaries()
    bounds_contra = bf_boundary_finder.region_boundaries(hemisphere="right_for_both", view_space_for_other_hemisphere="flatmap_butterfly")

    local_axon_cols = local_axon_df.columns[local_axon_df.columns.str.startswith("axon_")]
    local_axon_cols = [c for c in local_axon_cols if c not in ("axon_exit_theta",)]

    corr_df_list = []
    corr_sc_list = []
    for sc, met_types in MET_PRED_TYPES.items():
        specimen_ids = full_morph_lf_df.loc[
            full_morph_lf_df["predicted_met_type"].isin(met_types), :].index.values
        if len(specimen_ids) < 10:
            continue
        else:
            corr_sc_list.append(sc)
        sc_full_morph_lf_df = full_morph_lf_df.loc[specimen_ids, :]
        sc_local_axon_df = local_axon_df.loc[specimen_ids, :]
        sc_complete_axon_df = complete_axon_df.loc[specimen_ids, :]
        lf_cols = sc_full_morph_lf_df.columns[
            sc_full_morph_lf_df.columns.str.startswith("LF") &
            sc_full_morph_lf_df.columns.str.endswith(sc)]

        local_res = spearmanr(sc_full_morph_lf_df.loc[:, lf_cols],
            sc_local_axon_df.loc[:, local_axon_cols],
            nan_policy="omit")
        r_list = []
        pval_list = []
        lf_list = []
        var_list = []

        for i, lf_col in enumerate(lf_cols):
            r_list += local_res.statistic[i, len(lf_cols):].tolist()
            pval_list += local_res.pvalue[i, len(lf_cols):].tolist()
            lf_list += [lf_col] * len(local_axon_cols)
            var_list += local_axon_cols

        complete_res = spearmanr(sc_full_morph_lf_df.loc[:, lf_cols],
            sc_complete_axon_df,
            nan_policy="omit")
        for i, lf_col in enumerate(lf_cols):
            r_list += complete_res.statistic[i, len(lf_cols):].tolist()
            pval_list += complete_res.pvalue[i, len(lf_cols):].tolist()
            lf_list += [lf_col] * sc_complete_axon_df.shape[1]
            var_list += sc_complete_axon_df.columns.tolist()

        df = pd.DataFrame({
            "spearman_r": r_list,
            "pvalue": pval_list,
            "lf": lf_list,
            "variable": var_list,
        })
        df["subclass"] = sc
        corr_df_list.append(df)
    corr_df = pd.concat(corr_df_list)
    corr_df = corr_df.dropna()

    reject, pvals_adj, _, _ = multipletests(corr_df["pvalue"].values, method='fdr_bh')
    corr_df["reject"] = reject
    corr_df["pval_adj"] = pvals_adj
    print(corr_df)
    corr_df_filt = corr_df[corr_df["reject"]].copy()
    print(corr_df_filt)
    print(corr_df_filt["subclass"].value_counts())

    used_local_cols = [c for c in local_axon_cols if c in corr_df_filt["variable"].tolist()]
    used_complete_cols = [c for c in complete_axon_df.columns if c in corr_df_filt["variable"].tolist()]
    print(used_local_cols)
    print(used_complete_cols)

    ##### PLOTTING ##############
    fig = plt.figure(figsize=(7.5, 10))
    gs = gridspec.GridSpec(
        5, 1,
        height_ratios=(1, 1, 1, 1, 2.5),
        hspace=0.45,
    )

    cbar_width = 0.02

    # L2/3
    subclass = "L23-IT"
    lf_label = "L2/3 IT M-LF-1"
    lf_col = "LF1_L23-IT"
    feat_label = "local axon\ndepth PC-2"
    feat_col = "axon_depth_pc_1"
    gs_row = gridspec.GridSpecFromSubplotSpec(
        1, 5,
        width_ratios=(1, 4, 0.5, 0.75, 1.25),
        wspace=0.4,
        subplot_spec=gs[0]
    )
    specimen_ids = full_morph_lf_df.loc[full_morph_lf_df["predicted_met_type"].isin(MET_PRED_TYPES[subclass]), :].index.values
    sc_full_morph_lf_df = full_morph_lf_df.loc[specimen_ids, :]
    visp_spec_ids = sc_full_morph_lf_df.loc[sc_full_morph_lf_df["ccf_soma_location_nolayer"] == "VISp", :].index.values
    cell_colors = [MET_TYPE_COLORS[t] for t in full_morph_lf_df.loc[specimen_ids, "predicted_met_type"]]
    ax_morph = plt.subplot(gs_row[1])
    plotted_ids = plot_morph_lineup_with_lf(
        ax_morph,
        visp_spec_ids,
        full_morph_lf_df.loc[visp_spec_ids, lf_col].values,
        args["full_swc_dir"],
        layer_edges,
        cbar_label=lf_label,
        vmin=-5, vmax=5,
        n_to_plot=7,
        show_axon=True,
        morph_spacing=500,
        cbar_width=cbar_width,
    )

    ax_pc = ax_morph.inset_axes([1.3, 0, 0.15, 1], transform=ax_morph.transAxes)
    ax_morph.sharey(ax_pc)
    plot_axon_pc_weights(ax_pc, axon_pc_weights_df.loc[1, :], axon_pc_depths, layer_edges,
        xlabel="depth PC-2\nloadings")

    ax_lf = plt.subplot(gs_row[4])
    sns.regplot(
        x=full_morph_lf_df.loc[specimen_ids, lf_col],
        y=local_axon_df.loc[specimen_ids, feat_col],
        ci=None,
        scatter_kws=dict(s=5, color=cell_colors, edgecolors="white", linewidths=0.25),
        line_kws=dict(lw=1, color="black"),
        ax=ax_lf,
    )
    ax_lf.scatter(
        x=full_morph_lf_df.loc[plotted_ids, lf_col],
        y=local_axon_df.loc[plotted_ids, feat_col],
        c=[MET_TYPE_COLORS[t] for t in full_morph_lf_df.loc[plotted_ids, "predicted_met_type"]],
        edgecolors="black",
        s=7,
    )

    ax_lf.set_xlabel(lf_label, fontsize=6)
    ax_lf.set_ylabel(feat_label, fontsize=6)
    ax_lf.tick_params(axis='both', length=3, labelsize=6)
    sns.despine(ax=ax_lf)


    # L4 & L5 IT
    subclass = "L4-L5-IT"
    lf_label = "L4 & L5 IT M-LF-2"
    lf_col = "LF2_L4-L5-IT"
    feat_label = "local axon\ndepth PC-3"
    feat_col = "axon_depth_pc_2"
    gs_row = gridspec.GridSpecFromSubplotSpec(
        1, 4,
        width_ratios=(4, 0.5, 1, 2),
        wspace=0.4,
        subplot_spec=gs[1]
    )
    specimen_ids = full_morph_lf_df.loc[full_morph_lf_df["predicted_met_type"].isin(MET_PRED_TYPES[subclass]), :].index.values
    sc_full_morph_lf_df = full_morph_lf_df.loc[specimen_ids, :]
    visp_spec_ids = sc_full_morph_lf_df.loc[sc_full_morph_lf_df["ccf_soma_location_nolayer"] == "VISp", :].index.values
    cell_colors = [MET_TYPE_COLORS[t] for t in full_morph_lf_df.loc[specimen_ids, "predicted_met_type"]]
    ax_morph = plt.subplot(gs_row[0])
    plotted_ids = plot_morph_lineup_with_lf(
        ax_morph,
        visp_spec_ids,
        full_morph_lf_df.loc[visp_spec_ids, lf_col].values,
        args["full_swc_dir"],
        layer_edges,
        cbar_label=lf_label,
        vmin=-5, vmax=5,
        n_to_plot=8,
        show_axon=True,
        morph_spacing=500,
        cbar_width=cbar_width,
    )

    ax_pc = ax_morph.inset_axes([1.35, 0, 0.15, 1], transform=ax_morph.transAxes)
    ax_morph.sharey(ax_pc)
    plot_axon_pc_weights(ax_pc, axon_pc_weights_df.loc[2, :], axon_pc_depths, layer_edges,
        xlabel="depth PC-3\nloadings")

    ax_lf = plt.subplot(gs_row[3])
    sns.regplot(
        x=full_morph_lf_df.loc[specimen_ids, lf_col],
        y=local_axon_df.loc[specimen_ids, feat_col],
        ci=None,
        scatter_kws=dict(s=5, color=cell_colors, edgecolors="white", linewidths=0.25),
        line_kws=dict(lw=1, color="black"),
        ax=ax_lf,
    )
    ax_lf.scatter(
        x=full_morph_lf_df.loc[plotted_ids, lf_col],
        y=local_axon_df.loc[plotted_ids, feat_col],
        c=[MET_TYPE_COLORS[t] for t in full_morph_lf_df.loc[plotted_ids, "predicted_met_type"]],
        edgecolors="black",
        s=7,
    )

    ax_lf.set_xlabel(lf_label, fontsize=6)
    ax_lf.set_ylabel(feat_label, fontsize=6)
    ax_lf.tick_params(axis='both', length=3, labelsize=6)
    sns.despine(ax=ax_lf)


    # L6 CT
#     subclass = "L6-CT"
#     lf_label = "L6 CT M-LF-2"
#     lf_col = "LF2_L6-CT"
#     feat_label = "local axon\ndepth PC-5"
#     feat_col = "axon_depth_pc_4"
#     gs_row = gridspec.GridSpecFromSubplotSpec(
#         1, 4,
#         width_ratios=(4, 0.5, 1, 2),
#         wspace=0.4,
#         subplot_spec=gs[2]
#     )
#     specimen_ids = full_morph_lf_df.loc[full_morph_lf_df["predicted_met_type"].isin(MET_PRED_TYPES[subclass]), :].index.values
#     sc_full_morph_lf_df = full_morph_lf_df.loc[specimen_ids, :]
#     visp_spec_ids = sc_full_morph_lf_df.loc[sc_full_morph_lf_df["ccf_soma_location_nolayer"] == "VISp", :].index.values
#     cell_colors = [MET_TYPE_COLORS[t] for t in full_morph_lf_df.loc[specimen_ids, "predicted_met_type"]]
#     ax_morph = plt.subplot(gs_row[0])
#     plot_morph_lineup_with_lf(
#         ax_morph,
#         visp_spec_ids,
#         full_morph_lf_df.loc[visp_spec_ids, lf_col].values,
#         args["full_swc_dir"],
#         layer_edges,
#         cbar_label=lf_label,
#         vmin=-5, vmax=5,
#         n_to_plot=9,
#         show_axon=True,
#         morph_spacing=500,
#         cbar_width=cbar_width,
#     )
#
#     ax_pc = ax_morph.inset_axes([1.35, 0, 0.15, 1], transform=ax_morph.transAxes)
#     ax_morph.sharey(ax_pc)
#     plot_axon_pc_weights(ax_pc, axon_pc_weights_df.loc[4, :], axon_pc_depths, layer_edges,
#         xlabel="depth PC-5\nloadings")
#
#     ax_lf = plt.subplot(gs_row[3])
#     sns.regplot(
#         x=full_morph_lf_df.loc[specimen_ids, lf_col],
#         y=local_axon_df.loc[specimen_ids, feat_col],
#         ci=None,
#         scatter_kws=dict(s=5, color=cell_colors, edgecolors="white", linewidths=0.25),
#         line_kws=dict(lw=1, color="black"),
#         ax=ax_lf,
#     )
#     ax_lf.set_xlabel(lf_label, fontsize=6)
#     ax_lf.set_ylabel(feat_label, fontsize=6)
#     ax_lf.tick_params(axis='both', length=3, labelsize=6)
#     sns.despine(ax=ax_lf)


    # L2/3 IT
    subclass = "L23-IT"
    lf_label = "L2/3 IT M-LF-1"
    lf_col = "LF1_L23-IT"
    feat_label = "axon length in\nvisual areas (mm)"
    feat_col = "complete_axon_VIS_length"
    gs_row = gridspec.GridSpecFromSubplotSpec(
        1, 2,
        width_ratios=(5.5, 1.75),
        wspace=0.4,
        subplot_spec=gs[2]
    )
    specimen_ids = full_morph_lf_df.loc[full_morph_lf_df["predicted_met_type"].isin(MET_PRED_TYPES[subclass]), :].index.values
    sc_full_morph_lf_df = full_morph_lf_df.loc[specimen_ids, :]
    visp_spec_ids = sc_full_morph_lf_df.loc[sc_full_morph_lf_df["ccf_soma_location_nolayer"] == "VISp", :].index.values
    cell_colors = [MET_TYPE_COLORS[t] for t in full_morph_lf_df.loc[specimen_ids, "predicted_met_type"]]
    print(complete_axon_df.loc[full_morph_lf_df.loc[visp_spec_ids, lf_col].sort_values().index, feat_col])

    flat_h5f = h5py.File(args["l23_flat_morph_file"], "r")
    plotted_ids = plot_flat_morphs(
        gs_row[0],
        visp_spec_ids,
        full_morph_lf_df.loc[visp_spec_ids, lf_col].values,
        args["ccf_swc_dir"],
        flat_h5f,
        (bounds_ipsi, ),
        selected_areas=["VISp", "VISpm", "VISam", "VISa", "VISrl", "VISal", "VISl", "VISli", "VISpl", "VISpor"],
        cbar_label=lf_label,
        vmin=-5, vmax=5,
        n_to_plot=4,
        select_indices=[4, 7, 14, 20]
    )
    flat_h5f.close()
    print(complete_axon_df.loc[plotted_ids, feat_col] / 1000)

    ax_lf = plt.subplot(gs_row[1])
    sns.regplot(
        x=full_morph_lf_df.loc[specimen_ids, lf_col],
        y=complete_axon_df.loc[specimen_ids, feat_col] / 1000,
        ci=None,
        scatter_kws=dict(s=5, color=cell_colors, edgecolors="white", linewidths=0.25),
        line_kws=dict(lw=1, color="black"),
        ax=ax_lf,
    )
#     ax_lf.scatter(
#         x=full_morph_lf_df.loc[visp_spec_ids, lf_col],
#         y=complete_axon_df.loc[visp_spec_ids, feat_col] / 1000,
#         c="black",
#         s=8,
#     )
    ax_lf.scatter(
        x=full_morph_lf_df.loc[plotted_ids, lf_col],
        y=complete_axon_df.loc[plotted_ids, feat_col] / 1000,
        c=[MET_TYPE_COLORS[t] for t in full_morph_lf_df.loc[plotted_ids, "predicted_met_type"]],
        edgecolors="black",
        s=7,
    )
    ax_lf.set_xlabel(lf_label, fontsize=6)
    ax_lf.set_ylabel(feat_label, fontsize=6)
    ax_lf.tick_params(axis='both', length=3, labelsize=6)
    sns.despine(ax=ax_lf)


    # L4 & L5 IT
    subclass = "L4-L5-IT"
    lf_label = "L4 & L5 IT M-LF-1"
    lf_col = "LF1_L4-L5-IT"
    feat_label = "complete axon\nlength (mm)"
    feat_col = "complete_axon_total_length"
    gs_row = gridspec.GridSpecFromSubplotSpec(
        1, 2,
        width_ratios=(5.5, 1.75),
        wspace=0.4,
        subplot_spec=gs[3]
    )
    specimen_ids = full_morph_lf_df.loc[full_morph_lf_df["predicted_met_type"].isin(MET_PRED_TYPES[subclass]), :].index.values
    sc_full_morph_lf_df = full_morph_lf_df.loc[specimen_ids, :]
    visp_spec_ids = sc_full_morph_lf_df.loc[sc_full_morph_lf_df["ccf_soma_location_nolayer"] == "VISp", :].index.values
    cell_colors = [MET_TYPE_COLORS[t] for t in full_morph_lf_df.loc[specimen_ids, "predicted_met_type"]]

    indices = np.linspace(0, len(visp_spec_ids) - 1, num=4).astype(int)
    print(indices)

    sorted_by_val = np.argsort(full_morph_lf_df.loc[visp_spec_ids, lf_col])
    plotted_ids = visp_spec_ids[sorted_by_val][indices]
    print(plotted_ids)

    gs_img = gridspec.GridSpecFromSubplotSpec(
        1, len(plotted_ids),
        wspace=0.2,
        subplot_spec=gs_row[0]
    )

    for i, spec_id in enumerate(plotted_ids):
        ax = plt.subplot(gs_img[i])
        img = np.asarray(Image.open(os.path.join(args["neuroglancer_pngs_dir"], f"{spec_id}_closeup.png")))
        ax.imshow(img)
        sns.despine(ax=ax, left=True, bottom=True)
        ax.set(xticks=[], yticks=[])

    cmap = matplotlib.colormaps["magma"]
    norm = plt.Normalize(-5, 5)

    ax_cbar = ax.inset_axes([1.05, 0, 0.05, 1], transform=ax.transAxes)
    cbar = plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
             cax=ax_cbar, orientation='vertical')
    cbar.set_label(lf_label, size=6)
    cbar.outline.set_visible(False)
    ax_cbar.tick_params(axis='y', labelsize=6, length=3)


    ax_lf = plt.subplot(gs_row[1])
    sns.regplot(
        x=full_morph_lf_df.loc[specimen_ids, lf_col],
        y=complete_axon_df.loc[specimen_ids, feat_col] / 1000,
        ci=None,
        scatter_kws=dict(s=5, color=cell_colors, edgecolors="white", linewidths=0.25),
        line_kws=dict(lw=1, color="black"),
        ax=ax_lf,
    )
#     ax_lf.scatter(
#         x=full_morph_lf_df.loc[visp_spec_ids, lf_col],
#         y=complete_axon_df.loc[visp_spec_ids, feat_col] / 1000,
#         edgecolors="black",
#         c="black",
#         s=8,
#     )
    ax_lf.scatter(
        x=full_morph_lf_df.loc[plotted_ids, lf_col],
        y=complete_axon_df.loc[plotted_ids, feat_col] / 1000,
        c=[MET_TYPE_COLORS[t] for t in full_morph_lf_df.loc[plotted_ids, "predicted_met_type"]],
        edgecolors="black",
        s=7,
    )

    ax_lf.set_xlabel(lf_label, fontsize=6)
    ax_lf.set_ylabel(feat_label, fontsize=6)
    ax_lf.tick_params(axis='both', length=3, labelsize=6)
    sns.despine(ax=ax_lf)

    # L6b
#     subclass = "L6b"
#     lf_label = "L6b M-LF-1"
#     lf_col = "LF1_L6b"
#     feat_label = "complete axon\n max euclid. dist. (mm)"
#     feat_col = "complete_axon_max_euclidean_distance"
#     gs_row = gridspec.GridSpecFromSubplotSpec(
#         1, 2,
#         width_ratios=(5.5, 1.75),
#         wspace=0.4,
#         subplot_spec=gs[5]
#     )
#     specimen_ids = full_morph_lf_df.loc[full_morph_lf_df["predicted_met_type"].isin(MET_PRED_TYPES[subclass]), :].index.values
#     sc_full_morph_lf_df = full_morph_lf_df.loc[specimen_ids, :]
#     visp_spec_ids = sc_full_morph_lf_df.loc[sc_full_morph_lf_df["ccf_soma_location_nolayer"] == "VISp", :].index.values
#     cell_colors = [MET_TYPE_COLORS[t] for t in full_morph_lf_df.loc[specimen_ids, "predicted_met_type"]]
#     flat_h5f = h5py.File(args["l6b_flat_morph_file"], "r")
#     print(complete_axon_df.loc[full_morph_lf_df.loc[visp_spec_ids, lf_col].sort_values().index, feat_col])
#     plotted_ids = plot_flat_morphs(
#         gs_row[0],
#         visp_spec_ids,
#         full_morph_lf_df.loc[visp_spec_ids, lf_col].values,
#         args["ccf_swc_dir"],
#         flat_h5f,
#         (bounds_ipsi, bounds_contra),
# #         selected_areas=["VISp", "VISpm", "VISam", "VISa", "VISrl", "VISal", "VISl", "VISli", "VISpl", "VISpor"],
#         cbar_label=lf_label,
#         vmin=-5, vmax=5,
#         n_to_plot=4,
#         select_indices=[1, 4, 5, 6]
#     )
#     flat_h5f.close()
#     print(complete_axon_df.loc[plotted_ids, feat_col] / 1000)
#
#     ax_lf = plt.subplot(gs_row[1])
#     sns.regplot(
#         x=full_morph_lf_df.loc[specimen_ids, lf_col],
#         y=complete_axon_df.loc[specimen_ids, feat_col] / 1000,
#         ci=None,
#         scatter_kws=dict(s=5, color=cell_colors, edgecolors="white", linewidths=0.25),
#         line_kws=dict(lw=1, color="black"),
#         ax=ax_lf,
#     )
#     ax_lf.scatter(
#         x=full_morph_lf_df.loc[visp_spec_ids, lf_col],
#         y=complete_axon_df.loc[visp_spec_ids, feat_col] / 1000,
#         c="black",
#         s=8,
#     )
#     ax_lf.scatter(
#         x=full_morph_lf_df.loc[plotted_ids, lf_col],
#         y=complete_axon_df.loc[plotted_ids, feat_col] / 1000,
#         c="firebrick",
#         s=8,
#     )
#     ax_lf.set_xlabel(lf_label, fontsize=6)
#     ax_lf.set_ylabel(feat_label, fontsize=6)
#     ax_lf.tick_params(axis='both', length=3, labelsize=6)
#     sns.despine(ax=ax_lf)

    spacing = [1, 2, 2, 3, 1]
    gs_row = gridspec.GridSpecFromSubplotSpec(
        1, len(corr_sc_list) * 2 + 3,
        wspace=0.05,
        width_ratios=[4] + spacing + [7] + spacing + [2],
        subplot_spec=gs[4],
    )

    title_list = ["L2/3\nIT", "L4 &\nL5 IT", "L5 ET", "L6 CT", "L6b"]
    grouped_corr = corr_df_filt.groupby("subclass")
    i = 0
    for sc in corr_sc_list:
        print(sc)
        g = grouped_corr.get_group(sc)
        sc_pivot = g.pivot(columns=["lf"], index="variable", values="spearman_r")
        sc_pivot = sc_pivot.reindex(index=used_local_cols)
        ax = plt.subplot(gs_row[i + 1])
        if i == 0:
            yticklabels = [FEATURE_RELABEL[l] for l in used_local_cols]
        else:
            yticklabels = False


        xticklabels = [f"M-LF-{j + 1}" for j in range(sc_pivot.shape[1])]
        sns.heatmap(
            sc_pivot,
            square=True,
            cmap="RdYlBu_r",
            vmin=-1,
            vmax=1,
            yticklabels=yticklabels,
            xticklabels=xticklabels,
            cbar=False,
        )
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.tick_params("y", labelsize=5, length=0)
        ax.tick_params("x", rotation=90, labelsize=5)
        ax.set_title(title_list[i], loc="center", fontsize=7)


        sc_pivot = g.pivot(columns=["lf"], index="variable", values="spearman_r")
        sc_pivot = sc_pivot.reindex(index=used_complete_cols)
        ax = plt.subplot(gs_row[i + len(corr_sc_list) + 2])
        if i == 0:
            yticklabels = [FEATURE_RELABEL[l] for l in used_complete_cols]
        else:
            yticklabels = False


        xticklabels = [f"M-LF-{j + 1}" for j in range(sc_pivot.shape[1])]
        sns.heatmap(
            sc_pivot,
            square=True,
            cmap="RdYlBu_r",
            vmin=-1,
            vmax=1,
            yticklabels=yticklabels,
            xticklabels=xticklabels,
            cbar=False,
        )
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.tick_params("y", labelsize=5, length=0)
        ax.tick_params("x", rotation=90, labelsize=5)
        ax.set_title(title_list[i], loc="center", fontsize=6)

        i += 1
    ax_cbar = ax.inset_axes([3, 0.25, 1, 0.5], transform=ax.transAxes)
    norm = plt.Normalize(-1, 1)

    cbar = plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.colormaps["RdYlBu_r"]),
             cax=ax_cbar, orientation='vertical')
    cbar.set_label("Spearman correlation", size=6)
    cbar.outline.set_visible(False)
    ax_cbar.tick_params(axis='y', labelsize=6, length=3)

    h5f_no_diam.close()
    plt.savefig(args["output_file"], bbox_inches="tight", dpi=1200)

if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=FigWnmLfParameters)
    main(module.args)
