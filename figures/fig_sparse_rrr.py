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
import latent_factor_plot as lfp
from morph_plotting import basic_morph_plot


class FigSparseRrrParameters(ags.ArgSchema):
    ps_tx_anno_file = ags.fields.InputFile(
        description="Feather file with Patch-seq transcriptomic annotations",
    )
    ps_tx_data_file = ags.fields.InputFile(
        description="Feather file with Patch-seq transcriptomic data",
    )
    inferred_met_type_file = ags.fields.InputFile(
    )
    sparse_rrr_fit_file = ags.fields.InputFile(
    )
    sparse_rrr_feature_file = ags.fields.InputFile(
    )
    sparse_rrr_parameters_file = ags.fields.InputFile(
    )
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
    ap_waveform_file = ags.fields.InputFile(
        description="h5 file with AP waveforms",
    )
    ephys_info_file = ags.fields.InputFile(
        description="json file with ephys trace info for met cells",
    )
    l23_it_proj_pc_file = ags.fields.InputFile(
        description="csv file with target projection related eigengenes",
    )
    layer_depths_file = ags.fields.InputFile(
        description="json file with distances from top of layer to pia",
    )
    layer_aligned_swc_dir = ags.fields.InputDir(
        description="directory with layer-aligned swc files",
    )
    l6b_subthresh_traces_file = ags.fields.InputFile(
    )
    output_file = ags.fields.OutputFile(
    )


MET_TYPE_ORDER = [
	"L2/3 IT",
	"L4 IT",
	"L4/L5 IT",
	"L5 IT-1",
	"L5 IT-2",
	"L5 IT-3 Pld5",
	"L6 IT-1",
	"L6 IT-2",
	"L6 IT-3",
	"L5/L6 IT Car3",
	"L5 ET-1 Chrna6",
	"L5 ET-2",
	"L5 ET-3",
	"L5 NP",
	"L6 CT-1",
	"L6 CT-2",
	"L6b",
]


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


SUBCLASS_INFO = [
    {
        "set_of_types": ["L2/3 IT"],
        "filepart": "L23-IT",
    },
    {
        "set_of_types": ["L4 IT", "L4/L5 IT", "L5 IT-1", "L5 IT-2", "L5 IT-3 Pld5"],
        "filepart": "L4-L5-IT",
    },
    {
        "set_of_types": ["L6 IT-1", "L6 IT-2", "L6 IT-3"],
        "filepart": "L6-IT",
    },
    {
        "set_of_types": ["L5/L6 IT Car3"],
        "filepart": "L5L6-IT-Car3",
    },
    {
        "set_of_types": ["L5 ET-1 Chrna6", "L5 ET-2", "L5 ET-3"],
        "filepart": "L5-ET",
    },
    {
        "set_of_types": ["L5 NP"],
        "filepart": "L5-NP",
    },
    {
        "set_of_types": ["L6 CT-1", "L6 CT-2"],
        "filepart": "L6-CT",
    },
    {
        "set_of_types": ["L6b"],
        "filepart": "L6b",
    }
]

SUBCLASS_DISPLAY_NAMES = {
    "L23-IT": "L2/3\nIT",
    "L4-L5-IT": "L4 &\nL5 IT",
    "L6-IT": "L6\nIT",
    "L5L6-IT-Car3": "L5/L6\nIT Car3",
    "L5-ET": "L5\nET",
    "L5-NP": "L5\nNP",
    "L6-CT": "L6\nCT",
    "L6b": "L6b",
}

def get_sp_rrr_fit_info(filepart, modality, h5f):
    specimen_ids = h5f[filepart][modality]["specimen_ids"][:]
    genes = h5f[filepart][modality]["genes"][:]
    genes = np.array([s.decode() for s in genes])
    w = h5f[filepart][modality]["w"][:].T
    v = h5f[filepart][modality]["v"][:].T

    return specimen_ids, genes, w, v


def select_and_normalize_feature_data(specimen_ids, filepart, modality, select_features, data_df):
    features = select_features[filepart][modality]
    if modality == "ephys":
        feature_data = data_df.loc[specimen_ids, [str(ft) for ft in features]]
        feature_data.columns = select_features[filepart]["ephys_text"]
        feature_data.columns = [relabel_ephys_column(c) for c in feature_data.columns]
    else:
        feature_data = data_df.loc[specimen_ids, features]
        feature_data.columns = [relabel_morph_column(c) for c in feature_data.columns]
    feature_data = (feature_data - feature_data.mean(axis=0)) / feature_data.std(axis=0)
    return feature_data


def plots_for_modality( w, v, gene_data, feature_data, cell_colors, gs, lf_prefix,
        corr_radius=2.7):
    rank = w.shape[1]
    print("rank ", rank)
    if rank < 2:
        lfp.plot_single_rank_latent_factor(
            w, v,
            gene_data, feature_data,
            gs, lf_prefix,
            corr_radius=1,
            n_features_to_plot=5, n_genes_to_plot=5,
            scatter_colors=cell_colors
        )
    else:
        lfp.plot_side_by_side_latent_factors(
            w, v,
            gene_data, feature_data,
            gs, lf_prefix,
            corr_radius=corr_radius,
            n_features_to_plot=3, n_genes_to_plot=3,
            x_index=0, y_index=1,
            scatter_colors=cell_colors)


def relabel_ephys_column(c):
    EPHYS_FEATURE_RELABEL = {
        "first_ap_v": "AP Vm",
        "first_ap_dv": "AP dV/dt",
        "isi_shape": "ISI shape",
        "step_subthresh": "step subthresh.",
        "subthresh_norm": "subthresh. (norm.)",
        "inst_freq": "inst. freq.",
        "spiking_upstroke_downstroke_ratio": "upstroke:downstroke",
        "spiking_peak_v": "AP peak",
        "spiking_fast_trough_v": "AP fast trough",
        "spiking_threshold_v": "AP thresh.",
        "spiking_width": "AP width",
        "inst_freq_norm": "inst. freq. (norm.)",
    }

    split_c = c.split(" ")
    return f"{EPHYS_FEATURE_RELABEL[split_c[0]]} {int(split_c[1]) + 1}"

def relabel_morph_column(c):
    MORPH_FEATURE_RELABEL = {
        "apical_dendrite_bias_x": "apical horiz. bias",
        "apical_dendrite_bias_y": "apical vert. bias",
        "apical_dendrite_extent_x": "apical width",
        "apical_dendrite_extent_y": "apical height",
        "apical_dendrite_depth_pc_0": "apical profile (PC1)",
        "apical_dendrite_depth_pc_1": "apical profile (PC2)",
        "apical_dendrite_depth_pc_2": "apical profile (PC3)",
        "apical_dendrite_depth_pc_3": "apical profile (PC4)",
        "apical_dendrite_max_branch_order": "apical max. branch order",
        "apical_dendrite_max_euclidean_distance": "apical max euclid. dist.",
        "apical_dendrite_max_path_distance": "apical max path dist.",
        "apical_dendrite_mean_contraction": "apical contract.",
        "apical_dendrite_mean_diameter": "apical mean diam.",
        "apical_dendrite_num_branches": "apical num. branches",
        "apical_dendrite_frac_above_basal_dendrite": "apical pct. above basal",
        "apical_dendrite_frac_below_basal_dendrite": "apical pct. below basal",
        "apical_dendrite_frac_intersect_basal_dendrite": "apical pct. overlap basal",
        "apical_dendrite_soma_percentile_x": "apical soma percentile (horiz.)",
        "apical_dendrite_soma_percentile_y": "apical soma percentile (vert.)",
        "apical_dendrite_total_length": "apical total length",
        "apical_dendrite_total_surface_area": "apical surface area",
        "apical_dendrite_num_outer_bifurcations": "apical num. outer bifurc.",
        "apical_dendrite_mean_moments_along_max_distance_projection": "apical mean along max. proj.",
        "apical_dendrite_std_moments_along_max_distance_projection": "apical std. along max. proj.",
        "apical_dendrite_early_branch_path": "apical early branch path",
        "axon_exit_distance": "axon exit dist.",
        "axon_exit_theta": "axon exit angle",
        "basal_dendrite_bias_x": "basal horiz. bias",
        "basal_dendrite_bias_y": "basal vert. bias",
        "basal_dendrite_calculate_number_of_stems": "num. basal stems",
        "apical_dendrite_emd_with_basal_dendrite": "apical/basal EMD",
        "basal_dendrite_extent_x": "basal width",
        "basal_dendrite_extent_y": "basal height",
        "basal_dendrite_max_branch_order": "basal max. branch order",
        "basal_dendrite_max_euclidean_distance": "basal max. euclid. dist.",
        "basal_dendrite_max_path_distance": "basal max path dist.",
        "basal_dendrite_mean_contraction": "basal contract.",
        "basal_dendrite_mean_diameter": "basal mean diam.",
        "basal_dendrite_num_branches": "basal num. branches",
        "basal_dendrite_frac_above_apical_dendrite": "basal pct.\nabove apical",
        "basal_dendrite_frac_below_apical_dendrite": "basal pct.\nbelow apical",
        "basal_dendrite_frac_intersect_apical_dendrite": "basal pct.\noverlap apical",
        "basal_dendrite_soma_percentile_x": "basal soma percentile (horiz.)",
        "basal_dendrite_soma_percentile_y": "basal soma percentile (vert.)",
        "basal_dendrite_total_length": "basal total length",
        "basal_dendrite_total_surface_area": "basal surface area",
        "soma_aligned_dist_from_pia": "soma depth from pia",
    }

    return MORPH_FEATURE_RELABEL[c]


def translation_from_spc_to_feature(specimen_ids, spca_df, ephys_feat_df,
        spca_info_dict, spc_name, feat_name):
    x = spca_df.loc[specimen_ids, str(spca_info_dict[spc_name])]
    y = ephys_feat_df.loc[specimen_ids, feat_name]
    nan_mask = np.isnan(y)

    A = np.vstack([x[~nan_mask], np.ones_like(x[~nan_mask])]).T
    m, c = np.linalg.lstsq(A, y[~nan_mask], rcond=None)[0]
    return m, c


def plot_lf_vs_feature(ax, specimen_ids, lf_ind, feat_name, v,
        gene_lf, features, orig_df,
        alt_feat_df=None, alt_feat_name=None, spca_info_dict=None, m=None, c=None,
        n_span=50, scatter_color="k", xlabel="", ylabel=""):

    # Get latent factors for prediction that span the space (use medians for others than selected LF)
    median_lf = np.median(gene_lf, axis=0)
    span_max = np.abs(gene_lf).max(axis=0)
    lf_span = np.linspace(-span_max[lf_ind], span_max[lf_ind], num=n_span)
    full_lf_span = np.repeat(median_lf, n_span, axis=0).reshape(-1, n_span).T
    full_lf_span[:, lf_ind] = lf_span

    # predict model values (normalized to specific set of cells) from (gene) latent factors
    model_pred = full_lf_span @ v.T

    # Reverse normalization on model prediction to get into original feature space
    features_str = [str(f) for f in features]
    feat_centers = orig_df.loc[specimen_ids, features_str].mean()
    feat_stds = orig_df.loc[specimen_ids, features_str].std()
    model_pred_feat_space = model_pred * feat_stds.values[np.newaxis, :] + feat_centers.values[np.newaxis, :]

    # Convert from sPCA space to "classic" feature space, if requested
    if spca_info_dict is not None and m is not None and c is not None:
        spc_feature_ind = np.flatnonzero(np.array(features) == spca_info_dict[feat_name])[0]
        model_pred_feat_space = m * model_pred_feat_space[:, spc_feature_ind] + c
    else:
        feature_idx = features.index(feat_name)
        model_pred_feat_space = model_pred_feat_space[:, feature_idx]

    if alt_feat_df is not None and alt_feat_name is not None:
        feat_values = alt_feat_df.loc[specimen_ids, alt_feat_name]
    else:
        feat_values = orig_df.loc[specimen_ids, feat_name]

    ax.scatter(gene_lf[:, lf_ind], feat_values,
        c=scatter_color,
        s=5, edgecolors="white", linewidths=0.25)
    ax.plot(lf_span, model_pred_feat_space, linestyle="dashed", c="#333333")

    ax.set_xlabel(xlabel, fontsize=6)
    ax.set_ylabel(ylabel, fontsize=6)
    ax.tick_params(axis='both', labelsize=6)

    sns.despine(ax=ax)


def plot_binned_ap(ax, specimen_ids, values_for_binning, bins, ap_v, ap_v_spec_ids,
        show_scale_bars=False, vmin=None, vmax=None, legend_title=""):

    ap_cmap = matplotlib.colormaps["magma"]
    if vmin is None:
        vmin = values_for_binning.min()
    if vmax is None:
        vmax = values_for_binning.max()
    norm = plt.Normalize(vmin, vmax)

    binned_filt = np.digitize(values_for_binning, bins)

    for bin_idx in range(1, len(bins)):
        ids_in_bin = specimen_ids[binned_filt == bin_idx]
        mask = np.in1d(ap_v_spec_ids, ids_in_bin)

        avg_ap_v = ap_v[mask, :].mean(axis=0)
        std_ap_v = ap_v[mask, :].std(axis=0)
        n_ap_v = np.sum(mask)
        sem_ap_v = std_ap_v / np.sqrt(n_ap_v)

        color = ap_cmap(norm(values_for_binning[binned_filt == bin_idx].mean()))
        ax.plot(np.arange(400) * 0.02 - 4, avg_ap_v, c=color, lw=0.5, label=f"{bins[bin_idx - 1]} to {bins[bin_idx]}")
        ax.fill_between(np.arange(400) * 0.02 - 4, y1=avg_ap_v + sem_ap_v, y2=avg_ap_v - sem_ap_v,
            alpha=0.3, color=color, edgecolor='none', lw=0)

    sns.despine(ax=ax, left=True, bottom=True)
    ax.set(xticks=[], yticks=[])
    ax.set_xlim(left=-2.0, right=3.5)
    ax.set_ylim((-50, 50))
    if show_scale_bars:
        ax.plot([-1.8, -1.8, -0.8], [15, -5, -5], lw=1.5, c="k")
        ax.text(-1.6, 2, "20 mV\n1 ms", fontsize=6, va='baseline', ha='left')
    ax.legend(loc='upper right', fontsize=5, frameon=False,
        title=legend_title, title_fontsize=5, bbox_to_anchor=(1, 1.1))
    ax.tick_params(axis="y", labelsize=6)


def plot_binned_subthresh(specimen_ids, values_for_binning, bins, trace_file, ax,
    legend_title="", maxsize=60000, vmin=None, vmax=None):

    v_cmap = matplotlib.colormaps["magma"]
    if vmin is None:
        vmin = values_for_binning.min()
    if vmax is None:
        vmax = values_for_binning.max()
    norm = plt.Normalize(vmin, vmax)

    binned_filt = np.digitize(values_for_binning, bins)

    h5f = h5py.File(trace_file, "r")
    for bin_idx in range(1, len(bins)):
        e_ids_in_bin = specimen_ids[binned_filt == bin_idx]
        v_list = []
        for spec_id in e_ids_in_bin:
            if str(spec_id) not in h5f.keys():
                continue
            v = h5f[str(spec_id)]['v'][:]
            t = h5f[str(spec_id)]['t'][:]

            # check for faster sampling rate
            if t[1] - t[0] < 1 / 100000:
                # convert from 200 kHz to 50 kHz
                v = v[::4]
                t = t[::4]

            end_baseline_ind = np.flatnonzero(t >= 0.1)[0]
            v_baseline = np.mean(v[:end_baseline_ind])

            v -= v_baseline
            v = v[:maxsize - 1]
            t = t[:maxsize - 1]

            v_list.append(v)
        v_arr = np.vstack(v_list)
        avg_v = np.nanmean(v_arr, axis=0)
        std_v = np.nanstd(v_arr, axis=0)
        n_v = np.sum(~np.isnan(v_arr), axis=0)
        sem_v = std_v / np.sqrt(n_v)

        color = v_cmap(norm(values_for_binning[binned_filt == bin_idx].mean()))
        ax.plot(t, avg_v, c=color, lw=0.5, label=f"{bins[bin_idx - 1]} to {bins[bin_idx]}")
        ax.fill_between(t, y1=avg_v + sem_v, y2=avg_v - sem_v,
            alpha=0.3, color=color, edgecolor='none', lw=0)
    h5f.close()

    sns.despine(ax=ax, bottom=True, left=True)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel("")
    ax.set_xlabel("")

    ax.set_xlim(0, t[-1])
    ax.plot([-0.4, -0.4, -0.15], [0, -10, -10], c='k', lw=1, clip_on=False)
    ax.text(-0.35, -8, "10 mV\n250 ms", ha='left', va='baseline', fontsize=6)

    ax.legend(loc='upper center', fontsize=5, frameon=False,
        title=legend_title, title_fontsize=5, bbox_to_anchor=(0.5, 1.2))
    ax.tick_params(axis="y", length=3, labelsize=6)


def plot_morph_lineup_with_lf(ax, specimen_ids, lf_values, aligned_swc_dir, layer_edges,
        n_to_plot=10, path_color="#cccccc", morph_spacing=300, show_axon=False,
        vmin=None, vmax=None, cbar_label="", cbar_width=0.05, cbar_label_pos="default"):

    morph_cmap = matplotlib.colormaps["magma"]
    if vmin is None:
        vmin = lf_values.min()
    if vmax is None:
        vmax = lf_values.max()
    norm = plt.Normalize(vmin, vmax)

    if n_to_plot >= len(specimen_ids):
        indices = np.arange(len(specimen_ids)).astype(int)
    else:
        indices = np.linspace(0, len(specimen_ids) - 1, num=n_to_plot).astype(int)

    sorted_by_val = np.argsort(lf_values)
    xoffset = 0
    for idx in indices:
        spec_id = specimen_ids[sorted_by_val][idx]
        swc_path = os.path.join(aligned_swc_dir, f"{spec_id}.swc")
        morph = swc.read_swc(swc_path)
        color=morph_cmap(norm(lf_values[sorted_by_val][idx]))
        if not show_axon:
            basic_morph_plot(morph, ax=ax, xoffset=xoffset,
                morph_colors={3: color, 4: color})
        else:
            basic_morph_plot(morph, ax=ax, xoffset=xoffset,
                morph_colors={3: color, 4: color, 2: "#999999"})
        xoffset += morph_spacing

    sns.despine(ax=ax, bottom=True)
    ax.set_xticks([])
    ax.set_xlim(-morph_spacing / 1.25, xoffset - morph_spacing / 2)
    ax.set_aspect("equal")
    ax.set_ylabel("µm", rotation=0, fontsize=6)
    ax.tick_params(axis='y', labelsize=6)
    for e in layer_edges:
        ax.axhline(-e, zorder=-5, color=path_color, linewidth=0.75)

    ax_cbar = ax.inset_axes([1.05, 0, cbar_width, 1], transform=ax.transAxes)
    cbar = plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=morph_cmap),
             cax=ax_cbar, orientation='vertical')
    cbar.set_label(cbar_label, size=6)
    if cbar_label_pos == "below":
        ax_cbar.set_ylabel(cbar_label, size=6, rotation=0, pos="bottom", ha="right")
    cbar.outline.set_visible(False)
    ax_cbar.tick_params(axis='y', labelsize=6, length=3)

    return specimen_ids[sorted_by_val][indices]

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

    ephys_feat_df = pd.read_csv(args["ephys_features_file"], index_col=0)
    # convert AP width from seconds to ms
    ephys_feat_df["ap_mean_width_0_long_square"] *= 1000

    with open(args["spca_feature_info_file"], "r") as f:
        spca_info = json.load(f)
    ind_counter = 0
    spca_info_dict = {}
    for si in spca_info:
        for i in si['indices']:
            spca_info_dict[si['key'] + "_" + str(i)] = ind_counter
            ind_counter += 1

    with h5py.File(args["ap_waveform_file"], "r") as ap_h5f:
        ap_v = ap_h5f["ap_v"][:]
        ap_t = ap_h5f["ap_t"][:]
        ap_v_spec_ids = ap_h5f["specimen_id"][:]
        ap_thresh_deltas = ap_h5f["thresh_delta"][:]

    l23it_proj_pcs_df = pd.read_csv(args["l23_it_proj_pc_file"], index_col=0)

    with open(args["layer_depths_file"], "r") as f:
        layer_info = json.load(f)
    layer_edges = [0] + list(layer_info.values())

    ##### PLOTTING ##############

    fig = plt.figure(figsize=(8.5, 11))
    gs = gridspec.GridSpec(
        5, 2,
        height_ratios=(1, 1, 1, 0.5, 0.5),
        hspace=0.45,
        wspace=0.4,
    )
    fileparts = ("L23-IT", "L4-L5-IT", "L6-IT", "L5-ET", "L6-CT", "L6b")

    gs_first_row = gridspec.GridSpecFromSubplotSpec(
        1, 3,
        width_ratios=(0.3, 1, 1),
        wspace=0.3,
        subplot_spec=gs[0, :],
    )

    filepart_gs_specs = {
        "L23-IT": gs_first_row[1],
        "L4-L5-IT": gs_first_row[2],
        "L6-IT": gs[1, 0],
        "L5-ET": gs[1, 1],
        "L6-CT": gs[2, 0],
        "L6b": gs[2, 1],
    }
    filepart_corr_radius = {
        "L5-ET": 3.2,
    }

    for filepart in fileparts:
        gs_sub = gridspec.GridSpecFromSubplotSpec(
            2, 1,
            hspace=0.2,
            subplot_spec=filepart_gs_specs[filepart],
        )
        for i, modality in enumerate(("ephys", "morph")):
            print(filepart, modality)
            specimen_ids, genes, w, v = get_sp_rrr_fit_info(
                filepart, modality, h5f
            )
            cell_colors = [MET_TYPE_COLORS[t] for t in inf_met_df.loc[specimen_ids, "inferred_met_type"]]
            feature_data = select_and_normalize_feature_data(
                specimen_ids, filepart, modality,
                sp_rrr_features_info, feature_df_dict[modality])
            sample_ids = ps_anno_df.loc[specimen_ids, "sample_id"]
            gene_data = ps_data_df.loc[sample_ids, genes]

            if filepart in filepart_corr_radius:
                corr_radius = filepart_corr_radius[filepart]
            else:
                corr_radius = 2.7

            plots_for_modality(
                w, v,
                gene_data, feature_data,
                cell_colors, gs_sub[i], prefix_dict[modality],
                corr_radius=corr_radius
            )

    gs_lower_a = gridspec.GridSpecFromSubplotSpec(
        1, 6,
        width_ratios=(2, 0.3, 1, 1, 0.3, 1),
        subplot_spec=gs[3, :],
        wspace=0.1,
    )

    ax_r2 = plt.subplot(gs_lower_a[0])
    full_fileparts = ("L23-IT", "L4-L5-IT", "L6-IT", "L5L6-IT-Car3", "L5-ET", "L5-NP", "L6-CT", "L6b")
    values_for_df = []
    for filepart in full_fileparts:
        for modality in ("ephys", "morph"):
            values_for_df.append((
                SUBCLASS_DISPLAY_NAMES[filepart],
                modality,
                sp_rrr_parameters_info[filepart][modality]["sparse_rrr"]["r2_relaxed"]
            ))
    r2_df = pd.DataFrame(values_for_df, columns=("subclass", "modality", "r2"))
    sns.barplot(
        data=r2_df,
        x="subclass",
        y="r2",
        hue="modality",
        ax=ax_r2,
        legend=False,
    )
    ax_r2.set_ylim(top=0.25)
    ax_r2.tick_params(axis='both', labelsize=6)
    ax_r2.set_xlabel("")
    ax_r2.set_ylabel("CV-$R^2$", fontsize=6)
    sns.despine(ax=ax_r2)

    # L2/3 IT examples
    filepart = "L23-IT"
    modality = "ephys"
    specimen_ids, genes, w, v = get_sp_rrr_fit_info(
        filepart, modality, h5f
    )
    cell_colors = [MET_TYPE_COLORS[t] for t in inf_met_df.loc[specimen_ids, "inferred_met_type"]]
    features = sp_rrr_features_info[filepart][modality]
    feature_data = select_and_normalize_feature_data(
        specimen_ids, filepart, modality,
        sp_rrr_features_info, feature_df_dict[modality])
    sample_ids = ps_anno_df.loc[specimen_ids, "sample_id"]
    gene_data = ps_data_df.loc[sample_ids, genes]
    gene_lf = gene_data.values @ w

    m, c = translation_from_spc_to_feature(
        specimen_ids,
        ephys_df,
        ephys_feat_df,
        spca_info_dict,
        "spiking_width_0",
        "ap_mean_width_0_long_square",
    )

    ax = plt.subplot(gs_lower_a[2])
    plot_lf_vs_feature(
        ax,
        specimen_ids,
        0,
        "spiking_width_0",
        v,
        gene_lf,
        features,
        ephys_df,
        alt_feat_df=ephys_feat_df,
        alt_feat_name="ap_mean_width_0_long_square",
        spca_info_dict=spca_info_dict,
        m=m, c=c,
        n_span=50,
        scatter_color=cell_colors,
        xlabel="L2/3 IT E-LF-1", ylabel="mean AP width (ms)"
    )
    ax.set_ylim(0.5, 1.5)

    ax = plt.subplot(gs_lower_a[3])
    plot_binned_ap(
        ax,
        specimen_ids,
        gene_lf[:, 0],
        [-3, -1, 1, 3],
        ap_v, ap_v_spec_ids,
        show_scale_bars=True,
        legend_title="L2/3 IT\nE-LF-1",
    )

    ax = plt.subplot(gs_lower_a[5])
    sns.regplot(
        x=gene_lf[:, 0],
        y=l23it_proj_pcs_df.loc[specimen_ids, "PC1"],
        ci=None,
        scatter_kws=dict(s=5, color=cell_colors, edgecolors="white", linewidths=0.25),
        line_kws=dict(lw=1, color="black"),
        ax=ax,
    )
    ax.tick_params(axis='both', length=3, labelsize=6)
    ax.set_xlabel("L2/3 IT E-LF-1", size=6)
    ax.set_ylabel("projection-target eigengene", size=6)
    ax.set_ylim(-15, 20)
    ax.axhline(y=0, color='gray', linestyle='dotted', lw=0.5)
    ax.annotate(
        "more like\nVISpm-targeting",
        (1.1, 0),
        (1.1, -10),
        xycoords=ax.get_yaxis_transform(),
        arrowprops=dict(
            arrowstyle='<-',
            connectionstyle='arc3',
            relpos=(0, 1),
        ),
        va='top',
        ha='left',
        fontsize=6,
    )
    ax.annotate(
        "more like\nVISal-targeting",
        (1.1, 0),
        (1.1, 10),
        xycoords=ax.get_yaxis_transform(),
        textcoords=ax.get_yaxis_transform(),
        arrowprops=dict(
            facecolor='black',
            arrowstyle='<|-',
            connectionstyle='arc3',
            relpos=(0, 0),
        ),
        va='baseline',
        ha='left',
        fontsize=6,
    )
    sns.despine(ax=ax)

    #########

    gs_lower_b = gridspec.GridSpecFromSubplotSpec(
        1, 5,
        width_ratios=(1, 1, 1, 1, 2),
        subplot_spec=gs[4, :],
        wspace=0.5,
    )

    # L6b examples
    filepart = "L6b"
    modality = "ephys"
    specimen_ids, genes, w, v = get_sp_rrr_fit_info(
        filepart, modality, h5f
    )
    cell_colors = [MET_TYPE_COLORS[t] for t in inf_met_df.loc[specimen_ids, "inferred_met_type"]]
    features = sp_rrr_features_info[filepart][modality]
    feature_data = select_and_normalize_feature_data(
        specimen_ids, filepart, modality,
        sp_rrr_features_info, feature_df_dict[modality])
    sample_ids = ps_anno_df.loc[specimen_ids, "sample_id"]
    gene_data = ps_data_df.loc[sample_ids, genes]
    gene_lf = gene_data.values @ w

    m, c = translation_from_spc_to_feature(
        specimen_ids,
        ephys_df,
        ephys_feat_df,
        spca_info_dict,
        "step_subthresh_1",
        "input_resistance",
    )

    ax = plt.subplot(gs_lower_b[0])
    plot_lf_vs_feature(
        ax,
        specimen_ids,
        0,
        "step_subthresh_1",
        v,
        gene_lf,
        features,
        ephys_df,
        alt_feat_df=ephys_feat_df,
        alt_feat_name="input_resistance",
        spca_info_dict=spca_info_dict,
        m=m, c=c,
        n_span=50,
        scatter_color=cell_colors,
        xlabel="L6b E-LF-1", ylabel="input resistance (MΩ)"
    )

    ax = plt.subplot(gs_lower_b[1])
    plot_binned_subthresh(
        specimen_ids,
        gene_lf[:, 0],
        [-3, -1, 1, 3],
        args["l6b_subthresh_traces_file"],
        ax,
        legend_title="L6b\nE-LF-1"
    )


    # L6 CT examples
    filepart = "L6-CT"
    modality = "morph"
    specimen_ids, genes, w, v = get_sp_rrr_fit_info(
        filepart, modality, h5f
    )
    cell_colors = [MET_TYPE_COLORS[t] for t in inf_met_df.loc[specimen_ids, "inferred_met_type"]]
    features = sp_rrr_features_info[filepart][modality]
    feature_data = select_and_normalize_feature_data(
        specimen_ids, filepart, modality,
        sp_rrr_features_info, feature_df_dict[modality])
    sample_ids = ps_anno_df.loc[specimen_ids, "sample_id"]
    gene_data = ps_data_df.loc[sample_ids, genes]
    gene_lf = gene_data.values @ w

    ax = plt.subplot(gs_lower_b[2])
    plot_lf_vs_feature(
        ax,
        specimen_ids,
        0,
        "apical_dendrite_max_path_distance",
        v,
        gene_lf,
        features,
        morph_df,
        n_span=50,
        scatter_color=cell_colors,
        xlabel="L6 CT M-LF-1", ylabel="apical max. path dist. (µm)"
    )

    ax = plt.subplot(gs_lower_b[3])
    plot_lf_vs_feature(
        ax,
        specimen_ids,
        1,
        "soma_aligned_dist_from_pia",
        v,
        gene_lf,
        features,
        morph_df,
        n_span=50,
        scatter_color=cell_colors,
        xlabel="L6 CT M-LF-2", ylabel="soma depth (µm)"
    )
    ax.invert_yaxis()

    ax = plt.subplot(gs_lower_b[4])
    plot_morph_lineup_with_lf(
        ax,
        specimen_ids,
        gene_lf[:, 0],
        args["layer_aligned_swc_dir"],
        layer_edges,
        cbar_label="L6 CT M-LF-1",
        vmin=-4, vmax=4,
        n_to_plot=9,
    )

    plt.savefig(args["output_file"], dpi=300, bbox_inches="tight")

    h5f.close()



if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=FigSparseRrrParameters)
    main(module.args)

