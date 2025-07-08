import json
import os
import re
import h5py
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import matplotlib.patches as patches
import seaborn as sns
import argschema as ags
import allensdk.core.swc as swc
import ipfx.script_utils as su
from ipfx.epochs import get_stim_epoch
from scipy.stats import kruskal
from scikit_posthocs import posthoc_dunn
from morph_plotting import basic_morph_plot, adjust_lightness
from fig_met_it import (plot_morph_lineup, plot_depth_profiles, process_fi_curves,
    plot_avg_fi_for_mets, plot_avg_ap_for_mets, plot_gene_dots,
    natural_sort_key)

MET_TYPE_ORDER = [
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


class FigMetNonItParameters(ags.ArgSchema):
    inferred_met_type_file = ags.fields.InputFile(
        default="csv file with inferred met types",
    )
    morph_file = ags.fields.InputFile(
        description="csv file with unnormalized morph features",
    )
    aligned_depths_file = ags.fields.InputFile(
        description="csv file with aligned depth profiles",
    )
    layer_depths_file = ags.fields.InputFile(
        description="json file with distances from top of layer to pia",
    )
    layer_aligned_swc_dir = ags.fields.InputDir(
        description="directory with layer-aligned swc files",
    )
    ephys_features_file = ags.fields.InputFile(
        description="csv file with traditional ephys features",
    )
    ap_waveform_file = ags.fields.InputFile(
        description="h5 file with AP waveforms",
    )
    ephys_info_file = ags.fields.InputFile(
        description="json file with ephys trace info for met cells",
    )
    et_inst_rate_file = ags.fields.InputFile(
        description="csv file with inst rate measurements",
    )
    l5et_de_ion_channels_file = ags.fields.InputFile(
        description="csv file with de ion channel information for l5 et",
    )
    output_file = ags.fields.OutputFile(
        description="output file",
    )


def main(args):
    # Load type and data
    inf_met_df = pd.read_csv(args["inferred_met_type_file"], index_col=0)
    morph_df = pd.read_csv(args["morph_file"], index_col=0)
    merge_df = pd.merge(inf_met_df, morph_df[["soma_aligned_dist_from_pia"]],
        left_index=True, right_index=True)

    # Load morph information
    with open(args["layer_depths_file"], "r") as f:
        layer_info = json.load(f)
    layer_edges = [0] + list(layer_info.values())
    layer_mids = [(l1 + l2) / 2 for l1, l2 in zip(layer_edges[:-1], layer_edges[1:])]

    # Load depth profiles
    hist_df = pd.read_csv(args["aligned_depths_file"], index_col=0)
    hist_df *= 1.144 # approx. scaling to microns
    all_cols = sorted(hist_df.columns, key=natural_sort_key)
    basal_cols = [c for c in all_cols if c.startswith("3_")]
    apical_cols = [c for c in all_cols if c.startswith("4_")]

    fig = plt.figure(figsize=(4, 8))
    gs = gridspec.GridSpec(
        4, 1, height_ratios=(2.5, 2.5, 0.3, 3.5), hspace=0.1)

    # Load AP and ephys info
    with h5py.File(args["ap_waveform_file"], "r") as h5f:
        ap_v = h5f["ap_v"][:]
        ap_t = h5f["ap_t"][:]
        ap_spec_ids = h5f["specimen_id"][:]
        ap_thresh_deltas = h5f["thresh_delta"][:]

    all_non_it_ids = inf_met_df.index.values[inf_met_df.inferred_met_type.isin(MET_TYPE_ORDER)]
    all_non_it_mask =  np.in1d(ap_spec_ids, all_non_it_ids)
    all_non_it_avg_ap = ap_v[all_non_it_mask, :].mean(axis=0)

    ephys_feat_df = pd.read_csv(args["ephys_features_file"], index_col=0)
    sub_df_long_filled, bins  = process_fi_curves(ephys_feat_df)
    sub_df_long_met = pd.merge(sub_df_long_filled, inf_met_df,
        left_index=True, right_index=True, how='left')
    print("max stim amp: ", sub_df_long_met.stimulus_amplitude.max())
    print("max bin:", bins[-1])
    ephys_feat_df["ap_1_height_0_long_square"] = (
        ephys_feat_df["ap_1_peak_v_0_long_square"] -
        ephys_feat_df["ap_1_fast_trough_v_0_long_square"])
    ephys_feat_df["ap_1_width_0_long_square"] *= 1000 # to ms
    ephys_feat_df = pd.merge(ephys_feat_df, inf_met_df, left_index=True, right_index=True)

    with open(args['ephys_info_file'], "r") as f:
        ephys_info = json.load(f)
    ephys_info_by_specimen_id = {d["specimen_id"]: d for d in ephys_info if d is not None}

    l5et_de_df = pd.read_csv(args["l5et_de_ion_channels_file"], index_col=0)
    et_rate_df = pd.read_csv(args['et_inst_rate_file'], index_col=0)

    # Find sweeps for each cell with max inst to avg ratio
    et_rate_df["ratio"] = et_rate_df['max_inst_rate'] / et_rate_df['avg_rate']
    max_ratios = et_rate_df.reset_index().groupby("specimen_id")["ratio"].max()
    rate_mask = et_rate_df['ratio'].values == max_ratios[et_rate_df.index]
    et_rate_maxratio_df = et_rate_df.loc[rate_mask.values, :].copy()

    # Find min rate sweeps for each cell where number of spikes > 1
    et_rate_df_over_1 = et_rate_df.loc[et_rate_df['avg_rate'] >= 2, :].reset_index()
    et_rate_group_spec_id = et_rate_df_over_1.groupby("specimen_id")
    idx_min_avg_rate = et_rate_group_spec_id['avg_rate'].idxmin()
    et_rate_min_avg_rate_df = et_rate_df_over_1.loc[idx_min_avg_rate, :].set_index("specimen_id")
    et_rate_min_avg_rate_df = et_rate_min_avg_rate_df.merge(inf_met_df,
        left_index=True, right_index=True)

    # Plot example sweeps
    et_trace_examples = [
        826780116, # et-met-1
        880871943, # et-met-2
        912767246, # et-met-3, lower ratio
        605493156, # et-met-3, higher ratio
    ]


    # Morphology plots
    gs_morph = gridspec.GridSpecFromSubplotSpec(
        2, 2,
        subplot_spec=gs[0],
        wspace=0.1,
        width_ratios=(9, 0.5),
    )
    ax_morph_top = plt.subplot(gs_morph[0, 0])
    plot_morph_lineup(
        ax_morph_top, merge_df, MET_TYPE_ORDER[:4],
        args["layer_aligned_swc_dir"], layer_edges
    )
    ax_morph_bottom = plt.subplot(gs_morph[1, 0])
    plot_morph_lineup(
        ax_morph_bottom, merge_df, MET_TYPE_ORDER[4:],
        args["layer_aligned_swc_dir"], layer_edges
    )

    ax_profiles_top = ax_morph_top.inset_axes([1.05, 0, 0.1, 1],
        transform=ax_morph_top.transAxes)
    ax_morph_top.sharey(ax_profiles_top)
    plot_depth_profiles(ax_profiles_top, merge_df, MET_TYPE_ORDER[:4],
        hist_df, basal_cols, apical_cols, layer_edges)
    ax_profiles_top.set_title("Avg. dendrite\ndepth profiles", fontsize=5)
    ax_profiles_bottom = ax_morph_bottom.inset_axes([1.05, 0, 0.1, 1],
        transform=ax_morph_bottom.transAxes)
    ax_morph_bottom.sharey(ax_profiles_bottom)
    plot_depth_profiles(ax_profiles_bottom, merge_df, MET_TYPE_ORDER[4:],
        hist_df, basal_cols, apical_cols, layer_edges)

    # Electrophysiology plots
    gs_ephys = gridspec.GridSpecFromSubplotSpec(
        2, 3,
        subplot_spec=gs[1],
    )
    met_types_left = MET_TYPE_ORDER[:3]
    met_types_middle = ["L5 NP", "L6b"]
    met_types_right = ["L6 CT-1", "L6 CT-2"]
    plot_avg_ap_for_mets(
        plt.subplot(gs_ephys[0, 0]),
        inf_met_df,
        met_types_left,
        ap_v,
        ap_spec_ids,
        show_scale_bars=True
    )
    plot_avg_ap_for_mets(
        plt.subplot(gs_ephys[0, 1]),
        inf_met_df,
        met_types_middle,
        ap_v,
        ap_spec_ids,
        show_scale_bars=False
    )
    plot_avg_ap_for_mets(
        plt.subplot(gs_ephys[0, 2]),
        inf_met_df,
        met_types_right,
        ap_v,
        ap_spec_ids,
        show_scale_bars=False
    )
    for i in range(3):
        ax = plt.subplot(gs_ephys[0, i])
        ax.plot(
            np.arange(400) * 0.02 - 4, all_non_it_avg_ap,
            c="gray", linestyle="dotted", zorder=2, linewidth=0.75,
        )

    plot_avg_fi_for_mets(
        plt.subplot(gs_ephys[1, 0]),
        sub_df_long_met,
        inf_met_df,
        met_types_left,
        ylim=(0, 30),
        show_ylabel=True
    )
    plot_avg_fi_for_mets(
        plt.subplot(gs_ephys[1, 1]),
        sub_df_long_met,
        inf_met_df,
        met_types_middle,
        ylim=(0, 30),
        show_ylabel=False
    )
    plot_avg_fi_for_mets(
        plt.subplot(gs_ephys[1, 2]),
        sub_df_long_met,
        inf_met_df,
        met_types_right,
        ylim=(0, 30),
        show_ylabel=False
    )

    gs_l5et = gridspec.GridSpecFromSubplotSpec(
        4, 2,
        subplot_spec=gs[3],
        hspace=0.3, wspace=0.1,
        width_ratios=(1, 1.2),
        height_ratios=(1, 1, 0.2, 1.1),
    )

    gs_examples = gridspec.GridSpecFromSubplotSpec(
        2, 2,
        subplot_spec=gs_l5et[0, 0],
        hspace=0.2, wspace=0.4,
    )
    inset_shift = (
        0.01,
        0.01,
        0.03,
        0.045,
    )
    inset_width = 0.075

    for ind, spec_id in enumerate(et_trace_examples):
        ax = plt.subplot(gs_examples[ind % 2, ind // 2])

        sweep_number = et_rate_maxratio_df.at[spec_id, "sweep_number"]
        data_set = su.dataset_for_specimen_id(spec_id,
            'lims-nwb2', None, None)
        met_type = inf_met_df.at[spec_id, "inferred_met_type"]
        color = MET_TYPE_COLORS[met_type]
        swp = data_set.sweep(sweep_number)
        v = swp.v
        t = swp.t
        i = swp.i
        start, end = get_stim_epoch(i)
        ax.plot(t, v, lw=0.5, color=color)
        if spec_id == et_trace_examples[0]:
            ax.plot([t[start] - 0.8, t[start] - 0.8, t[start] - 0.3],
                [20, -30, -30], lw=1, c='k', clip_on=False)
            ax.text(t[start] - 0.7, -18, "50 mV\n500 ms", fontsize=5, va='baseline', ha='left')
        if spec_id != et_trace_examples[3]:
            title_str = met_type
            if title_str.count(" ") > 1:
                title_split = title_str.split(" ")
                title_str = title_split[0] + " " + title_split[1] + "\n" + title_split[2]
            ax.set_title(title_str, color=color, fontsize=5, va='top')

        rect_for_inset = patches.Rectangle(
            (t[start] + inset_shift[ind], -80),
            inset_width, 140,
            clip_on=False,
            edgecolor="#999999",
            fill=False,
            linewidth=0.25,
            zorder=5,
        )
        rect_inset_border = patches.Rectangle(
            (t[start] + inset_shift[ind], -80),
            inset_width, 140,
            clip_on=False,
            edgecolor="#999999",
            fill=False,
            linewidth=0.25,
            zorder=5,
        )

        ax.add_patch(rect_for_inset)

        ax.set_ylim(-80, 60)
        ax.set_xlim(t[start] - 0.1, t[end] + 0.1)
        ax_inset = ax.inset_axes([1.03, 0, 0.25, 1])
        ax.sharey(ax_inset)
        ax_inset.plot(t, v, lw=0.5, color=color)
        ax_inset.set_xlim(
            t[start] + inset_shift[ind],
            t[start] + inset_shift[ind] + inset_width
        )
        ax_inset.add_patch(rect_inset_border)

        sns.despine(ax=ax, left=True, bottom=True)
        sns.despine(ax=ax_inset, left=True, bottom=True)
        ax.set_yticks([])
        ax.set_xticks([])
        ax_inset.set_xticks([])

    ax_l5et_scatter = plt.subplot(gs_l5et[1, 0])
    sns.scatterplot(
        data=et_rate_min_avg_rate_df,
        x="avg_rate",
        y="max_inst_rate",
        hue="inferred_met_type",
        hue_order=["L5 ET-2", "L5 ET-1 Chrna6", "L5 ET-3"],
        palette=MET_TYPE_COLORS,
        s=5,
        edgecolors='white',
        alpha=0.5,
        legend=False,
        ax=ax_l5et_scatter,
    )
    sns.despine(ax=ax_l5et_scatter)
    ax_l5et_scatter.tick_params(axis="both", labelsize=6, length=3)
    ax_l5et_scatter.set_xlabel("num. spikes", fontsize=6)
    ax_l5et_scatter.set_ylabel("max. instantaneous rate\n(spikes/s)", fontsize=6)


    plot_gene_dots(
        l5et_de_df.loc[l5et_de_df["ps_is_de"] & l5et_de_df["ref_is_de"], :],
        "gene",
        ["ps_mean_L5.ET.1.Chrna6", "ps_mean_L5.ET.2", "ps_mean_L5.ET.3"],
        ["ps_present_L5.ET.1.Chrna6", "ps_present_L5.ET.2", "ps_present_L5.ET.3"],
        ["L5 ET-1 Chrna6", "L5 ET-2", "L5 ET-3"],
        gs_l5et[3, 0],
        scatter_factor=15,
    )


    plt.savefig(args["output_file"], bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=FigMetNonItParameters)
    main(module.args)

