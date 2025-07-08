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
import seaborn as sns
import argschema as ags
import allensdk.core.swc as swc
import ipfx.script_utils as su
from scipy.stats import kruskal
from scikit_posthocs import posthoc_dunn
from morph_plotting import basic_morph_plot, adjust_lightness


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


class FigMetItParameters(ags.ArgSchema):
    inferred_met_type_file = ags.fields.InputFile(
        description="csv file with inferred met types",
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
    hump_measure_file = ags.fields.InputFile(
        description="csv file with hump measurements",
    )
    subthresh_fit_file = ags.fields.InputFile(
        description="csv file with measurements of subthreshold depolarizing sweeps",
    )
    risetime_file = ags.fields.InputFile(
        description="csv file with risetimes of subthresh depol sweeps",
    )
    l6it_de_ion_channels_file = ags.fields.InputFile(
        description="csv file with de ion channel information for l6 it",
    )
    output_file = ags.fields.OutputFile(
        default="fig_met_it_Feb2025.pdf",
    )


def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]


def plot_morph_lineup(ax, merge_df, met_types, aligned_swc_dir, layer_edges,
        plot_quantiles=None, path_color="#cccccc", morph_spacing=300):
    if plot_quantiles is None:
        plot_quantiles = np.linspace(0, 1, 5)


    xoffset = 0
    for met in met_types:
        print(met)
        spec_ids = merge_df.loc[merge_df["met_type"] == met, :].sort_values("soma_aligned_dist_from_pia").index

        inds = np.arange(len(spec_ids))
        if len(inds) <= len(plot_quantiles):
            plot_inds = inds
        else:
            plot_inds = np.quantile(inds, plot_quantiles).astype(int)
            if plot_inds[0] + 1 < plot_inds[1]:
                plot_inds[0] += 1
            if plot_inds[-1] - 1 > plot_inds[-2]:
                plot_inds[-1] -= 1

        print("Plotted morphs for met-type", met)
        print(spec_ids.values[plot_inds])
        color = MET_TYPE_COLORS[met]
        morph_locs = []
        for spec_id in spec_ids.values[plot_inds]:
            swc_path = os.path.join(aligned_swc_dir, f"{spec_id}.swc")
            morph = swc.read_swc(swc_path)
            basic_morph_plot(morph, ax=ax, xoffset=xoffset,
                morph_colors={3: adjust_lightness(color, 0.5), 4: color})
            morph_locs.append(xoffset)
            xoffset += morph_spacing
        title_str = met
        if title_str.count(" ") > 1:
            title_split = title_str.split(" ")
            title_str = title_split[0] + " " + title_split[1] + "\n" + title_split[2]
        ax.text(np.mean(morph_locs), 100, title_str, fontsize=6, color=color, ha="center")

    sns.despine(ax=ax, bottom=True)
    ax.set_xticks([])
    ax.set_xlim(-morph_spacing / 1.25, xoffset - morph_spacing / 2)
    ax.set_aspect("equal")
    ax.set_ylabel("µm", rotation=0, fontsize=7)
    ax.tick_params(axis='y', labelsize=6)
    for e in layer_edges:
        ax.axhline(-e, zorder=-5, color=path_color, linewidth=0.5)


def plot_depth_profiles(ax, merge_df, met_types, hist_df, basal_cols, apical_cols,
        layer_edges, plot_quantiles=None, path_color="#cccccc", spacing=200, bin_width=5):
    if plot_quantiles is None:
        plot_quantiles = np.linspace(0, 1, 5)

    xoffset = 0
    for met in met_types:
        spec_ids = merge_df.loc[merge_df["met_type"] == met, :].sort_values("soma_aligned_dist_from_pia").index

        color = MET_TYPE_COLORS[met]
        sub_depth_df = hist_df.loc[
            hist_df.index.intersection(spec_ids), basal_cols + apical_cols]
        avg_depth = sub_depth_df.mean(axis=0)
        basal_avg = avg_depth[basal_cols].values
        apical_avg = avg_depth[apical_cols].values
        all_avg = basal_avg + apical_avg
        zero_mask = all_avg > 0
        ax.plot(
            all_avg[zero_mask] + xoffset,
            -np.arange(len(all_avg))[zero_mask] * bin_width,
            c=color, linewidth=0.5, zorder=10
        )
        xoffset += spacing

    sns.despine(ax=ax, bottom=True, left=True)
    ax.set_xticks([])
    ax.tick_params(axis="y", left=False, labelleft=False)
    for e in layer_edges:
        ax.axhline(-e, zorder=-5, color=path_color, linewidth=0.5)


def process_fi_curves(ephys_feat_df):
    amp_cols = ephys_feat_df.columns[
        ephys_feat_df.columns.str.startswith("stimulus_amplitude_")]
    rate_cols = ephys_feat_df.columns[
        ephys_feat_df.columns.str.startswith("avg_rate_")]
    sub_df = ephys_feat_df[rate_cols.union(amp_cols)].reset_index()
    sub_df.columns = sub_df.columns.str.replace("_long_square", "")
    sub_df_long = pd.wide_to_long(
        sub_df,
        stubnames=["avg_rate", "stimulus_amplitude"],
        i="specimen_id", j="step",
        sep="_").dropna().reset_index().set_index("specimen_id"
    )

    bins = np.arange(-20, 460, 20)
    sub_df_long["amp_bin"] = pd.cut(sub_df_long.stimulus_amplitude, bins=bins)

    new_zero_rows = []
    for n, g in sub_df_long.groupby("specimen_id"):
        # print(g)
        min_bin = g["amp_bin"].min()
        lower_bins = sub_df_long["amp_bin"].cat.categories[sub_df_long["amp_bin"].cat.categories < min_bin]
        for lb in lower_bins:
            new_zero_rows.append({
                "specimen_id": n,
                "step": -1,
                "stimulus_amplitude": np.nan,
                "amp_bin": lb,
                "avg_rate": 0,
            })
    new_df = pd.DataFrame(new_zero_rows).set_index("specimen_id")
    sub_df_long_filled = pd.concat([sub_df_long, new_df])
    return sub_df_long_filled, bins


def plot_avg_ap_for_mets(ax, inf_met_df, met_types, ap_v, ap_spec_ids,
        show_scale_bars=False):
    label_position = 1.05
    for met_type in met_types:
        met_type_ids = inf_met_df.index.values[inf_met_df.inferred_met_type == met_type]
        met_mask = np.in1d(ap_spec_ids, met_type_ids)
        ax.plot(np.arange(400) * 0.02 - 4, ap_v[met_mask, :].mean(axis=0),
                lw=0.75, c=MET_TYPE_COLORS[met_type], zorder=5)
        ax.text(1.0, label_position, met_type, ha='right', va='top',
            color=MET_TYPE_COLORS[met_type], fontsize=5, transform=ax.transAxes)
        label_position -= 0.125
        sns.despine(ax=ax, left=True, bottom=True)
        ax.set(xticks=[], yticks=[])
        ax.set_xlim(left=-2.0, right=3.5)
        ax.set_ylim((-50, 50))
        if show_scale_bars:
            ax.plot([-1.8, -1.8, -0.8], [15, -5, -5], lw=1.5, c="k")
            ax.text(-1.6, 2, "20 mV\n1 ms", fontsize=5, va='baseline', ha='left')


def plot_avg_fi_for_mets(ax, sub_df_long_met, inf_met_df, met_types,
        min_n=5, xlim=(0, 250), ylim=(0, 25), show_ylabel=True, show_xlabel=True):
    grouped = sub_df_long_met.groupby("inferred_met_type")
    for n in met_types:
        g = grouped.get_group(n)
        avg_by_cell = g.groupby(["amp_bin", "specimen_id"])["avg_rate"].mean().reset_index()
        avg_rates = avg_by_cell.groupby(["amp_bin"])["avg_rate"].mean()
        count_mask = avg_by_cell.groupby("amp_bin")["avg_rate"].count() >= min_n
        err_rates = avg_by_cell.groupby("amp_bin")["avg_rate"].std() / np.sqrt(avg_by_cell.groupby("amp_bin")["avg_rate"].count())
        within_xlim_mask = avg_rates.index.right.values < xlim[1]

        ax.errorbar(
            x=avg_rates.index.right[count_mask & within_xlim_mask],
            y=avg_rates[count_mask & within_xlim_mask],
            yerr=err_rates[count_mask & within_xlim_mask],
            capsize=3,
            capthick=0.75,
            color=MET_TYPE_COLORS[n],
            lw=0.75,
        )
        ax.scatter(x=avg_rates.index.right[count_mask], s=5, c=MET_TYPE_COLORS[n], y=avg_rates[count_mask])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    sns.despine(ax=ax)
    ax.tick_params(axis="both", labelsize=6, length=3)
    if show_ylabel:
        ax.set_ylabel("firing rate (spikes/s)", size=6)
    if show_xlabel:
        ax.set_xlabel("stimulus amplitude (pA)", size=6)


def plot_l6it_morph_scatter(ax, inf_met_df, morph_df,
        met_types=["L6 IT-1", "L6 IT-2", "L6 IT-3", "L5/L6 IT Car3"]):
    l6_it_ids = inf_met_df.loc[
        inf_met_df['inferred_met_type'].isin(met_types)].index.values
    l6_it_morph_ids = morph_df.index.intersection(l6_it_ids)
    l6_morph_df = morph_df.loc[l6_it_morph_ids, :].copy()
    l6_morph_df = l6_morph_df.merge(inf_met_df, left_index=True, right_index=True)

    sns.scatterplot(
        data=l6_morph_df,
        x="apical_dendrite_bias_y",
        y="apical_dendrite_max_path_distance",
        hue="met_type",
        palette=MET_TYPE_COLORS,
        s=5,
        edgecolors='white',
        alpha=0.8,
        legend=False,
        ax=ax
    )
    ax.tick_params(axis='both', labelsize=6)
    ax.set_ylabel("apical\nmax path dist. (µm)", fontsize=6)
    ax.set_xlabel("apical vertical bias (µm)", fontsize=6)
    ax.text(
        350, 800,
        "L6 IT-1",
        fontsize=5,
        color=MET_TYPE_COLORS["L6 IT-1"],
        va='baseline',
        ha='left',
    )
    ax.text(
        400, 450,
        "L6 IT-2",
        fontsize=5,
        color=MET_TYPE_COLORS["L6 IT-2"],
        va='baseline',
        ha='left',
    )
    ax.text(
        -200, 600,
        "L6 IT-3",
        fontsize=5,
        color=MET_TYPE_COLORS["L6 IT-3"],
        va='baseline',
        ha='left',
    )
    ax.text(
        50, 230,
        "L5/L6 IT Car3",
        fontsize=5,
        color=MET_TYPE_COLORS["L5/L6 IT Car3"],
        va='baseline',
        ha='left',
    )
    sns.despine(ax=ax)


def plot_l6it_ephys_examples(gs_subplot, inf_met_df,
        ephys_info_by_specimen_id, depol_subthresh_df, risetime_df,
        ephys_example_ids=[794963377, 901060224, 1000110992, 973832650]):

    gs_examples = gridspec.GridSpecFromSubplotSpec(
        2, 2, gs_subplot,
        wspace=0.05,
        hspace=0.2,
    )
    for i, specimen_id in enumerate(ephys_example_ids):
        met_type = inf_met_df.at[specimen_id, "inferred_met_type"]
        color = MET_TYPE_COLORS[met_type]
        v_ax = plt.subplot(gs_examples[i])
        data_set = su.dataset_for_specimen_id(specimen_id,
                'lims-nwb2', None, None)

        # find sweeps of interest
        depol_df = depol_subthresh_df.loc[(depol_subthresh_df['specimen_id'] == specimen_id) &
            (depol_subthresh_df['stim_amp'] > 0), :].sort_values("stim_amp", ascending=False)
        sub_depol_sweep_num = int(depol_df['sweep_number'].values[0])
        m90_sweep_num = ephys_info_by_specimen_id[specimen_id]['subthresh_sweep_minus90']
        rheo_sweep_num = ephys_info_by_specimen_id[specimen_id]['rheo_sweep']

        for s in (m90_sweep_num, sub_depol_sweep_num):
            swp = data_set.sweep(s)
            v_ax.plot(swp.t, swp.v, c=color, lw=0.5, linestyle='solid', zorder=10)

        risetime = risetime_df.at[specimen_id, "risetime_10pct_90pct"]
        baseline = risetime_df.at[specimen_id, "baseline_v"]
        t_10 = risetime_df.at[specimen_id, "t_10pct"]
        t_10_ind = np.flatnonzero(swp.t >= t_10)[0]
        t_90 = risetime_df.at[specimen_id, "t_90pct"]
        t_90_ind = np.flatnonzero(swp.t >= t_90)[0]
        v_ax.plot(swp.t[t_10_ind:t_90_ind], swp.v[t_10_ind:t_90_ind], lw=0.75, c='k', zorder=20)

        swp = data_set.sweep(rheo_sweep_num)
        v_ax.plot(swp.t, swp.v, c=color, lw=0.25, linestyle='solid', zorder=5)

        v_ax.text(0.7, baseline, f"{int(risetime * 1e3)} ms rise",
            fontsize=3.5, ha='left', va='bottom')
        sns.despine(ax=v_ax, left=True, bottom=True)

        v_ax.set_xlim(0.4, 1.8)
        v_ax.set_ylim(-100, -25)
        v_ax.set_xticks([])
        if i % 2 == 0:
            sns.despine(ax=v_ax, left=False, bottom=True)
            v_ax.tick_params(axis='y', labelsize=6)
            v_ax.set_ylabel("mV", rotation=0, fontsize=6)
        else:
            v_ax.set_yticks([])
            sns.despine(ax=v_ax, left=True, bottom=True)
        if i == 3:
            v_ax.plot((0.8, 1.3), (-98, -98), c='k')
            v_ax.text(1.05, -105, "500 ms", va='top', ha='center', fontsize=4)
        elif i < 2:
            v_ax.text(
                0.05, 1.05,
                met_type,
                fontsize=5,
                color=color,
                va='baseline',
                ha='left',
                transform=v_ax.transAxes
            )
        else:
            v_ax.text(
                0.05, -0.05,
                met_type,
                fontsize=5,
                color=color,
                va='top',
                ha='left',
                transform=v_ax.transAxes
            )

def plot_l6it_ephys_property(ax, prop_df, prop_name, prop_label,
        met_types=["L6 IT-1", "L6 IT-2", "L6 IT-3", "L5/L6 IT Car3"]):
    sns.stripplot(
        data=prop_df,
        x="inferred_met_type",
        y=prop_name,
        hue="inferred_met_type",
        order=met_types,
        hue_order=met_types,
        palette=MET_TYPE_COLORS,
        alpha=0.5,
        size=2,
        ax=ax,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        sns.pointplot(
            data=prop_df,
            x="inferred_met_type",
            y=prop_name,
            hue="met_type",
            order=met_types,
            hue_order=met_types,
            palette=MET_TYPE_COLORS,
            markers="_", scale=0.8, errorbar=None,
            ax=ax,
        )
    ax.get_legend().remove()
    ax.tick_params(axis='both', labelsize=6)
    ax.set_xlabel(prop_label, fontsize=6)
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_ylim(bottom=0)
    sns.despine(ax=ax, bottom=True)

    groups = prop_df.groupby("inferred_met_type")
    value_list = [groups.get_group(g)[prop_name].values for g in met_types]
    kw_stat, p_val = kruskal(*value_list)

    if p_val < 0.05:
        print("KW test p = ", p_val)
        # Figure out dimensions for the annotations
        ylim_diff = np.diff(ax.get_ylim())
        annot_height = 0.05 * ylim_diff
        max_val = prop_df.loc[~prop_df["inferred_met_type"].isnull(), prop_name].max()

        plot_val = max_val + annot_height
        dunn_results = posthoc_dunn(
            prop_df.loc[~prop_df["inferred_met_type"].isnull(), :],
            val_col=prop_name,
            group_col="inferred_met_type",
            p_adjust="holm"
        )
        dunn_results = dunn_results.loc[met_types, met_types]
        for i in range(len(met_types)):
            for j in range(i + 1, len(met_types)):
                if dunn_results.iloc[i, j] < 0.05:
                    print(met_types[i], met_types[j], dunn_results.iloc[i, j])
                    line_x = [i, i, j, j]
                    line_y = [plot_val, plot_val + annot_height,
                        plot_val + annot_height, plot_val]
                    ax.plot(line_x, line_y, linewidth=0.5, c='k', clip_on=False)
                    ax.text(
                        i + (j - i) / 2, plot_val + annot_height / 4,
                        "*", va='baseline', ha='center', fontsize=7)
                    plot_val += annot_height * 2
        ax.set_ylim(top=plot_val - annot_height)


def plot_gene_dots(df, gene_col, mean_cols, present_cols, type_labels, subplot_spec,
    scatter_factor=8):

    gs = gridspec.GridSpecFromSubplotSpec(3, 4, subplot_spec,
        width_ratios=(0.7, 0.05, 0.15, 0.5),
        height_ratios=(0.8, 0.2, 1))
    ax = plt.subplot(gs[:, 0])
    genes = df[gene_col].values
    for i, g in enumerate(genes):
        max_val = df.iloc[i, :][mean_cols].max()
        ax.scatter(
            x=np.arange(len(type_labels)),
            y=[i] * len(type_labels),
            c=df.iloc[i, :][mean_cols].values.astype(float),
            s=df.iloc[i, :][present_cols].values.astype(float) * scatter_factor,
            vmin=0,
            vmax=max_val,
            edgecolors="black",
            linewidths=0.25,
            cmap='RdYlBu_r',
        )

    ax.set_yticks(range(len(genes)))
    ax.set_yticklabels(genes, fontsize=5)
    ax.set_xticks(range(len(type_labels)))
    ax.set_xticklabels(type_labels, rotation=90, fontsize=5)
    ax.set_ylim(-0.5, len(genes) - 0.5)
    ax.set_xlim(-1, len(type_labels))
    ax.tick_params("both", length=0, width=0)
    ax.invert_yaxis()
    sns.despine(ax=ax, left=True, bottom=True)

    ax_gradient_legend = plt.subplot(gs[0, 2])
    gradient = np.linspace(0, 1, 256)[::-1]
    gradient = np.vstack((gradient, gradient)).T
    ax_gradient_legend.imshow(gradient, aspect="auto", cmap="RdYlBu_r")
    ax_gradient_legend.text(0.5, -.1, "min", va='top', ha='center', fontsize=5,
        transform=ax_gradient_legend.transAxes)
    ax_gradient_legend.text(0.5, 1.1, "max", va='baseline', ha='center', fontsize=5,
        transform=ax_gradient_legend.transAxes)
    ax_gradient_legend.set(xticks=[], yticks=[])
    ax_gradient_legend.yaxis.set_label_position("right")
    ax_gradient_legend.set_ylabel("normalized\ngene\nexpression", fontsize=5, rotation=0, ha="left", va="center")
    sns.despine(ax=ax_gradient_legend, left=True, bottom=True)

    ax_dot_legend = plt.subplot(gs[2, 2])
    ax_dot_legend.scatter(y=[0, 0.5, 1], x=[0, 0, 0], s=scatter_factor * np.array([0.1, 0.5, 1]), c="k", linewidth=0.25)
    ax_dot_legend.set_yticks([0, 0.5, 1])
    ax_dot_legend.set_yticklabels([0.1, 0.5, 1])
    ax_dot_legend.set_ylim(-0.5, 1.5)
    ax_dot_legend.set_xlim(-1, 1)
    ax_dot_legend.tick_params(length=0, labelsize=5, pad=1,
        labelleft=False, labelright=True)
    ax_dot_legend.set_xticks([])
    ax_dot_legend.text(2.2, 0.5, "proportion\nof cells", ha='left', va='center', fontsize=5,
        transform=ax_dot_legend.transAxes)
    sns.despine(ax=ax_dot_legend, left=True, bottom=True)


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
        4, 1, height_ratios=(2.5, 3, 0.3, 3), hspace=0.1)

    # Load AP and ephys info
    with h5py.File(args["ap_waveform_file"], "r") as h5f:
        ap_v = h5f["ap_v"][:]
        ap_t = h5f["ap_t"][:]
        ap_spec_ids = h5f["specimen_id"][:]
        ap_thresh_deltas = h5f["thresh_delta"][:]

    all_it_ids = inf_met_df.index.values[inf_met_df.inferred_met_type.isin(MET_TYPE_ORDER)]
    all_it_mask =  np.in1d(ap_spec_ids, all_it_ids)
    all_it_avg_ap = ap_v[all_it_mask, :].mean(axis=0)

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

    hump_measure_df = pd.read_csv(args['hump_measure_file'], index_col=0)
    depol_subthresh_df = pd.read_csv(args['subthresh_fit_file'], index_col=0)
    risetime_df = pd.read_csv(args['risetime_file'], index_col=0)

    l6_prop_df = risetime_df.merge(
        ephys_feat_df[['tau', 'input_resistance']], left_index=True, right_index=True)
    l6_prop_df = l6_prop_df.merge(hump_measure_df, left_index=True, right_index=True)
    l6_prop_df = l6_prop_df.merge(inf_met_df, left_index=True, right_index=True)
    l6_prop_df['tau'] = l6_prop_df['tau'] * 1e3 # convert to ms
    l6_prop_df['risetime_10pct_90pct'] = l6_prop_df['risetime_10pct_90pct'] * 1e3 # convert to ms

    l6it_de_df = pd.read_csv(args["l6it_de_ion_channels_file"], index_col=0)

    # Morphology plots
    gs_morph = gridspec.GridSpecFromSubplotSpec(
        2, 2,
        subplot_spec=gs[0],
        wspace=0.1,
        width_ratios=(9, 0.5),
    )
    ax_morph_top = plt.subplot(gs_morph[0, 0])
    plot_morph_lineup(
        ax_morph_top, merge_df, MET_TYPE_ORDER[:5],
        args["layer_aligned_swc_dir"], layer_edges
    )
    ax_morph_bottom = plt.subplot(gs_morph[1, 0])
    plot_morph_lineup(
        ax_morph_bottom, merge_df, MET_TYPE_ORDER[5:],
        args["layer_aligned_swc_dir"], layer_edges
    )

    ax_profiles_top = ax_morph_top.inset_axes([1.05, 0, 0.1, 1],
        transform=ax_morph_top.transAxes)
    ax_morph_top.sharey(ax_profiles_top)
    plot_depth_profiles(ax_profiles_top, merge_df, MET_TYPE_ORDER[:5],
        hist_df, basal_cols, apical_cols, layer_edges)
    ax_profiles_top.set_title("Avg. dendrite\ndepth profiles", fontsize=5)
    ax_profiles_bottom = ax_morph_bottom.inset_axes([1.05, 0, 0.1, 1],
        transform=ax_morph_bottom.transAxes)
    ax_morph_bottom.sharey(ax_profiles_bottom)
    plot_depth_profiles(ax_profiles_bottom, merge_df, MET_TYPE_ORDER[5:],
        hist_df, basal_cols, apical_cols, layer_edges)

    # Electrophysiology plots
    gs_ephys = gridspec.GridSpecFromSubplotSpec(
        2, 3,
        subplot_spec=gs[1],
    )
    met_types_left = MET_TYPE_ORDER[:3]
    met_types_middle = MET_TYPE_ORDER[3:6]
    met_types_right = MET_TYPE_ORDER[6:]
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
            np.arange(400) * 0.02 - 4, all_it_avg_ap,
            c="gray", linestyle="dotted", zorder=2, linewidth=0.75,
        )

    plot_avg_fi_for_mets(
        plt.subplot(gs_ephys[1, 0]),
        sub_df_long_met,
        inf_met_df,
        met_types_left,
        ylim=(0, 20),
        show_ylabel=True
    )
    plot_avg_fi_for_mets(
        plt.subplot(gs_ephys[1, 1]),
        sub_df_long_met,
        inf_met_df,
        met_types_middle,
        ylim=(0, 20),
        show_ylabel=False
    )
    plot_avg_fi_for_mets(
        plt.subplot(gs_ephys[1, 2]),
        sub_df_long_met,
        inf_met_df,
        met_types_right,
        ylim=(0, 20),
        show_ylabel=False
    )


    gs_l6it = gridspec.GridSpecFromSubplotSpec(
        2, 2,
        subplot_spec=gs[3],
        hspace=0.6, wspace=0.4,
    )

    ax_morph_scatter = plt.subplot(gs_l6it[1, 1])
    plot_l6it_morph_scatter(ax_morph_scatter, inf_met_df, morph_df)

    plot_l6it_ephys_examples(gs_l6it[0, 0], inf_met_df,
        ephys_info_by_specimen_id, depol_subthresh_df, risetime_df)

    gs_l6it_ephys_prop = gridspec.GridSpecFromSubplotSpec(
        1, 2,
        subplot_spec=gs_l6it[0, 1],
        wspace=0.5,
    )
    plot_l6it_ephys_property(
        plt.subplot(gs_l6it_ephys_prop[0]),
        l6_prop_df, "risetime_10pct_90pct", "rise\ntime (ms)",
    )
    plot_l6it_ephys_property(
        plt.subplot(gs_l6it_ephys_prop[1]),
        l6_prop_df, "hump_amplitude", "hump\namp. (mV)",
    )

    plot_gene_dots(
        l6it_de_df.loc[l6it_de_df["ps_is_de"] & l6it_de_df["ref_is_de"], :],
        "gene",
        ["ps_mean_L6.IT.1", "ps_mean_L6.IT.2", "ps_mean_L6.IT.3", "ps_mean_L5.L6.IT.Car3"],
        ["ps_present_L6.IT.1", "ps_present_L6.IT.2", "ps_present_L6.IT.3", "ps_present_L5.L6.IT.Car3"],
        ["L6 IT-1", "L6 IT-2", "L6 IT-3", "L5/L6 IT Car3"],
        gs_l6it[1, 0],
        scatter_factor=20,
    )


    plt.savefig(args["output_file"], bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=FigMetItParameters)
    main(module.args)

