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
from matplotlib.lines import Line2D
from fig_ephys_supplement_it import plot_prop_for_mets, stats_prop_for_mets


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


class FigMetItEphysSupplementParameters(ags.ArgSchema):
    inferred_met_type_file = ags.fields.InputFile(
        default="../derived_data/inferred_met_types_Jan2025.csv"
    )
    ephys_features_file = ags.fields.InputFile(
        default="../derived_data/exc_mMET_ephys_features_2025-01-31.csv",
        description="csv file with traditional ephys features",
    )
    ephys_info_file = ags.fields.InputFile(
        default="../derived_data/spca_complete_cell_ephys_sweep_info.json",
        description="json file with ephys trace info for met cells",
    )
    subthresh_info_file = ags.fields.InputFile(
        default="../derived_data/spca_complete_subthresh_info.json",
        description="json file with subthreshold response info for met cells",
    )
    output_file = ags.fields.OutputFile(
        default="fig_ephys_supplement_nonit.pdf",
    )


def main(args):
    inf_met_df = pd.read_csv(args["inferred_met_type_file"], index_col=0)

    ephys_feat_df = pd.read_csv(args["ephys_features_file"], index_col=0)
    ephys_feat_df["ap_1_height_0_long_square"] = (
        ephys_feat_df["ap_1_peak_v_0_long_square"] -
        ephys_feat_df["ap_1_fast_trough_v_0_long_square"])
    ephys_feat_df["ap_1_width_0_long_square"] *= 1000 # to ms
    ephys_feat_df["tau"] *= 1000 # to ms
    ephys_feat_df = pd.merge(ephys_feat_df, inf_met_df, left_index=True, right_index=True)

    with open(args['ephys_info_file'], "r") as f:
        ephys_info = json.load(f)
    ephys_info_by_specimen_id = {d["specimen_id"]: d for d in ephys_info if d is not None}


    with open(args["subthresh_info_file"], "r") as f:
        subthresh_data = json.load(f)

    list_for_df = []
    for k, v in subthresh_data.items():
        for swp_info in v["subthresh_deflect"]:
            swp_info["specimen_id"] = int(k)
            list_for_df.append(swp_info)
        baseline_entry = {
            "specimen_id": int(k),
            "sweep_number": -1,
            "stimulus_amplitude": 0,
            "peak_deflect_v": v["v_baseline"],
            "avg_peak_deflect_v": v["v_baseline"],
            "steady_v": v["v_baseline"],
        }
        list_for_df.append(baseline_entry)

    deflect_df = pd.DataFrame(list_for_df).set_index("specimen_id")
    amp_counts = deflect_df["stimulus_amplitude"].map(np.round).astype(int).value_counts()
    amps_to_use = [-110, -90, -70, -50, -30, 0]
    print(amp_counts[amps_to_use])
    deflect_df["stimulus_amplitude"] = deflect_df["stimulus_amplitude"].map(np.round).astype(int)
    deflect_df_met = pd.merge(deflect_df, inf_met_df, left_index=True, right_index=True, how='left')
    deflect_df_met = deflect_df_met.loc[deflect_df_met["stimulus_amplitude"].isin(amps_to_use), :]

    fig = plt.figure(figsize=(7.5, 9))
    gs = gridspec.GridSpec(
        5, 1, height_ratios=(1, 0.3, 2.5, 0.3, 1), hspace=0.1)

    gs_ephys_prop = gridspec.GridSpecFromSubplotSpec(
        1, 4,
        subplot_spec=gs[0]
    )
    plot_prop_for_mets(plt.subplot(gs_ephys_prop[0]),
        MET_TYPE_ORDER, ephys_feat_df, "ap_1_height_0_long_square", "AP height (mV)",
        show_yticklabels=True)
    stats_prop_for_mets(MET_TYPE_ORDER, ephys_feat_df, "ap_1_height_0_long_square", "AP height (mV)")

    plot_prop_for_mets(plt.subplot(gs_ephys_prop[1]),
        MET_TYPE_ORDER, ephys_feat_df, "ap_1_width_0_long_square", "AP width (ms)",
        show_yticklabels=False, xlim=(0.5, 2))
    stats_prop_for_mets(MET_TYPE_ORDER, ephys_feat_df, "ap_1_width_0_long_square", "AP width (ms)")

    plot_prop_for_mets(plt.subplot(gs_ephys_prop[2]),
        MET_TYPE_ORDER, ephys_feat_df, "stimulus_amplitude_0_long_square", "rheobase (pA)",
        show_yticklabels=False)
    stats_prop_for_mets(MET_TYPE_ORDER, ephys_feat_df, "stimulus_amplitude_0_long_square", "rheobase (pA)")

    plot_prop_for_mets(plt.subplot(gs_ephys_prop[3]),
        MET_TYPE_ORDER, ephys_feat_df, "fi_linear_fit_slope", "slope of f-I curve\n(spikes/s/pA)",
        show_yticklabels=False)
    stats_prop_for_mets(MET_TYPE_ORDER, ephys_feat_df, "fi_linear_fit_slope", "slope of f-I curve\n(spikes/s/pA)")


    gs_subthresh = gridspec.GridSpecFromSubplotSpec(
        2, 4,
        subplot_spec=gs[2],
        wspace=0.3,
        hspace=0.5,
    )
    grouped_deflect = deflect_df_met.groupby("inferred_met_type", observed=False)
    for i, met_type in enumerate(MET_TYPE_ORDER):
        g = grouped_deflect.get_group(met_type)
        color = MET_TYPE_COLORS[met_type]
        ax = plt.subplot(gs_subthresh[i // 4, i % 4])
        if i == 0:
            ax_first = ax
            custom_lines = [Line2D([0], [0], color="black", lw=0.75, linestyle="solid"),
                            Line2D([0], [0], color="black", lw=0.75, linestyle="dotted")]
            ax.legend(custom_lines, ["peak", "steady-state"], fontsize=6, frameon=False)

        else:
            ax.sharex(ax_first)
            ax.sharey(ax_first)

        for feature, linestyle in zip(("avg_peak_deflect_v", "steady_v"), ("solid", "dotted")):
            avg_by_cell = g.groupby(["stimulus_amplitude", "specimen_id"], observed=False)[feature].mean().reset_index()
            avg = avg_by_cell.groupby(["stimulus_amplitude"], observed=False)[feature].mean()
            err = avg_by_cell.groupby("stimulus_amplitude", observed=False)[feature].std() / np.sqrt(avg_by_cell.groupby("stimulus_amplitude", observed=False)[feature].count())
            ax.errorbar(x=amps_to_use, y=avg[amps_to_use],
                         yerr=err[amps_to_use],
                         capsize=3,
                         color=color,
                         lw=0.75,
                         linestyle=linestyle,
            )
            ax.scatter(x=amps_to_use, y=avg[amps_to_use],
                s=10, c=color)

            sns.despine(ax=ax)
            if i % 5 == 0:
                ax.set_ylabel("mV", rotation=0, fontsize=7)
            if i // 5 == 1:
                ax.set_xlabel("stimulus amplitude (pA)", fontsize=7)
            ax.tick_params("both", labelsize=6)
            ax.set_title(met_type, fontsize=8)

    gs_subthresh_prop = gridspec.GridSpecFromSubplotSpec(
        1, 4,
        subplot_spec=gs[4]
    )
    plot_prop_for_mets(plt.subplot(gs_subthresh_prop[0]),
        MET_TYPE_ORDER, ephys_feat_df, "v_baseline", "resting potential (mV)",
        show_yticklabels=True)
    stats_prop_for_mets(MET_TYPE_ORDER, ephys_feat_df, "v_baseline", "resting potential (mV)")

    plot_prop_for_mets(plt.subplot(gs_subthresh_prop[1]),
        MET_TYPE_ORDER, ephys_feat_df, "input_resistance", "input resistance (MΩ)",
        show_yticklabels=False, xlim=(0, 500))
    stats_prop_for_mets(MET_TYPE_ORDER, ephys_feat_df, "input_resistance", "input resistance (MΩ)")

    plot_prop_for_mets(plt.subplot(gs_subthresh_prop[2]),
        MET_TYPE_ORDER, ephys_feat_df, "tau", "membrane time constant (ms)",
        show_yticklabels=False, xlim=(0, 50))
    stats_prop_for_mets(MET_TYPE_ORDER, ephys_feat_df, "tau", "membrane time constant (ms)")

    plot_prop_for_mets(plt.subplot(gs_subthresh_prop[3]),
        MET_TYPE_ORDER, ephys_feat_df, "sag_nearest_minus_100", "sag",
        show_yticklabels=False)
    stats_prop_for_mets(MET_TYPE_ORDER, ephys_feat_df, "sag_nearest_minus_100", "sag")

    plt.savefig(args["output_file"], bbox_inches="tight", dpi=300)



if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=FigMetItEphysSupplementParameters)
    main(module.args)
