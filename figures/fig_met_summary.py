import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import argschema as ags
import ipfx.script_utils as su
import allensdk.core.swc as swc
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


class FigMetSummaryParameters(ags.ArgSchema):
    met_selection_file = ags.fields.InputFile(
        description="json file with representative cell specimen IDs per met-type",
    )
    layer_depths_file = ags.fields.InputFile(
        description="json file with distances from top of layer to pia",
    )
    layer_aligned_swc_dir = ags.fields.InputDir(
        description="directory with layer-aligned swc files",
    )
    met_ephys_info_file = ags.fields.InputFile(
        description="json file with ephys trace info for met cells",
    )
    output_file = ags.fields.OutputFile(
        description="output file",
    )


def main(args):
    with open(args['met_selection_file'], 'r') as f:
        met_rep_cells = json.load(f)

    with open(args["layer_depths_file"], "r") as f:
        layer_info = json.load(f)
    layer_edges = [0] + list(layer_info.values())

    with open(args['met_ephys_info_file'], "r") as f:
        ephys_info = json.load(f)
    ephys_info_by_specimen_id = {d["specimen_id"]: d for d in ephys_info if d is not None}

    fig = plt.figure(figsize=(8, 2))
    gs_overall = gridspec.GridSpec(
        1,
        1,
    )

    plot_row(gs_overall, 0, MET_TYPE_ORDER,
        met_rep_cells, layer_edges, args['layer_aligned_swc_dir'],
        ephys_info_by_specimen_id)

    plt.savefig(args['output_file'], dpi=300, bbox_inches='tight')


def plot_row(gs_overall, row_ind, met_types, met_rep_cells, layer_edges,
        layer_aligned_swc_dir, ephys_info_by_specimen_id):
    gs_row = gridspec.GridSpecFromSubplotSpec(
        2,
        1,
        hspace=0.1,
        height_ratios=[2, 1],
        subplot_spec=gs_overall[row_ind],
    )
    ax_morph = plt.subplot(gs_row[0])
    ax_ephys = plt.subplot(gs_row[1])

    m_xoffset = 0
    m_spacing = 600
    e_xoffset = 0
    e_spacing = 1.55
    last_met_type = met_types[0]
    for met_type in met_types:
        print(met_type)
        met_example_id = met_rep_cells[met_type]
        color = MET_TYPE_COLORS[met_type]
        swc_path = os.path.join(layer_aligned_swc_dir, f"{met_example_id}.swc")
        morph = swc.read_swc(swc_path)
        basic_morph_plot(morph, ax=ax_morph, xoffset=m_xoffset,
            morph_colors={3: adjust_lightness(color, 0.5), 4: color})
        basic_ephys_plot(ephys_info_by_specimen_id[met_example_id], ax=ax_ephys, xoffset=e_xoffset,
            color=color)

        title_text = met_type
        if title_text.count(" ") > 1:
            ws_split = title_text.split(" ")
            title_text = ws_split[0] + " " + ws_split[1] + "\n" + ws_split[2]
        ax_morph.text(m_xoffset, 40, title_text,
            fontsize=5, color=color, va="baseline", ha="center")

        m_xoffset += m_spacing
        e_xoffset += e_spacing
        last_met_type = met_type

    ax_ephys.set_ylim(-100, 50)
    ax_ephys.set_xlim(-e_spacing * 0.75, e_xoffset - e_spacing * 0.25)
    ax_ephys.set_xticks([])
    ax_ephys.tick_params(axis='y', labelsize=5, length=3, direction="out",
        right=True, left=False, labelright=True, labelleft=False)
    ax_ephys.set_yticks([-100, -50, 0, 50])
    ax_ephys.set_ylabel("mV", rotation=0, fontsize=5)
    ax_ephys.yaxis.set_label_position("right")
    ax_ephys.spines["left"].set_visible(False)
    ax_ephys.spines["bottom"].set_visible(False)
    ax_ephys.spines["top"].set_visible(False)

    if row_ind == 0:
        ax_ephys.plot([e_xoffset - e_spacing - 0.5, e_xoffset - e_spacing], [-98, -98], linewidth=1.5, color='black')

    ax_morph.set_ylim(-1050, 10)
    ax_morph.plot([m_xoffset - m_spacing / 2 + 100] * 2, [-100, -300], clip_on=False, linewidth=1.5, color="black")
    ax_morph.set_xlim(-m_spacing * 0.75, m_xoffset - m_spacing * 0.25)
    ax_morph.set_yticks([])
    ax_morph.set_xticks([])
    sns.despine(ax=ax_morph, left=True, bottom=True)
    for edge in layer_edges:
        ax_morph.axhline(-edge, color="lightgray", linewidth=0.5, zorder=-1)
    layer_midpoints = [(l1 + l2) / 2 for l1, l2 in zip(layer_edges[:-1], layer_edges[1:])]
    layer_label_dict = dict(zip(["L1", "L2/3", "L4", "L5", "L6a", "L6b"], layer_midpoints))
    for l, y in layer_label_dict.items():
        ax_morph.text(1.01, -y, l, transform=ax_morph.get_yaxis_transform(), fontsize=5,
            color="black", va="center", ha="left")
    ax_morph.set_aspect("equal")


def basic_ephys_plot(ephys_sweep_info, ax, xoffset, color, extra_t=0.1):
    data_set = su.dataset_for_specimen_id(ephys_sweep_info['specimen_id'],
        'lims-nwb2', None, None)

    if ephys_sweep_info["subthresh_sweep_minus70"] is not None:
        print(ephys_sweep_info['specimen_id'], ":", "using minus 70 sweep")
        subthresh_sweep_num = ephys_sweep_info["subthresh_sweep_minus70"]
    elif ephys_sweep_info["subthresh_sweep_minus90"] is not None:
        print(ephys_sweep_info['specimen_id'], ":", "using minus 90 sweep")
        subthresh_sweep_num = ephys_sweep_info["subthresh_sweep_minus90"]
    else:
        subthresh_sweep_num = None

    if subthresh_sweep_num is not None:
        swp = data_set.sweep(subthresh_sweep_num)
        v = swp.v
        t = swp.t

        start_ind = np.flatnonzero(t >= ephys_sweep_info["start"] - extra_t)[0]
        end_ind = np.flatnonzero(t >= ephys_sweep_info["end"] + extra_t)[0]
        v = v[start_ind:end_ind]
        t = t[start_ind:end_ind]
        t -= t[0]

        ax.plot(t + xoffset - t[len(t) // 2], v, lw=0.5, color=color)

    if ephys_sweep_info["plus30or40_sweep"] is not None:
        swp = data_set.sweep(ephys_sweep_info["plus30or40_sweep"])
        v = swp.v
        t = swp.t

        start_ind = np.flatnonzero(t >= ephys_sweep_info["start"] - extra_t)[0]
        end_ind = np.flatnonzero(t >= ephys_sweep_info["end"] + extra_t)[0]
        v = v[start_ind:end_ind]
        t = t[start_ind:end_ind]
        t -= t[0]

        ax.plot(t + xoffset - t[len(t) // 2], v, lw=0.5, color=color)


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=FigMetSummaryParameters)
    main(module.args)
