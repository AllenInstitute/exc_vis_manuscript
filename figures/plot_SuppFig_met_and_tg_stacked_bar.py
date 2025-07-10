import os
import pandas
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import json
import warnings
from matplotlib.gridspec import GridSpec
from matplotlib.colors import to_hex  
warnings.filterwarnings('ignore')
from matplotlib.patches import Patch

def stacked_barplot(
    input_df,
    x_var,
    y_var,
    sorted_x_var,
    input_color_dict,
    ax,
    legend_yloc=0.5,
    plot_all_types_in_lookup=True,
    legend=True,
    sorted_y_var=None,
    # Additional style kwargs with defaults
    axis_label_fontsize=12,
    tick_labelsize=10,
    legend_fontsize=10,
    legend_title_fontsize=12,
    legend_marker_size=10,
    legend_tilte = "",
    hatch_types = ['L5 NP', 'L6b']
):
    import matplotlib as mpl
    mpl.rcParams['hatch.linewidth'] = 0.1  # previous pdf hatch linewidth
    # mpl.rcParams['hatch.linewidth'] = 1.0  # previous svg hatch linewidth

    stacked_df = input_df.groupby([x_var, y_var]).size().unstack()

    sorted_stacked = pd.DataFrame()
    plotted_types = []
    for pm in sorted_x_var:
        if pm in stacked_df.index:
            plotted_types.append(pm)
            sorted_stacked = pd.concat([sorted_stacked, stacked_df.loc[[pm]]])

    sorted_stacked_norm = sorted_stacked.div(sorted_stacked.sum(axis=1), axis=0)

    ct_dict = sorted_stacked.sum(axis=1).astype(int).to_dict()
    sorted_stacked_norm.index = [f"{i} (n={ct_dict[i]})" for i in sorted_stacked_norm.index]

    # sorted_stacked_norm.plot(
    #     kind='bar',
    #     stacked=True,
    #     color=[input_color_dict[c] for c in sorted_stacked.columns],
    #     ax=ax,
    #     legend=legend,
    # )
    bar_plot = sorted_stacked_norm.plot(
        kind='bar',
        stacked=True,
        color=[input_color_dict[c] for c in sorted_stacked.columns],
        ax=ax,
        legend=legend,
    )

    bar_containers = ax.containers
    for container, label in zip(bar_containers, sorted_stacked.columns):
        if label in hatch_types:
            for bar in container:
                bar.set_hatch('////')  # You can change hatch style here
                # bar.set_edgecolor('black')
                bar.set_linewidth(0.0001)

    if legend:
        if plot_all_types_in_lookup:
            all_labels = sorted_y_var if sorted_y_var is not None else input_color_dict.keys()
            handles = [
                Patch(color=input_color_dict[label], label=label)
                for label in all_labels
            ]
            ax.legend(
                handles=handles,
                bbox_to_anchor=(1, legend_yloc),
                loc='center left',
                fontsize=legend_fontsize,
                title=legend_tilte,
                title_fontsize=legend_title_fontsize,
                handleheight=legend_marker_size / 10,
                handlelength=1.5,
            )
        else:
            ax.legend(
                bbox_to_anchor=(1, legend_yloc),
                loc='center left',
                title=legend_tilte,
                fontsize=legend_fontsize,
                title_fontsize=legend_title_fontsize,
            )

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_ylabel("Fraction of Cells", fontsize=axis_label_fontsize)
    ax.set_xlabel("")  # Optional: adjust as needed
    ax.tick_params(axis='x', labelsize=tick_labelsize)
    ax.tick_params(axis='y', labelsize=tick_labelsize)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='center')
    
def main():  
      
    with open("../data/ScriptArgs.json","r") as f:
        args = json.load(f)

    with open(args['color_file'],'r') as f:
        color_dict=json.load(f)

    sorted_met_types = ['L2/3 IT',
    'L4 IT',
    'L4/L5 IT',
    'L5 IT-1',
    'L5 IT-2',
    'L5 IT-3 Pld5',
    'L6 IT-1',
    'L6 IT-2',
    'L6 IT-3',
    'L5/L6 IT Car3',
    'L5 ET-1 Chrna6',
    'L5 ET-2',
    'L5 ET-3',
    'L5 NP',
    'L6 CT-1',
    'L6 CT-2',
    'L6b']


    fmost_meta_df = pd.read_csv(args['fmost_metadata_file'],index_col=0)
    fmost_meta_df[['ccf_soma_location',
    'ccf_soma_x',
    'ccf_soma_y',
    'ccf_soma_z',
    'cre_line']]

    fmost_meta_df = pd.read_csv(args['fmost_metadata_file'],index_col=0)
    fmost_meta_df['cre_line_shorthand'] = fmost_meta_df.cre_line.map(lambda x:x.split("-")[0].split(';')[0])
    fmost_under_represented_tgs = [k for k,v in fmost_meta_df.cre_line_shorthand.value_counts().items() if v<5]
    fmost_meta_df.loc[fmost_meta_df['cre_line_shorthand'].isin(fmost_under_represented_tgs), 'cre_line_shorthand'] = "Low N"

    sorted_fmost_tgs = sorted(fmost_meta_df.cre_line_shorthand.unique())
    sorted_fmost_tgs.remove("Low N")
    sorted_fmost_tgs= sorted_fmost_tgs + ["Low N"]


    patch_seq_tg_meta = pd.read_csv(args['patchseq_genotype_metadata'],index_col=0)
    patch_seq_tg_meta = patch_seq_tg_meta[patch_seq_tg_meta['Morphology (manual)']==True]
    # patch_seq_tg_meta = patch_seq_tg_meta[patch_seq_tg_meta['Reporter']=='tdt+']

    patch_seq_tg_meta['cre_line_shorthand'] = patch_seq_tg_meta['Genotype'].map(lambda x:x.split("-")[0])
    patch_seq_tg_meta = patch_seq_tg_meta[~patch_seq_tg_meta['MET-type'].isnull()]


    patch_seq_tg_meta.loc[patch_seq_tg_meta['Reporter']=='tdt-', 'cre_line_shorthand'] = "tdt-"

    sorted_pseq_tgs = sorted(patch_seq_tg_meta.cre_line_shorthand.unique())

    under_represented_tgs = [s for s in sorted_pseq_tgs if patch_seq_tg_meta.cre_line_shorthand.value_counts().to_dict()[s]<=5]
    patch_seq_tg_meta.loc[patch_seq_tg_meta['cre_line_shorthand'].isin(under_represented_tgs), 'cre_line_shorthand'] = "Low N"


    sorted_pseq_tgs = sorted(patch_seq_tg_meta.cre_line_shorthand.unique().tolist()) 
    sorted_pseq_tgs.remove("Low N")
    sorted_pseq_tgs.remove("tdt-")
    sorted_pseq_tgs = sorted_pseq_tgs +  ["Low N", 'tdt-'] 


    patch_seq_tg_meta_filter = patch_seq_tg_meta.copy()
    ### patch_seq_tg_meta_filter = patch_seq_tg_meta[patch_seq_tg_meta['cre_line_shorthand'].isin(sorted_pseq_tgs)]


    base_palette = sns.color_palette("tab20", 20)  # First 20
    extra_colors = sns.color_palette("Dark2", 9)   # 7 more distinct ones

    # Merge into final palette
    final_palette = base_palette + extra_colors  # 27 total

    # Convert to hex for mapping
    all_types = sorted(set(sorted_fmost_tgs + sorted_pseq_tgs))  # Unique labels
    all_types.remove("Low N")
    all_types.remove("tdt-")

    # all_types = all_types + ['tdt-', 'Low N']

    tg_type_to_color = {label: to_hex(color) for label, color in zip(all_types, final_palette)}

    tg_type_to_color['Low N'] = "#332E2E"
    tg_type_to_color['tdt-'] = "#614E4E"


    axis_label_fontsize=6
    tick_labelsize=4
    legend_fontsize=4
    legend_title_fontsize=5
    legend_marker_size=3

    outdir = "./"
    os.makedirs(outdir,exist_ok=True)

    fig = plt.figure(figsize=(6, 3))  # Adjust height to your content

    spacer_size = 0.15
    gs = GridSpec(nrows=5, ncols=3, height_ratios=[1, spacer_size,spacer_size,spacer_size, 1], hspace=0.65,
                width_ratios = [1, 0.2, 1])

    # Create axes explicitly where needed
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[4,0])

    # First plot (WNM)
    stacked_barplot(
        fmost_meta_df, 
        ax=ax1,
        x_var="predicted_met_type", 
        y_var="cre_line_shorthand",
        sorted_x_var=sorted_met_types,
        legend_yloc=-0.85,
        legend_tilte='Tg Line',
        input_color_dict=tg_type_to_color,
        axis_label_fontsize=axis_label_fontsize,
        tick_labelsize=tick_labelsize,
        legend_fontsize=legend_fontsize,
        legend_title_fontsize=legend_title_fontsize,
        legend_marker_size=legend_marker_size,
    )
    ax1.set_title("WNM")

    # Second plot (Patch-Seq)
    stacked_barplot(
        patch_seq_tg_meta_filter, 
        ax=ax2,
        x_var="MET-type", 
        y_var="cre_line_shorthand",
        sorted_x_var=sorted_met_types,
        legend=False,
        input_color_dict=tg_type_to_color,
        axis_label_fontsize=axis_label_fontsize,
        tick_labelsize=tick_labelsize,
        legend_fontsize=legend_fontsize,
        legend_title_fontsize=legend_title_fontsize,
        legend_marker_size=legend_marker_size,
    )
    ax2.set_title("Patch-Seq")





    # Create axes explicitly where needed
    ax11 = fig.add_subplot(gs[0,2])
    ax21 = fig.add_subplot(gs[4,2])

    # First plot (WNM)
    stacked_barplot(
        fmost_meta_df, 
        ax=ax11,
        x_var = "cre_line_shorthand", 
        y_var = "predicted_met_type",
        sorted_x_var = sorted_fmost_tgs,
        plot_all_types_in_lookup=False,
        input_color_dict=color_dict,
        legend_yloc=-1.1,
        legend_tilte='MET-Type',
        axis_label_fontsize=axis_label_fontsize,
        tick_labelsize=tick_labelsize,
        legend_fontsize=legend_fontsize,
        legend_title_fontsize=legend_title_fontsize,
        legend_marker_size=legend_marker_size,
    )
    ax11.set_title("WNM")

    # Second plot (Patch-Seq)
    stacked_barplot(
        patch_seq_tg_meta, 
        ax=ax21,
        x_var = "cre_line_shorthand", 
        y_var = "MET-type",
        sorted_x_var = sorted_pseq_tgs,
        plot_all_types_in_lookup=False,
        input_color_dict = color_dict,
        legend=False,
        axis_label_fontsize=axis_label_fontsize,
        tick_labelsize=tick_labelsize,
        legend_fontsize=legend_fontsize,
        legend_title_fontsize=legend_title_fontsize,
        legend_marker_size=legend_marker_size,
    )
    ax21.set_title("Patch-Seq")

    plt.tight_layout()

    ofile = os.path.join(outdir,"plot_SuppFig_met_and_tg_stacked_bar.pdf")
    fig.savefig(ofile,dpi=600,bbox_inches='tight')
    plt.close()
if __name__=='__main__':
    main()