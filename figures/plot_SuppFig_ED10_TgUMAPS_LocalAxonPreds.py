import pandas as pd
import umap
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import json
import seaborn as sns
import os
from matplotlib import cm
import matplotlib.patches as mpatches
from scipy import stats
import matplotlib as mpl
from matplotlib import cm
import ast

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



def main():
    
    with open("../data/ScriptArgs.json","r") as f:
        args = json.load(f)

    color_file = args['color_file'] 
    with open(color_file,"r") as f:
        color_dict = json.load(f)
        

    local_axon_pred_pmet_confmat_file = args['wnm_local_axon_predicting_pred_met_type_confmat']
    fmost_metadata_file = args['fmost_metadata_file']
    fmost_pseq_umap_csv = args['fmost_pseq_combined_dend_umap_coords']
    pseq_tg_metadata_file = args['patchseq_genotype_metadata']
    fmost_only_umap_csv = args['fmost_only_morpho_umap_embeddings']
    ivscc_met_labels_file = args['pseq_met_labels_file']
    
    
    
    fmost_meta_df = pd.read_csv(fmost_metadata_file,index_col=0)
    fmost_met_dict = fmost_meta_df.predicted_met_type.to_dict()


    ivscc_met_labels = pd.read_csv(ivscc_met_labels_file, index_col=0)
    ivscc_met_labels_dict = {str(k):v for k,v in ivscc_met_labels.met_type.to_dict().items()}
    
    combined_met_dict = {**fmost_met_dict, **ivscc_met_labels_dict}
    
    fmost_pseq_umap_csv = os.path.abspath(fmost_pseq_umap_csv)

    stacked_umap_df = pd.read_csv(fmost_pseq_umap_csv)

        
    stacked_umap_df["met_type"] = stacked_umap_df.specimen_id.map(combined_met_dict)


    fmost_meta_df['cre_line_shorthand'] = fmost_meta_df.cre_line.map(lambda x:x.split("-")[0].split(';')[0])
    fmost_under_represented_tgs = [k for k,v in fmost_meta_df.cre_line_shorthand.value_counts().items() if v<5]
    fmost_meta_df.loc[fmost_meta_df['cre_line_shorthand'].isin(fmost_under_represented_tgs), 'cre_line_shorthand'] = "Low N"

    sorted_fmost_tgs = sorted(fmost_meta_df.cre_line_shorthand.unique())
    sorted_fmost_tgs.remove("Low N")
    sorted_fmost_tgs= sorted_fmost_tgs + ["Low N"]


    patch_seq_tg_meta = pd.read_csv(pseq_tg_metadata_file,index_col=0)
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



    combined_cre_label_dict = {**patch_seq_tg_meta['cre_line_shorthand'].to_dict(), **fmost_meta_df['cre_line_shorthand'].to_dict()}
    stacked_umap_df["cre_line"] = stacked_umap_df.specimen_id.map(combined_cre_label_dict)


    fmost_only_umap_csv = os.path.abspath(fmost_only_umap_csv)

    fmost_only_umap_df = pd.read_csv(fmost_only_umap_csv,index_col=0)
    fmost_only_umap_df = fmost_only_umap_df.merge(fmost_meta_df,left_index=True,right_index=True)
    
    fmost_only_umap_df["met_color"] = fmost_only_umap_df.predicted_met_type.map(color_dict)

    stacked_umap_df["met_color"] = stacked_umap_df.met_type.map(color_dict)
    stacked_umap_df['tg_color'] = stacked_umap_df.cre_line.map(tg_type_to_color)

    this_stacked_df_fmost = stacked_umap_df[stacked_umap_df['specimen_id'].astype(str).str.contains(".swc")]

    this_stacked_df_ivscc = stacked_umap_df[~stacked_umap_df['specimen_id'].astype(str).str.contains(".swc")]

    conf_mat = pd.read_csv(local_axon_pred_pmet_confmat_file, index_col=0)
        
    legend_dot_size=3

    tg_handles =[Line2D( [0],[0], label=k, marker='o',markersize=legend_dot_size,markeredgecolor=c, markerfacecolor=c, linestyle='') for k,c in tg_type_to_color.items()]
    pt = Line2D([0], [0], label='fMOST', marker='o', markersize=legend_dot_size, 
            markeredgecolor="k", markerfacecolor="white", linestyle='')
    tg_handles.append(pt)

    sorted_pmets = sorted(stacked_umap_df.met_type.unique(), key = lambda x: ( [sc in x for sc in ["IT","ET", "NP","CT",'L6b']].index(True),x))
    
    met_handle_list = []
    fmost_only = []# ["L6b","ET-MET-1*"]
    for pmet in sorted_pmets:
        mtdf = stacked_umap_df[stacked_umap_df['met_type']==pmet]
        c = mtdf['met_color'].values[0]
        pmet_rename = pmet.replace("PT-MET","ET-MET")
        edgecolor=c
        # if pmet in fmost_only:
        #     edgecolor = "k"
        
        pt = Line2D([0], [0], label=pmet_rename, marker='o', markersize=legend_dot_size, 
            markeredgecolor=edgecolor, markerfacecolor=c, linestyle='')
        met_handle_list.append(pt)
        
    pt = Line2D([0], [0], label='fMOST', marker='o', markersize=legend_dot_size, 
            markeredgecolor="k", markerfacecolor="white", linestyle='')
    met_handle_list.append(pt)
    
    
    
    
    label_fontsize = 7
    legend_fontsize = 7
    legend_title_fontsize = 8
    annot_fontsize = 7
    tilte_fs = 8

    fig = plt.gcf()
    gs = GridSpec(2, 2, wspace=1.2, hspace=1)

    dot_size = 4
    sc_lw = 0.25

    # --- First subplot ---
    ax_0 = fig.add_subplot(gs[0, 0])
    ax_0.scatter(this_stacked_df_ivscc['combined_umap_embedding_1'],
                this_stacked_df_ivscc['combined_umap_embedding_2'],
                c=this_stacked_df_ivscc['met_color'],
                s=dot_size)

    legend = ax_0.legend(handles=met_handle_list,
                bbox_to_anchor=(1, 0.5),
                loc='center left',
                prop={'size': legend_fontsize},
                title='MET-Type')
    legend.get_title().set_fontsize(legend_title_fontsize)

    ax_0.scatter(this_stacked_df_fmost['combined_umap_embedding_1'],
                this_stacked_df_fmost['combined_umap_embedding_2'],
                c=this_stacked_df_fmost['met_color'],
                edgecolor='k',
                linewidths=sc_lw,
                s=dot_size)
    ax_0.set_title("Patch-seq and WNM\nUMAP by MET-type",fontsize=tilte_fs)


    # --- Second subplot ---
    ax_1 = fig.add_subplot(gs[0, 1])
    ax_1.scatter(this_stacked_df_ivscc['combined_umap_embedding_1'],
                this_stacked_df_ivscc['combined_umap_embedding_2'],
                c=this_stacked_df_ivscc['tg_color'],
                s=dot_size)

    ax_1.scatter(this_stacked_df_fmost['combined_umap_embedding_1'],
                this_stacked_df_fmost['combined_umap_embedding_2'],
                c=this_stacked_df_fmost['tg_color'],
                edgecolor='k',
                linewidths=sc_lw,
                s=dot_size)

    legend = ax_1.legend(handles=tg_handles,
                bbox_to_anchor=(1, 0.5),
                loc='center left',
                prop={'size': legend_fontsize},
                title='Tg. Line')
    legend.get_title().set_fontsize(legend_title_fontsize)
    ax_1.set_title("Patch-seq and WNM\nUMAP by Tg. Line",fontsize=tilte_fs)


    # --- Third subplot ---
    ax_2 = fig.add_subplot(gs[1, 0])
    ax_2.scatter(fmost_only_umap_df['local_and_complete_axon_only_umap_embedding_1'],
                fmost_only_umap_df['local_and_complete_axon_only_umap_embedding_2'],
                c=fmost_only_umap_df['met_color'],
                s=dot_size)

    # --- Shared axis formatting ---
    for ax in [ax_0, ax_1, ax_2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel("M-UMAP 1", fontsize=label_fontsize)
        ax.set_ylabel("M-UMAP 2", fontsize=label_fontsize)
        ax.tick_params(axis='both', which='both', length=0, labelsize=label_fontsize)
        ax.set_xticks([])
        ax.set_yticks([])
    ax_2.set_title("WNM local axon UMAP",fontsize=tilte_fs)

    # --- Heatmap subplot ---
    hm_ax = fig.add_subplot(gs[1, 1])

    # Only annotate diagonal with values
    vals = conf_mat.values.astype(object)
    for i in range(len(vals)):
        for j in range(len(vals)):
            vals[i, j] = str(round(vals[i, j], 2)) if i == j else ''
    diagonal_vals = vals.astype(str)

    con = sns.heatmap(conf_mat,
                    annot=diagonal_vals,
                    fmt='',
                    xticklabels=conf_mat.columns,
                    yticklabels=conf_mat.index,
                    vmin=0, vmax=1.0,
                    ax=hm_ax,
                    annot_kws={"size": 5})

    hm_ax.set_xticklabels(hm_ax.get_xticklabels(), rotation=90, ha='right', fontsize=label_fontsize)
    hm_ax.set_yticklabels(hm_ax.get_yticklabels(), rotation=0, ha='right', fontsize=label_fontsize)
    con.set_xlabel('Prediction', fontsize=label_fontsize)
    con.set_ylabel('Truth', fontsize=label_fontsize)
    hm_ax.set_title("Axon morphology predicts predicted MET-type",fontsize=tilte_fs)
    colorbar = con.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=7)  # Change to your desired size

    # --- Final figure settings ---
    fig.set_size_inches(7, 6)
    fig.savefig("plot_SuppFig_ED10_TgUMAPS_LocalAxonPreds.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    
if __name__ == "__main__":
    main()