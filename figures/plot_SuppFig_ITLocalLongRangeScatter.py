import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


import scikit_posthocs  as sp

import json

import warnings
warnings.filterwarnings('ignore')
def get_tt_color_dict(hex_color):

    contrast_level = 0.6

    bright_rgb = (tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))) 

    darker_rgb = tuple([int((contrast_level*x)) for x in bright_rgb])
 
    #morph colors

    bright_rgb_morph = bright_rgb + (255,)

    darker_rgb_morph = darker_rgb + (255,)
 
    bright_rgba = tuple([(((1/255)*x)) for x in bright_rgb_morph])

    darker_rgba = tuple([(((1/255)*x)) for x in darker_rgb_morph])
 
    bright_rgb_std = bright_rgb + (5,)        

    darker_rgb_std = darker_rgb + (5,)
 
    list_bright_rgb_std = list(bright_rgb_std)

    list_bright_rgb_std[1] = list_bright_rgb_std[1] + ((255- list_bright_rgb_std[1])/3.5)

    list_bright_rgb_std[2] = list_bright_rgb_std[2] + ((255- list_bright_rgb_std[2])/3.5)
 
    list_darker_rgb_std = list(darker_rgb_std)

    list_darker_rgb_std[1] = list_darker_rgb_std[1] + ((255- list_darker_rgb_std[1])/3.5)

    list_darker_rgb_std[2] = list_darker_rgb_std[2] + ((255- list_darker_rgb_std[2])/3.5)
 
    bright_rgba_std = tuple([(((1/255)*x)) for x in list_bright_rgb_std])

    darker_rgba_std = tuple([(((1/255)*x)) for x in list_darker_rgb_std])

    tt_color_dict = {3:darker_rgba,4:bright_rgba,33:darker_rgba_std,44:bright_rgba_std}

    return tt_color_dict
 
def get_secondary_color(hex_color):

    this_met_cdict = get_tt_color_dict(hex_color[1:])

    secondary_rgba_color = [int(i*255) for i in this_met_cdict[3]]

    secondary_hex_color = '#{:02x}{:02x}{:02x}{:02x}'.format( secondary_rgba_color[0],secondary_rgba_color[1],secondary_rgba_color[2],secondary_rgba_color[3])

    return secondary_hex_color


def main():
    
    with open("../data/ScriptArgs.json","r") as f:
        args = json.load(f)

    metadata_file = args['fmost_metadata_file'] #"../FullMorphMetaData.csv"
    color_file = args['color_file'] 
    local_axon_feature_path = args['fmost_raw_local_axon_feature_file']
    proj_mat_file = args['fmost_projection_matrix_roll_up']


    fmost_complete_axon_feat_file = args['fmost_entire_axon_features']
    with open(color_file,"r") as f:
        met_color_dict = json.load(f)
        
    proj_df=pd.read_csv(proj_mat_file,index_col=0)
    meta = pd.read_csv(metadata_file,index_col=0)
    local_axon_df  = pd.read_csv(local_axon_feature_path,index_col=0)
    local_axon_df.index=local_axon_df.index.map(lambda x:x+'.swc')
    complete_df = pd.read_csv(fmost_complete_axon_feat_file,index_col=0)

    meta = meta.merge(local_axon_df[['axon_total_length']],left_index=True,right_index=True)
    merged_df = meta.merge(complete_df,left_index=True,right_index=True)

    merged_df['total_number_of_targets'] = merged_df['complete_axon_total_number_of_targets']
    merged_df['total_long_range_axon_length'] = merged_df['complete_axon_total_length']

    odir = "./"

    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.gridspec as gridspec
    from copy import copy

    plt.rc('axes', titlesize=8) 
    plt.rc('axes', labelsize=6)
    plt.rc('xtick', labelsize=6)
    plt.rc('ytick', labelsize=6)



    dotsize = 7
    lgnd_dotsize = 1
    # Create a figure with the specified grid layout
    fig = plt.figure(figsize=(7, 1))
    n_cols = 12
    n_rows = 2
    dist_vert_width = 0.15  # Adjust the vertical spacing
    dist_horiz_height = 0.2
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, wspace=0.05, hspace=0.01,
                        width_ratios=[1, dist_vert_width, 1, dist_vert_width, 1, dist_vert_width]*2,
                        height_ratios=[dist_horiz_height, 1])

    of_interest = ['L2/3 IT', 'L4 IT', 'L4/L5 IT', 'L5 IT-2', 'L5/L6 IT Car3', 'L6b']
    it_df = merged_df[merged_df['predicted_met_type'].isin(of_interest)]

    prev_scatter_axe = None
    all_scatter_axes = []
    miny,maxy=np.inf,0
    for i, mt in enumerate(of_interest):
        this_met_df = it_df[it_df['predicted_met_type'] == mt].copy()
        
        this_met_df['axon_total_length'] = this_met_df['axon_total_length']*0.001
        
        visp_met_df = this_met_df[this_met_df['ccf_soma_location_nolayer']=='VISp']
        hva_met_df = this_met_df[this_met_df['ccf_soma_location_nolayer']!='VISp']
        

        if this_met_df['axon_total_length'].min()<miny:
            miny=this_met_df['axon_total_length'].min()

        if this_met_df['axon_total_length'].max()>maxy:
            maxy=this_met_df['axon_total_length'].max()
            
        visp_color = met_color_dict[mt]
        hva_color = get_secondary_color(visp_color)  # met_color_dict[mt]
        # Scatterplot
        
        if prev_scatter_axe is not None:
            scatter_ax = fig.add_subplot(gs[1, i * 2])
    #         scatter_ax.set_yticklabels([])  # Hide y-tick labels for subsequent scatter plots
    #         scatter_ax.set_ylabel("") 
        else:
            scatter_ax = fig.add_subplot(gs[1, i * 2])
            
            
        sns.regplot(data=this_met_df,
                            x='total_number_of_targets',
                            y='axon_total_length',
                            ax=scatter_ax,
                            ci=None,
                            scatter_kws=dict(s=0, color="white",
                            edgecolors='white',  alpha=0.8),
                            line_kws=dict(lw=1, color='black'))
                    
        visp_scatter = sns.scatterplot(data=visp_met_df,
                        x='total_number_of_targets',
                        y='axon_total_length',
                        ax=scatter_ax,
                        color=visp_color,
                                    s=dotsize,
                                    label='VISp')
        hva_scatter = sns.scatterplot(data=hva_met_df,
                            x='total_number_of_targets',
                            y='axon_total_length',
                            ax=scatter_ax,
                            color=hva_color,
                                    s=dotsize,
                                    label='HVA')
        
        visp_ptch = mlines.Line2D([], [], color=visp_color, marker='o', linestyle='None',
                            markersize=lgnd_dotsize, label='VISp')
        hva_ptch = mlines.Line2D([], [], color=hva_color, marker='o', linestyle='None',
                            markersize=lgnd_dotsize, label='HVA')
        scatter_ax.legend(handles=[visp_ptch,hva_ptch],prop={'size': 4})

        
        if prev_scatter_axe is not None:
            scatter_ax.set_ylabel("")
            scatter_ax.set_yticklabels([]) 
            scatter_ax.set_yticks([]) 
        else:
            scatter_ax.set_ylabel("Local Axon Length (mm)")
        prev_scatter_axe = copy(scatter_ax)

        scatter_ax.set_xlabel("Total Num. Targets")
        all_scatter_axes.append(scatter_ax)
        
        # KDE Plot above (sharing the x-axis)
        kde_ax_above = fig.add_subplot(gs[0, i * 2], sharex=scatter_ax)
        sns.kdeplot(data=visp_met_df['total_number_of_targets'], 
                    ax=kde_ax_above, 
                    color=visp_color, 
                    vertical=False,
                )
        sns.kdeplot(data=hva_met_df['total_number_of_targets'], 
                    ax=kde_ax_above, 
                    color=hva_color, 
                    vertical=False,
                )
        

        kde_ax_right = fig.add_subplot(gs[1, (i * 2) + 1], sharey=scatter_ax)
        sns.kdeplot(data=visp_met_df['axon_total_length'], 
                    ax=kde_ax_right, 
                    color=visp_color, 
                    vertical=True)
        sns.kdeplot(data=hva_met_df['axon_total_length'], 
                    ax=kde_ax_right, 
                    color=hva_color, 
                    vertical=True)

        kde_ax_right.axis('off')
        kde_ax_above.axis('off')
        
        scatter_ax.set_xlim((0,30))
    #     scatter_ax.set_ylim((10000,47000))
        scatter_ax.set_ylim((7,47))
        
        
        rsp,pva = stats.spearmanr(this_met_df["total_number_of_targets"], this_met_df["axon_total_length"])
        rsp = round(rsp,3)
        
        st,pval_x = stats.mannwhitneyu(visp_met_df['total_number_of_targets'], hva_met_df['total_number_of_targets'])
        st,pval_y = stats.mannwhitneyu(visp_met_df['axon_total_length'], hva_met_df['axon_total_length'])
        
        pv1 =  r"($p_\mathrm{{x}}$ = {0:.2e})".format(pval_x)
        pv2 =  r"($p_\mathrm{{y}}$ = {0:.2e})".format(pval_y)
        title = mt+' ' +r"$r_\mathrm{{s}}$ = {0:.2f}".format(rsp) +'\n'+pv1 + "\n"+pv2
        kde_ax_above.set_title(title,fontsize=6)

    plt.tight_layout()
    # plt.show()
    fig.savefig("plot_SuppFig_ITLocalLongRangeScatter.pdf" ,dpi=600,bbox_inches='tight')
    # plt.clf()
    plt.close()

if __name__ == "__main__":
    main()