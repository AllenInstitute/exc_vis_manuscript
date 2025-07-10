import json
import os
import re
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import seaborn as sns
from morph_utils.measurements import rightextent, leftextent
from neuron_morphology.swc_io import *
import allensdk.core.swc as swc
from morph_utils.measurements import leftextent, rightextent
import warnings
warnings.filterwarnings('ignore')
from skeleton_keys.io import load_default_layer_template
import matplotlib.colors as mc
import colorsys






def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]


def basic_morph_plot(morph, ax, morph_colors={3: "firebrick", 4: "salmon", 2: "steelblue"},
                     side=False, xoffset=0, alpha=1.0, line_w_dict = {3:.175, 4:.175, 2:.125}):
    for compartment, color in morph_colors.items():
        lines_x = []
        lines_y = []
        for c in [n for n in morph.nodes() if n['type'] == compartment]:#morph.compartment_list_by_type(compartment):
            if c["parent"] == -1:
                continue
            p = morph.node_by_id(c['parent'])#morph.compartment_index[c["parent"]]
            if side:
                lines_x += [p["z"] + xoffset, c["z"] + xoffset, None]
            else:
                lines_x += [p["x"] + xoffset, c["x"] + xoffset, None]
            lines_y += [p["y"], c["y"], None]
        line_w = line_w_dict[compartment]
        ax.plot(lines_x, lines_y, c=color, linewidth=line_w, zorder=compartment, alpha=alpha)
    return ax



def adjust_lightness(color, amount=0.5):
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])



with open("../data/ScriptArgs.json","r") as f:
    args = json.load(f)


with open(args['color_file'],'r') as f:
    MET_TYPE_COLORS = json.load(f)
    
aligned_swc_dir = args['fmost_layer_aligned_swc_dir']


meta_df = pd.read_csv(args['fmost_metadata_file'], index_col=0)

feature_df = pd.read_csv(args['fmost_raw_local_dend_feature_file'], index_col=0)
depth_df = feature_df[['soma_aligned_dist_from_pia']]
depth_df.index = depth_df.index+'.swc'

meta_df = meta_df.merge(depth_df,left_index=True,right_index=True)

hist_df = pd.read_csv(args['fmost_raw_local_histogram_file'],index_col=0)
hist_df.index = hist_df.index.map(lambda x:x+'.swc')
hist_df *= 1.144 # approx. scaling to microns
all_cols = sorted(hist_df.columns, key=natural_sort_key)
axon_cols = [c for c in all_cols if c.startswith("2_")]
basal_cols = [c for c in all_cols if c.startswith("3_")]
apic_cols = [c for c in all_cols if c.startswith("4_")]


merge_df = meta_df.merge(hist_df,left_index=True,right_index=True)
# merge_df = merge_df[merge_df['ccf_soma_location_nolayer']=='VISp']


it_met_types = sorted([i for  i in merge_df['predicted_met_type'].unique() if "IT" in i])
non_it_met_types = sorted([i for  i in merge_df['predicted_met_type'].unique() if "IT" not in i])

non_it_grouped_met_types = [ 
    ['L5 ET-1 Chrna6', 'L5 ET-2', 'L5 ET-3'],
    ['L5 NP', 'L6 CT-1', 'L6 CT-2', "L6b"],
]

it_grouped_met_types = [
    ['L2/3 IT','L4 IT'],
    ['L4/L5 IT','L5 IT-2'],
    ['L6 IT-1','L6 IT-2'], 
    ['L6 IT-3','L5/L6 IT Car3'] 
]

it_grouped_met_type_sizes = [
    (3.1496, 0.3937),
    (3.4252, 0.3937),
    (2.1654, 0.3937),
    (1.7717, 0.3937)
]

layer_info = load_default_layer_template()
layer_edges = [0] + list(layer_info.values())
path_color = "#cccccc"
plot_quantiles = np.linspace(0, 1, 4)
plot_axon_hist = True

drop_ids = ['191813_6745-X2825-Y19481_reg.swc', '191812_8013-X7581-Y5690_reg.swc', 
            '191812_7717-X6201-Y6443_reg.swc','220308_7775-X8240-Y16877_reg.swc',
           '182709_7486-X4577-Y22503_reg.swc']

# n_rows = len(it_grouped_met_types)
# fig, axes = plt.subplots(n_rows, 1, figsize=(2.5, 0.6*n_rows ), 
#                         )
depth_dict = {}
axon_depth_dict = {}
bin_width = 5
plotted_ids = {}
lw=0.5
row_ct=-1
morph_spacer = -50

for met_type_list, figure_size in zip(it_grouped_met_types, it_grouped_met_type_sizes):

    row_ct+=1
    x_offset = 0
    morph_spacing = 600
    
#     ax=axes[row_ct]
    fig,ax=plt.gcf(),plt.gca()
    for mt in met_type_list:

        print(mt)
        this_met_df = merge_df[merge_df['predicted_met_type']==mt]
        visp_df = this_met_df[this_met_df['ccf_soma_location_nolayer']=='VISp']
        if not visp_df.empty:
            this_met_df = visp_df
            
        
        this_met_df = this_met_df.loc[~this_met_df.index.isin(drop_ids)]
        spec_ids = this_met_df.sort_values("soma_aligned_dist_from_pia").index


        sub_depth_df = hist_df.loc[hist_df.index.intersection(spec_ids), basal_cols + apic_cols + axon_cols]
        avg_depth = sub_depth_df.mean(axis=0)
        basal_avg = avg_depth[basal_cols].values
        apic_avg = avg_depth[apic_cols].values
        all_avg = basal_avg + apic_avg
        depth_dict[mt] = all_avg
        axon_depth_dict[mt] = avg_depth[axon_cols].values
        
        
        inds = np.arange(len(spec_ids))
        plot_inds = np.quantile(inds, plot_quantiles).astype(int)
        plot_inds = sorted(set(plot_inds))

        plotted_ids[mt] = this_met_df.loc[spec_ids[plot_inds]]['soma_aligned_dist_from_pia'].to_dict()


        color = MET_TYPE_COLORS[mt]
        for spec_id in spec_ids.values[plot_inds]:
            sloc=this_met_df.loc[spec_id]['ccf_soma_location_nolayer']
            print(sloc, spec_id)
            swc_path = os.path.join(aligned_swc_dir, f"{spec_id}")
#             morph = swc.read_swc(swc_path)
            new_morph = morphology_from_swc(swc_path)
            x_offset += leftextent(new_morph, [2,3,4])
            basic_morph_plot(new_morph, ax=ax, xoffset=x_offset, morph_colors={3: adjust_lightness(color), 4: color, 2:'grey'})
            x_offset += rightextent(new_morph, [2,3,4])
    #         basic_morph_plot(morph, ax=ax, xoffset=x_offset, morph_colors={3: color, 4: color, 2:'grey'})
            x_offset += morph_spacer
        
    
#     new_morph = morphology_from_swc(swc_path)
#     x_offset -= morph_spacing
#     x_offset += rightextent(new_morph, [2,3,4])
#     x_offset += morph_spacing/2

    sns.despine(ax=ax, bottom=True)
    #### uncomment
    ax.set_xticks([])
    
#     ax.set_xlim(-morph_spacing / 1.25, x_offset - morph_spacing / 2)
#     ax.set_aspect("equal")
    ax.set_ylabel("µm", rotation=0, fontsize=7)
    ax.tick_params(axis='y', labelsize=6)

    

    
    morph_x_limit = x_offset
    # x_offset = x_offset + 3*morph_spacer
    x_offset+= 500
    
    histogram_start_x = x_offset
    for mt in met_type_list:
        if mt==met_type_list[0]:
            x_offset+=100
            
        color = MET_TYPE_COLORS[mt]
        zero_mask = depth_dict[mt] > 0
        ax.plot(depth_dict[mt][zero_mask] + x_offset, -np.arange(len(depth_dict[mt]))[zero_mask] * bin_width,
                c=color, linewidth=1, zorder=10)
        x_offset += max(depth_dict[mt])+100
        
        if plot_axon_hist:
            color = 'grey'
            zero_mask = axon_depth_dict[mt] > 0
            ax.plot(axon_depth_dict[mt][zero_mask] + x_offset, -np.arange(len(axon_depth_dict[mt]))[zero_mask] * bin_width,
                    c=color, linewidth=1, zorder=10)
            x_offset += max(axon_depth_dict[mt])+250
    histogram_stop_x = x_offset
    
    ax.set_aspect('equal')
    for e in layer_edges:
        ax.plot([histogram_start_x, histogram_stop_x], [-e,-e],c=path_color,zorder=-100,lw=lw)
        ax.plot([0, morph_x_limit], [-e,-e],c=path_color,zorder=-100,lw=lw)
    print(ax.get_xlim())
    
    ax.set_xlim((-150, 10409.73878216749))
    print(ax.get_xlim())
    
    fig.set_size_inches(3.4252, 0.3937)
    fig.savefig(f"./plot_morphology_lineup_it_VISp_Row{row_ct}.pdf",dpi=300,bbox_inches='tight')
    plt.clf()
    plt.close()
 
 
 
non_it_grouped_met_types = [['L5 ET-1 Chrna6'],
 ['L5 ET-2'],
 ['L5 ET-3'],
 ['L5 NP'],
 ['L6 CT-1'],
 ['L6 CT-2'],
 ['L6b']]

layer_info = load_default_layer_template()
layer_edges = [0] + list(layer_info.values())
path_color = "#cccccc"
plot_quantiles = np.linspace(0, 1, 4)
plot_axon_hist = True

drop_ids = ['191813_6745-X2825-Y19481_reg.swc', '191812_8013-X7581-Y5690_reg.swc', 
            '191812_7717-X6201-Y6443_reg.swc','220308_7775-X8240-Y16877_reg.swc',
           '182709_7486-X4577-Y22503_reg.swc', '194069_7827-X3508-Y20244_reg.swc']

# n_rows = len(it_grouped_met_types)
# fig, axes = plt.subplots(n_rows, 1, figsize=(2.5, 0.6*n_rows ), 
#                         )
depth_dict = {}
axon_depth_dict = {}
bin_width = 5
plotted_ids = {}
lw=0.5
row_ct=-1
morph_spacer = -50

for met_type_list in non_it_grouped_met_types:# figure_size in zip(it_grouped_met_types, it_grouped_met_type_sizes):

    row_ct+=1
    x_offset = 0
    morph_spacing = 600
    
#     ax=axes[row_ct]
    fig,ax=plt.gcf(),plt.gca()
    for mt in met_type_list:

        print(mt)
        this_met_df = merge_df[merge_df['predicted_met_type']==mt]
        visp_df = this_met_df[this_met_df['ccf_soma_location_nolayer']=='VISp']
        if not visp_df.empty:
            this_met_df = visp_df
            
        
        this_met_df = this_met_df.loc[~this_met_df.index.isin(drop_ids)]
        spec_ids = this_met_df.sort_values("soma_aligned_dist_from_pia").index


        sub_depth_df = hist_df.loc[hist_df.index.intersection(spec_ids), basal_cols + apic_cols + axon_cols]
        avg_depth = sub_depth_df.mean(axis=0)
        basal_avg = avg_depth[basal_cols].values
        apic_avg = avg_depth[apic_cols].values
        all_avg = basal_avg + apic_avg
        depth_dict[mt] = all_avg
        axon_depth_dict[mt] = avg_depth[axon_cols].values
        
        
        inds = np.arange(len(spec_ids))
        plot_inds = np.quantile(inds, plot_quantiles).astype(int)
        plot_inds = sorted(set(plot_inds))

        plotted_ids[mt] = this_met_df.loc[spec_ids[plot_inds]]['soma_aligned_dist_from_pia'].to_dict()


        color = MET_TYPE_COLORS[mt]
        for spec_id in spec_ids.values[plot_inds]:
            sloc=this_met_df.loc[spec_id]['ccf_soma_location_nolayer']
            print(sloc, spec_id)
            swc_path = os.path.join(aligned_swc_dir, f"{spec_id}")
#             morph = swc.read_swc(swc_path)
            new_morph = morphology_from_swc(swc_path)
            x_offset += leftextent(new_morph, [2,3,4])
            basic_morph_plot(new_morph, ax=ax, xoffset=x_offset, morph_colors={3: adjust_lightness(color), 4: color, 2:'grey'})
            x_offset += rightextent(new_morph, [2,3,4])
    #         basic_morph_plot(morph, ax=ax, xoffset=x_offset, morph_colors={3: color, 4: color, 2:'grey'})
            x_offset += morph_spacer
        
    
#     new_morph = morphology_from_swc(swc_path)
#     x_offset -= morph_spacing
#     x_offset += rightextent(new_morph, [2,3,4])
#     x_offset += morph_spacing/2

    sns.despine(ax=ax, bottom=True)
    #### uncomment
    ax.set_xticks([])
    
#     ax.set_xlim(-morph_spacing / 1.25, x_offset - morph_spacing / 2)
#     ax.set_aspect("equal")
    ax.set_ylabel("µm", rotation=0, fontsize=5)
    ax.tick_params(axis='y', labelsize=4)

    

    
    morph_x_limit = x_offset
    # x_offset = x_offset + 3*morph_spacer
    x_offset+= 500
    
    histogram_start_x = x_offset
    for mt in met_type_list:
        if mt==met_type_list[0]:
            x_offset+=100
            
        color = MET_TYPE_COLORS[mt]
        zero_mask = depth_dict[mt] > 0
        ax.plot(depth_dict[mt][zero_mask] + x_offset, -np.arange(len(depth_dict[mt]))[zero_mask] * bin_width,
                c=color, linewidth=1, zorder=10)
        x_offset += max(depth_dict[mt])+100
        
        if plot_axon_hist:
            color = 'grey'
            zero_mask = axon_depth_dict[mt] > 0
            ax.plot(axon_depth_dict[mt][zero_mask] + x_offset, -np.arange(len(axon_depth_dict[mt]))[zero_mask] * bin_width,
                    c=color, linewidth=1, zorder=10)
            x_offset += max(axon_depth_dict[mt])+250
    histogram_stop_x = x_offset

    # print(met_type_list, histogram_stop_x)
    
    
    ax.set_aspect('equal')
    for e in layer_edges:
        ax.plot([histogram_start_x, histogram_stop_x], [-e,-e],c=path_color,zorder=-100,lw=lw)
        ax.plot([0, morph_x_limit], [-e,-e],c=path_color,zorder=-100,lw=lw)
    # print(ax.get_xlim())
    
    ax.set_xlim((-150, 5852.5726828192055))
    print(ax.get_xlim())
    
    fig.set_size_inches(2.04724, 0.511811)
    fig.savefig(f"plot_morphology_lineup_Non_it_VISp_Row{row_ct}.pdf",dpi=300,bbox_inches='tight')
    plt.clf()
    plt.close()
    

