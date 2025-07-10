
from allensdk.core.reference_space_cache import ReferenceSpaceCache
from copy import copy
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from morph_utils.ccf import ACRONYM_MAP
from morph_utils.ccf import load_structure_graph
from scipy import ndimage
import ast
import json
import matplotlib
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from matplotlib.patches import Rectangle

import numpy as np
from matplotlib.patches import Rectangle

import warnings
warnings.filterwarnings('ignore')

def imshow_vector(ax, img_arr, origin='upper'):
    """
    Recreates imshow using vector Rectangles for editable PDFs.

    Parameters:
        ax      -- matplotlib axes
        img_arr -- (H, W, 3 or 4) NumPy array of RGB or RGBA
        origin  -- 'upper' (default) or 'lower' to control Y origin
    """
    height, width = img_arr.shape[:2]

    for row in range(height):
        for col in range(width):
            rgba = img_arr[row, col]
            color = rgba[:3]
            alpha = rgba[3] if img_arr.shape[2] == 4 else 1.0

            if alpha > 0:
                # Flip Y if origin is 'upper' (like imshow does)
                y = height - 1 - row if origin == 'upper' else row
                rect = Rectangle((col, y), 1, 1, facecolor=color, edgecolor='none')
                ax.add_patch(rect)

    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('auto')
    ax.set_xticks([])
    ax.set_yticks([])

    # IMPORTANT: do NOT invert y-axis again; already handled above


def plot_projection_histogram(
    these_releveant_proj_cols,
    sg_df,
    met_df,
    ax,
    met_color,
        inverse_acronym_map = ACRONYM_MAP
):
    """'
    Processes projection data, generates plots, and handles contra/ipsi structures.

    Args:
        these_releveant_proj_cols: List of projection column names.
        sg_df: DataFrame with structure data (e.g., colors).
        met_df: DataFrame with metadata for mean/SEM calculations.
        ax: Matplotlib axis for contra/ipsi visualization.
        axe: Matplotlib axis for metadata plots.
        inverse_acronym_map: Mapping of structure acronyms to IDs.
        stripe_alpha: Default alpha for plotting.
        higher_alpha_structs: List of structures with higher alpha values.
        lower_alpha_structs: List of structures with lower alpha values.
        roll_up_ontology_label_yloc: Y-location for ontology label text (optional).
        roll_up_ontology_label_size: Font size for ontology label text (optional).
        met_color: Metadata color for plotting.
    """
    # Function code here

    projcols = copy(these_releveant_proj_cols)
    these_releveant_proj_cols = list(reversed(projcols))

    stripe_alpha = 0.07
    higher_alpha = 0.07
    lower_alpha = 0.07

    higher_alpha_structs = ['CNU','CTX']
    lower_alpha_structs = []

    ctx_id  = sg_df.loc['Isocortex']['id']
    iso_ctx_ch = sg_df[sg_df.structure_id_path.map(lambda x : ctx_id in x) ].index.tolist()

    these_contra_column = [c for c in these_releveant_proj_cols if "contra" in c]
    if these_contra_column:
        contra_start_index = these_releveant_proj_cols.index([c for c in these_releveant_proj_cols if "contra" in c][0])
    else:
        contra_start_index = None



    manual_label_records = {}
    meta_groups = ["VISp"] + iso_ctx_ch + ["HPF","OLF","CTXsp",  "CNU", "TH", "HY", "MB", "HB"]
    meta_groups.remove("PTLp")


    met_mean = np.log(met_df[these_releveant_proj_cols]+1).mean(axis=0)
    met_sem = np.log(met_df[these_releveant_proj_cols]+1).sem(axis=0)

    # if met_type=='L5 ET-2':
    #     alph=0.6
    # else:
    alph=1
    ys=np.arange(0,len(met_mean))
    ys = ys+0.5
    ax.plot(met_mean, ys, c= met_color,alpha=alph, lw=0.75)

    ax.fill_betweenx(ys,
                    met_mean - met_sem,
                    met_mean + met_sem,
                    color=met_color, 
                    linewidth=0, 
                    alpha=0.4)
    ax.set_ylim( (0,len(these_releveant_proj_cols)))





def plot_config_dict(config_dict,
                     all_altitude_sorted_proj_cols,
                     all_azimuth_sorted_proj_cols,
                     all_sorted_proj_cols,
                     non_it_structures_to_drop,
                     sg_df,
                     data_color_dict,
                     ):
    
    data_df = config_dict['data_df']
    output_file = config_dict['output_file']
    heatmap_column = config_dict['heatmap_column']
    heatmap_tick_name = config_dict['heatmap_tick_name']
    cbar_label = config_dict['cbar_label']
    categorical_adder_col = config_dict['categorical_adder_col']
    categorical_adder_col_tick_name = config_dict['categorical_adder_col_tick_name']
    met_type_list = config_dict['met_type_list']
    intensity_proj_mat_bool = config_dict['intensity_proj_mat_bool']
    DROP_MISMAPS = config_dict['DROP_MISMAPS']
    legend_center_locations_frac = config_dict['legend_center_locations_frac']
    legend_radii = config_dict['legend_radii']
    sort_order = config_dict['sort_order']
    heatmap_cmap = config_dict['heatmap_cmap']
    heatmap_alpha = config_dict['heatmap_alpha']
    proj_y_tick_loc = config_dict['proj_y_tick_loc'] 
    hist_width_per_cell = config_dict['hist_width_per_cell']
    spacer_between_proj_mat_and_proj_hist = config_dict['spacer_between_proj_mat_and_proj_hist']
    



    data_df = data_df[data_df['predicted_met_type'].isin(met_type_list)]

    adder_cols = [ config_dict[c] for c in sort_order ]
    adder_cols = [c for c in adder_cols if c is not None]

    if intensity_proj_mat_bool:
        proj_cmap = cm.viridis
        axhline_color = "white"
    else:
        axhline_color="lightgrey"

    pmet_met_col = 'predicted_met_type'
    sorted_mets = sorted(met_type_list,key = lambda x: ([sc in x for sc in ['IT',"ET","NP",'CT','L6b']].index(True),x) )

    sorted_pmets = [s for s in sorted_mets if s in data_df.predicted_met_type.unique()]

    assert all([p in sorted_pmets for p in sorted(data_df[pmet_met_col].unique())])


    sorted_df = pd.DataFrame()
    if DROP_MISMAPS:
        for pmet in sorted_pmets:
            pmet_df =  data_df[data_df[pmet_met_col]==pmet]
            t=1
            pmet_df = pmet_df.sort_values(by=adder_cols)
            sorted_df = sorted_df.append(pmet_df)


    else:
        print("TODO, without dropping mismaps (i think we threw these at either the front or the back of their met type depending what made more sense based on projection>?)")

    if 'altit' in heatmap_column: 
        these_sorted_proj_cols = all_altitude_sorted_proj_cols
    elif  'azimuth' in heatmap_column:
        these_sorted_proj_cols = all_azimuth_sorted_proj_cols
    else:
        these_sorted_proj_cols = all_sorted_proj_cols
        del these_sorted_proj_cols

    relevant_proj_cols = [c for c in these_sorted_proj_cols if sorted_df[c].max() != 0]
    relevant_proj_cols = [c for c in relevant_proj_cols if "fiber tracts" not in c]

    droppers = [c for c in relevant_proj_cols if sorted_df[c].astype(bool).sum()<3]
    resurected_cols = []
    for c in droppers:
        ct_dict = sorted_df.groupby('predicted_met_type')[c].apply(lambda x: x.astype(bool).sum()).to_dict()
        low_n_types = [k for k,v in sorted_df['predicted_met_type'].value_counts().to_dict().items() if v<3]
        if any([ct_dict[m] !=0 for m in low_n_types]):
            resurected_cols.append(c)    
    [droppers.remove(r) for r in resurected_cols]
    relevant_proj_cols = [c for c in relevant_proj_cols if c not in droppers]
    relevant_proj_cols = [c for c in relevant_proj_cols if c not in non_it_structures_to_drop]

    n_ipsi_cols = len([p for p in relevant_proj_cols if "ipsi" in p])
    n_contra_cols = len([p for p in relevant_proj_cols if "contra" in p])

    ipsi_metalabel_idx = int(n_ipsi_cols/2)
    contra_metalabel_idx = n_ipsi_cols+int(n_contra_cols/2)



    n_cells = len(sorted_df)

    fig=plt.gcf()
    # 3.93701 x 3.34646 IT

    fig.set_size_inches(3.93701, 3.34646)

#     fig = plt.figure(1, figsize=(2,4), dpi=300)

    added_ax_for_hist = 1
    categorical_added_ax_for_cbar = 0 #20 if categorical_adder_col is not None else 0
    continuous_added_ax_for_cbar = 0 #10
    max_add_cbar = 0 #max([categorical_added_ax_for_cbar,continuous_added_ax_for_cbar ])
    spacer_lateral = 15
    spacer_horiz = 0
    num_adder_rows = len(adder_cols)

    spacer_between_proj_mat_and_proj_hist = spacer_between_proj_mat_and_proj_hist
    num_proj_hists = len(sorted_pmets)
    if hist_width_per_cell is not None:
        width_per_hist = int(n_cells * hist_width_per_cell)
    else:
        width_per_hist = 9
        
    hist_spacer = 1
    total_cols_added_for_hist_addition = spacer_between_proj_mat_and_proj_hist + (num_proj_hists*width_per_hist) + ((num_proj_hists-1)*hist_spacer)

    total_num_cols = n_cells  + total_cols_added_for_hist_addition  + max_add_cbar  #+ spacer_lateral
    total_num_rows = len(adder_cols) + len(relevant_proj_cols) + spacer_horiz + added_ax_for_hist + spacer_horiz
#     total_num_rows = len(adder_cols) + len(relevant_proj_cols)  + added_ax_for_hist + spacer_horiz

    # helps the TG colorbar have more space like the heatmap 
    max_add_cbar = max_add_cbar-3
    grid = plt.GridSpec(total_num_rows,
                        total_num_cols,
                        wspace=0.0, hspace=0.0,
                        )

    legend_center_locations = [int(total_num_rows*i) for i in legend_center_locations_frac]


    cfig = [

        {
            "column":heatmap_column,
            "type":"continuous",
            "palette":heatmap_cmap,
            "ytick_name":heatmap_tick_name,
            "cbar_label":cbar_label, #"Soma Depth (µm)"

        }
    ]
    strip_axes = []
    if categorical_adder_col is not None:

        cat_dict = {
                "column":categorical_adder_col,
                "type":"categorical",
                "palette":data_color_dict,
                "ytick_name":categorical_adder_col_tick_name,
                "cbar_label":"Cre Line",
            }
        cfig = [cat_dict]+cfig

    imgs=[]
    for strip_ax_ct, cfig_dict in enumerate(cfig):


        column_name = cfig_dict['column']
        strip_ax = fig.add_subplot(grid[strip_ax_ct,:n_cells])
        strip_axes.append(strip_ax)

        strip_values = sorted_df[column_name].values.reshape(1,len(sorted_df))

        strip_cbar_center = legend_center_locations[strip_ax_ct]
        strip_cbar_rad = legend_radii[strip_ax_ct]

        cbar_ax_width = categorical_added_ax_for_cbar
        if cfig_dict['type']=='continuous':
            cbar_ax_width = continuous_added_ax_for_cbar

        # strip_cbar_ax = fig.add_subplot(grid[strip_cbar_center-strip_cbar_rad:strip_cbar_center+strip_cbar_rad,
        #                               -max_add_cbar: min([-1, -max_add_cbar+cbar_ax_width]) ])

        my_cmap = cfig_dict['palette']
        max_depth = sorted_df[column_name].max() 
        vmin = sorted_df[column_name].min() 
        norm=None
        if "soma_distance_from_pia" == column_name:
            max_depth = depths['wm']
            vmin=0



        if cfig_dict['type']=='continuous':

            norm = mpl.colors.Normalize(vmin=0, vmax=1) #vmin=vmin, vmax=max_depth)
            image_strip = norm(strip_values)[0].data #np.array([my_cmap( ((max_depth-d)/(max_depth-vmin)) )  for d in strip_values[0]])  
            image_strip = np.array([my_cmap(i) for i in image_strip])
            img_alpha = heatmap_alpha

        else:
            img_alpha = 1
            unique_vals = sorted_df[column_name].unique()
            if len(unique_vals)>len(my_cmap):
                print("WARNING NOT ENOUGH COLORS PROVIDED FOR NUMBER OF UNIQUE LABELS")

            color_dict = {}
            handle_list = []
            for cte,uni in enumerate(unique_vals):
                if uni not in my_cmap:
                    color_dict[uni] = cm.tab10(cte)
                else:
                    color_dict[uni]=my_cmap[uni]

                ptch = mpatches.Patch(color = color_dict[uni], label=uni)
                handle_list.append(ptch)
            image_strip = np.array([color_dict[d] for d in strip_values[0]])  

        image_strip = image_strip.reshape((1,n_cells,4))
        imgs.append(image_strip)
        imshow_vector(strip_ax, image_strip)
        # img = strip_ax.imshow(image_strip,aspect='auto',alpha=img_alpha)
        strip_ax.spines['bottom'].set_visible(False)
        strip_ax.spines['left'].set_visible(False)

        strip_ax.set_yticks([0.5])
        strip_ax.set_yticklabels([cfig_dict['ytick_name']],rotation=360,fontsize=5,ha='center')
        strip_ax.yaxis.tick_right()  # Move ticks to the right
        strip_ax.yaxis.set_label_position("right")  # Ensure the labels are on the right
        # Adjust tick labels manually to add padding
        for label in strip_ax.get_yticklabels():
            label.set_x(1.1)  # Move labels further out (1.0 is the default)

        strip_ax.spines['top'].set_visible(False)

    strip_ax.get_xaxis().set_visible(False)


    if intensity_proj_mat_bool:

        max_val = sorted_df[relevant_proj_cols].max().max()
        norm_version=True
        if max_val<=1:
            vmax=1
            proj_unit="(norm)"
        else:
            vmax=max_val
            norm_version=False
            proj_unit="(um)"

        intensity_cbar_center = legend_center_locations[1]
        intensity_cbar_rad = legend_radii[1]
        intensity_cbar_ax = fig.add_subplot(grid[intensity_cbar_center-intensity_cbar_rad:intensity_cbar_center+intensity_cbar_rad,
                                            -added_ax_for_cbar : ])
        norm = mpl.colors.Normalize(vmin=0, vmax=max_val)
        cb2 = mpl.colorbar.ColorbarBase(intensity_cbar_ax, cmap=proj_cmap,
                                    norm=norm,
                                    orientation='vertical')

        if not log_transform:
            cb2.set_label('Proj Intensity {}'.format(proj_unit))
        else:
            cb2.set_label('Proj Intensity (log(n+1))'.format(proj_unit))

    # construct an image array where color alpha is determined by projection intensity
    # Projection Heatmap
    proj_img_ax = fig.add_subplot(grid[num_adder_rows: (len(relevant_proj_cols)),:n_cells ])

    bg_color = [0.9372549019607843, 0.9372549019607843, 0.9411764705882353, 1.0]
    proj_img_ax.set_facecolor(bg_color)


    img_arr = np.zeros((len(relevant_proj_cols),n_cells, 4))
    for row_idx, proj_col  in enumerate(relevant_proj_cols):


        if intensity_proj_mat_bool:
            for counter, p_val in enumerate(sorted_df[proj_col]):

                if p_val != 0:
                    rgb_color = proj_cmap(p_val/vmax) # because we will pass in either raw matrix or norm matrix and vmax=1 for norm
                else:
                    rgb_color = [0.82745098039,0.82745098039,0.82745098039]

                img_arr[row_idx, counter , 0] = rgb_color[0]
                img_arr[row_idx, counter , 1] = rgb_color[1]
                img_arr[row_idx, counter , 2] = rgb_color[2]
                img_arr[row_idx, counter , 3] = 1

                strip_ax


        else:
            rgb_triplet = sg_df.loc[proj_col.replace("ipsi_","").replace("contra_",""),'rgb_triplet']
            try:
                rgb_triplet = ast.literal_eval(rgb_triplet)
            except:
                True
            
            r_val = rgb_triplet[0]/255
            g_val = rgb_triplet[1]/255
            b_val = rgb_triplet[2]/255

            r_col = np.array([r_val]*n_cells)
            g_col = np.array([g_val]*n_cells)
            b_col = np.array([b_val]*n_cells)

            alpha_vals = sorted_df[proj_col]
            # Make alpha vals all 1
            alpha_vals = sorted_df[proj_col].astype(bool)
            img_arr[row_idx, : , 0] = r_col
            img_arr[row_idx, : , 1] = g_col
            img_arr[row_idx, : , 2] = b_col
            img_arr[row_idx, : , 3] = alpha_vals

    # NEW PLOTTING FOR ILLUSTRATOR
    imshow_vector(proj_img_ax, img_arr, 'upper')
    
    # proj_img_ax.imshow(img_arr,aspect='auto')
    
    
    
    proj_img_ax.spines['top'].set_visible(False)
    if n_ipsi_cols!= len(relevant_proj_cols):

        proj_img_ax.axhline(len(relevant_proj_cols) - n_ipsi_cols, c='lightgrey',linestyle='--',lw=0.75)

    # Major ticks
    modified_proj_cols = []
    for c in relevant_proj_cols: 
        if "fiber" in c:
            modified_proj_cols.append(c)
        else:
            modified_proj_cols.append(c.replace("ipsi_","").replace("contra_",""))

    proj_img_ax.set_yticks(np.arange(0.5, len(relevant_proj_cols)+0.5, 1),)
    proj_img_ax.set_yticklabels(list(reversed(modified_proj_cols)),rotation=360, fontsize=5, ha='center')

#     proj_img_ax.tick_params(axis='y', which='both', pad=500)  # Add padding (in points) between ticks and labels
    proj_img_ax.yaxis.tick_right()  # Move y-axis ticks to the right
#     proj_img_ax.yaxis.set_label_position("right")  # Move y-axis label to the right

    proj_img_ax.set_xticks([])

    for label in proj_img_ax.get_yticklabels():
        label.set_x(proj_y_tick_loc)  # Move labels further out (1.0 is the default)

    
#     proj_img_ax.text(-25,ipsi_metalabel_idx, "Ipsilateral",rotation=90, horizontalalignment='center',
#                     verticalalignment='center',fontsize=6)
#     proj_img_ax.text(-25,contra_metalabel_idx, "Contralateral",rotation=90, horizontalalignment='center',
#                     verticalalignment='center',fontsize=6)

    proj_img_ax.spines['left'].set_visible(False)
    proj_img_ax.spines['top'].set_visible(False)
    proj_img_ax.spines['bottom'].set_visible(False)

    proj_img_ax.spines['right'].set_linewidth(0.25)
    strip_ax.spines['right'].set_linewidth(0.25)






    #HISTOGRAM
    # Number targets per cell HISTOGRAM
    num_structs_per_cell = sorted_df[these_sorted_proj_cols].astype(bool).sum(axis=1).to_dict()

    # if we want total axon length
    # num_structs_per_cell = sorted_df[relevant_proj_cols].sum(axis=1).to_dict()
    # num_structs_per_cell = {k:v*0.001 for k,v in num_structs_per_cell.items()}

    hist_vals = [v for v in num_structs_per_cell.values()]

    histogram_ax = fig.add_subplot(grid[ len(relevant_proj_cols)+spacer_horiz:,:n_cells ])
    xs = np.arange(0.5, n_cells+0.5, 1)
    bar_plt = histogram_ax.bar(x=xs, height=hist_vals,width=0.1)

    # histogram_ax.set_xlim((len(hist_vals), 0))
    histogram_ax.set_xticks([])
    histogram_ax.invert_yaxis()
#     histogram_ax.set_yticks([5,10])
    histogram_ax.set_yticklabels(histogram_ax.get_yticklabels(), fontsize=5)


    for axis in ['top','bottom','left','right']:
        histogram_ax.spines[axis].set_linewidth(0.2)
    # histogram_ax.xaxis.set_ticks_position("top")
    histogram_ax.set_xlim(proj_img_ax.get_xlim())

    # Separating Lines
    vert_line_idx=0 #-0.5
    prev_idx = 0
    soma_depth_img_ylabels = ['']*n_cells

    pmet_counter = -1
    vert_line_w = 0.225
    proj_hist_axes = []
    for pmet_lbl in sorted_pmets:
        pmet_counter+=1

        pmet_df =  sorted_df[sorted_df[pmet_met_col]==pmet_lbl]
        pmet_df = pmet_df.sort_values(by=adder_cols)

        vert_line_idx +=len(pmet_df)
        if pmet_lbl != sorted_pmets[-1]:

            proj_img_ax.axvline(vert_line_idx, c="k",lw=vert_line_w)
            for axes in strip_axes:
                axes.axvline(vert_line_idx, c='k',lw=vert_line_w)

            histogram_ax.axvline(vert_line_idx, c='k',lw=vert_line_w)

        next_idx = prev_idx+len(pmet_df)
        pmet_color = data_color_dict[pmet_lbl]
        center = prev_idx + int((next_idx-prev_idx)/2)
        soma_depth_img_ylabels[center-1] = "{} (n={})".format(pmet_lbl,len(pmet_df))

        for idx in range(prev_idx,next_idx):

            bar_plt[idx].set_color(pmet_color)
        prev_idx = next_idx 

        start_col = n_cells+spacer_between_proj_mat_and_proj_hist + (pmet_counter*(width_per_hist+hist_spacer))
        finish_col = start_col + width_per_hist
        this_proj_hist_ax = fig.add_subplot(grid[ num_adder_rows: (len(relevant_proj_cols)), 
                                                start_col:finish_col ])

        plot_projection_histogram(these_releveant_proj_cols=relevant_proj_cols,
        sg_df=sg_df,
        met_df=pmet_df,
        ax=this_proj_hist_ax,
        met_color = pmet_color)
        title_formated = pmet_lbl
        if "Car3" in pmet_lbl:
            title_formated = "L5/L6 IT\nCar3"
        if "L5 ET-1 Chrna6" in pmet_lbl:
            title_formated = "L5 ET-1\nChrna6"
            
        this_proj_hist_ax.set_title(title_formated, rotation=45,ha='center',color=pmet_color,size=5)
        this_proj_hist_ax.axhline(len(relevant_proj_cols)-n_ipsi_cols,c='lightgrey',linestyle='--',lw=0.75)

        this_proj_hist_ax.spines['top'].set_visible(False)
        this_proj_hist_ax.spines['right'].set_visible(False)
        this_proj_hist_ax.spines['bottom'].set_linewidth(0.5)
        this_proj_hist_ax.xaxis.set_tick_params(width=0.25, pad=0.33)
        if pmet_counter!=0:
            this_proj_hist_ax.set_yticks([])
            this_proj_hist_ax.set_xticklabels([])

        else:
            ys=np.arange(0, len(relevant_proj_cols), 1)
            ys=ys+0.5
            this_proj_hist_ax.set_yticks(ys)
            this_proj_hist_ax.set_yticklabels([""]*len(ys),rotation=360, fontsize=5, ha='center')
            this_proj_hist_ax.set_xticklabels(this_proj_hist_ax.get_xticklabels(),fontsize=5)
            this_proj_hist_ax.set_xticklabels(this_proj_hist_ax.get_xticklabels(),fontsize=5)
            this_proj_hist_ax.set_xlabel("Log(n+1) Proj. Length ",fontsize=5)


        proj_hist_axes.append(this_proj_hist_ax)


    mark_up_axes = strip_axes[0] # will decorate the top most one
    mark_up_axes.set_xticks(np.arange(0, n_cells, 1))
    mark_up_axes.tick_params(bottom=False,top=False)#axis='x', colors='white')
    mark_up_axes.set_xticklabels(soma_depth_img_ylabels,
                                    rotation=45, 
                                    fontsize=5,
                                    horizontalalignment='left',
                                    verticalalignment='bottom')
    mark_up_axes.xaxis.set_ticks_position('top') 
    mark_up_axes.tick_params(axis='x',         
        which='both',      
        bottom=False,      
        top=False,        
        labelbottom=False)

    #manually adjust some of the tiny overlapping labels
    special_label_horizontal_alignments = {
        "L5 IT-1":"left",
        "L5/L6 IT Car3":"right",
        "L6 IT-1":"right"
    }
    xTick_objects = mark_up_axes.xaxis.get_major_ticks()
    for t_cter, t in enumerate(mark_up_axes.get_xticklabels()):
        text = t.get_text()
        text = text.split(" (n")[0]
        if text!="":        
            if text in list(special_label_horizontal_alignments.keys()):
                xTick_objects[t_cter].label1.set_horizontalalignment(special_label_horizontal_alignments[text]) 



    for txt in mark_up_axes.get_xticklabels():
        txt_val =txt.get_text()
        if txt_val != "":
            met = txt_val.split("(n")[0].strip()
            clr = data_color_dict[met]
            txt.set_color(clr)

    y_adj_axes = strip_axes+[histogram_ax,proj_img_ax ] + proj_hist_axes
    for adj_ax in y_adj_axes:

        adj_ax.yaxis.set_tick_params(width=0.25)
        adj_ax.yaxis.set_tick_params(width=0.25)
        adj_ax.tick_params(axis='y', which='major', pad=0.5, length=2)
        adj_ax.spines['left'].set_linewidth(0.5)

    for ax_i in proj_hist_axes:
        ax_i.spines['bottom'].set_linewidth(0.5)
    histogram_ax.spines['left'].set_linewidth(0.25)
    histogram_ax.spines['bottom'].set_visible(False)
    histogram_ax.spines['right'].set_visible(False)
    print(output_file)
    # plotted_structures[output_file] = relevant_proj_cols
    fig.savefig(output_file,bbox_inches='tight',transparent=True, dpi=300)
    # plt.show()
    # plt.clf()
    plt.close()
    del fig
    
    
def main():
        
            
    with open("../data/ScriptArgs.json","r") as f:
        args = json.load(f)


    retno_df = pd.read_csv(args['vis_top_flatmap_retnotopy_data'],index_col=0)
    az_min, az_max = retno_df.azimuth.min(), retno_df.azimuth.max()
    alt_min, alt_max = retno_df.altitude.min(), retno_df.altitude.max()



    metadata = args['fmost_metadata_file']
    fmost_metadata = pd.read_csv(metadata,index_col=0)

    with open(args['color_file'],"r") as f:
        data_color_dict = json.load(f)

    proj_df=pd.read_csv(args['fmost_projection_matrix_roll_up'],index_col=0)
    proj_df['swc_path'] = proj_df.index


    # with open(args['vis_structure_order_path'],'r') as f:
    #     vis_struct_order = json.load(f)

    sg_df = load_structure_graph()




    # sort projection columns by ipsi VIS ipsi NonVIS contra VIS contra NonVIS
    # within VIS sort by Harris 2019 visual heirarchy
    unformatted_proj_cols = [c for c in proj_df.columns if any([i in c for i in ['ipsi','contra']])]
    rename_dict = {c:c.replace(",","").strip() for c in unformatted_proj_cols}
    proj_df = proj_df.rename(columns=rename_dict)

    proj_cols = [c for c in proj_df.columns if any([i in c for i in ['ipsi','contra']]) and "Out Of Cortex" not in c]

    ipsi_vis_cols = [c for c in proj_cols if "ipsi_VIS" in c and c!= 'ipsi_VISC']
    sorted_ipsi_vis_cols = sorted(ipsi_vis_cols)#,key=lambda x:vis_struct_order[x.replace("ipsi_","")])

    ipsi_cols = [c for c in proj_cols if (c not in sorted_ipsi_vis_cols) and ("ipsi" in c)]
    sorted_ipsi_cols = sorted(ipsi_cols, key = lambda x: sg_df.loc[x.replace("ipsi_","")]['graph_order'])

    contra_vis_cols = [c for c in proj_cols if "contra_VIS" in c and c!= 'contra_VISC']
    sorted_contra_vis_cols = sorted(contra_vis_cols)#,key=lambda x:vis_struct_order[x.replace("contra_",'')])

    contra_cols = [c for c in proj_cols if (c not in sorted_contra_vis_cols) and ("contra" in c)]
    sorted_contra_cols = sorted(contra_cols, key = lambda x: sg_df.loc[x.replace("contra_","")]['graph_order'])


    #alternatively sort VIS structures by retnotopic values
    region_mean_altitude = retno_df.groupby("region")[['altitude']].mean()
    ascending_altitude_regions = region_mean_altitude.sort_values(by='altitude').index.tolist()
    ascending_altitude_regions.remove("VISp")

    region_mean_azimuth = retno_df.groupby("region")[['azimuth']].mean()
    ascending_azimuth_regions = region_mean_azimuth.sort_values(by='azimuth').index.tolist()
    ascending_azimuth_regions.remove("VISp")

    alt_sorted_ipsi_vis_cols = ['ipsi_VISp'] + [f'ipsi_{c}' for c in ascending_altitude_regions]
    alt_sorted_contra_vis_cols = ['contra_VISp'] + [f'contra_{c}' for c in ascending_altitude_regions]

    azim_sorted_ipsi_vis_cols = ['ipsi_VISp'] + [f'ipsi_{c}' for c in ascending_azimuth_regions]
    azim_sorted_contra_vis_cols = ['contra_VISp'] + [f'contra_{c}' for c in ascending_azimuth_regions]



    all_altitude_sorted_proj_cols = alt_sorted_ipsi_vis_cols + sorted_ipsi_cols + alt_sorted_contra_vis_cols + sorted_contra_cols
    all_azimuth_sorted_proj_cols = azim_sorted_ipsi_vis_cols + sorted_ipsi_cols + azim_sorted_contra_vis_cols + sorted_contra_cols
    all_sorted_proj_cols = sorted_ipsi_vis_cols + sorted_ipsi_cols + sorted_contra_vis_cols + sorted_contra_cols



    azimuth_cmap = plt.cm.magma #pink_cmap #warm_neutrals_cmap#plt.cm.YlOrBr   
    azimuth_alpha = 1 # 0.6
    altitude_cmap = plt.cm.viridis # warm_neutrals_cmap#plt.cm.OrRd 
    altitude_alpha = 1 #plt.cm.viridis #1 # 0.75



    visualization_df = fmost_metadata.merge(proj_df,left_index=True,right_index=True)
    visualization_df['cre_line']=visualization_df['cre_line'].apply(lambda x:x.split("-")[0])
    visualization_df['cre_line']=visualization_df['cre_line'].apply(lambda x:x.split(";")[0])

    it_visp_df = visualization_df[ (visualization_df['ccf_soma_location_nolayer']=='VISp') ].copy()
    it_visp_df = it_visp_df[it_visp_df['predicted_met_type'].str.contains("IT")]

    it_visp_df['altitude_norm'] = (it_visp_df.altitude - alt_min)/ (alt_max - alt_min)
    it_visp_df['azimuth_norm'] = (it_visp_df.azimuth - az_min)/ (az_max - az_min)

    visp_df = visualization_df[visualization_df['ccf_soma_location_nolayer']=='VISp']

    it_met_types = set([m for m in visualization_df.predicted_met_type if "IT" in m])
    non_it_met_types = set([m for m in visualization_df.predicted_met_type if "IT" in m])


    configs = [

        {
            "data_df":it_visp_df,
            "output_file":"./plot_proj_mat_histos_VISp_IT_Azimuth.pdf",
            "heatmap_column":"azimuth_norm",
            "heatmap_tick_name":"Azimuth",
            "cbar_label":"Azimuth",
            "heatmap_cmap":azimuth_cmap,
            "heatmap_alpha":azimuth_alpha,
            "categorical_adder_col":None,
            "categorical_adder_col_tick_name":None,#"Cre Line",
            "met_type_list":it_met_types,
            "intensity_proj_mat_bool":False,
            "DROP_MISMAPS": True,
            "legend_center_locations_frac": [0.2, 0.5],
            "legend_radii": [2,3],
            "sort_order":["heatmap_column", "categorical_adder_col"],
            'proj_y_tick_loc':1.11,
            'fig_size':[3.93701, 3.34646],
            'hist_width_per_cell':None,
            'spacer_between_proj_mat_and_proj_hist':16,

        },

        {
            "data_df":it_visp_df,
            "output_file":"./plot_proj_mat_histos_VISp_IT_Altitude.pdf",
            "heatmap_column":"altitude_norm",
            "heatmap_tick_name":"Altitude",
            "cbar_label":"Altitude",
            "heatmap_cmap":altitude_cmap,
            "heatmap_alpha":altitude_alpha,
            "categorical_adder_col":None,
            "categorical_adder_col_tick_name":None,#"Cre Line",
            "met_type_list":it_met_types,
            "intensity_proj_mat_bool":False,
            "DROP_MISMAPS": True,
            "legend_center_locations_frac": [0.2, 0.5],
            "legend_radii": [2,3],
            "sort_order":["heatmap_column", "categorical_adder_col"],
            'proj_y_tick_loc':1.11,
            'fig_size':[3.93701, 3.34646],
            'hist_width_per_cell':None,
            'spacer_between_proj_mat_and_proj_hist':16,
        }

    ]
    del visp_df

    non_it_structures_to_drop = """ipsi_VPM
    ipsi_MGd
    ipsi_MGv
    ipsi_PO
    ipsi_POL
    ipsi_AD
    ipsi_IntG
    ipsi_SubG
    ipsi_STN
    ipsi_SNr
    ipsi_SCiw
    ipsi_MPT
    ipsi_OP
    ipsi_NLL
    ipsi_TRN
    ipsi_TH
    ipsi_P
    ipsi_MB
    """
    non_it_structures_to_drop = non_it_structures_to_drop.split("\n")

    for config_dict in configs:

        plot_config_dict(config_dict,
                     all_altitude_sorted_proj_cols=all_altitude_sorted_proj_cols,
                     all_azimuth_sorted_proj_cols=all_azimuth_sorted_proj_cols,
                     all_sorted_proj_cols=all_sorted_proj_cols,
                     non_it_structures_to_drop=non_it_structures_to_drop,
                     sg_df=sg_df,
                     data_color_dict=data_color_dict
                    )
        



    #
    #
    #   Non-IT Projection Matrix
    #
    #

    hist_width_per_cell = 0.15517241379310345 # width_per_hist/n_cells

    visualization_df = fmost_metadata.merge(proj_df,left_index=True,right_index=True)
    visualization_df['cre_line']=visualization_df['cre_line'].apply(lambda x:x.split("-")[0])
    visualization_df['cre_line']=visualization_df['cre_line'].apply(lambda x:x.split(";")[0])

    visp_df = visualization_df[ (visualization_df['ccf_soma_location_nolayer']=='VISp') | 
                    (visualization_df['predicted_met_type'].isin(['L5 NP'])) #(['L5 ET-2','L5 NP']) )
                    ]

    #actually taking n=1 L5 ET-2 out
    # visp_df = visp_df[visp_df['predicted_met_type']!='L5 ET-2']

    visp_df['altitude_norm'] = (visp_df.altitude-alt_min)/ (alt_max-alt_min)
    visp_df['azimuth_norm'] = (visp_df.azimuth-az_min)/ (az_max-az_min)


    it_met_types = set([m for m in visualization_df.predicted_met_type if "IT" in m])
    non_it_met_types = set([m for m in visualization_df.predicted_met_type if "IT" not in m])



    configs = [

        {
            "data_df":visp_df,
            "output_file":"plot_proj_mat_histos_VISp_NonIT_Azimuth.pdf",
            "heatmap_column":"azimuth_norm",
            "heatmap_tick_name":"Azimuth",
            "cbar_label":"Azimuth",
            "heatmap_cmap":azimuth_cmap,
            "heatmap_alpha":azimuth_alpha,
            "categorical_adder_col":None,
            "categorical_adder_col_tick_name":None,#"Cre Line",
            "met_type_list":non_it_met_types,
            "intensity_proj_mat_bool":False,
            "DROP_MISMAPS": True,
            "legend_center_locations_frac": [0.2, 0.5],
            "legend_radii": [3,3],
            "sort_order":["heatmap_column", "categorical_adder_col"],
            'proj_y_tick_loc':1.11,
            'fig_size':[3.93701, 3.54331],
            'hist_width_per_cell':hist_width_per_cell,
            'spacer_between_proj_mat_and_proj_hist':30,

        },


        {
            "data_df":visp_df,
            "output_file":"./plot_proj_mat_histos_VISp_NonIT_Altitude.pdf",
            "heatmap_column":"altitude_norm",
            "heatmap_tick_name":"Altitude",
            "cbar_label":"Altitude",
            "heatmap_cmap":altitude_cmap,
            "heatmap_alpha":altitude_alpha,
            "categorical_adder_col":None,
            "categorical_adder_col_tick_name":None,
            "met_type_list":non_it_met_types,
            "intensity_proj_mat_bool":False,
            "DROP_MISMAPS": True,
            "legend_center_locations_frac": [0.2, 0.5],
            "legend_radii": [3,3],
            "sort_order":["heatmap_column", "categorical_adder_col"],
            'proj_y_tick_loc':1.11,
            'fig_size':[3.93701, 3.54331],
            'hist_width_per_cell':hist_width_per_cell,
            'spacer_between_proj_mat_and_proj_hist':30,
        }

    ]
    del visp_df

    for config_dict in configs:

        plot_config_dict(config_dict,
                     all_altitude_sorted_proj_cols=all_altitude_sorted_proj_cols,
                     all_azimuth_sorted_proj_cols=all_azimuth_sorted_proj_cols,
                     all_sorted_proj_cols=all_sorted_proj_cols,
                     non_it_structures_to_drop=non_it_structures_to_drop,
                     sg_df=sg_df,
                     data_color_dict=data_color_dict
                    )
        
if __name__=='__main__':
    main()