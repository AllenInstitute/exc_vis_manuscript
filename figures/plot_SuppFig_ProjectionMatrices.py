import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import os
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
import matplotlib.patches as mpatches
import json
import matplotlib as mpl
from scipy import ndimage
from matplotlib import cm
import ast
from morph_utils.ccf import load_structure_graph

def main():
    
    with open("../data/ScriptArgs.json","r") as f:
        args = json.load(f)

#    alt_azi_lookup_df = pd.read_csv(args['vis_top_flatmap_retnotopy_data'])
    retno_df = pd.read_csv(args['vis_top_flatmap_retnotopy_data'],index_col=0)
    az_min, az_max = retno_df.azimuth.min(), retno_df.azimuth.max()
    alt_min, alt_max = retno_df.altitude.min(), retno_df.altitude.max()


    metadata = args['fmost_metadata_file']
    fmost_metadata = pd.read_csv(metadata,index_col=0)


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
    
    subclass_cdict = {
        "IT-NP-6b":plt.cm.Accent(0),
        "ET":plt.cm.Accent(1),
        "CT":plt.cm.Accent(2),
        'None':plt.cm.Accent(10),
    }
        
    visualization_df = fmost_metadata.merge(proj_df,left_index=True,right_index=True)
    visualization_df['cre_line']=visualization_df['cre_line'].apply(lambda x:x.split("-")[0])
    visualization_df['cre_line']=visualization_df['cre_line'].apply(lambda x:x.split(";")[0])

    # visualization_df = visualization_df[visualization_df['predicted_met_type'].str.contains("IT")]

    visualization_df['altitude_norm'] = (visualization_df.altitude - alt_min)/ (alt_max - alt_min)
    visualization_df['azimuth_norm'] = (visualization_df.azimuth - az_min)/ (az_max - az_min)


    it_met_types = set([m for m in visualization_df.predicted_met_type if "IT" in m])
    non_it_met_types = set([m for m in visualization_df.predicted_met_type if "IT" in m])

    from matplotlib.colors import LinearSegmentedColormap

    warm_neutrals_cmap = LinearSegmentedColormap.from_list(
        "warm_neutrals", ["#ffffff", "#DED9D2", "#B8ACA0", "#8D7967"], N=256
    )

    azimuth_cmap = plt.cm.magma #pink_cmap #warm_neutrals_cmap#plt.cm.YlOrBr   
    azimuth_alpha = 1 # 0.6
    altitude_cmap = plt.cm.viridis # warm_neutrals_cmap#plt.cm.OrRd 
    altitude_alpha = 1 #plt.cm.viridis #1 # 0.75


    cat_dicts = [

                    {
                        "column":"ccf_soma_location_nolayer",
                        "type":"categorical",
                        "palette":data_color_dict,
                        "ytick_name":"Soma Location",
                        "cbar":True,
                        "cbar_label":"Soma Location",
                    },
        
                    {
                        "column":"auto_projection_subclass",
                        "type":"categorical",
                        "palette":subclass_cdict,
                        "ytick_name":"Proj. Subclass",
                        "cbar":True,
                        "cbar_label":"Subclass",
                    },
        
                    {
                        "column":"dend_derived_predicted_subclass",
                        "type":"categorical",
                        "palette":subclass_cdict,
                        "ytick_name":"Dend. Subclass",
                        "cbar":False,
                        "cbar_label":None,
                    },
        
                    {
                        "column":"local_axon_derived_subclass",
                        "type":"categorical",
                        "palette":subclass_cdict,
                        "ytick_name":"Local Axon Subclass",
                        "cbar":False,
                        "cbar_label":None,
                    },

        
    ]

    configs = [
        
        # {
        #     "data_df":visualization_df,
        #     "output_file":"./SuppFig_ProjMat_IT_Azimuth.pdf",
        #     "heatmap_column":"azimuth_norm",
        #     "heatmap_tick_name":"Azimuth",
        #     "cbar_label":"Azimuth",
        #     "heatmap_cmap":azimuth_cmap,
        #     "heatmap_alpha":azimuth_alpha,
        #     "cat_dicts":cat_dicts,
        #     # "categorical_adder_col":"ccf_soma_location_nolayer",
        #     # "categorical_adder_col_tick_name":"Structure",
        #     "met_type_list":it_met_types,
        #     "intensity_proj_mat_bool":False,
        #     "DROP_MISMAPS": True,
        #     "legend_center_locations_frac": [0.2, 0.5, 0.7],
        #     "legend_radii": [2,2,3],
        #     "sort_order":['ccf_soma_location_nolayer', "azimuth_norm"],
        #     "categorical_sort_order": ["VISp"] +ascending_azimuth_regions,

        # },
        
        {
            "data_df":visualization_df,
            "output_file":"./plot_SuppFig_ProjMat_IT_Altitude.pdf",
            "heatmap_column":"altitude_norm",
            "heatmap_tick_name":"Altitude",
            "cbar_label":"Altitude",
            "heatmap_cmap":altitude_cmap,
            "heatmap_alpha":altitude_alpha,
            "cat_dicts":cat_dicts,
            # "categorical_adder_col":"ccf_soma_location_nolayer",
            # "categorical_adder_col_tick_name":"Structure",
            "met_type_list":it_met_types,
            "intensity_proj_mat_bool":False,
            "DROP_MISMAPS": True,
            "legend_center_locations_frac": [0.2, 0.5, 0.7],
            "legend_radii": [2,2,3],
            "sort_order":['ccf_soma_location_nolayer', "altitude_norm"],
            "categorical_sort_order": ["VISp"] +ascending_altitude_regions,

        }
        
    ]

    this_df = visualization_df[visualization_df['predicted_met_type'].isin(it_met_types)]
    inch_per_sp = 4/this_df.shape[0]

    plotted_structures = {}
    yfontsize = 4
    hist_yfontsize = 4
    for config_dict in configs:
        
        data_df = config_dict['data_df']
        output_file = config_dict['output_file']
        heatmap_column = config_dict['heatmap_column']
        heatmap_tick_name = config_dict['heatmap_tick_name']
        cbar_label = config_dict['cbar_label']
        cat_dicts = config_dict['cat_dicts']
        # categorical_adder_col = config_dict['categorical_adder_col']
        # categorical_adder_col_tick_name = config_dict['categorical_adder_col_tick_name']
        met_type_list = config_dict['met_type_list']
        intensity_proj_mat_bool = config_dict['intensity_proj_mat_bool']
        DROP_MISMAPS = config_dict['DROP_MISMAPS']
        legend_center_locations_frac = config_dict['legend_center_locations_frac']
        legend_radii = config_dict['legend_radii']
        sort_order = config_dict['sort_order']
        heatmap_cmap = config_dict['heatmap_cmap']
        heatmap_alpha = config_dict['heatmap_alpha']
        categorical_sort_order = config_dict['categorical_sort_order']

        categorical_adder_col = 'ccf_soma_location_nolayer'
        assert all([c in categorical_sort_order for c in data_df[categorical_adder_col].unique()])
        
        data_df = data_df[data_df['predicted_met_type'].isin(met_type_list)]
        
        # adder_cols = [ config_dict[c] for c in sort_order ]
        # adder_cols = [c for c in adder_cols if c is not None]
        
        if intensity_proj_mat_bool:
            proj_cmap = cm.viridis
            axhline_color = "white"
        else:
            axhline_color="lightgrey"
            
        pmet_met_col = 'predicted_met_type'
        sorted_mets = sorted(met_type_list,key = lambda x: ([sc in x for sc in ['IT',"ET","NP",'CT','L6b']].index(True),x) )

        sorted_pmets = [s for s in sorted_mets if s in data_df.predicted_met_type.unique()]

        assert all([p in sorted_pmets for p in sorted(data_df[pmet_met_col].unique())])

        
        
        projection_matrix_meta_order = ["IT","ET","NP","CT","L6b"]
        sorted_df = pd.DataFrame()
        for pmet in sorted_pmets:
            pmet_df =  data_df[data_df[pmet_met_col]==pmet]
            for cat in categorical_sort_order:
                this_cat = pmet_df[pmet_df[categorical_adder_col]==cat]
                this_cat = this_cat.sort_values(by=heatmap_column)
                sorted_df = sorted_df.append(this_cat)
                
            # pmet_df = pmet_df.sort_values(by=adder_cols)
            # sorted_df = sorted_df.append(pmet_df)


        # else:
        #     print("TODO, without dropping mismaps (i think we threw these at either the front or the back of their met type depending what made more sense based on projection>?)")

        if 'altit' in heatmap_column: 
            these_sorted_proj_cols = all_altitude_sorted_proj_cols
        elif  'azimuth' in heatmap_column:
            these_sorted_proj_cols = all_azimuth_sorted_proj_cols
        else:
            these_sorted_proj_cols = all_sorted_proj_cols
            del these_sorted_proj_cols

        relevant_proj_cols = [c for c in these_sorted_proj_cols if sorted_df[c].max() != 0]
        relevant_proj_cols = [c for c in relevant_proj_cols if "fiber tracts" not in c]

        droppers = []# [c for c in relevant_proj_cols if sorted_df[c].astype(bool).sum()<3]
        resurected_cols = []
        for c in droppers:
            ct_dict = sorted_df.groupby('predicted_met_type')[c].apply(lambda x: x.astype(bool).sum()).to_dict()
            if [k for k,v in ct_dict.items() if v==2]:
                resurected_cols.append(c)
        [droppers.remove(r) for r in resurected_cols]
        relevant_proj_cols = [c for c in relevant_proj_cols if c not in droppers]
        
        n_ipsi_cols = len([p for p in relevant_proj_cols if "ipsi" in p])
        n_contra_cols = len([p for p in relevant_proj_cols if "contra" in p])

        ipsi_metalabel_idx = int(n_ipsi_cols/2)
        contra_metalabel_idx = n_ipsi_cols+int(n_contra_cols/2)
        
        

        n_cells = len(sorted_df)

        fig=plt.gcf()

        # 7.08661 " wide X 6.69291
        fig.set_size_inches(inch_per_sp*n_cells, 6.69)
        
    #     fig = plt.figure(1, figsize=(2,4), dpi=300)

        added_ax_for_hist = 3
        categorical_added_ax_for_cbar = 20 if categorical_adder_col is not None else 0
        continuous_added_ax_for_cbar = 10
        max_add_cbar = max([categorical_added_ax_for_cbar,continuous_added_ax_for_cbar ])
        spacer_lateral = 20
        spacer_horiz = 0

        num_adder_rows = 0
        if heatmap_column is not None:
            num_adder_rows+=1
        num_adder_rows+=len(cat_dicts)


        total_num_cols = n_cells + spacer_lateral + max_add_cbar 
        total_num_rows = num_adder_rows + len(relevant_proj_cols) + spacer_horiz + added_ax_for_hist + spacer_horiz
    #     total_num_rows = len(adder_cols) + len(relevant_proj_cols)  + added_ax_for_hist + spacer_horiz

        inches_per_column = 15/total_num_cols
        inches_per_row = 21/total_num_rows

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
                "cbar":True,
                "ytick_name":heatmap_tick_name,
                "cbar_label":cbar_label, #"Soma Depth (µm)"

            }
        ]
        strip_axes = []
        if cat_dicts is not None:
            cfig = cat_dicts+cfig

        imgs=[]
        strip_ax_with_cbar_ct = -1
        for strip_ax_ct, cfig_dict in enumerate(cfig):


            column_name = cfig_dict['column']
            strip_ax = fig.add_subplot(grid[strip_ax_ct,:n_cells])
            strip_axes.append(strip_ax)

            strip_values = sorted_df[column_name].values.reshape(1,len(sorted_df))

            strip_cbar_center = None
            strip_cbar_rad = None
            if cfig_dict['cbar']:
                strip_ax_with_cbar_ct+=1
                strip_cbar_center = legend_center_locations[strip_ax_with_cbar_ct]
                strip_cbar_rad = legend_radii[strip_ax_with_cbar_ct]

            cbar_ax_width = categorical_added_ax_for_cbar
            if cfig_dict['type']=='continuous':
                cbar_ax_width = continuous_added_ax_for_cbar

            if cfig_dict['cbar']==True:
                    
                strip_cbar_ax = fig.add_subplot(grid[strip_cbar_center-strip_cbar_rad:strip_cbar_center+strip_cbar_rad,
                                            -max_add_cbar: min([-1, -max_add_cbar+cbar_ax_width]) ])
            else:
                strip_cbar_ax = None
            
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
                print(column_name, unique_vals)
                if len(unique_vals)>len(my_cmap):
                    print("WARNING NOT ENOUGH COLORS PROVIDED FOR NUMBER OF UNIQUE LABELS")

                color_dict = {}
                handle_list = []
                for cte,uni in enumerate(unique_vals):
                    # if uni not in my_cmap:
                    #     print
                    #     color_dict[uni] = cm.tab10(cte)
                    # else:
                    color_dict[uni]=my_cmap[uni]

                    ptch = mpatches.Patch(color = color_dict[uni], label=uni)
                    handle_list.append(ptch)
                image_strip = np.array([color_dict[d] for d in strip_values[0]])  

            image_strip = image_strip.reshape((1,n_cells,4))
            imgs.append(image_strip)
            img = strip_ax.imshow(image_strip,aspect='auto',alpha=img_alpha)
            strip_ax.spines['bottom'].set_visible(False)
            strip_ax.spines['right'].set_visible(False)

            strip_ax.set_yticks([0])
            strip_ax.set_yticklabels([cfig_dict['ytick_name']],rotation=360,fontsize=yfontsize)#,ha='center')
            # strip_ax.yaxis.tick_right()  # Move ticks to the right
            # strip_ax.yaxis.set_label_position("right")  # Ensure the labels are on the right
            # Adjust tick labels manually to add padding
            # for label in strip_ax.get_yticklabels():
            #     label.set_x(1.1)  # Move labels further out (1.0 is the default)


            if strip_cbar_ax is not None:
                    
                if cfig_dict['type']=='continuous':
                    if norm is not None:
                        cb1 = mpl.colorbar.ColorbarBase(strip_cbar_ax, cmap=my_cmap,alpha=img_alpha,
                                                    norm=norm,
                                                        
                                                    orientation='vertical')
                    else:
                        cb1 = mpl.colorbar.ColorbarBase(strip_cbar_ax, cmap=my_cmap,alpha=img_alpha,
                                        orientation='vertical')
        
                    cb1.set_label(cfig_dict['cbar_label'], fontsize=5)
                    cb1.ax.tick_params(labelsize=5) 
        #             cb1.set_yticklabels(cb1.get_yticklabels(), fontsize=5)
                    
                    if "soma_distance_from_pia" == column_name:
                        cb1.set_ticks([0,250,500,750,1000])
                        strip_cbar_ax.invert_yaxis()
        
                else:
        
                    strip_cbar_ax.legend(handles=handle_list,
                            title=cfig_dict['cbar_label'],
                                        fontsize=5,
                                        title_fontproperties={ 'size': 6},
                            loc='center')
                    strip_cbar_ax.axis('off')
                

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


        proj_img_ax.imshow(img_arr,aspect='auto')
        proj_img_ax.spines['top'].set_visible(False)
        proj_img_ax.axhline(n_ipsi_cols-0.5,c='lightgrey',linestyle='--',lw=0.75)

        # Major ticks
        modified_proj_cols = []
        for c in relevant_proj_cols: 
            if "fiber" in c:
                modified_proj_cols.append(c)
            else:
                modified_proj_cols.append(c.replace("ipsi_","").replace("contra_",""))

        proj_img_ax.set_yticks(np.arange(0, len(relevant_proj_cols), 1),)
        proj_img_ax.set_yticklabels(modified_proj_cols,rotation=360, fontsize=yfontsize)#, ha='center')
        
    #     proj_img_ax.tick_params(axis='y', which='both', pad=500)  # Add padding (in points) between ticks and labels
        # proj_img_ax.yaxis.tick_right()  # Move y-axis ticks to the right
    #     proj_img_ax.yaxis.set_label_position("right")  # Move y-axis label to the right

        proj_img_ax.set_xticks([])
        
        # for label in proj_img_ax.get_yticklabels():
        #     label.set_x(1.1)  # Move labels further out (1.0 is the default)
        

    #     proj_img_ax.text(-25,ipsi_metalabel_idx, "Ipsilateral",rotation=90, horizontalalignment='center',
    #                     verticalalignment='center',fontsize=6)
    #     proj_img_ax.text(-25,contra_metalabel_idx, "Contralateral",rotation=90, horizontalalignment='center',
    #                     verticalalignment='center',fontsize=6)

        proj_img_ax.spines['right'].set_visible(False)
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
        xs = np.arange(0, n_cells, 1)
        bar_plt = histogram_ax.bar(x=xs, height=hist_vals,width=0.001)

        # histogram_ax.set_xlim((len(hist_vals), 0))
        histogram_ax.set_xticks([])
        histogram_ax.invert_yaxis()
    #     histogram_ax.set_yticks([5,10])
        histogram_ax.set_yticklabels(histogram_ax.get_yticklabels(), fontsize=hist_yfontsize)
        
        
        for axis in ['top','bottom','left','right']:
            histogram_ax.spines[axis].set_linewidth(0.2)
        # histogram_ax.xaxis.set_ticks_position("top")
        histogram_ax.set_xlim(proj_img_ax.get_xlim())

        # Separating Lines
        vert_line_idx=-0.5
        prev_idx = 0
        soma_depth_img_ylabels = ['']*n_cells

        vert_line_w = 0.225
        for pmet_lbl in sorted_pmets:


            pmet_df =  sorted_df[sorted_df[pmet_met_col]==pmet_lbl]
            # pmet_df = pmet_df.sort_values(by=adder_cols)

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

        # if not DROP_MISMAPS:
        #     mismap_inds = np.where(sorted_df['mismap_status'].values==True)[0]
        #     for i in mismap_inds:
        #         mismap_sp = sorted_df['swc_path'].values[i]
        #         left_right_call = left_right_records[mismap_sp]
        #         pusher = 0.5 if left_right_call == "left" else -0.5
        #         proj_img_ax.axvline(i+pusher,c='lightgrey',linestyle='--',lw=0.75)


        for txt in mark_up_axes.get_xticklabels():
            txt_val =txt.get_text()
            if txt_val != "":
                met = txt_val.split("(n")[0].strip()
                clr = data_color_dict[met]
                txt.set_color(clr)

        for del_ax in strip_axes[1:]:
            del_ax.set_xticks([])

        
        y_adj_axes = strip_axes+[histogram_ax,proj_img_ax ]
        for adj_ax in y_adj_axes:

            adj_ax.yaxis.set_tick_params(width=0.25)
            adj_ax.yaxis.set_tick_params(width=0.25)
            adj_ax.tick_params(axis='y', which='major', pad=0.5, length=2)
            adj_ax.spines['left'].set_linewidth(0.5)
            
        histogram_ax.spines['bottom'].set_visible(False)
        histogram_ax.spines['right'].set_visible(False)
        print(output_file)
        plotted_structures[output_file] = relevant_proj_cols
        
        fig.savefig(output_file,bbox_inches='tight',dpi=300)
        # plt.show()
        plt.clf()
        plt.close()
        del fig
        
        
        
    #
    #
    #
    #  Non - IT Full Proj Mat
    # 
    #
    #
    #
    visualization_df = fmost_metadata.merge(proj_df,left_index=True,right_index=True)
    visualization_df['cre_line']=visualization_df['cre_line'].apply(lambda x:x.split("-")[0])
    visualization_df['cre_line']=visualization_df['cre_line'].apply(lambda x:x.split(";")[0])
    # visualization_df = visualization_df[visualization_df['predicted_met_type'].str.contains("IT")]

    visualization_df['altitude_norm'] = (visualization_df.altitude - alt_min)/ (alt_max - alt_min)
    visualization_df['azimuth_norm'] = (visualization_df.azimuth - az_min)/ (az_max - az_min)


    non_it_met_types = set([m for m in visualization_df.predicted_met_type if "IT" not in m])

    azimuth_cmap = plt.cm.magma #pink_cmap #warm_neutrals_cmap#plt.cm.YlOrBr   
    azimuth_alpha = 1 # 0.6
    altitude_cmap = plt.cm.viridis # warm_neutrals_cmap#plt.cm.OrRd 
    altitude_alpha = 1 #plt.cm.viridis #1 # 0.75



    cat_dicts = [

                    {
                        "column":"ccf_soma_location_nolayer",
                        "type":"categorical",
                        "palette":data_color_dict,
                        "ytick_name":"Soma Location",
                        "cbar":True,
                        "cbar_label":"Soma Location",
                    },
        
                    {
                        "column":"auto_projection_subclass",
                        "type":"categorical",
                        "palette":subclass_cdict,
                        "ytick_name":"Proj. Subclass",
                        "cbar":True,
                        "cbar_label":"Subclass",
                    },
        
                    {
                        "column":"dend_derived_predicted_subclass",
                        "type":"categorical",
                        "palette":subclass_cdict,
                        "ytick_name":"Dend. Subclass",
                        "cbar":False,
                        "cbar_label":None,
                    },
        
                    {
                        "column":"local_axon_derived_subclass",
                        "type":"categorical",
                        "palette":subclass_cdict,
                        "ytick_name":"Local Axon Subclass",
                        "cbar":False,
                        "cbar_label":None,
                    },

        
    ]

    configs = [
        
        {
            "data_df":visualization_df,
            "output_file":"./plot_SuppFig_ProjMat_NonIT_Azimuth.pdf",
            "heatmap_column":"azimuth_norm",
            "heatmap_tick_name":"Azimuth",
            "cbar_label":"Azimuth",
            "heatmap_cmap":azimuth_cmap,
            "heatmap_alpha":azimuth_alpha,
            "cat_dicts":cat_dicts,
            # "categorical_adder_col":"ccf_soma_location_nolayer",
            # "categorical_adder_col_tick_name":"Structure",
            "met_type_list":non_it_met_types,
            "intensity_proj_mat_bool":False,
            "DROP_MISMAPS": True,
            "legend_center_locations_frac": [0.1, 0.27, 0.4],
            "legend_radii": [2,2,8],
            "sort_order":['ccf_soma_location_nolayer', "azimuth_norm"],
            "categorical_sort_order": ["VISp"] +ascending_azimuth_regions,

        },
        
        # {
        #     "data_df":visualization_df,
        #     "output_file":"./SuppFig_ProjMat_NonIT_Altitude.pdf",
        #     "heatmap_column":"altitude_norm",
        #     "heatmap_tick_name":"Altitude",
        #     "cbar_label":"Altitude",
        #     "heatmap_cmap":altitude_cmap,
        #     "heatmap_alpha":altitude_alpha,
        #     "cat_dicts":cat_dicts,
        #     # "categorical_adder_col":"ccf_soma_location_nolayer",
        #     # "categorical_adder_col_tick_name":"Structure",
        #     "met_type_list":non_it_met_types,
        #     "intensity_proj_mat_bool":False,
        #     "DROP_MISMAPS": True,
        #     "legend_center_locations_frac": [0.1, 0.27, 0.4],
        #     "legend_radii": [2,2,8],
        #     "sort_order":['ccf_soma_location_nolayer', "altitude_norm"],
        #     "categorical_sort_order": ["VISp"] +ascending_altitude_regions,

        # }
        
    ]

    plotted_structures = {}
    yfontsize = 3
    hist_yfontsize = 3
    for config_dict in configs:
        
        data_df = config_dict['data_df']
        output_file = config_dict['output_file']
        heatmap_column = config_dict['heatmap_column']
        heatmap_tick_name = config_dict['heatmap_tick_name']
        cbar_label = config_dict['cbar_label']
        cat_dicts = config_dict['cat_dicts']
        # categorical_adder_col = config_dict['categorical_adder_col']
        # categorical_adder_col_tick_name = config_dict['categorical_adder_col_tick_name']
        met_type_list = config_dict['met_type_list']
        intensity_proj_mat_bool = config_dict['intensity_proj_mat_bool']
        DROP_MISMAPS = config_dict['DROP_MISMAPS']
        legend_center_locations_frac = config_dict['legend_center_locations_frac']
        legend_radii = config_dict['legend_radii']
        sort_order = config_dict['sort_order']
        heatmap_cmap = config_dict['heatmap_cmap']
        heatmap_alpha = config_dict['heatmap_alpha']
        categorical_sort_order = config_dict['categorical_sort_order']
        
        categorical_adder_col = 'ccf_soma_location_nolayer'
        
        assert all([c in categorical_sort_order for c in data_df[categorical_adder_col].unique()])
        
        data_df = data_df[data_df['predicted_met_type'].isin(met_type_list)]
        
        # adder_cols = [ config_dict[c] for c in sort_order ]
        # adder_cols = [c for c in adder_cols if c is not None]
        
        if intensity_proj_mat_bool:
            proj_cmap = cm.viridis
            axhline_color = "white"
        else:
            axhline_color="lightgrey"
            
        pmet_met_col = 'predicted_met_type'
        sorted_mets = sorted(met_type_list,key = lambda x: ([sc in x for sc in ['IT',"ET","NP",'CT','L6b']].index(True),x) )

        sorted_pmets = [s for s in sorted_mets if s in data_df.predicted_met_type.unique()]

        assert all([p in sorted_pmets for p in sorted(data_df[pmet_met_col].unique())])

        
        
        projection_matrix_meta_order = ["IT","ET","NP","CT","L6b"]
        sorted_df = pd.DataFrame()
        for pmet in sorted_pmets:
            pmet_df =  data_df[data_df[pmet_met_col]==pmet]
            for cat in categorical_sort_order:
                this_cat = pmet_df[pmet_df[categorical_adder_col]==cat]
                this_cat = this_cat.sort_values(by=heatmap_column)
                sorted_df = sorted_df.append(this_cat)
                
            # pmet_df = pmet_df.sort_values(by=adder_cols)
            # sorted_df = sorted_df.append(pmet_df)


        # else:
        #     print("TODO, without dropping mismaps (i think we threw these at either the front or the back of their met type depending what made more sense based on projection>?)")

        if 'altit' in heatmap_column: 
            these_sorted_proj_cols = all_altitude_sorted_proj_cols
        elif  'azimuth' in heatmap_column:
            these_sorted_proj_cols = all_azimuth_sorted_proj_cols
        else:
            these_sorted_proj_cols = all_sorted_proj_cols
            del these_sorted_proj_cols

        relevant_proj_cols = [c for c in these_sorted_proj_cols if sorted_df[c].max() != 0]
        relevant_proj_cols = [c for c in relevant_proj_cols if "fiber tracts" not in c]

        droppers = []# [c for c in relevant_proj_cols if sorted_df[c].astype(bool).sum()<3]
        resurected_cols = []
        for c in droppers:
            ct_dict = sorted_df.groupby('predicted_met_type')[c].apply(lambda x: x.astype(bool).sum()).to_dict()
            if [k for k,v in ct_dict.items() if v==2]:
                resurected_cols.append(c)
        [droppers.remove(r) for r in resurected_cols]
        relevant_proj_cols = [c for c in relevant_proj_cols if c not in droppers]
        
        n_ipsi_cols = len([p for p in relevant_proj_cols if "ipsi" in p])
        n_contra_cols = len([p for p in relevant_proj_cols if "contra" in p])

        ipsi_metalabel_idx = int(n_ipsi_cols/2)
        contra_metalabel_idx = n_ipsi_cols+int(n_contra_cols/2)
        
        

        n_cells = len(sorted_df)

        
        fig=plt.gcf()
        # fig_height = min([ inch_per_sp*n_cells, 7])
        # print("fig height",fig_height)
        fig.set_size_inches(6.69291, 6.69)

        
        added_ax_for_hist = 3
        categorical_added_ax_for_cbar = 20 if categorical_adder_col is not None else 0
        continuous_added_ax_for_cbar = 10
        max_add_cbar = max([categorical_added_ax_for_cbar,continuous_added_ax_for_cbar ])
        spacer_lateral = 20
        spacer_horiz = 0

        num_adder_rows = 0
        if heatmap_column is not None:
            num_adder_rows+=1
        num_adder_rows+=len(cat_dicts)


        total_num_cols = n_cells + spacer_lateral + max_add_cbar 
        total_num_rows = num_adder_rows + len(relevant_proj_cols) + spacer_horiz + added_ax_for_hist + spacer_horiz
    #     total_num_rows = len(adder_cols) + len(relevant_proj_cols)  + added_ax_for_hist + spacer_horiz

        inches_per_column = 15/total_num_cols
        inches_per_row = 21/total_num_rows

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
                "cbar":True,
                "ytick_name":heatmap_tick_name,
                "cbar_label":cbar_label, #"Soma Depth (µm)"

            }
        ]
        strip_axes = []
        if cat_dicts is not None:
            cfig = cat_dicts+cfig

        imgs=[]
        strip_ax_with_cbar_ct = -1
        for strip_ax_ct, cfig_dict in enumerate(cfig):


            column_name = cfig_dict['column']
            strip_ax = fig.add_subplot(grid[strip_ax_ct,:n_cells])
            strip_axes.append(strip_ax)

            strip_values = sorted_df[column_name].values.reshape(1,len(sorted_df))

            strip_cbar_center = None
            strip_cbar_rad = None
            if cfig_dict['cbar']:
                strip_ax_with_cbar_ct+=1
                strip_cbar_center = legend_center_locations[strip_ax_with_cbar_ct]
                strip_cbar_rad = legend_radii[strip_ax_with_cbar_ct]

            cbar_ax_width = categorical_added_ax_for_cbar
            if cfig_dict['type']=='continuous':
                cbar_ax_width = continuous_added_ax_for_cbar

            if cfig_dict['cbar']==True:
                    
                strip_cbar_ax = fig.add_subplot(grid[strip_cbar_center-strip_cbar_rad:strip_cbar_center+strip_cbar_rad,
                                            -max_add_cbar: min([-1, -max_add_cbar+cbar_ax_width]) ])
            else:
                strip_cbar_ax = None
            
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
                print(column_name, unique_vals)
                if len(unique_vals)>len(my_cmap):
                    print("WARNING NOT ENOUGH COLORS PROVIDED FOR NUMBER OF UNIQUE LABELS")

                color_dict = {}
                handle_list = []
                for cte,uni in enumerate(unique_vals):
                    # if uni not in my_cmap:
                    #     print
                    #     color_dict[uni] = cm.tab10(cte)
                    # else:
                    color_dict[uni]=my_cmap[uni]

                    ptch = mpatches.Patch(color = color_dict[uni], label=uni)
                    handle_list.append(ptch)
                image_strip = np.array([color_dict[d] for d in strip_values[0]])  

            image_strip = image_strip.reshape((1,n_cells,4))
            imgs.append(image_strip)
            img = strip_ax.imshow(image_strip,aspect='auto',alpha=img_alpha)
            strip_ax.spines['bottom'].set_visible(False)
            strip_ax.spines['right'].set_visible(False)

            strip_ax.set_yticks([0])
            strip_ax.set_yticklabels([cfig_dict['ytick_name']],rotation=360,fontsize=yfontsize)#,ha='center')
            # strip_ax.yaxis.tick_right()  # Move ticks to the right
            # strip_ax.yaxis.set_label_position("right")  # Ensure the labels are on the right
            # Adjust tick labels manually to add padding
            # for label in strip_ax.get_yticklabels():
            #     label.set_x(1.1)  # Move labels further out (1.0 is the default)


            if strip_cbar_ax is not None:
                    
                if cfig_dict['type']=='continuous':
                    if norm is not None:
                        cb1 = mpl.colorbar.ColorbarBase(strip_cbar_ax, cmap=my_cmap,alpha=img_alpha,
                                                    norm=norm,
                                                        
                                                    orientation='vertical')
                    else:
                        cb1 = mpl.colorbar.ColorbarBase(strip_cbar_ax, cmap=my_cmap,alpha=img_alpha,
                                        orientation='vertical')
        
                    cb1.set_label(cfig_dict['cbar_label'], fontsize=5)
                    cb1.ax.tick_params(labelsize=5) 
        #             cb1.set_yticklabels(cb1.get_yticklabels(), fontsize=5)
                    
                    if "soma_distance_from_pia" == column_name:
                        cb1.set_ticks([0,250,500,750,1000])
                        strip_cbar_ax.invert_yaxis()
        
                else:
        
                    strip_cbar_ax.legend(handles=handle_list,
                            title=cfig_dict['cbar_label'],
                                        fontsize=5,
                                        title_fontproperties={ 'size': 6},
                            loc='center')
                    strip_cbar_ax.axis('off')
                

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


        proj_img_ax.imshow(img_arr,aspect='auto')
        proj_img_ax.spines['top'].set_visible(False)
        proj_img_ax.axhline(n_ipsi_cols-0.5,c='lightgrey',linestyle='--',lw=0.75)

        # Major ticks
        modified_proj_cols = []
        for c in relevant_proj_cols: 
            if "fiber" in c:
                modified_proj_cols.append(c)
            else:
                modified_proj_cols.append(c.replace("ipsi_","").replace("contra_",""))

        proj_img_ax.set_yticks(np.arange(0, len(relevant_proj_cols), 1),)
        proj_img_ax.set_yticklabels(modified_proj_cols,rotation=360, fontsize=yfontsize)#, ha='center')
        
    #     proj_img_ax.tick_params(axis='y', which='both', pad=500)  # Add padding (in points) between ticks and labels
        # proj_img_ax.yaxis.tick_right()  # Move y-axis ticks to the right
    #     proj_img_ax.yaxis.set_label_position("right")  # Move y-axis label to the right

        proj_img_ax.set_xticks([])
        
        # for label in proj_img_ax.get_yticklabels():
        #     label.set_x(1.1)  # Move labels further out (1.0 is the default)
        

    #     proj_img_ax.text(-25,ipsi_metalabel_idx, "Ipsilateral",rotation=90, horizontalalignment='center',
    #                     verticalalignment='center',fontsize=6)
    #     proj_img_ax.text(-25,contra_metalabel_idx, "Contralateral",rotation=90, horizontalalignment='center',
    #                     verticalalignment='center',fontsize=6)

        proj_img_ax.spines['right'].set_visible(False)
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
        xs = np.arange(0, n_cells, 1)
        bar_plt = histogram_ax.bar(x=xs, height=hist_vals,width=0.001)

        # histogram_ax.set_xlim((len(hist_vals), 0))
        histogram_ax.set_xticks([])
        histogram_ax.invert_yaxis()
    #     histogram_ax.set_yticks([5,10])
        histogram_ax.set_yticklabels(histogram_ax.get_yticklabels(), fontsize=hist_yfontsize)
        
        
        for axis in ['top','bottom','left','right']:
            histogram_ax.spines[axis].set_linewidth(0.2)
        # histogram_ax.xaxis.set_ticks_position("top")
        histogram_ax.set_xlim(proj_img_ax.get_xlim())

        # Separating Lines
        vert_line_idx=-0.5
        prev_idx = 0
        soma_depth_img_ylabels = ['']*n_cells

        vert_line_w = 0.225
        for pmet_lbl in sorted_pmets:


            pmet_df =  sorted_df[sorted_df[pmet_met_col]==pmet_lbl]
            # pmet_df = pmet_df.sort_values(by=adder_cols)

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



        mark_up_axes = strip_axes[0] # will decorate the top most one
        mark_up_axes.set_xticks(np.arange(0, n_cells, 1))
        mark_up_axes.tick_params(bottom=False,top=False)#axis='x', colors='white')
        mark_up_axes.set_xticklabels(soma_depth_img_ylabels,
                                    rotation=20, 
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

        # if not DROP_MISMAPS:
        #     mismap_inds = np.where(sorted_df['mismap_status'].values==True)[0]
        #     for i in mismap_inds:
        #         mismap_sp = sorted_df['swc_path'].values[i]
        #         left_right_call = left_right_records[mismap_sp]
        #         pusher = 0.5 if left_right_call == "left" else -0.5
        #         proj_img_ax.axvline(i+pusher,c='lightgrey',linestyle='--',lw=0.75)


        for txt in mark_up_axes.get_xticklabels():
            txt_val =txt.get_text()
            if txt_val != "":
                met = txt_val.split("(n")[0].strip()
                clr = data_color_dict[met]
                txt.set_color(clr)

        for del_ax in strip_axes[1:]:
            del_ax.set_xticks([])
            
        y_adj_axes = strip_axes+[histogram_ax,proj_img_ax ]
        for adj_ax in y_adj_axes:

            adj_ax.yaxis.set_tick_params(width=0.25)
            adj_ax.yaxis.set_tick_params(width=0.25)
            adj_ax.tick_params(axis='y', which='major', pad=0.5, length=2)
            adj_ax.spines['left'].set_linewidth(0.5)
            
        histogram_ax.spines['bottom'].set_visible(False)
        histogram_ax.spines['right'].set_visible(False)
        print(output_file)
        plotted_structures[output_file] = relevant_proj_cols
        
        fig.savefig(output_file,bbox_inches='tight',dpi=300)
        # plt.show()
        plt.clf()
        plt.close()
        del fig
        
   
if __name__=='__main__':
    main()     