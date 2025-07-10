#!/usr/bin/env python3

from allensdk.core.reference_space_cache import ReferenceSpaceCache
from ccf_streamlines.angle import coordinates_to_voxels
from ccf_streamlines.angle import find_closest_streamline
from ccf_streamlines.coordinates import coordinates_to_voxels
from matplotlib.patches import Patch
from morph_utils.ccf import coordinates_to_voxels, open_ccf_annotation
from morph_utils.ccf import load_structure_graph,ACRONYM_MAP
from neuron_morphology.swc_io import *
from pathlib import Path
from scipy import ndimage
from scipy.interpolate import griddata
from scipy.spatial import KDTree
from shapely.geometry import Polygon, Point
from skeleton_keys.io import load_default_layer_template
from tqdm import tqdm
import ccf_streamlines.morphology as ccfmorph
import ccf_streamlines.projection as ccfproj
import json
import math
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns


def main():

    with open("../data/ScriptArgs.json","r") as f:
        args = json.load(f)

    with open(args['color_file'],'r') as f:
        color_dict = json.load(f)

    meta_data_file = args['fmost_metadata_file']
    proj_file = args['fmost_projection_matrix_roll_up'] 
    structure_borders_file = args['ccf_top_view_structure_boundaries']
    complete_axon_features = args['fmost_entire_axon_features']
    fmost_top_view_soma_locations = args['ccf_top_view_fmost_soma_locations']
    retnotopy_grid_data = args['vis_top_flatmap_retnotopy_data']
    
    
    fmost_meta_data = pd.read_csv(meta_data_file, index_col=0)
    projection_df = pd.read_csv(proj_file,index_col=0)
    complete_axon_feats = pd.read_csv(complete_axon_features,index_col=0)
    
    
    projection_df["amount_contralateral_axon"] = projection_df[[c for c in projection_df.columns if c.startswith("contra")]].sum(axis=1)
    total_contra_axon_len_per_cell = projection_df.amount_contralateral_axon.to_dict()

    num_targets_per_cell = complete_axon_feats['complete_axon_total_number_of_targets'].to_dict()

    # file was generated already
    res_df_top = pd.read_csv(fmost_top_view_soma_locations)

    # and so was the top view of VIS structure boundaries
    with open(structure_borders_file,'r') as f:
        bf_left_boundaries = json.load(f)

    merged_df = res_df_top.merge(fmost_meta_data,left_on='swc_path',right_index=True)
    merged_df['number_of_projetion_targets'] =merged_df.swc_path.map(num_targets_per_cell)


    it_met_types = merged_df[merged_df['predicted_met_type'].str.contains("IT")]['predicted_met_type'].unique().tolist()
    non_it_met_types = merged_df[~merged_df['predicted_met_type'].str.contains("IT")]['predicted_met_type'].unique().tolist()

    it_met_types = merged_df[merged_df['predicted_met_type'].str.contains("IT")]['predicted_met_type'].unique().tolist()
    non_it_met_types = merged_df[~merged_df['predicted_met_type'].str.contains("IT")]['predicted_met_type'].unique().tolist()


    met_types_lists = {

        "IT":it_met_types,
        "ET_MET_Types": ['L5 ET-1 Chrna6', 'L5 ET-3', 'L5 ET-2'],
        "CT_NP_L6b":['L5 NP', 'L6 CT-1', 'L6 CT-2', 'L6b']

    }

    vis_boundaries = {k:np.array(v) for k,v in bf_left_boundaries.items() if ("VIS" in k and k!="VISC") or ("RSPagl" in k)}


    # load retnotopic atlas data and generate grid data 

    retno_df = pd.read_csv(retnotopy_grid_data,index_col=0)

    # Simulate a tall array of 2D points and intensity values
    np.random.seed(0)
    num_points = 1000
    x = retno_df['top_voxel_x'].values
    y = retno_df['top_voxel_y'].values
    altitude_intensity = retno_df['altitude'].values

    # Create a grid for interpolation
    grid_x, grid_y = np.meshgrid(
        np.linspace(min(x), max(x), 200),  # Adjust the resolution
        np.linspace(min(y), max(y), 200)
    )

    # Interpolate intensity values onto the grid
    altitude_grid_intensity = griddata(
        points=(x, y),             # Original data points
        values=altitude_intensity,          # Associated intensity values
        xi=(grid_x, grid_y),       # Grid points for interpolation
    )


    vis_boundary_polygons = {k:Polygon(v) for k,v in vis_boundaries.items()}
    polygon_points = np.array(list(zip(retno_df['top_voxel_x'], retno_df['top_voxel_y'])))
    polygon = Polygon(polygon_points)  # Create a shapely polygon
    flat_grid_x, flat_grid_y = grid_x.flatten(), grid_y.flatten()
    points = np.vstack((flat_grid_x, flat_grid_y)).T

    mask=[]
    for p in tqdm(points):

        bool_res= any([poly.contains(Point(p)) for poly in vis_boundary_polygons.values()])
        mask.append(bool_res)

    mask_arr = np.array(mask)
    mask_grid = mask_arr.reshape(grid_x.shape)  # Reshape mask to match grid


    # # Apply mask to the altitude grid
    altitude_grid_intensity[~mask_grid] = np.nan  # Mask points outside the polygon


    # Simulate a tall array of 2D points and intensity values
    np.random.seed(0)
    num_points = 1000
    x = retno_df['top_voxel_x'].values
    y = retno_df['top_voxel_y'].values
    azimuth_intensity = retno_df['azimuth'].values

    # Interpolate intensity values onto the grid
    azimuth_grid_intensity = griddata(
        points=(x, y),             # Original data points
        values=azimuth_intensity,          # Associated intensity values
        xi=(grid_x, grid_y),       # Grid points for interpolation
    )
    # Apply mask to the altitude grid
    azimuth_grid_intensity[~mask_grid] = np.nan  # Mask points outside the polygon



    def round_up_to_nearest_5(n):
        return int(math.ceil(n / 5.0)) * 5

    outdir = "./"

    alph=0.9
    dotsize=15
    n_targ_max = merged_df["number_of_projetion_targets"].max()
    n_targ_min = merged_df["number_of_projetion_targets"].min()

    for config_name, list_of_mets in met_types_lists.items():

        for visp_filter_str in ["", "VISpOnly"]:

            this_df = merged_df[merged_df['predicted_met_type'].isin(list_of_mets)]

            adder_str = "plot_ccf_flatmaps_SuppFig"
            if visp_filter_str == 'VISpOnly':
                this_df = this_df[this_df['ccf_soma_location_nolayer']=='VISp']
                adder_str = "plot_ccf_flatmaps_MainFig"

            n_targ_max = this_df["number_of_projetion_targets"].max()
            n_targ_min = this_df["number_of_projetion_targets"].min()

            describe_dict = this_df["number_of_projetion_targets"].describe()

            # ALTITUDE PLOT
            fig, axes = plt.gcf(), plt.gca()
            for k, boundary_coords in vis_boundaries.items():
                axes.plot(*boundary_coords.T, c="black", lw=.25)        


            min_targ=n_targ_min
            min_targ = min([0,min_targ])
            max_targ=n_targ_max
            norm_dot_size = (this_df.number_of_projetion_targets-min_targ)/(max_targ-min_targ)
            norm_dot_size = norm_dot_size*dotsize
            sc = axes.scatter(this_df.slab_0, 
                                   this_df.slab_1, 
                                   s=norm_dot_size,
                                   alpha=alph, 
                                   edgecolors='none', 
                                   c=this_df['predicted_met_type'].map(color_dict), 
                                   zorder=1000
                                  )

            sizes = [round_up_to_nearest_5(n_targ_min), round_up_to_nearest_5(describe_dict['50%']), round_up_to_nearest_5(n_targ_max)]
            if sizes[0] == sizes[1]:
                new_sizes = [sizes[0], sizes[1]+5, sizes[2]]
                sizes = new_sizes
            markers = []
            for v in sizes:
                norm_v = (v-min_targ)/(max_targ-min_targ)
                marker = mlines.Line2D([], [], color='k', marker='.', linestyle='None',
                                      markersize=(dotsize*norm_v*4)**0.5, label=str(v))
                markers.append(marker) 

            # axes.legend(handles=markers,bbox_to_anchor=(1.45,0.5), loc="center left",title='# of Targets')
            legend = axes.legend(
                handles=markers, 
                bbox_to_anchor=(1, 0.5), 
                loc="center left", 
                title='Targets',
                fontsize=6,  # Reduce text size
                handletextpad=0.5,  # Reduce space between marker and text
                borderpad=0.5,  # Reduce padding inside legend box
                labelspacing=0.4,  # Reduce space between legend entries
            )
            legend.get_title().set_fontsize(7)  # Set title font size separately

            axes.invert_yaxis()
            axes.set_aspect('equal')
            axes.axis('off')
            axes.set_title(f'{visp_filter_str} {config_name} (n={this_df.shape[0]})')

            fig.set_size_inches(1,1)
            # ofile = os.path.join(outdir,adder_str + "SomaFlatmap" + config_name+f"{visp_filter_str}.png")
            # fig.savefig(ofile,dpi=300,bbox_inches='tight')

            ofile = os.path.join(outdir, adder_str + "SomaFlatmap" + config_name+f"{visp_filter_str}.pdf")
            fig.savefig(ofile,dpi=300,bbox_inches='tight')

            # plt.show()
            plt.clf()
            plt.close()






    merged_df["total_contra_axon_length"] = merged_df.swc_path.map(total_contra_axon_len_per_cell)
    merged_df["log_total_contra_axon_length"] = merged_df.total_contra_axon_length.map(lambda x: np.log(x+1))


    alph=0.9
    dotsize=50
    axon_max = merged_df["log_total_contra_axon_length"].max()
    axon_min = merged_df["log_total_contra_axon_length"].min()

    contra_dict = {
        "AllCells":['L5 ET-2',
     'L5 ET-3',
     'L5 ET-1 Chrna6',
     'L6 CT-1',
     'L6 CT-2',
     'L5/L6 IT Car3',
     'L6 IT-3',
     'L4/L5 IT',
     'L6 IT-2',
     'L2/3 IT',
     'L5 IT-2',
     'L4 IT',
     'L5 NP',
     'L6b',
     'L6 IT-1']
    }


    for config_name, list_of_mets in contra_dict.items():

        for visp_filter_str in [""]: #,["", "VISpOnly"]:

            this_df = merged_df[merged_df['predicted_met_type'].isin(list_of_mets)]

            if visp_filter_str == 'VISpOnly':
                this_df = this_df[this_df['ccf_soma_location_nolayer']=='VISp']

            # CONTRA ONLY
            this_df = this_df[this_df['total_contra_axon_length']!=0]
            axon_max = this_df["total_contra_axon_length"].max()
            axon_min = this_df["total_contra_axon_length"].min()

            # print(visp_filter_str)
            # print(this_df.log_total_contra_axon_length.describe())
            # print(this_df.shape)
            # print()
            fig, axes = plt.gcf(), plt.gca()
            for k, boundary_coords in vis_boundaries.items():
                axes.plot(*boundary_coords.T, c="black", lw=.25)        


            described_dict = this_df.total_contra_axon_length.describe().to_dict()
            norm_dot_size = (this_df.total_contra_axon_length-axon_min)/(axon_max-axon_min)
            norm_dot_size = norm_dot_size*dotsize
            sc = axes.scatter(this_df.slab_0, 
                                   this_df.slab_1, 
                                   s=norm_dot_size,
                                   alpha=alph, 
                                   edgecolors='none', 
                                   c=this_df['predicted_met_type'].map(color_dict), 
                                   zorder=1000
                                  )
            def mini_proc(n):
                micron = 0.001*n
                if micron>1:
                    return round(micron)
                else:
                    return round(micron,3)

            sizes = [ 500, described_dict['50%'],  50000]
            markers = []
            for v in sizes:
                norm_v = (v-axon_min)/(axon_max-axon_min)
                # print(v, norm_v, mini_proc(v))
                marker = mlines.Line2D([], [], color='k', marker='.', linestyle='None',
                                      markersize=(dotsize*norm_v*4)**0.5, label=str(mini_proc(v)))
                markers.append(marker) 

            # axes.legend(handles=markers,bbox_to_anchor=(1.45,0.5), loc="center left",title='# of Targets')
            legend = axes.legend(
                handles=markers, 
                bbox_to_anchor=(1, 0.5), 
                loc="center left", 
                title='Contra Axon (mm)',
                fontsize=6,  # Reduce text size
                handletextpad=0.5,  # Reduce space between marker and text
                borderpad=0.5,  # Reduce padding inside legend box
                labelspacing=0.6,  # Reduce space between legend entries
            )
            legend.get_title().set_fontsize(7)  # Set title font size separately

            axes.invert_yaxis()
            axes.set_aspect('equal')
            axes.axis('off')

            axes.set_title(f'Contra Only {visp_filter_str} {config_name} (n={this_df.shape[0]})',size=4)


            fig.set_size_inches(1,1)
            # ofile = os.path.join(outdir,config_name+f"_ALTITUDE_{visp_filter_str}.png")
            # ofile = os.path.join(outdir,"plot_ccf_flatmaps_ContraOnly"+config_name+f"{visp_filter_str}.png")
            # fig.savefig(ofile,dpi=300,bbox_inches='tight')

            ofile = os.path.join(outdir,"plot_ccf_flatmaps_ContraOnly"+config_name+f"{visp_filter_str}.pdf")
            # ofile = os.path.join(outdir,config_name+f"_ALTITUDE_{visp_filter_str}.pdf")
            fig.savefig(ofile,dpi=300,bbox_inches='tight')

            # plt.show()
            plt.clf()
            plt.close()






    # azimuth and altitude reference plots
    azimuth_cmap = plt.cm.magma #pink_cmap #warm_neutrals_cmap#plt.cm.YlOrBr   
    azimuth_alpha = 1 # 0.6
    altitude_cmap = plt.cm.viridis # warm_neutrals_cmap#plt.cm.OrRd 
    altitude_alpha = 1 #plt.cm.viridis #1 # 0.75

    def normalize_with_nans(arr):
        valid_mask = ~np.isnan(arr)
        min_val = np.nanmin(arr)
        max_val = np.nanmax(arr)

        normalized = (arr - min_val) / (max_val - min_val)

        return normalized

    # ALTITUDE PLOT
    fig, axes = plt.gcf(), plt.gca()
    for k, boundary_coords in vis_boundaries.items():
        axes.plot(*boundary_coords.T, c="black", lw=.25)        



    sizes = [10,25,50]
    markers = []
    axes.invert_yaxis()
    axes.set_aspect('equal')
    axes.axis('off')

    mesh = plt.pcolormesh(grid_x, grid_y, 
                          altitude_grid_intensity, 
                          shading='gouraud',
                          cmap=altitude_cmap,
                          )

    cbar = plt.colorbar(mesh, ax=axes, label='Altitude')
    for spine in cbar.ax.spines.values():
        spine.set_linewidth(.5)  # Set to your desired width
    cbar.ax.tick_params(width=.5)  #


    plt.grid(False)

    axes.set_title('Altitude')
    fig.set_size_inches(1,1)
    # ofile = os.path.join(outdir,f"plot_ccf_flatmaps_altitutde_ref.png")
    # fig.savefig(ofile,dpi=300,bbox_inches='tight')
    ofile = os.path.join(outdir,f"plot_ccf_flatmaps_altitutde_ref.pdf")
    fig.savefig(ofile,dpi=300,bbox_inches='tight')

    plt.clf()
    plt.close()





    fig, axes = plt.gcf(), plt.gca()
    for k, boundary_coords in vis_boundaries.items():
        axes.plot(*boundary_coords.T, c="black", lw=0.25)        

    axes.invert_yaxis()
    axes.set_aspect('equal')
    axes.axis('off')

    mesh = plt.pcolormesh(grid_x, grid_y, 
                          azimuth_grid_intensity, 
                          shading='gouraud',
                          cmap=azimuth_cmap,
                          )

    #     plt.colorbar(im,fraction=0.046, pad=0.04)
    cbar = plt.colorbar(mesh, ax=axes, label='Azimuth')
    for spine in cbar.ax.spines.values():
        spine.set_linewidth(.5)  
    cbar.ax.tick_params(width=.5)  #

    axes.set_title(f'Azimuth')
    fig.set_size_inches(1,1)
    # ofile = os.path.join(outdir,"plot_ccf_flatmaps_aximuth_ref.png")
    # fig.savefig(ofile,dpi=300,bbox_inches='tight')
    ofile = os.path.join(outdir,f"plot_ccf_flatmaps_aximuth_ref.pdf")
    fig.savefig(ofile,dpi=300,bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    main()
