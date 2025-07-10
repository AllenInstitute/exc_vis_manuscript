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


visp_id = ACRONYM_MAP['VISp']

def check_coord_out_of_cortex(coordinate, structure_id, atlas_volume, closest_surface_voxel_file, surface_paths_file,
                                tree, isocortex_perim_coords):
    """
    Check if a given ccf coordinate (microns) is located outside a certain CCF structure

    :param coordinate: array (1x3)
        Location of coordinate to check (microns)
    :param structure_id: int
        ccf structure id to check if the input coordinate is in. E.g. if structure_id=315,
        this script would check if the input coordinate is located in iso-cortex
    :param atlas_volume: 3d atlas array
        Region-annotated CCF atlas volume
    :param closest_surface_voxel_file: str
        Closest surface voxel reference HDF5 file path for angle calculation
    :param surface_paths_file: str
        Surface paths (streamlines) HDF5 file path for slice angle calculation:
    :param tree: structure tree

    :return:
    out_of_cortex : bool
        True if the input coordinate is out of cortex, otherwise is False
    nearest_cortex_coord : array/None
        Nearest iso-cortex coordinate if the input coordinate is out of cortex, otherwise is None

    """
    # get structure of input voxel
    voxel = coordinates_to_voxels(coordinate.reshape(1, 3))[0]
    voxel_struct_id = atlas_volume[voxel[0], voxel[1], voxel[2]]

    # find all descendant structures of input structure_id
    structure_ids = tree.descendant_ids([structure_id])[0]

    nearest_cortex_coord = None
    out_of_cortex = False
    if voxel_struct_id not in structure_ids:

        out_of_cortex = True
        nearest_cortex_coord, nearest_cortex_voxel = find_neaerst_isocortex_structure(voxel,
                                                                                        atlas_volume,
                                                                                        structure_ids,
                                                                                        closest_surface_voxel_file,
                                                                                        surface_paths_file,
                                                                                        isocortex_perim_coords)
        if nearest_cortex_coord is not None:

            nearest_cortex_coord = np.array([nearest_cortex_coord[0], nearest_cortex_coord[1], nearest_cortex_coord[2]])



    return out_of_cortex, nearest_cortex_coord


def find_neaerst_isocortex_structure(out_of_cortex_voxel, atlas_volume, isocortex_ids,
                                        closest_surface_voxel_file, surface_paths_file, perim_coords, atlas_resolution=10.):
    """
    Given an out of cortex voxel, this will find the nearest iso-cortex voxel.

    :param out_of_cortex_voxel: array (1x3)
        voxel location of point that is out of cortex
    :param atlas_volume: 3d atlas array
        Region-annotated CCF atlas volume
    :param isocortex_ids: list
        list of iso-cortex structure ids (ints)
    :param closest_surface_voxel_file: str
        Closest surface voxel reference HDF5 file path for angle calculation
    :param surface_paths_file: str
        Surface paths (streamlines) HDF5 file path for slice angle calculation
    :param atlas_resolution: float, default 10.
        Voxel size of atlas volume (microns)

    :return:
    soma_coords : array
        closest isocortex point (micron)
    new_soma_voxel : array
        closest isocortex point (voxel)
    """
    print("Finding nearest isocortex voxel for the OOC voxel at:")
    print(out_of_cortex_voxel)


    # Find nearest isocortex voxel
    perim_kd_tree = KDTree(perim_coords)
    dists, inds = perim_kd_tree.query(out_of_cortex_voxel.reshape(1, 3), k=1)
    print("Nearest Isocortex Voxel is {} Voxels away".format(dists[0]))
    perim_coord_index = inds[0]

    closest_cortex_voxel = perim_coords[perim_coord_index]
    print("Input voxel: {}".format(out_of_cortex_voxel))
    print("Closest voxel: {}".format(closest_cortex_voxel))


    closest_cortex_coord = atlas_resolution * closest_cortex_voxel
    streamline = find_closest_streamline(closest_cortex_coord, closest_surface_voxel_file, surface_paths_file)
    if len(streamline)==0:

        # b/c going to low reso, we need to bump up y  find_closest_streamline 
        closest_cortex_coord = atlas_resolution * closest_cortex_voxel
        closest_cortex_coord = [closest_cortex_coord[0], closest_cortex_coord[1]-10, closest_cortex_coord[2]]

        streamline = find_closest_streamline(closest_cortex_coord, closest_surface_voxel_file, surface_paths_file)

    if len(streamline)==0:
        return None, None
#         closest_cortex_coord = atlas_resolution * closest_cortex_voxel
#         closest_cortex_coord = [closest_cortex_coord[0]-10, closest_cortex_coord[1]-20, closest_cortex_coord[2]-10]

#         streamline = find_closest_streamline(closest_cortex_coord, closest_surface_voxel_file, surface_paths_file)



    # go one up to prevent this coord from existing directly on the white matter boundary
    closest_cortex_coord = streamline[-2]
    new_soma_voxel = closest_cortex_coord / atlas_resolution

    return closest_cortex_coord, new_soma_voxel.astype(int)


    


def main():
    
    output_vis_boundary_file = "../derived_data/vis_top_projcetion_boundaries.json"
    
    with open("../data/ScriptArgs.json","r") as f:
        args = json.load(f)

    with open(args['color_file'],'r') as f:
        color_dict=json.load(f)
    
    

    meta_data_file = args['fmost_metadata_file']
    closest_surface_voxel_file = os.path.join(args['flatmap_input_dir'], "closest_surface_voxel_lookup.h5")
    surface_paths_file = os.path.join(args['flatmap_input_dir'], "surface_paths_10_v3.h5") 

    fmost_meta_data = pd.read_csv(meta_data_file, index_col=0)
    # proj_file = args['fmost_projection_matrix_roll_up'] 
    # projection_df = pd.read_csv(proj_file,index_col=0)
    # projection_df["amount_contralateral_axon"] = projection_df[[c for c in projection_df.columns if c.startswith("contra")]].sum(axis=1)


    
    annotation = open_ccf_annotation(with_nrrd=True)

    cache = ReferenceSpaceCache(
        manifest=os.path.join("allen_ccf", "manifest.json"),  # downloaded files are stored relative to here
        resolution=10,
        reference_space_key="annotation/ccf_2017"  # use the latest version of the CCF
    )
    rsp = cache.get_reference_space()
    name_map = rsp.structure_tree.get_name_map()
    acronym_map = {v:k for k,v in rsp.structure_tree.get_id_acronym_map().items()}
    inverse_acronym_map = {v:k for k,v in acronym_map.items()}
    tree = rsp.structure_tree

    layer_tops  = load_default_layer_template()
    layer_thicknesses = {
            'Isocortex layer 1': layer_tops['2/3'],
            'Isocortex layer 2/3': layer_tops['4'] - layer_tops['2/3'],
            'Isocortex layer 4': layer_tops['5'] - layer_tops['4'],
            'Isocortex layer 5': layer_tops['6a'] - layer_tops['5'],
            'Isocortex layer 6a': layer_tops['6b'] - layer_tops['6a'],
            'Isocortex layer 6b': layer_tops['wm'] - layer_tops['6b'],
    }

    ccf_coord_proj_top = ccfproj.IsocortexCoordinateProjector(
        projection_file = os.path.join(args['flatmap_input_dir'],"top.h5"),
        surface_paths_file = os.path.join(args['flatmap_input_dir'],"surface_paths_10_v3.h5"),
        closest_surface_voxel_reference_file= os.path.join(args['flatmap_input_dir'],"closest_surface_voxel_lookup.h5"),
        layer_thicknesses=layer_thicknesses,
        streamline_layer_thickness_file= os.path.join(args['flatmap_input_dir'],"cortical_layers_10_v2.h5"),
    )

    bf_boundary_finder_top = ccfproj.BoundaryFinder(
        projected_atlas_file=os.path.join(args['flatmap_input_dir'],"top.nrrd"),
        labels_file=os.path.join(args['flatmap_input_dir'],"labelDescription_ITKSNAPColor.txt"),
    )

    bf_left_boundaries = bf_boundary_finder_top.region_boundaries()
    bf_left_boundaries = {k:v.tolist() for k,v in bf_left_boundaries.items()}
    with open(output_vis_boundary_file,"w") as f:
        json.dump(bf_left_boundaries, f)


    proj_butterfly_slab = ccfproj.Isocortex3dProjector(
        # Similar inputs as the 2d version...
        os.path.join(args['flatmap_input_dir'],"flatmap_butterfly.h5"),
        os.path.join(args['flatmap_input_dir'],"surface_paths_10_v3.h5"),

        hemisphere="both",
        view_space_for_other_hemisphere=False, #'flatmap_butterfly', different from documentation

        # Additional information for thickness calculations
        thickness_type="normalized_layers", # each layer will have the same thickness everwhere
        layer_thicknesses=layer_thicknesses,
        streamline_layer_thickness_file=os.path.join(args['flatmap_input_dir'],"cortical_layers_10_v2.h5"),
    )


    # run dummy data through to get shapes
    atlas_shape = (1320, 800, 1140)
    somas_atlas = np.zeros(atlas_shape)
    somas_atlas[20,20,20]=255
    morph_normalized_layers = proj_butterfly_slab.project_volume(somas_atlas)

    main_max = morph_normalized_layers.max(axis=2).T
    top_max = morph_normalized_layers.max(axis=1).T
    left_max = morph_normalized_layers.max(axis=0)

    # Same plotting code as before...
    main_shape = main_max.shape
    top_shape = top_max.shape
    left_shape = left_max.shape

    ooc_df = fmost_meta_data[~fmost_meta_data['ccf_soma_location'].str.contains("VIS")]

    vis_ids = tree.descendant_ids([669])[0]

    vis_mask = np.isin(annotation, vis_ids)
    struct = ndimage.generate_binary_structure(3, 3)
    eroded_cortex_mask = ndimage.morphology.binary_erosion(vis_mask, structure=struct).astype(int)

    vis_perimeter = np.subtract(vis_mask, eroded_cortex_mask)
    vis_perim_coords = np.array(np.where(vis_perimeter != 0)).T



    all_somas_list = []
    sps = []
    failures = []
    for swc_fn, row in tqdm(fmost_meta_data.iterrows()):

        soma_struct_acry = row.ccf_soma_location_nolayer
        soma_id = ACRONYM_MAP[soma_struct_acry]


        soma_arr = np.array([row.ccf_soma_x, row.ccf_soma_y, row.ccf_soma_z ])
        try:

            out_of_cortex, nearest_cortex_coord = check_coord_out_of_cortex(soma_arr, 
                                                                            669, #VIS structure ID 
                                                                            annotation, 
                                                                            closest_surface_voxel_file, 
                                                                            surface_paths_file, 
                                                                            tree,
                                                                        vis_perim_coords
                                                                        )

            if not out_of_cortex:
                all_somas_list.append([row.ccf_soma_x, row.ccf_soma_y, row.ccf_soma_z])
            else:
                all_somas_list.append([nearest_cortex_coord[0],nearest_cortex_coord[1],nearest_cortex_coord[2]])

            sps.append(swc_fn)
            
        except:
            failures.append(swc_fn)

    def move_soma(soma_z):
        resolution=10
        volume_shape=[1320, 800, 1140]
        z_size = volume_shape[2]*resolution
        z_midline = z_size / 2
        new_soma_z = soma_z
        if soma_z > z_midline:
            new_soma_z = int(z_size - soma_z)
        return new_soma_z

    all_somas_list_unilat = [ [l[0], l[1], move_soma(l[2]) ] for l in all_somas_list]

    all_coords_slab_top = ccf_coord_proj_top.project_coordinates(
            np.array(all_somas_list_unilat),
            thickness_type='normalized_layers',
            drop_voxels_outside_view_streamlines=False,
            )

    res_df_top = pd.DataFrame(all_coords_slab_top,columns=['slab_0','slab_1','slap_2'])
    res_df_top['swc_path']=sps
    
    res_df_top.to_csv(args['ccf_top_view_fmost_soma_locations'])
    
    
if __name__ == "__main__":
    main()
