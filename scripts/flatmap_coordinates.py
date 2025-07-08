import os
import json
import nrrd
import lims_utils
import numpy as np
import pandas as pd
import ccf_streamlines.projection as ccfproj
import argschema as ags
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache



class FlatmapCoordinatesParameters(ags.ArgSchema):
    ps_tx_anno_file = ags.fields.InputFile(
        description="Feather file with Patch-seq transcriptomic annotations",
    )
    specimen_id_file = ags.fields.InputFile()
    projection_file = ags.fields.InputFile()
    surface_paths_file = ags.fields.InputFile()
    closest_surface_voxel_reference_file = ags.fields.InputFile()
    streamline_layer_thickness_file = ags.fields.InputFile()
    layer_depths_file = ags.fields.InputFile(
        description="json file with distances from top of layer to pia",
    )
    atlas_file = ags.fields.InputFile()
    output_file = ags.fields.OutputFile()

def coordinates_for_specimens(spec_id_list):
    sql = """
    select specimen_id, x, y, z from cell_soma_locations
    where specimen_id = any(%s)
    """
    results = lims_utils.query(sql, (spec_id_list,))
    coords_df = pd.DataFrame(results, columns=["specimen_id", "x", "y", "z"]).dropna().set_index("specimen_id")
    return coords_df


def structures_for_specimens(spec_id_list):
    sql = """
    select sp.id, str.acronym from specimens sp
    join structures str on str.id = sp.structure_id
    where sp.id = any(%s)
    """
    results = lims_utils.query(sql, (spec_id_list,))
    struct_df = pd.DataFrame(results, columns=["specimen_id", "structure"]).dropna().set_index("specimen_id")
    return struct_df


def get_structures_of_streamline_tops(coords_df, proj, atlas, tree):
    _, _, _, matching_surface_voxel_ind = proj._get_collapsed_voxels_and_surface_voxels(coords_df.values)
    top_struct_ids = atlas.flat[matching_surface_voxel_ind]
    struct_acronyms = [d['acronym'] for d in tree.get_structures_by_id(top_struct_ids)]
    return pd.DataFrame({"top_struct": struct_acronyms}, index=coords_df.index)


def main(args):
    # Load data
    tx_anno_df = pd.read_feather(args['ps_tx_anno_file'])
    tx_anno_df["spec_id_label"] = pd.to_numeric(tx_anno_df["spec_id_label"])
    tx_anno_df.set_index("spec_id_label", inplace=True)

    specimen_ids = np.loadtxt(args['specimen_id_file']).astype(int)

    with open(args['layer_depths_file'], "r") as f:
        layer_tops = json.load(f)
    layer_thicknesses = {
        'Isocortex layer 1': layer_tops['2/3'],
        'Isocortex layer 2/3': layer_tops['4'] - layer_tops['2/3'],
        'Isocortex layer 4': layer_tops['5'] - layer_tops['4'],
        'Isocortex layer 5': layer_tops['6a'] - layer_tops['5'],
        'Isocortex layer 6a': layer_tops['6b'] - layer_tops['6a'],
        'Isocortex layer 6b': layer_tops['wm'] - layer_tops['6b'],
    }

    print(f"Starting with {len(specimen_ids)} cells")
    ephys_tx_pass_ids = specimen_ids[tx_anno_df.loc[specimen_ids, "Tree_call_label"].isin(["Core", "I1", "I2", "I3"])]
    print(f"Dropping {len(specimen_ids) - len(ephys_tx_pass_ids)} PoorQ cells")


    struct_df = structures_for_specimens(ephys_tx_pass_ids.tolist())
    mcc = MouseConnectivityCache(resolution=10)
    structure_tree = mcc.get_structure_tree()

    allowed_regions = ["VIS", "PTLp"]
    print("allowed regions:", allowed_regions)
    allowed_ids = [d['id'] for d in structure_tree.get_structures_by_acronym(allowed_regions)]

    regions_of_cells = struct_df['structure'].unique()
    allowed_map = {}
    region_ids = [d['id'] for d in structure_tree.get_structures_by_acronym(regions_of_cells)]

    for r, r_id in zip(regions_of_cells, region_ids):
        allowed_map[r] = np.any([structure_tree.structure_descends_from(r_id, p_id)
            for p_id in allowed_ids])
    struct_df['in_allowed_region'] = struct_df['structure'].map(allowed_map)
    print("cells outside allowed regions")
    print(struct_df.loc[~struct_df['in_allowed_region'], :])
    in_vis_areas_ids = struct_df.index[struct_df['in_allowed_region']]

    coords_df = coordinates_for_specimens(in_vis_areas_ids.tolist())
    print(coords_df.shape)

    ccf_coord_proj = ccfproj.IsocortexCoordinateProjector(
        args["surface_paths_file"],
        closest_surface_voxel_reference_file=args["closest_surface_voxel_reference_file"],
        layer_thicknesses=layer_thicknesses,
        streamline_layer_thickness_file=args["streamline_layer_thickness_file"],
        projection_file=args["projection_file"],
    )

    ccf_flatmap_coords = ccf_coord_proj.project_coordinates(
        coords_df.values,
        scale="microns",
        thickness_type="normalized_layers",
        hemisphere="left",
    )

    atlas, _ = nrrd.read(args['atlas_file'])
    top_struct_df = get_structures_of_streamline_tops(coords_df, ccf_coord_proj, atlas, structure_tree)

    regions_of_cells = top_struct_df['top_struct'].unique()
    allowed_map = {}
    region_ids = [d['id'] for d in structure_tree.get_structures_by_acronym(regions_of_cells)]
    for r, r_id in zip(regions_of_cells, region_ids):
        allowed_map[r] = np.any([structure_tree.structure_descends_from(r_id, p_id)
            for p_id in allowed_ids])
    top_struct_df['top_in_allowed_region'] = top_struct_df['top_struct'].map(allowed_map)

    ccf_flatmap_coords_df = pd.DataFrame(
        ccf_flatmap_coords,
        columns=["x", "y", "depth"],
        index=coords_df.index,
    )

    ccf_flatmap_coords_df = ccf_flatmap_coords_df.join(struct_df).join(top_struct_df)


    ccf_flatmap_coords_df.to_csv(args['output_file'])


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=FlatmapCoordinatesParameters)
    main(module.args)
