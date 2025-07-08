import os
import h5py
import numpy as np
import pandas as pd
import argschema as ags
from ccf_streamlines.coordinates import coordinates_to_voxels
from ccf_streamlines.projection import _matching_voxel_indices
from scipy.spatial.distance import cdist


class StreamlineTopCoordsParameters(ags.ArgSchema):
    surface_paths_file = ags.fields.InputFile()
    closest_surface_voxel_reference_file = ags.fields.InputFile()
    resolution = ags.fields.Integer(
        default=10
    )
    input_file = ags.fields.InputFile()
    output_file = ags.fields.OutputFile()


def main(args):
    with h5py.File(args["surface_paths_file"], "r") as path_f:
#             paths = path_f["paths"][:]
        volume_shape = tuple(path_f['volume lookup flat'].attrs['original shape'])

    with h5py.File(args["closest_surface_voxel_reference_file"], "r") as f:
        closest_dset = f["closest surface voxel"]
        closest_surface_voxels = closest_dset[:]

    resolution = (args['resolution'], args['resolution'], args['resolution'])
    df = pd.read_csv(args['input_file'], index_col=0).dropna(subset=["ccf_soma_x", "ccf_soma_y", "ccf_soma_z"])
    coords_xyz = df[["ccf_soma_x", "ccf_soma_y", "ccf_soma_z"]].values
    voxels = coordinates_to_voxels(coords_xyz)

    # reflect voxels to left hemisphere
    z_size = volume_shape[2]
    z_midline = z_size / 2
    voxels[voxels[:, 2] > z_midline, 2] = z_size - voxels[voxels[:, 2] > z_midline, 2]

    # Also reflect coordinates
    reflect_coords = coords_xyz.copy()
    reflect_coords[reflect_coords[:, 2] > z_midline * resolution[2], 2] = z_size * resolution[2] - reflect_coords[reflect_coords[:, 2] > z_midline * resolution[2], 2]


    # Find the surface voxels that best match the voxels
    voxel_inds = np.ravel_multi_index(
        tuple(voxels[:, i] for i in range(voxels.shape[1])),
        volume_shape
    )
    matching_surface_voxel_ind = _matching_voxel_indices(
        voxel_inds,
        closest_surface_voxels)

    has_values = matching_surface_voxel_ind != 0

    # Find nearest isocortex voxels for other cells
    if (~has_values).sum() > 0:
        n_missing =  (~has_values).sum()
        print(f"Finding nearest matches for {n_missing} cell{'s' if n_missing > 1 else ''} outside isocortex")
        missing_voxels = voxels[~has_values, :]
        iso_coords = np.unravel_index(closest_surface_voxels[:, 0], volume_shape)
        iso_coords_arr = np.vstack(iso_coords).T
        distances = cdist(missing_voxels, iso_coords_arr)
        near_inds = closest_surface_voxels[distances.argmin(axis=1), 0]
        matching_missing_ind = _matching_voxel_indices(
            near_inds,
            closest_surface_voxels)
        matching_surface_voxel_ind[~has_values] = matching_missing_ind


    # Get 3D coordinates of those surface voxels
    surface_voxel_coords = np.unravel_index(
        matching_surface_voxel_ind, volume_shape)
    surface_voxel_coords = np.array(surface_voxel_coords).T
    surface_voxel_coords_um = surface_voxel_coords * args['resolution']
    # create output file

    out_df = pd.DataFrame({
        "specimen_id": df.index.values,
        "surface_ccf_x": surface_voxel_coords_um[:, 0],
        "surface_ccf_y": surface_voxel_coords_um[:, 1],
        "surface_ccf_z": surface_voxel_coords_um[:, 2],
    })
    out_df.to_csv(args['output_file'], index=False)


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=StreamlineTopCoordsParameters)
    main(module.args)
