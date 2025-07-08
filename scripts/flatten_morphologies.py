import os
import h5py
import argschema as ags
import pandas as pd
import ccf_streamlines.morphology as ccfmorph
from ccf_streamlines.projection import BoundaryFinder, IsocortexCoordinateProjector
from tqdm import tqdm


class FlattenMorphologiesParameters(ags.ArgSchema):
    input_file = ags.fields.InputFile()
    projection_file = ags.fields.InputFile()
    surface_paths_file = ags.fields.InputFile()
    closest_surface_voxel_reference_file = ags.fields.InputFile()
    ccf_morph_dir = ags.fields.InputDir()
    met_type_selection = ags.fields.List(ags.fields.String,
        cli_as_single_argument=True,
        default=[]
    )
    output_file = ags.fields.OutputFile()


def main(args):
    input_df = pd.read_csv(args["input_file"], index_col=0)
    if len(args["met_type_selection"]) > 0:
        input_df = input_df.loc[input_df["predicted_met_type"].isin(args["met_type_selection"])]
    specimen_ids = input_df.index.unique()

    flatmap_proj = IsocortexCoordinateProjector(
        projection_file=args["projection_file"],
        surface_paths_file=args["surface_paths_file"],
        closest_surface_voxel_reference_file=args["closest_surface_voxel_reference_file"],
    )

    h5f = h5py.File(args["output_file"], "w")

    for my_id in tqdm(specimen_ids):
        path = os.path.join(args["ccf_morph_dir"], f"{my_id}.swc")
        morph_df = ccfmorph.load_swc_as_dataframe(path)
        print(my_id)
        print(morph_df.shape)
        coords = flatmap_proj.project_coordinates(
            morph_df[['x', 'y', 'z']].values,
            view_space_for_other_hemisphere="flatmap_butterfly"
        )
        print(coords.shape)
        h5f.create_dataset(my_id, data=coords)
    h5f.close()


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=FlattenMorphologiesParameters)
    main(module.args)
