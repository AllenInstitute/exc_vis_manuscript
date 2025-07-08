import numpy as np
import pandas as pd
import feather
import pathlib
import pprint
import argschema as ags

class UmapSeedingParameters(ags.ArgSchema):
    specimens_file = ags.fields.InputFile(
        description="txt file with specimens to use (optional)",
        required=False,
        allow_none=True,
        default=None)
    anno_file = ags.fields.InputFile(
        description="feather file with transcriptomic annotations",
        allow_none=True,
        default=None)
    initial_positions_umap_file =  ags.fields.InputFile(
        description="CSV of UMAP coordinates to use in determining initial positions",
        allow_none=True,
        default=None)
    merge_on = ags.fields.String(
        description="index to use for merging with initial_positions_umap_file (sample_id or spec_id_label)",
        allow_none=True,
        default="sample_id")
    initial_positions_feature_file = ags.fields.InputFile(
        description="feather or csv file with transcriptomic annotations",
        allow_none=True,
        default=None)
    initial_positions_feature = ags.fields.String(
        description="feature to use for seeding initial positions (ex. Tree_first_cl_label)",
        allow_none=True,
        default=None)
    initial_positions_stat = ags.fields.String(
        description="statistic to measure from initial_positions_feature (ex. mean, median)",
        allow_none=True,
        default=None)
    output_file = ags.fields.OutputFile(description="CSV with UMAP coordinates")

def main(specimens_file, anno_file, initial_positions_umap_file, merge_on, initial_positions_feature_file, initial_positions_feature, initial_positions_stat, output_file, **kwargs):
    
    # Load Patch-seq annotation file to get spec ID to sample ID conversion
    anno_df = feather.read_dataframe(anno_file)
    anno_df["spec_id_label"] = anno_df["spec_id_label"].astype(int)
    
    # Load final Patch-seq specimen list (and convert to sample ids)
    with open(specimens_file, "r") as fn:
        specimen_ids = [int(x.strip("\n")) for x in fn.readlines()]
    anno_cols = ["spec_id_label","sample_id"]
    specimen_df = anno_df[anno_df["spec_id_label"].isin(specimen_ids)][anno_cols]

    print(F"Seeding initial positions based on {initial_positions_feature} {initial_positions_stat} of cells in {initial_positions_umap_file}")
    # Is the label coming from csv with met-types or from the transcriptomics anno file?
    file_extension = pathlib.Path(initial_positions_feature_file).suffix
    if file_extension == ".feather":
        initial_positions_feature_df = feather.read_dataframe(initial_positions_feature_file)
        initial_positions_feature_df.set_index("spec_id_label", inplace=True, drop=False)
        initial_positions_feature_df.index = initial_positions_feature_df.index.astype(int)
    elif file_extension == ".csv":
        initial_positions_feature_df = pd.read_csv(initial_positions_feature_file, index_col=0)
    initial_positions_feature_df = initial_positions_feature_df[initial_positions_feature_df.index.isin(specimen_ids)]

    # Get mean/median/etc position in UMAP for each type
    initial_positions_umap_df = pd.read_csv(initial_positions_umap_file)
    initial_positions_umap_df.rename(columns={"specimen_id":"spec_id_label"}, inplace=True)
    initial_positions_umap_df = initial_positions_umap_df.merge(specimen_df, how="right", on=merge_on)
    initial_positions_umap_df.set_index(keys="spec_id_label", inplace=True)   
    initial_positions_umap_df = initial_positions_umap_df.join(initial_positions_feature_df[initial_positions_feature], how="left")
    agg_df = initial_positions_umap_df.groupby(by=[initial_positions_feature]).agg({
                                                    "x":initial_positions_stat,
                                                    "y":initial_positions_stat})
    agg_df["xy"] = list(zip(agg_df["x"],agg_df["y"]))
    
    # Merge with cells
    seed_df = pd.merge(
        initial_positions_feature_df[initial_positions_feature].to_frame(), 
        agg_df[["x","y"]], 
        how="left", left_on=initial_positions_feature, right_index=True
        )
    fill_values = {initial_positions_feature:"", "x":0, "y":0}
    seed_df_filled = seed_df.fillna(value=fill_values)

    # Rename PT to ET to match convention
    seed_df_filled[initial_positions_feature] = seed_df_filled[initial_positions_feature].apply(lambda x: x.replace("PT","ET"))
    seed_df_filled.to_csv(output_file)

if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=UmapSeedingParameters)
    main(**module.args)