#!/usr/bin/env python
from mimetypes import init
from random import seed
from matplotlib.cm import ScalarMappable
import numpy as np
import pandas as pd
import mouse_met_figs.utils as utils
import umap
import feather
import argschema as ags

class UmapParameters(ags.ArgSchema):
    features_file = ags.fields.InputFile(
        description="CSV of ephys or morpho features, with cell specimen id in index_col=0")
    min_dist = ags.fields.Float(default=0.1, allow_none=True, description="min_dist parameter for UMAP")
    n_neighbors = ags.fields.Integer(default=15, allow_none=True, description="n_neighbors parameter for UMAP")
    initial_positions_file = ags.fields.InputFile(
        description="csv with UMAP seeds by initial_positions_feature",
        allow_none=True,
        default=None)
    anno_file = ags.fields.InputFile(
        description="feather file with transcriptomic annotations",
        allow_none=True,
        default=None)
    facs_anno_file = ags.fields.InputFile(
        description="feather file with reference FACS transcriptomic annotations")
    specimens_file = ags.fields.InputFile(
        description="txt file with specimens to use (optional)",
        required=False,
        allow_none=True,
        default=None)
    required_tree_calls = ags.fields.List(ags.fields.String,
        default=None,
        allow_none=True,
        description="tree call categories to restrict data (optional)",
        cli_as_single_argument=True)
    required_subclasses = ags.fields.List(ags.fields.String, 
        default=None,
        allow_none=True,
        description="transcriptomic subclasses to restrict data (optional)",
        cli_as_single_argument=True)
    random_state = ags.fields.Integer(
        description="random seed",
        allow_none=True,
        default=None)
    output_file = ags.fields.OutputFile(description="CSV with UMAP coordinates")


def main(features_file, min_dist, n_neighbors, initial_positions_file, anno_file, facs_anno_file, specimens_file, required_tree_calls, required_subclasses, random_state, output_file, **kwargs):
    # Create t-type to subclass dict
    facs_anno_df = feather.read_dataframe(facs_anno_file)
    subclass_dict = utils.dict_from_facs(facs_anno_df, key_column="cluster_label", value_column="subclass_label")
    
    anno_df = feather.read_dataframe(anno_file)
    anno_df.set_index("sample_id", inplace=True, drop=False)
    ps_ids = anno_df[anno_df["collection_label"]!="FACS"].index.tolist()
    sample_id_to_spec_id_dict = anno_df.loc[ps_ids,"spec_id_label"].apply(str).to_dict()

    df = pd.read_csv(features_file, index_col=0)
    print(F"Original shape of data: {df.shape}")
    df.dropna(axis=1, inplace=True)
    df.index = df.index.astype(str)

    # Filter data if necessary with tree call, subclass and specimen list masks
    anno_df.loc[:,"subclass"] = anno_df["Tree_first_cl_label"].apply(lambda x: subclass_dict[x] if x in subclass_dict.keys() else "None")
    if (specimens_file != None) & (required_tree_calls != None) & (required_subclasses != None):
        tree_mask, subclass_mask, specimens_mask = utils.get_tx_masks(anno_df, specimens_file, required_tree_calls, required_subclasses)
        tx_mask = tree_mask & subclass_mask & specimens_mask
        good_t_cells = anno_df.loc[tx_mask, "spec_id_label"].astype(str).tolist()
        print(F"Transcriptomic mask sum: {sum(tx_mask)}")
        df = df[df.index.isin(good_t_cells)]
    print(F"UMAP embedding to be applied to data of shape: {df.shape}")
    
    cell_ids = df.index.astype(int).tolist()
    if initial_positions_file is not None:
        print(F"Seeding initial positions based on {initial_positions_file}")
        seeds_df = pd.read_csv(initial_positions_file, index_col=0)
        cell_seeds = np.array(list(zip(seeds_df.loc[cell_ids,'x'], seeds_df.loc[cell_ids,'y'])))

        embedding = umap.UMAP(random_state=random_state, min_dist=min_dist, 
                             n_neighbors=n_neighbors, init=cell_seeds).fit_transform(df.values)

    else:
        embedding = umap.UMAP(random_state=random_state, min_dist=min_dist, 
        n_neighbors=n_neighbors).fit_transform(df.values)
    
    out_df = pd.DataFrame(
        {
            "x": embedding[:, 0],
            "y": embedding[:, 1]
        },
        index=df.index,
        )
    out_df.to_csv(output_file)

if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=UmapParameters)
    main(**module.args)
