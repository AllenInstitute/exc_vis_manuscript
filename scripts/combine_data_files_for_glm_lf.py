import os
import pandas as pd
import argschema as ags
from functools import reduce


class CombineDataFilesForGlmLfParameters(ags.ArgSchema):
    input_surface_file = ags.fields.InputFile()
    input_latent_factor_file = ags.fields.InputFile()
    input_meta_file = ags.fields.InputFile()
    input_proj_file = ags.fields.InputFile()
    output_file = ags.fields.OutputFile()


MET_GROUP_NAMES = (
    "L23-IT",
    "L4-L5-IT",
    "L6-IT",
    "L5-ET",
    "L6-CT",
    "L6b",
)

MET_GROUP_PRED_TYPES = {
    "L23-IT": ("L2/3 IT",),
    "L4-L5-IT": ("L4 IT", "L4/L5 IT", "L5 IT-1", "L5 IT-2", "L5 IT-3 Pld5"),
    "L6-IT": ("L6 IT-1", "L6 IT-2", "L6 IT-3"),
    "L5-ET": ("L5 ET-1 Chrna6", "L5 ET-2", "L5 ET-3"),
    "L6-CT": ("L6 CT-1", "L6 CT-2",),
    "L6b": ("L6b",),
}


def main(args):
    surface_df = pd.read_csv(args['input_surface_file'], index_col=0)
    surface_df.index = [i[:-4] for i in surface_df.index]

    lf_df =  pd.read_csv(args['input_latent_factor_file'], index_col=0)
    lf_df.index = [i[:-4] for i in lf_df.index]

    meta_df = pd.read_csv(args['input_meta_file'], index_col=0)
    meta_df.index = [i[:-4] for i in meta_df.index]
    meta_df['soma_region'] = meta_df['ccf_soma_location_nolayer']
    meta_df = meta_df[['soma_region', "predicted_met_type"]]

    print(meta_df.loc[meta_df["soma_region"] == "VISp", "predicted_met_type"].value_counts())
    print(meta_df.shape)
    meta_df = meta_df.loc[meta_df["soma_region"] == "VISp", :]
    print(meta_df.shape)

    proj_df = pd.read_csv(args['input_proj_file'], index_col=0)
    proj_df.index = [i[:-4] for i in proj_df.index]

    df_list = [meta_df, lf_df, surface_df, proj_df]
    combo_df = reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True), df_list)
    print(combo_df.shape)
    combo_df.to_csv(args["output_file"])


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=CombineDataFilesForGlmLfParameters)
    main(module.args)
