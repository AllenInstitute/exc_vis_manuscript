import os
import argschema as ags
import numpy as np
import pandas as pd


SUBCLASS_INFO = [
    {
        "set_of_types": ["L2/3 IT"],
        "filepart": "L23-IT",
    },
    {
        "set_of_types": ["L4 IT", "L4/L5 IT", "L5 IT-1", "L5 IT-2", "L5 IT-3 Pld5"],
        "filepart": "L4-L5-IT",
    },
    {
        "set_of_types": ["L6 IT-1", "L6 IT-2", "L6 IT-3"],
        "filepart": "L6-IT",
    },
    {
        "set_of_types": ["L5/L6 IT Car3"],
        "filepart": "L5L6-IT-Car3",
    },
    {
        "set_of_types": ["L5 ET-1 Chrna6", "L5 ET-2", "L5 ET-3"],
        "filepart": "L5-ET",
    },
    {
        "set_of_types": ["L5 NP"],
        "filepart": "L5-NP",
    },
    {
        "set_of_types": ["L6 CT-1", "L6 CT-2"],
        "filepart": "L6-CT",
    },
    {
        "set_of_types": ["L6b"],
        "filepart": "L6b",
    },
    {
        "set_of_types": ["L5 ET-1 Chrna6"],
        "filepart": "L5-ET-Chrna6",
    },
    {
        "set_of_types": ["L5 ET-2", "L5 ET-3"],
        "filepart": "L5-ET-non-Chrna6",
    },
]


class TransformPsTxWithRefPcsParameters(ags.ArgSchema):
    ps_tx_anno_file = ags.fields.InputFile(
        description="Feather file with Patch-seq transcriptomic annotations",
    )
    ps_tx_data_file = ags.fields.InputFile(
        description="Feather file with Patch-seq transcriptomic data",
    )
    met_type_file = ags.fields.InputFile(
        description="csv file with met type text labels",
    )
    inf_met_type_file = ags.fields.InputFile(
        description="csv file with inferred met type text labels",
    )
    ref_pc_dir = ags.fields.InputDir(
        description="directory with ref tx pca files",
    )
    output_pc_dir = ags.fields.OutputDir(
        description="directory for transformed patch-seq tx pc files",
    )


def tx_pc_transform(subclass_info, ref_pc_dir, tx_anno_df, tx_data_df,
        inf_met_type_df, output_pc_dir):
    # Load tx PC
    filepart = subclass_info['filepart']
    pc_weights = pd.read_csv(os.path.join(ref_pc_dir, f"{filepart}_tx_pca_weights.csv"), index_col=0)
    pc_centers = pd.read_csv(os.path.join(ref_pc_dir, f"{filepart}_tx_pca_centers.csv"), index_col=0)

    # ID cells to analyze
    specimen_ids = inf_met_type_df.loc[inf_met_type_df["inferred_met_type"].isin(subclass_info['set_of_types']), :].index.values
    sample_ids = tx_anno_df.loc[specimen_ids, "sample_id"].values

    # Transform with tx PC
    gene_data = tx_data_df.loc[sample_ids, pc_weights.index].values
    gene_data = np.log2(gene_data + 1) - pc_centers.loc[pc_weights.index, "x"].values
    pc_ps_transformed = gene_data @ pc_weights.values
    pc_ps_transformed_df = pd.DataFrame(pc_ps_transformed, index=specimen_ids, columns=pc_weights.columns)

    pc_ps_transformed_df.to_csv(os.path.join(output_pc_dir, f"{filepart}_ps_transformed_pcs.csv"))


def main(args):
    # Load data
    tx_anno_df = pd.read_feather(args['ps_tx_anno_file'])
    tx_anno_df["spec_id_label"] = pd.to_numeric(tx_anno_df["spec_id_label"])
    tx_anno_df.set_index("spec_id_label", inplace=True)

    print("loading tx data")
    tx_data_df = pd.read_feather(args['ps_tx_data_file']).set_index("sample_id")

    inf_met_type_df = pd.read_csv(args['inf_met_type_file'], index_col=0)
    for si in SUBCLASS_INFO:
        print(si['filepart'])
        tx_pc_transform(
            si, args['ref_pc_dir'],
            tx_anno_df, tx_data_df,
            inf_met_type_df, args['output_pc_dir'])


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=TransformPsTxWithRefPcsParameters)
    main(module.args)
