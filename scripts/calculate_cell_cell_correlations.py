import argschema as ags
import json
import pandas as pd
import numpy as np

class CalcCellCellCorrParameters(ags.ArgSchema):
    me_labels_file = ags.fields.InputFile()
    ephys_data_file = ags.fields.InputFile()
    morph_data_file = ags.fields.InputFile()
    marker_gene_file = ags.fields.InputFile()
    tx_data_file = ags.fields.InputFile()
    tx_anno_file = ags.fields.InputFile()
    ephys_corr_file = ags.fields.OutputFile()
    morph_corr_file = ags.fields.OutputFile()
    genes_corr_file = ags.fields.OutputFile()


def main(args):
    me_types_df = pd.read_csv(args["me_labels_file"], index_col=0)

    ephys_df = pd.read_csv(args["ephys_data_file"], index_col=0)
    morph_df = pd.read_csv(args["morph_data_file"], index_col=0)

    markers_df = pd.read_csv(args["marker_gene_file"], index_col=0)
    markers = markers_df["x"].values

    anno_df = pd.read_feather(args["tx_anno_file"])
    anno_df["spec_id_label"] = pd.to_numeric(anno_df["spec_id_label"])
    met_sample_ids = anno_df.set_index("spec_id_label").loc[
        me_types_df.index, "sample_id"].values

    tx_df = pd.read_feather(
        args["tx_data_file"],
        columns=["sample_id"] + markers.tolist()
    ).set_index("sample_id")

    genes_df = tx_df.loc[met_sample_ids, :].copy()


    # Calculate correlations
    ephys_corr = np.corrcoef(ephys_df.loc[me_types_df.index, :].values)
    morph_corr = np.corrcoef(morph_df.loc[me_types_df.index, :].values)
    genes_corr = np.corrcoef(genes_df.apply(np.log1p).values)

    pd.DataFrame(ephys_corr, index=me_types_df.index, columns=me_types_df.index).to_csv(
        args["ephys_corr_file"])
    pd.DataFrame(morph_corr, index=me_types_df.index, columns=me_types_df.index).to_csv(
        args["morph_corr_file"])
    pd.DataFrame(genes_corr, index=me_types_df.index, columns=me_types_df.index).to_csv(
        args["genes_corr_file"])


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=CalcCellCellCorrParameters)
    main(module.args)
