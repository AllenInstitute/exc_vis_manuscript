import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argschema as ags
import simple_sankey as sankey


MET_TYPE_ORDER = [
	"L2/3 IT",
	"L4 IT",
	"L4/L5 IT",
	"L5 IT-1",
	"L5 IT-2",
	"L5 IT-3 Pld5",
	"L6 IT-1",
	"L6 IT-2",
	"L6 IT-3",
	"L5/L6 IT Car3",
	"L5 ET-1 Chrna6",
	"L5 ET-2",
	"L5 ET-3",
	"L5 NP",
	"L6 CT-1",
	"L6 CT-2",
	"L6b",
]

IT_MET_TYPE_ORDER = [
	"L2/3 IT",
	"L4 IT",
	"L4/L5 IT",
	"L5 IT-1",
	"L5 IT-2",
	"L5 IT-3 Pld5",
	"L6 IT-1",
	"L6 IT-2",
	"L6 IT-3",
	"L5/L6 IT Car3",
]

NON_IT_MET_TYPE_ORDER = [
	"L5 ET-1 Chrna6",
	"L5 ET-2",
	"L5 ET-3",
	"L5 NP",
	"L6 CT-1",
	"L6 CT-2",
	"L6b",
]


class FigTtoMetRiverParameters(ags.ArgSchema):
    ps_tx_anno_file = ags.fields.InputFile(
        description="Feather file with Patch-seq transcriptomic annotations",
    )
    met_type_file = ags.fields.InputFile(
        description="csv file with met type text labels",
    )
    rename_pt_to_et = ags.fields.Boolean(
        default=True,
        description="whether to switch names from L5 PT to L5 ET",
    )
    include_cell_numbers = ags.fields.Boolean(
        default=True,
        description="whether to plot cell numbers for t and met types",
    )
    output_all_file = ags.fields.OutputFile(
        description="output file with all types",
    )
    output_it_file = ags.fields.OutputFile(
        description="output file with IT types",
    )
    output_non_it_file = ags.fields.OutputFile(
        description="output file with types apart from IT",
    )

def make_plot(met_type_df, met_type_order, filename,
        ttype_ids, ttype_colors, fig_width=10, include_cell_numbers=True):
    # river plot with all met-types
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, 0.6))

    text_labels = sankey.sankey(
        met_type_df["met_type"].values,
        met_type_df["ttype"].values,
        rightLabels=sorted(met_type_df["ttype"].unique().tolist(), key=lambda x: ttype_ids[x]),
        leftLabels=met_type_order,
        ax=ax,
        rearrange=True,
        orientation="horizontal",
        fontsize=8,
        rightColor=True,
        colorDict=ttype_colors,
        aspect=2.5,
        returnLabels=True,
    )

    if include_cell_numbers:
        ttype_counts = met_type_df["ttype"].value_counts()
        met_type_counts = met_type_df['met_type'].value_counts()

        for l in text_labels[1]:
            l['text'] = l['text'] + f" ({ttype_counts[l['text']]})"
        for l in text_labels[0]:
            l['text'] = l['text'] + f" ({met_type_counts[l['text']]})"

    for l in text_labels[1]:
        ax.text(
            l['x'],
            l['y'],
            l['text'],
            {'ha': 'center', 'va': 'bottom'},
            fontsize=8,
            rotation=90,
        )
    for l in text_labels[0]:
        ax.text(
            l['x'],
            l['y'],
            l['text'],
            {'ha': 'center', 'va': 'top'},
            fontsize=8,
            rotation=90,
        )

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def main(args):
    # Load annotations
    tx_anno_df = pd.read_feather(args['ps_tx_anno_file'])
    tx_anno_df["spec_id_label"] = pd.to_numeric(tx_anno_df["spec_id_label"])
    tx_anno_df.set_index("spec_id_label", inplace=True)
    tx_anno_df["Tree_first_cl_label"] = [t.replace("PT", "ET") if type(t) is str else t for t in tx_anno_df["Tree_first_cl_label"]]
    tx_anno_df["cluster_label"] = [t.replace("PT", "ET") if type(t) is str else t for t in tx_anno_df["cluster_label"]]

    ttype_colors = dict(zip(tx_anno_df["cluster_label"], tx_anno_df["cluster_color"]))
    ttype_ids = dict(zip(tx_anno_df["cluster_label"], tx_anno_df["cluster_id"]))

    met_type_df = pd.read_csv(args['met_type_file'], index_col=0)
    met_type_df["ttype"] = tx_anno_df.loc[met_type_df.index, "Tree_first_cl_label"].tolist()

    ttype_colors.update({m: 'gray' for m in MET_TYPE_ORDER})

    # river plot with all met-types
    make_plot(
        met_type_df,
        MET_TYPE_ORDER,
        args["output_all_file"],
        ttype_ids,
        ttype_colors,
        include_cell_numbers=args["include_cell_numbers"],
    )

    # river plot with IT met-types
    it_met_type_df = met_type_df.loc[met_type_df["met_type"].isin(IT_MET_TYPE_ORDER), :].copy()
    make_plot(
        it_met_type_df,
        IT_MET_TYPE_ORDER,
        args["output_it_file"],
        ttype_ids,
        ttype_colors,
        include_cell_numbers=args["include_cell_numbers"],
        fig_width=6,
    )

    # river plot with non-IT met-types
    non_it_met_type_df = met_type_df.loc[met_type_df["met_type"].isin(NON_IT_MET_TYPE_ORDER), :].copy()
    make_plot(
        non_it_met_type_df,
        NON_IT_MET_TYPE_ORDER,
        args["output_non_it_file"],
        ttype_ids,
        ttype_colors,
        include_cell_numbers=args["include_cell_numbers"],
        fig_width=4,
    )


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=FigTtoMetRiverParameters)
    main(module.args)

