#!/usr/bin/env python
import argschema as ags
import numpy as np
import pandas as pd
import feather
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import seaborn as sns

sns.set(style="white", context="paper", font="Helvetica", rc={"grid.linewidth": 0.5}, font_scale=1.3)
matplotlib.rc('font', family='Helvetica')

class FigUMAPPanelParameters(ags.ArgSchema):
    umap_file = ags.fields.InputFile(
        description="CSV with UMAP coordinates for modality-specific data"
    )
    modality = ags.fields.String(
        description="description of modality being plotted"
    )
    patchseq_anno_file = ags.fields.InputFile(
        description="feather file with Patch-seq transcriptomic annotations"
    )
    facs_anno_file = ags.fields.InputFile(
        description="feather file with reference FACS transcriptomic annotations"
    )
    specimens_file = ags.fields.InputFile(
        description="txt file with Patch-seq specimens to use"
    )
    focus_subclasses = ags.fields.List(
        ags.fields.String,
        default=[],
        description="if not empty -> subclasses to focus on (other subclasses will be gray)",
        cli_as_single_argument=True
    )
    add_subclass_labels = ags.fields.Bool(
       description="whether to add subclass labels at pre-determined locations",
       default=True
    )
    add_title = ags.fields.Bool(
        description="whether to add title above figure",
        default=False,
        allow_none=True
    )
    add_legend =  ags.fields.Bool(
        description="whether to add x and y axis labels in corner of figure",
        default=False,
        allow_none=True
    )
    output_file = ags.fields.OutputFile(
        description="output figure file name"
    )
    

def main(umap_file, modality, patchseq_anno_file, facs_anno_file, specimens_file, focus_subclasses, add_subclass_labels, add_title, add_legend, output_file, **kwargs):

    # CONSTANTS
    TX_LABEL_LOCATIONS = {
        'L2/3\nIT': (0.63, 0.1),
        'L4 IT': (0.24, 0.08),
        'L5 IT': (0.6, 0.48),
        'L6 IT': (0.45, 0.95),
        'L6 IT\nCar3': (0.08, 0.85),
        'L5 ET': (0.60, 0.65),
        'L5 ET\nChrna6': (0.75, 0.9),
        'L6 CT': (0.65, 0.55),
        'L6b': (0.65, 0.2),
        'NP': (0.8, 0.65)
    }
    EPHYS_LABEL_LOCATIONS = {
        'L2/3\nIT': (0.07, 0.38),
        'L4 IT': (0.9, 0.42),
        'L5 IT': (0.68, 0.1),
        'L6b': (0.32, 0.01),
        'L6\nIT': (0.0, 0.05),
        'L6 IT\nCar3': (0.8, 0.52),
        'L5 ET': (0.25, 0.68),
        'L5 ET Chrna6': (0.82, 0.56),
        'L6\nCT': (0.94, 0.32),
        'L6 IT': (0.25, 0.52),
        'NP': (0.78, 0.78)
    }
    EPHYS_LINE_LOCATIONS = {
        'L4 IT': [[[0.82, 0.51], [0.43, 0.35]]],
        'L5 IT': [[[0.68, 0.5], [0.16, 0.23]], [[0.68, 0.72], [0.16, 0.18]]],
        'L6 IT': [[[0.05, 0.08], [0.1, 0.17]],[[0.05, 0.4], [0.1, 0.05]]],
        'L6 IT Car3': [[[0.72, 0.51], [0.55, 0.42]]]
    }
    MORPHO_LABEL_LOCATIONS = {
        'L2/3 IT': (0.59, 0.08),
        'L4 IT': (0.18, 0.5),
        'L5 IT': (0.25, 0.62),
        'L6 IT': (0.62, 0.2),
        'L6 IT\nCar3': (0.1, 0.13),
        'L6b': (0.2, 0.42),
        'L6 CT': (0.9, 0.26),
        'L5\nET': (0.82, 0.80),
        'L5 ET\nChrna6': (0.27, 0.74),
        'NP': (0.86, 0.62)
    }

    MORPHO_LINE_LOCATIONS = {
        'L4 IT': [[[0.27, 0.33], [0.5, 0.5]], [[0.27, 0.33], [0.5, 0.3]]]
    }

    fs=9
    label_fs=14
    title_fs=14
    ms_dict = {
        "transcriptomics_FACS":2,
        "transcriptomics_Patch-seq":4,
        "electrophysiology_Patch-seq":8,
        "morphology_Patch-seq":8
    }
    subclass_label_fs = 13


    # Load FACS transcriptomic annotations
    facs_anno_df = feather.read_dataframe(facs_anno_file)
    facs_anno_df.set_index("sample_id", inplace=True, drop=True)
    # Replace 'PT' with 'ET' to match convention
    facs_anno_df.loc[:,"cluster_label"] = facs_anno_df["cluster_label"].apply(lambda x: x.replace(" PT "," ET ") if x != None else x)
    facs_sample_ids = facs_anno_df.index.tolist()

    # Set up color palettes for t-types and subclasses
    ttype_color_dict = facs_anno_df.set_index(keys="cluster_label")["cluster_color"].to_dict()
    # Clean up some subclass colors (for custom subclass labeling for a few cases)
    custom_color_dict = {}
    custom_color_dict["L2/3 IT"] = ttype_color_dict["L2/3 IT VISp Agmat"]
    custom_color_dict["L4 IT"] = ttype_color_dict["L4 IT VISp Rspo1"]
    custom_color_dict["L5 IT"] = ttype_color_dict["L5 IT VISp Batf3"]
    custom_color_dict["L6 IT"] = ttype_color_dict["L6 IT VISp Penk Col27a1"]
    custom_color_dict["L6 IT Car3"] = ttype_color_dict["L6 IT VISp Car3"]
    custom_color_dict["L5 ET"] = ttype_color_dict["L5 ET VISp Krt80"]
    custom_color_dict["L5 ET Chrna6"] = ttype_color_dict["L5 ET VISp Chrna6"]
    custom_color_dict["L6 CT"] = ttype_color_dict["L6 CT VISp Ctxn3 Brinp3"]
    custom_color_dict["L6b"] = ttype_color_dict["L6b VISp Col8a1 Rxfp1"]
    custom_color_dict["NP"] = ttype_color_dict["L5 NP VISp Trhr Met"]

    # Load Patch-seq transcriptomic annotations
    patchseq_anno_df = pd.read_feather(patchseq_anno_file, columns=["sample_id", "spec_id_label","Tree_call_label","Tree_first_cl_label","subclass_label"])
    patchseq_anno_df.dropna(subset=["spec_id_label","sample_id"], inplace=True)
    patchseq_anno_df["spec_id_label"] = patchseq_anno_df["spec_id_label"].astype(int)
    spec_to_sample_dict = patchseq_anno_df.set_index(keys="spec_id_label")["sample_id"].to_dict()
    patchseq_anno_df.set_index("sample_id", inplace=True, drop=False)
    
    # Replace 'PT' with 'ET'
    patchseq_anno_df.loc[:,"Tree_first_cl_label"] = patchseq_anno_df["Tree_first_cl_label"].apply(lambda x: x.replace(" PT "," ET ") if x != None else x)
    patchseq_anno_df.rename(columns={"Tree_first_cl_label":"cluster_label"}, inplace=True)
    patchseq_anno_df.loc[:,"subclass_label"] = patchseq_anno_df["subclass_label"].apply(lambda x: x.replace(" PT"," ET") if x != None else x)
   
    # Load final Patch-seq specimen list (and covert to sample ids)
    with open(specimens_file, "r") as fn:
        patchseq_specimen_ids = [int(x.strip("\n")) for x in fn.readlines()]
    patchseq_sample_ids = list(map(lambda x:spec_to_sample_dict[x], patchseq_specimen_ids))


    temp_df = pd.read_csv(umap_file, index_col=0)
    if "transcriptomics" in modality:
        print(F"{temp_df.shape[0]} cells in co-embedded transcriptomic UMAP")
        facs_mask = temp_df.index.isin(facs_sample_ids)
        patchseq_mask = temp_df.index.isin(patchseq_sample_ids)
        print(F"({sum(facs_mask)} FACS and {sum(patchseq_mask)} Patch-seq cells)")

        facs_umap_df = temp_df[facs_mask].join(
            facs_anno_df[["cluster_label","subclass_label"]], 
            how="left"
            )
        patchseq_umap_df = temp_df[patchseq_mask].join(
            patchseq_anno_df[["cluster_label","subclass_label","Tree_call_label"]],
            how="left"
            )

        if modality == "transcriptomics_FACS":
            if len(focus_subclasses) == 0:
                umap_df = facs_umap_df
                gray_umap_df = None
                label_locations = TX_LABEL_LOCATIONS
            else:
                umap_df = facs_umap_df[facs_umap_df["subclass_label"].isin(focus_subclasses)].copy()
                gray_umap_df = facs_umap_df[~facs_umap_df["subclass_label"].isin(focus_subclasses)].copy()
                label_locations = {k:v for k,v in TX_LABEL_LOCATIONS.items() if k.replace("\n"," ") in focus_subclasses}
            label_lines = None
        elif modality == "transcriptomics_Patch-seq":
            if len(focus_subclasses) == 0:
                umap_df = patchseq_umap_df
                gray_umap_df = None
                label_locations = TX_LABEL_LOCATIONS
            else:
                umap_df = patchseq_umap_df[patchseq_umap_df["subclass_label"].isin(focus_subclasses)].copy()
                gray_umap_df = patchseq_umap_df[~patchseq_umap_df["subclass_label"].isin(focus_subclasses)].copy()
                label_locations = {k:v for k,v in TX_LABEL_LOCATIONS.items() if k.replace("\n"," ") in focus_subclasses}
            label_lines = None
        else:
            print(F"Invalid modality string: {modality}")

    else:
        umap_df = temp_df.join(
                patchseq_anno_df.set_index(keys="spec_id_label")[["cluster_label", "subclass_label"]],
                how="left"
                )
        if len(focus_subclasses) == 0:
            gray_umap_df = None

            if modality == "electrophysiology_Patch-seq":
                label_locations = EPHYS_LABEL_LOCATIONS
                if add_subclass_labels:
                    label_lines = EPHYS_LINE_LOCATIONS
                else:
                    label_lines = None
            elif modality == "morphology_Patch-seq":
                label_locations = MORPHO_LABEL_LOCATIONS
                if add_subclass_labels:
                    label_lines = MORPHO_LINE_LOCATIONS
                else:
                    label_lines = None
            else:
                print(F"Invalid modality string: {modality}")
        else:
            gray_umap_df = umap_df[~umap_df["subclass_label"].isin(focus_subclasses)].copy()
            umap_df = umap_df[umap_df["subclass_label"].isin(focus_subclasses)].copy()

            if modality == "electrophysiology_Patch-seq":
                label_locations = {k:v for k,v in EPHYS_LABEL_LOCATIONS.items() if k.replace("\n"," ") in focus_subclasses}
                if add_subclass_labels:
                    label_lines = EPHYS_LINE_LOCATIONS
                else:
                    label_lines = None
            elif modality == "morphology_Patch-seq":
                label_locations = {k:v for k,v in MORPHO_LABEL_LOCATIONS.items() if k.replace("\n"," ") in focus_subclasses}
                if add_subclass_labels:
                    label_lines = MORPHO_LINE_LOCATIONS
                else:
                    label_lines = None
            else:
                print(F"Invalid modality string: {modality}")
            

    # Plot UMAP
    fig = plt.figure(figsize=(4, 5))
    g_main = gridspec.GridSpec(2,1, height_ratios=(0.2,1.), hspace=0.2)
    g_top = gridspec.GridSpecFromSubplotSpec(2,1, height_ratios=(1,1), hspace=0.2, subplot_spec=g_main[0])

    ax = plt.subplot(g_main[1])
    if gray_umap_df is not None:
        print(F"\nPlotting {gray_umap_df.shape[0] + umap_df.shape[0]} cells in {modality} UMAP.")
        sns.scatterplot(
            data=gray_umap_df, 
            x="x", y="y", 
            s=ms_dict[modality]-2, 
            linewidth=0, 
            color="lightgray"
            )
    else:
        print(F"\nPlotting {umap_df.shape[0]} cells in {modality} UMAP.")

    print(F"Labeling {umap_df.shape[0]} cells in {modality} UMAP by t-type.\n")
    sns.scatterplot(
        data=umap_df, 
        x="x", y="y",
        hue="cluster_label",
        s=ms_dict[modality],
        linewidth=0, 
        palette=ttype_color_dict, 
        legend=False
        )
    
    ax.set_xlabel("", fontsize=label_fs);
    ax.set_ylabel("", fontsize=label_fs);
    ax.set_yticks([]);
    ax.set_xticks([]);    
    sns.despine(ax=ax, bottom=True, left=True, top=True, right=True);


    if add_subclass_labels:
        # Add subclass labels
        for scl in list(label_locations.keys()):
            if (len(focus_subclasses) == 0) | (scl.replace("\n"," ") in focus_subclasses): 
                ax.text(
                    x=label_locations[scl][0],
                    y=label_locations[scl][1],
                    s=F"{scl}",
                    fontdict={
                        "fontsize":subclass_label_fs, 
                        "color":custom_color_dict[scl.replace("\n"," ")],
                    },
                    horizontalalignment='center',
                    transform=ax.transAxes
                )

    if label_lines != None:
        for scl in list(label_lines.keys()):
            if (len(focus_subclasses) == 0) |  (scl.replace("\n"," ") in focus_subclasses): 
                scl_lines = label_lines[scl]
                for l in scl_lines:
                    fig.add_artist(
                        Line2D(
                            l[0], l[1],
                            color='black',
                            linewidth=1.,
                            transform=ax.transAxes
                        )
                )


    # Add title
    if add_title:
        ax.text(
            x=0.5,y=1.1, 
            s=F"{modality.split('_')[0].capitalize()}",
            ha="center",
            fontsize=title_fs,
            transform=ax.transAxes
        )
        ax.text(
            x=0.5,y=1.03, 
            s=F"{modality.split('_')[1]} n={umap_df.shape[0]}",
            ha="center",
            fontsize=label_fs-3,
            transform=ax.transAxes
        )

    # Add little corner legend with xy labels
    if add_legend:
        # Y axis arrow
        ax.arrow(
            x=.8, y=0.02, dx=0.0, dy=.07, 
            transform=ax.transAxes, 
            lw=2.2, fc="k", ec="k", 
            head_width=0.018, head_length=0.018
        )
        # X axis arrow
        ax.arrow(
            x=.8, y=0.02, dx=-.06, dy=0, 
            transform=ax.transAxes, 
            lw=2.2, fc="k", ec="k", 
            head_width=0.014, head_length=0.015
        )
        # X axis label
        ax.text(
            .88, -0.05, 
            F"{modality.split('_')[0][0].capitalize()}-UMAP-x", 
            rotation=0,
            horizontalalignment='right',
            verticalalignment='center', 
            transform=ax.transAxes,
            fontsize=label_fs-3
        );
        # Y axis label
        ax.text(
            .935, 0.02, 
            F"{modality.split('_')[0][0].capitalize()}-UMAP-y", 
            rotation=0,
            horizontalalignment='center',
            verticalalignment='bottom', 
            multialignment='left',
            transform=ax.transAxes,
            fontsize=label_fs-3
        );


    plt.savefig(output_file, bbox_inches="tight", dpi=150)


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=FigUMAPPanelParameters)
    main(**module.args)