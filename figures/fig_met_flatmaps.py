import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import seaborn as sns
import ccf_streamlines.projection as ccfproj
import argschema as ags
from matplotlib.collections import PolyCollection


class FigMetFlatmapsParameters(ags.ArgSchema):
    inf_met_type_file = ags.fields.InputFile(
        description="csv file with inferred met type text labels",
    )
    ccf_flat_coords_file = ags.fields.InputFile(
        description="csv file with ccf flatmap coordinates",
    )
    dist_pval_file = ags.fields.InputFile()
    projected_atlas_file = ags.fields.InputFile()
    atlas_labels_file = ags.fields.InputFile()
    output_file = ags.fields.OutputFile()


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



MET_TYPE_COLORS = {
    'L2/3 IT': '#7AE6AB',
    'L4 IT': '#00979D',
    'L4/L5 IT': '#00DDC5',
    'L5 IT-1': '#00A809',
    'L5 IT-2': '#00FF00',
    'L5 IT-3 Pld5': '#26BF64',
    'L6 IT-1': '#C2E32C',
    'L6 IT-2': '#96E32C',
    'L6 IT-3': '#A19922',
    'L5/L6 IT Car3': '#5100FF',
    'L5 ET-1 Chrna6': '#0000FF',
    'L5 ET-2': '#22737F',
    'L5 ET-3': '#29E043',
    'L5 NP': '#73CA95',
    'L6 CT-1': '#74CAFF',
    'L6 CT-2': '#578EBF',
    'L6b': '#2B7880',
}


def main(args):
    inf_met_type_df = pd.read_csv(args['inf_met_type_file'], index_col=0)
    ccf_flat_coords_df = pd.read_csv(args['ccf_flat_coords_file'], index_col=0)
    pval_df = pd.read_csv(args['dist_pval_file'], index_col=0)
    print("total cells")
    print(ccf_flat_coords_df.shape[0])

    # Some cells belong to a visual area but are in a streamline that ends up in a non-visual area
    # We will suppress plotting those here
    ccf_flat_coords_df = ccf_flat_coords_df.loc[ccf_flat_coords_df['top_in_allowed_region'], :]

    bf_flatmap = ccfproj.BoundaryFinder(
        projected_atlas_file=args["projected_atlas_file"],
        labels_file=args["atlas_labels_file"],
    )

    flatmap_left_boundaries = bf_flatmap.region_boundaries(
        ["VISp", "VISpm", "VISam", "VISa", "VISrl", "VISal", "VISl", "VISli", "VISpl", "VISpor", "RSPagl"],
    )

    fig = plt.figure(figsize=(6.5, 9),)
    gs = gridspec.GridSpec(6, 4, height_ratios=(1.5, 1, 1, 1, 1, 1.2))

    # area names
    label_adj_dict = {
        "VISpm": {
            "dx": 0,
            "dy": -100,
            "rotation": -60,
        },
        "VISam": {
            "dx": 0,
            "dy": 0,
            "rotation": 15,
        },

        "RSPagl": {
            "dx": 450,
            "dy": 0,
            "rotation": 80,
        },
        "VISpl": {
            "dx": -600,
            "dy": -500,
            "rotation": 0,
        },
        "VISli": {
            "dx": 0,
            "dy": 0,
            "rotation": 45,
        },
        "VISl": {
            "dx": 0,
            "dy": 0,
            "rotation": 45,
        },
    }


    ax = plt.subplot(gs[0, :2])
    for k, boundary_coords in flatmap_left_boundaries.items():
        ax.plot(*(boundary_coords.T * 10), c="gray", lw=0.5) # to microns
        x, y = (boundary_coords * 10).mean(axis=0)
        if k not in label_adj_dict:
            ax.text(x, y, k, fontsize=6, color='gray', ha='center', va='center')
        else:
            adj = label_adj_dict[k]
            ax.text(x + adj['dx'], y + adj['dy'], k, rotation=adj['rotation'],
            fontsize=6, color='gray', ha='center', va='center')
    ax.set_aspect('equal')
    ax.invert_yaxis()
    sns.despine(ax=ax, left=True, bottom=True)
    ax.set(xticks=[], yticks=[])
    ax.set_title("visual areas flat map", fontsize=9, va='top')

    # all points
    ax = plt.subplot(gs[0, 2:])
    for k, boundary_coords in flatmap_left_boundaries.items():
        ax.plot(*(boundary_coords.T * 10), c="gray", lw=0.5) # to microns
    ax.scatter(
        ccf_flat_coords_df["x"],
        ccf_flat_coords_df["y"],
        c='k',
        s=4,
        edgecolors="white",
        lw=0.25,
    )
    sns.kdeplot(
        x=ccf_flat_coords_df["x"],
        y=ccf_flat_coords_df["y"],
        fill=False,
        thresh=0.2,
        levels=5,
        lw=0.75,
        color="firebrick",
        ax=ax,
    )
    ax.set_title("all cells", fontsize=9, va='top')
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_aspect('equal')
    ax.invert_yaxis()
    sns.despine(ax=ax, left=True, bottom=True)
    ax.set(xticks=[], yticks=[])


    for i, met_type in enumerate(MET_TYPE_ORDER):
        ax = plt.subplot(gs[(i // 4) + 1, i % 4])
        color = MET_TYPE_COLORS[met_type]
        for k, boundary_coords in flatmap_left_boundaries.items():
            ax.plot(*(boundary_coords.T * 10), c="gray", lw=0.5) # to microns

        specimen_ids = inf_met_type_df.index[inf_met_type_df['inferred_met_type'] == met_type]
        common_ids = ccf_flat_coords_df.index.intersection(specimen_ids)
        print(met_type)
        print(len(common_ids))
        print(ccf_flat_coords_df.loc[common_ids, "structure"].value_counts())
        ax.scatter(
            ccf_flat_coords_df.loc[common_ids, "x"],
            ccf_flat_coords_df.loc[common_ids, "y"],
            c=color,
            s=4,
            edgecolors="white",
            lw=0.25,
        )
        ax.set_title(met_type, fontsize=9, va='top', color=color)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        sns.despine(ax=ax, left=True, bottom=True)
        ax.set(xticks=[], yticks=[])

        pval = pval_df.at[met_type, "pval_adj"]
        if pval < 0.05:
            ax.set_xlabel(r"$p_\mathrm{adj}$ = " + f"{pval:.5f}", fontsize=7)


    # Panel labels
    ax.text(
        gs.get_grid_positions(fig)[2][0],
        gs.get_grid_positions(fig)[1][0],
        "a", transform=fig.transFigure,
        fontsize=14, fontweight="bold", va="baseline", ha="right")
    ax.text(
        gs.get_grid_positions(fig)[2][2],
        gs.get_grid_positions(fig)[1][0],
        "b", transform=fig.transFigure,
        fontsize=14, fontweight="bold", va="baseline", ha="right")
    ax.text(
        gs.get_grid_positions(fig)[2][0],
        gs.get_grid_positions(fig)[1][1],
        "c", transform=fig.transFigure,
        fontsize=14, fontweight="bold", va="baseline", ha="right")

    plt.savefig(args['output_file'], dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=FigMetFlatmapsParameters)
    main(module.args)
