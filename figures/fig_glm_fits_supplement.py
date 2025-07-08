import os
import h5py
import json
import nrrd
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import argschema as ags
from ccf_streamlines.projection import BoundaryFinder
from skimage.measure import find_contours
from matplotlib.lines import Line2D



class FigGlmFitsSupplementParameters(ags.ArgSchema):
    glm_results_dir = ags.fields.InputDir()
    wnm_data_file = ags.fields.InputFile()
    projected_atlas_file = ags.fields.InputFile()
    atlas_labels_file = ags.fields.InputFile()
    flatmap_file = ags.fields.InputFile()
    annot_file = ags.fields.InputFile()
    output_file = ags.fields.OutputFile()

REGION_CATEGORIES = {
    "higher visual areas": (
        "ipsi_VISpm",
        "ipsi_VISam",
        "ipsi_VISa",
        "ipsi_VISrl",
        "ipsi_VISal",
        "ipsi_VISl",
        "ipsi_VISli",
        "ipsi_VISpl",
        "ipsi_VISpor",
        "contra_VISp",
    ),
    "other isocortex": (
        "ipsi_RSPd",
        "ipsi_RSPagl",
        "ipsi_ACAd",
        "ipsi_MOs",
    ),
    "CP, ZI, PG": (
        "ipsi_CP",
        "ipsi_ZI",
        "ipsi_PG",
    ),
    "thalamus": (
        "ipsi_LGd-co",
        "ipsi_LGd-sh",
        "ipsi_LGd-ip",
        "ipsi_LP",
        "ipsi_POL",
        "ipsi_LD",
        "ipsi_IGL",
        "ipsi_LGv",
        "ipsi_SubG",
        "ipsi_RT",
    ),
    "midbrain": (
        "ipsi_SCig",
        "ipsi_SCsg",
        "ipsi_SCzo",
        "ipsi_SCop",
        "ipsi_SCiw",
        "ipsi_APN",
        "ipsi_MPT",
        "ipsi_NOT",
        "ipsi_OP",
        "ipsi_PPT",
    ),
}

REGION_CATEGORY_LOOKUP = {i: k for k, v in REGION_CATEGORIES.items() for i in v}

REGION_CATEGORY_COLORS = {
    'higher visual areas': "#07858D",
    'other isocortex': "#07858D",
    'CP, ZI, PG': "#666666",
    "thalamus": "#F48F9F",
    "midbrain": "#D088BA",
}


REGION_COLORS = {k: REGION_CATEGORY_COLORS[v] for k, v in REGION_CATEGORY_LOOKUP.items()}
REGION_COLORS["ipsi_CP"] = "#9BD5F4"
REGION_COLORS["ipsi_ZI"] = "#F0493B"
REGION_COLORS["ipsi_PG"] = "#FAAD70"

REGION_CATEGORY_ORDER = (
    'higher visual areas',
    'other isocortex',
    'CP, ZI, PG',
    "thalamus",
    "midbrain",
)

SUBCLASSES_ORDER = [
    "L23-IT",
    "L4-L5-IT",
    "L5-ET",
    "L6-CT",
]

SUBCLASS_NAMES = {
    "L23-IT": "L2/3 IT",
    "L4-L5-IT": "L4 & L5 IT",
    "L5-ET": "L5 ET",
    "L6-CT": "L6 CT",
}

SUBCLASS_MET_TYPES = {
    "L23-IT": ("L2/3 IT",),
    "L4-L5-IT": ("L4 IT", "L4/L5 IT", "L5 IT-1", "L5 IT-2", "L5 IT-3 Pld5"),
    "L5-ET": ("L5 ET-1 Chrna6", "L5 ET-2", "L5 ET-3"),
    "L6-CT": ("L6 CT-1", "L6 CT-2"),
}

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


SUBCORT_STRUCT_IDS = {
 'CP': 672,
 'ZI': 797,
 'PG': 931,
 'LGd-co': 496345668,
 'LGd-sh': 496345664,
 'LGd-ip': 496345672,
 'LP': 218,
 'POL': 1029,
 'LD': 155,
 'IGL': 27,
 'LGv': 178,
 'SubG': 321,
 'RT': 262,
 'SCig': 10,
 'SCsg': 842,
 'SCzo': 834,
 'SCop': 851,
 'SCiw': 17,
 'APN': 215,
 'MPT': 531,
 'NOT': 628,
 'OP': 706,
 'PPT': 1061
}



def main(args):
    aicc_df_list = []
    coef_df_list = []
    loo_df_list = []
    for sc in SUBCLASSES_ORDER:
        df = pd.read_csv(os.path.join(args["glm_results_dir"], f"{sc}_aicc_loglik.csv"),
            keep_default_na=False, na_values=("NA",))
        df["subclass"] = sc
        aicc_df_list.append(df)

        df = pd.read_csv(os.path.join(args["glm_results_dir"], f"{sc}_coef.csv"),
            keep_default_na=False, na_values=("NA",))
        df["subclass"] = sc
        coef_df_list.append(df)

        df = pd.read_csv(os.path.join(args["glm_results_dir"], f"{sc}_loo_results.csv"),
            keep_default_na=False, na_values=("NA",))
        df["subclass"] = sc
        loo_df_list.append(df)

    all_aicc_df = pd.concat(aicc_df_list)
    all_aicc_df = all_aicc_df.pivot(index=["subclass", "region"], columns="model_type", values="AICc")
    all_aicc_df["subclass_name"] = [SUBCLASS_NAMES[sc] for sc in all_aicc_df.reset_index()["subclass"]]

    all_aicc_df["delta_surf_null"] = all_aicc_df["surface"] - all_aicc_df["null"]
    all_aicc_df["delta_lf_null"] = all_aicc_df["lf"] - all_aicc_df["null"]
    all_aicc_df["delta_full_null"] = all_aicc_df["full"] - all_aicc_df["null"]
    all_aicc_df["region_category"] = [REGION_CATEGORY_LOOKUP[r]
        for r in all_aicc_df.index.get_level_values("region")]

    all_loglik_df = pd.concat(aicc_df_list)
    all_loglik_df = all_loglik_df.pivot(index=["subclass", "region"], columns="model_type", values="logLik")
    all_loglik_df["subclass_name"] = [SUBCLASS_NAMES[sc] for sc in all_loglik_df.reset_index()["subclass"]]

    null_loglik = all_loglik_df["null"].values
    model_type_cols = ["null", "surface", "lf", "full"]
    best_models = all_aicc_df[model_type_cols].idxmin(axis=1)
    best_loglik = all_loglik_df[model_type_cols].max(axis=1).values
    pseudo_r2 = 1 - best_loglik / null_loglik
    train_df = pd.DataFrame({
        "pR2": pseudo_r2,
        "best_model_type": best_models,
        "subclass_name": all_loglik_df["subclass_name"].values},
        index=all_loglik_df.index)
    train_df["region_category"] = [REGION_CATEGORY_LOOKUP[r]
        for r in train_df.index.get_level_values("region")]

    all_coef_df = pd.concat(coef_df_list)

    all_loo_df = pd.concat(loo_df_list)
    all_loo_df["pR2"] = 1 - all_loo_df["ll_fit"] / all_loo_df["ll_null"]

    wnm_data_df = pd.read_csv(args["wnm_data_file"], index_col=0)

    # Determine best models and how many regions to plot
    best_models = best_models[best_models != "null"]
    print(best_models)
    print("n models", best_models.shape[0])

    n_col = 5
    n_row = 8

    if n_col * n_row <  best_models.shape[0]:
        print("Need to expand the grid...quitting")
        return

    all_regions = []
    for r_list in REGION_CATEGORIES.values():
        all_regions += list(r_list)
    region_ordering = {r: i for i, r in enumerate(all_regions)}

    # Atlas info

    bf_boundary_finder = BoundaryFinder(
        projected_atlas_file=args["projected_atlas_file"],
        labels_file=args["atlas_labels_file"],
    )
    bounds_ipsi = bf_boundary_finder.region_boundaries()
    bounds_contra = bf_boundary_finder.region_boundaries(
        hemisphere="right_for_both",
        view_space_for_other_hemisphere="flatmap_butterfly")


    # Flatmap coordinates
    h5f = h5py.File(args["flatmap_file"], "r")
    annot, _ = nrrd.read(args["annot_file"])
    annot_hemi = annot[:, :, :1140 // 2]
    flatmap_lookup = h5f['view lookup'][:]
    flatmap_annot_ids = annot.flat[flatmap_lookup[:, 1]]
    visp_children = [593, 821, 721, 778, 33, 305]
    flatmap_mask = np.in1d(flatmap_annot_ids, visp_children)
    flatmap_3d_coords = np.unravel_index(flatmap_lookup[:, 1], h5f.attrs['size'])
    flatmap_3d_coords = np.array(flatmap_3d_coords).T
    flatmap_3d_coords *= 10 # to microns
    z_mid = (annot.shape[2] // 2) * 10
    flatmap_left_mask = flatmap_3d_coords[:, 2] < z_mid
    flatmap_2d_coords = np.unravel_index(flatmap_lookup[:, 0], h5f.attrs['view size'])
    flatmap_2d_coords = np.array(flatmap_2d_coords).T
    h5f.close()

    flatmap_2d_coords = flatmap_2d_coords[flatmap_mask & flatmap_left_mask, :]
    flatmap_3d_coords = flatmap_3d_coords[flatmap_mask & flatmap_left_mask, :]


    ##### PLOTTING ##############
    fig = plt.figure(figsize=(7.5, 10))
    gs = gridspec.GridSpec(
        n_row, n_col,
        wspace=0.4,
        hspace=0.85,
    )
    lf_cols = all_coef_df.columns[all_coef_df.columns.str.startswith("LF")]

    model_plot_cols = {
        "full": 2,
        "lf": 1,
        "surface": 1,
    }
    model_plot_ratios = {
        "full": (1, 3),
        "lf": (1, ),
        "surface": (1, ),
    }
    sc_x_locs = {
        "full": -2.5,
        "surface": -0.8,
        "lf": -0.6,
    }
    sc_count_starts = {
        "L23-IT": 0,
        "L6-CT": 2,
        "L4-L5-IT": 5,
        "L5-ET": 20,
    }

    for sc in SUBCLASSES_ORDER:
        gs_counter = sc_count_starts[sc]
        best_models_sc = best_models[sc]

        sc_data = wnm_data_df.loc[
            wnm_data_df.predicted_met_type.isin(SUBCLASS_MET_TYPES[sc]), :].copy()
        sc_lf_cols = sc_data.columns[sc_data.columns.str.startswith("LF")]
        sc_lf_cols = sc_lf_cols[sc_lf_cols.str.endswith(sc)]
        lf_values = sc_data.loc[:, sc_lf_cols].median().values

        first_of_subclass_flag = True
        sc_regions = best_models_sc.index.tolist()
        sc_regions_ordered = sorted(sc_regions, key=lambda x: region_ordering[x])
        for r in sc_regions_ordered:
            model_type = best_models_sc[r]
            print(sc, r, model_type)

            subplot_spec = gs[gs_counter // n_col, gs_counter % n_col]
            gs_model = gridspec.GridSpecFromSubplotSpec(
                1, model_plot_cols[model_type],
                subplot_spec=subplot_spec,
                width_ratios=model_plot_ratios[model_type],
                wspace=0.4,
            )

            if model_type in ("lf", "full"):
                ax_lf = plt.subplot(gs_model[0])
                lf_coefs = all_coef_df.loc[
                    (all_coef_df.region == r) &
                    (all_coef_df.subclass == sc) &
                    (all_coef_df.model_type == model_type),
                    lf_cols
                ]
                lf_coefs = lf_coefs.dropna(axis=1)
                lf_odds = np.exp(lf_coefs)
                print(lf_odds)

                ax_lf.plot(np.arange(lf_odds.shape[1]), np.squeeze(lf_odds),
                    'o-', c='k', markersize=2)
                ax_lf.set_xticks(np.arange(lf_odds.shape[1]))
                ax_lf.set_xticklabels([f"{i}" for i in np.arange(lf_odds.shape[1]) + 1])
                ax_lf.set_yscale("log")
                ax_lf.set_ylim(1/4, 4)
                ax_lf.set_xlim(-0.25, lf_odds.shape[1] - 0.75)
                ax_lf.axhline(1, linestyle="dotted", color="gray", lw=0.75)
                ax_lf.set_ylabel("odds", fontsize=5, labelpad=3.0)
                ax_lf.set_xlabel("M-LF", fontsize=5, labelpad=3.0)
                ax_lf.tick_params("both", labelsize=5)

                if model_type == "lf":
                    ax_lf.set_aspect(1.5)

                sns.despine(ax=ax_lf)

            if model_type in ("surface", "full"):
                ax_surf = plt.subplot(gs_model[model_plot_cols[model_type] - 1])
                surf_coefs = all_coef_df.loc[
                    (all_coef_df.region == r) &
                    (all_coef_df.model_type == model_type) &
                    (all_coef_df.subclass == sc),
                    ["surface_ccf_x", "surface_ccf_y", "surface_ccf_z"]].values.T
                log_odds_change = np.dot(flatmap_3d_coords[::5, :], surf_coefs) # downsample the surface coordinates

                intercept = all_coef_df.loc[
                    (all_coef_df.region == r) &
                    (all_coef_df.subclass == sc) &
                    (all_coef_df.model_type == model_type),
                    ["(Intercept)"]
                ].values.T
                log_odds = intercept + log_odds_change

                if model_type == "full":
                    lf_coefs = all_coef_df.loc[
                        (all_coef_df.subclass == sc) &
                        (all_coef_df.region == r) &
                        (all_coef_df.model_type == model_type),
                        lf_cols
                        ]
                    lf_coefs = lf_coefs.dropna(axis=1)
                    log_odds_change = np.dot(np.squeeze(lf_coefs.values.T), lf_values)
                    log_odds += log_odds_change
                est_pred_prob = np.exp(log_odds) / (1 + np.exp(log_odds))
                ax_surf.scatter(
                    flatmap_2d_coords[::5, 0],
                    flatmap_2d_coords[::5, 1],
                    s=1,
                    edgecolors="none",
                    c=est_pred_prob,
                    cmap="PuBu",
                    vmin=0, vmax=1,
                    zorder=5,
                    rasterized=True,
                )
                ax_surf.set_aspect("equal")
                ax_surf.set(xticks=[], yticks=[])
                ax_surf.invert_yaxis()
                sns.despine(ax=ax_surf, left=True, bottom=True)

            if first_of_subclass_flag:
                if model_type in ("lf", "full"):
                    ax = ax_lf
                else:
                    ax = ax_surf
                ax.text(sc_x_locs[model_type], 1.15, SUBCLASS_NAMES[sc], fontweight="bold", fontsize=9,
                    ha="left", transform=ax.transAxes)
                first_of_subclass_flag = False

            if r.startswith("ipsi_"):
                r_text = r.split("_")[1]
            elif r.startswith("contra_"):
                r_text = "contra " + r.split("_")[1]
            if model_type in ("lf", "surface"):
                if model_type == "lf":
                    ax = ax_lf
                elif model_type == "surface":
                    ax = ax_surf
                ax.set_title(f"to {r_text}", fontsize=7, color=REGION_COLORS[r])
            elif model_type == "full":
                ax = ax_surf
                ax.text(0.1, 1.15, f"to {r_text}",
                    color=REGION_COLORS[r], fontsize=7,
                    ha="center", transform=ax.transAxes
                )
            if gs_counter == 0:
                ax_first = ax

            gs_counter += 1

    ax_cbar = ax_first.inset_axes([1.01, 0.167, 0.1, 0.666],
        transform=ax_first.transAxes)
    cbar = plt.colorbar(
        matplotlib.cm.ScalarMappable(cmap=matplotlib.colormaps["PuBu"]),
        cax=ax_cbar, orientation='vertical')
    cbar.set_label("probability", size=5)
    cbar.outline.set_visible(False)
    ax_cbar.tick_params(labelsize=5, length=2)

    print(ax_first.get_xlim(), ax_first.get_ylim())
    ax_first.annotate("A", (400, 1300), (400, 1200),
        arrowprops={"arrowstyle": "<-", "shrinkB": 0, "shrinkA": 1},
        fontsize=6,
        ha="center",
        zorder=15,
        annotation_clip=False,
    )
    ax_first.annotate("M", (400, 1300), (500, 1300),
        arrowprops={"arrowstyle": "<-", "shrinkB": 0, "shrinkA": 1},
        fontsize=6,
        va="center",
        zorder=15,
        annotation_clip=False,
    )

    plt.savefig(args["output_file"], dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=FigGlmFitsSupplementParameters)
    main(module.args)
