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
import ccf_streamlines.morphology as ccfmorph
from skimage.measure import find_contours
from fig_sparse_rrr import select_and_normalize_feature_data, get_sp_rrr_fit_info

class FigGlmPredictionsParameters(ags.ArgSchema):
    glm_results_dir = ags.fields.InputDir()
    wnm_data_file = ags.fields.InputFile()
    inferred_met_type_file = ags.fields.InputFile()
    morph_file = ags.fields.InputFile(
        description="csv file with unnormalized morph features",
    )
    sparse_rrr_fit_file = ags.fields.InputFile()
    sparse_rrr_feature_file = ags.fields.InputFile()
    projected_atlas_file = ags.fields.InputFile()
    atlas_labels_file = ags.fields.InputFile()
    annot_file = ags.fields.InputFile()
    l4l5it_flat_morph_file = ags.fields.InputFile()
    l5et_flat_morph_file = ags.fields.InputFile()
    ccf_morph_dir = ags.fields.InputDir()
    output_file = ags.fields.OutputFile()


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
    'PPT': 1061,
}

VISP_3D_LOCATIONS_FOR_PREDS = np.array([
    [10180,  1410,  2710],
    [ 8780,   940,  2400],
    [ 9370,   760,  3080],
    [ 9670,   600,  3970],
    [ 7960,   550,  3090]]
)

VISP_2D_LOCATIONS_FOR_PREDS = np.array([
    [ 530, 1183],
    [ 580, 1035],
    [ 636, 1116],
    [ 721, 1175],
    [ 689,  980]]
)

PROB_CMAP = "PuBu"


def predict_prob_values(best_models, coef_df, surface_coords, lf_values):
    # figure out best type of model for each area
    model_type_cols = ["null", "surface", "lf", "full"]
    lf_cols = coef_df.columns[coef_df.columns.str.startswith("LF")]

    predicted_probs = {}
    for region, model_type in best_models.items():
        intercept = coef_df.loc[
            (coef_df.region == region) &
            (coef_df.model_type == model_type),
            ["(Intercept)"]].values.T

        if model_type == "null":
            log_odds = intercept
        elif model_type in ("surface", "full"):
            surf_coefs = coef_df.loc[
                (coef_df.region == region) &
                (coef_df.model_type == model_type),
                ["surface_ccf_x", "surface_ccf_y", "surface_ccf_z"]].values.T
            log_odds_change = np.dot(surface_coords, surf_coefs)
            log_odds = intercept + log_odds_change
        elif model_type in ("lf", "full"):
            lf_coefs = coef_df.loc[
                (coef_df.region == region) &
                (coef_df.model_type == model_type),
                lf_cols
                ]
            lf_coefs = lf_coefs.dropna(axis=1)
            log_odds_change = np.dot(np.squeeze(lf_coefs.values.T), lf_values)
            log_odds = intercept + log_odds_change
        else:
            print("not implemented model type")
            continue
        predicted_probs[region] = np.exp(log_odds) / (1 + np.exp(log_odds))
        predicted_probs[region] = predicted_probs[region][0, 0]
    return predicted_probs


def plot_flatmap_probs(ax, pred_values, bounds_ipsi=None, bounds_contra=None, highlight_regions=[]):
    cmap = matplotlib.colormaps[PROB_CMAP]

    if bounds_ipsi is not None:
        plot_flatmap_side(ax, cmap, pred_values, bounds_ipsi, "ipsi_", highlight_regions)

    if bounds_contra is not None:
        plot_flatmap_side(ax, cmap, pred_values, bounds_contra, "contra_", highlight_regions)

    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set(xticks=[], yticks=[])
    sns.despine(ax=ax, left=True, bottom=True)


def plot_flatmap_side(ax, cmap, pred_values, bounds, prefix, highlight_regions=[]):
    pred_regions = list(pred_values.keys())
    for k in bounds:
        if prefix + k in pred_regions:
            val = pred_values[prefix + k]
            color = cmap(val)
        else:
            color = "none"
        boundary_coords = bounds[k]
        if prefix + k in highlight_regions:
            edgecolor = "k"
            lw = 0.25
            zorder = 11
        else:
            edgecolor = ".5"
            lw = 0.25
            zorder = 10
        ax.fill(*boundary_coords.T, color=color, edgecolor=edgecolor, lw=lw, zorder=zorder)


def plot_top_probs(ax, pred_values, struct_contours_dict, background_contours,
        prefix="ipsi_", highlight_regions=[]):
    CUSTOM_ZORDER = {
        "ZI": 2,
        "LP": 8,
        "LGv": 9,
        "LD": 11,
        "OP": 11,
        "LGd-ip": 12,
    }

    cmap = matplotlib.colormaps[PROB_CMAP]

    for b in background_contours:
        ax.fill(*b.T, color="#dddddd", edgecolor='0.5',
            zorder=1, lw=0.25)

    for k, struct_contours in struct_contours_dict.items():
        val = pred_values[prefix + k]
        color = cmap(val)
        if k in CUSTOM_ZORDER.keys():
            zorder = CUSTOM_ZORDER[k]
        else:
            zorder = 10
        if "ipsi_" + k in highlight_regions:
            edgecolor = "k"
            lw = 0.25
        else:
            edgecolor = ".5"
            lw = 0.25
        for b in struct_contours:
            ax.fill(*b.T, color=color, edgecolor=edgecolor,
                zorder=zorder, lw=lw)

    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set(xticks=[], yticks=[])
    sns.despine(ax=ax, left=True, bottom=True)


def plot_ccf_morph(morph, ax, alpha=1.0, scale_factor=1, zorder=1, color="black",
        x_ind=4, y_ind=2):
    lines_x = []
    lines_y = []
    morph_vals = morph.values
    for i in range(morph_vals.shape[0]):
        if np.isnan(morph_vals[i, x_ind]) or np.isnan(morph_vals[i, y_ind]):
            continue
        parent_id = morph_vals[i, 6]
        if parent_id == -1:
            continue
        p_ind = np.flatnonzero(morph_vals[:, 0] == parent_id)[0]
        if np.isnan(morph_vals[p_ind, x_ind]) or np.isnan(morph_vals[p_ind, y_ind]):
            continue
        lines_x += [morph_vals[p_ind, x_ind] / scale_factor, morph_vals[i, x_ind] / scale_factor, None]
        lines_y += [morph_vals[p_ind, y_ind] / scale_factor, morph_vals[i, y_ind] / scale_factor, None]
    ax.plot(lines_x, lines_y, linewidth=0.25, alpha=alpha, zorder=zorder, color=color)
    return ax



def plot_flatmap_morph(morph, flatmap_coords, ax, alpha=1.0, scale_factor=1, zorder=1, color="black"):
    lines_x = []
    lines_y = []
    morph_vals = morph.values
    for i in range(morph_vals.shape[0]):
        if np.isnan(flatmap_coords[i, 0]) or np.isnan(flatmap_coords[i, 1]):
            continue
        parent_id = morph_vals[i, 6]
        if parent_id == -1:
            continue
        p_ind = np.flatnonzero(morph_vals[:, 0] == parent_id)[0]
        if np.isnan(flatmap_coords[p_ind, 0]) or np.isnan(flatmap_coords[p_ind, 1]):
            continue
        lines_x += [flatmap_coords[p_ind, 0] / scale_factor, flatmap_coords[i, 0] / scale_factor, None]
        lines_y += [flatmap_coords[p_ind, 1] / scale_factor, flatmap_coords[i, 1] / scale_factor, None]
    ax.plot(lines_x, lines_y, linewidth=0.25, alpha=alpha, zorder=zorder, color=color)
    return ax


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

    all_coef_df = pd.concat(coef_df_list)

    all_loo_df = pd.concat(loo_df_list)
    all_loo_df["pR2"] = 1 - all_loo_df["ll_fit"] / all_loo_df["ll_null"]

    wnm_data_df = pd.read_csv(args["wnm_data_file"], index_col=0)

    morph_df = pd.read_csv(args["morph_file"], index_col=0)
    with open(args["sparse_rrr_feature_file"], "r") as f:
        sp_rrr_features_info = json.load(f)
    h5f = h5py.File(args["sparse_rrr_fit_file"], "r")

    inf_met_df = pd.read_csv(args["inferred_met_type_file"], index_col=0)



    bf_boundary_finder = BoundaryFinder(
        projected_atlas_file=args["projected_atlas_file"],
        labels_file=args["atlas_labels_file"],
    )
    bounds_ipsi = bf_boundary_finder.region_boundaries()
    bounds_contra = bf_boundary_finder.region_boundaries(
        hemisphere="right_for_both",
        view_space_for_other_hemisphere="flatmap_butterfly")

    annot, _ = nrrd.read(args["annot_file"])
    annot_hemi = annot[:, :, :1140 // 2]


    cp_masked = (annot_hemi == SUBCORT_STRUCT_IDS["CP"]).astype(int)
    top_background_projection = (annot_hemi.sum(axis=1) > 0).astype(int)
    top_cp_projection = (cp_masked.sum(axis=1) > 0).astype(int)
    top_cp_contours = find_contours(top_cp_projection.T, level=0.5)
    top_background_contours = find_contours(top_background_projection.T, level=0.5)

    top_whole_background_projection = (annot.sum(axis=1) > 0).astype(int)
    top_whole_background_contours = find_contours(top_whole_background_projection.T, level=0.5)

    #### PLOTTING ######
    fig = plt.figure(figsize=(7.5, 8))
    gs = gridspec.GridSpec(
        3, 2,
        hspace=0.2,
        wspace=0.2,
    )

    # L4 & L5 IT
    sc = "L4-L5-IT"

    l4l5it_data = wnm_data_df.loc[
        wnm_data_df.predicted_met_type.isin(SUBCLASS_MET_TYPES[sc]), :].copy()
    l4l5it_data.columns = [c.replace("-", "_") for c in l4l5it_data.columns]
    l4l5_lf_cols = l4l5it_data.columns[l4l5it_data.columns.str.startswith("LF")]
    l4l5_lf_cols = l4l5_lf_cols[l4l5_lf_cols.str.endswith(sc)]
    surf_cols = l4l5it_data.columns[l4l5it_data.columns.str.startswith("surface")]
    best_models = train_df.loc[sc, "best_model_type"]

    ## Effect of surface

    gs_l4l5it_surf = gridspec.GridSpecFromSubplotSpec(
        2, 3,
        hspace=0.2,
        wspace=0,
        subplot_spec=gs[0, 1],
    )

    surf_axes = [
        plt.subplot(gs_l4l5it_surf[1, 0]),
        plt.subplot(gs_l4l5it_surf[0, 0]),
        plt.subplot(gs_l4l5it_surf[:, 1]),
        plt.subplot(gs_l4l5it_surf[1, 2]),
        plt.subplot(gs_l4l5it_surf[0, 2])
    ]

    best_models_surf = best_models[best_models.isin(["surface", "full"])]
    print(sc, "surface")
    print(best_models_surf)
    for i in range(len(VISP_2D_LOCATIONS_FOR_PREDS)):
        pred_values = predict_prob_values(
            best_models_surf,
            all_coef_df.groupby("subclass").get_group(sc),
            VISP_3D_LOCATIONS_FOR_PREDS[i, :],
            l4l5it_data.loc[:, l4l5_lf_cols].median().values,
        )
        ax = surf_axes[i]
        plot_flatmap_probs(ax, pred_values,
            bounds_ipsi=bounds_ipsi, bounds_contra=bounds_contra)
        ax.scatter(
            [VISP_2D_LOCATIONS_FOR_PREDS[i, 0]],
            [VISP_2D_LOCATIONS_FOR_PREDS[i, 1]],
            s=8,
            marker="x",
            color="firebrick",
            zorder=20,
            linewidths=0.75,
        )


    ax = surf_axes[1]
    xy_text_dict = {
        "ipsi_MOs": (0.4, 1.0),
        "ipsi_VISpl": (0, -0.1),
        'ipsi_VISrl': (-0.15, 0.3),
        "ipsi_VISa": (0.45, 0.2),
        "ipsi_VISam": (0.35, 0.05),
        "ipsi_VISpm": (0.25, -0.15),
        "contra_VISp": (0.85, -0.1),
    }
    xy_dict = {
        "ipsi_VISpl": (460, 1200),
    }
    for n in best_models_surf.index:
        hemi, struct = n.split("_")
        if hemi == "ipsi":
            b = bounds_ipsi[struct]
        elif hemi == "contra":
            b = bounds_contra[struct]
        if n not in xy_dict:
            xy_dict[n] = b.mean(axis=0)
        ax.annotate(
            n.split("_")[1],
            xy_dict[n],
            xy_text_dict[n],
            xycoords=ax.transData,
            textcoords=ax.transAxes,
            fontsize=4,
            arrowprops={
                "arrowstyle": "-",
                "lw": 0.25,
                "shrinkB": 0,
                "shrinkA": 0,
            },
            zorder=50,
        )

    # Effect of latent factors

    gs_l4l5it_lf = gridspec.GridSpecFromSubplotSpec(
        2, 7,
        hspace=0.05,
        wspace=0.1,
        width_ratios=(1.5, 3.25, 0.75, 0.2, 1.5, 3.25, 0.75),
        subplot_spec=gs[0, 0],
    )

    l4l5it_lf_example_values = np.array([
        [-1.5, 0],
        [-0.5, 2],
        [-5, -1],
        [1, -2],
    ])

    lf_scatter_axes = [
        plt.subplot(gs_l4l5it_lf[0, 0]),
        plt.subplot(gs_l4l5it_lf[0, 4]),
        plt.subplot(gs_l4l5it_lf[1, 0]),
        plt.subplot(gs_l4l5it_lf[1, 4]),
    ]

    lf_flatmap_axes = [
        plt.subplot(gs_l4l5it_lf[0, 1]),
        plt.subplot(gs_l4l5it_lf[0, 5]),
        plt.subplot(gs_l4l5it_lf[1, 1]),
        plt.subplot(gs_l4l5it_lf[1, 5]),
    ]

    lf_top_axes = [
        plt.subplot(gs_l4l5it_lf[0, 2]),
        plt.subplot(gs_l4l5it_lf[0, 6]),
        plt.subplot(gs_l4l5it_lf[1, 2]),
        plt.subplot(gs_l4l5it_lf[1, 6]),
    ]


    best_models_lf = best_models[best_models.isin(["lf", "full"])]
    print(sc, "lf")
    print(best_models_lf)

    specimen_ids, genes, w, v = get_sp_rrr_fit_info(
        sc, "morph", h5f
    )
    feature_data = select_and_normalize_feature_data(
        specimen_ids, sc, "morph",
        sp_rrr_features_info, morph_df)
    feature_lf = feature_data.values @ v
    feature_lf_std = feature_lf.std(axis=0)
    top_contours = {"CP": top_cp_contours}

    for i in range(l4l5it_lf_example_values.shape[0]):
        ax_scatter = lf_scatter_axes[i]
        ax_scatter.scatter(
            feature_lf[:, 0] / feature_lf_std[0],
            feature_lf[:, 1] / feature_lf_std[1],
            c=[MET_TYPE_COLORS[t] for t in inf_met_df.loc[specimen_ids, "inferred_met_type"]],
            s=1,
            edgecolors="white",
            lw=0.25,
        )
        ax_scatter.scatter(
            l4l5it_data["LF1_L4_L5_IT"] / feature_lf_std[0],
            l4l5it_data["LF2_L4_L5_IT"] / feature_lf_std[1],
            c=[MET_TYPE_COLORS[t] for t in l4l5it_data["predicted_met_type"]],
            s=3,
            edgecolors="white",
            lw=0.25,
        )
        ax_scatter.scatter(
            [l4l5it_lf_example_values[i, 0] / feature_lf_std[0]],
            [l4l5it_lf_example_values[i, 1] / feature_lf_std[1]],
            s=8,
            marker="x",
            color="firebrick",
            zorder=20,
            linewidths=0.75,
        )

        corr_circle = plt.Circle((0, 0), 2.7, edgecolor="#999999",
            fill=False, linestyle="dotted")
        ax_scatter.add_patch(corr_circle)

        ax_scatter.set_aspect("equal")
        sns.despine(ax=ax_scatter, left=True, bottom=True)
        ax_scatter.set(xticks=[], yticks=[])

        pred_values = predict_prob_values(
            best_models_lf,
            all_coef_df.groupby("subclass").get_group(sc),
            l4l5it_data.loc[:, surf_cols].mean().values,
            l4l5it_lf_example_values[i, :],
        )
        ax = lf_flatmap_axes[i]
        plot_flatmap_probs(ax, pred_values,
            bounds_ipsi=bounds_ipsi, bounds_contra=bounds_contra)

        ax = lf_top_axes[i]
        plot_top_probs(ax, pred_values, top_contours, top_background_contours)

    ax = lf_flatmap_axes[0]
    xy_text_dict = {
        "ipsi_MOs": (0.2, 1.0),
        "ipsi_ACAd": (0.45, 0.95),
        "ipsi_VISpor": (-0.1, -0.1),
        "ipsi_RSPagl": (0.45, -0.05),
        "ipsi_RSPd": (0.3, -0.2),
        "ipsi_VISpm": (0.45, 0.3),
        "contra_VISp": (0.85, -0.1),
    }
    xy_dict = {
        "ipsi_RSPd": (800, 1275),
        "ipsi_RSPagl": (820, 1100),
    }
    for n in best_models_lf.index:
        hemi, struct = n.split("_")
        if hemi == "ipsi" and struct in bounds_ipsi:
            b = bounds_ipsi[struct]
        elif hemi == "contra" and struct in bounds_contra:
            b = bounds_contra[struct]
        else:
            continue
        if n not in xy_dict:
            xy_dict[n] = b.mean(axis=0)
        print(n, xy_dict[n])

        ax.annotate(
            n.split("_")[1],
            xy_dict[n],
            xy_text_dict[n],
            xycoords=ax.transData,
            textcoords=ax.transAxes,
            fontsize=4,
            arrowprops={
                "arrowstyle": "-",
                "lw": 0.25,
                "shrinkB": 0,
                "shrinkA": 0,
            },
            zorder=50,
        )

    ax = lf_top_axes[0]
    xy_text_dict = {
        "ipsi_CP": (-0.05, 0.9),
    }
    xy_dict = {}
    for n in best_models_lf.index:
        hemi, struct = n.split("_")
        if hemi == "ipsi" and struct in top_contours:
            print(hemi, struct)
            b = top_contours[struct][0]
        else:
            continue
        if n not in xy_dict:
            xy_dict[n] = b.mean(axis=0)
        ax.annotate(
            n.split("_")[1],
            xy_dict[n],
            xy_text_dict[n],
            xycoords=ax.transData,
            textcoords=ax.transAxes,
            fontsize=4,
            arrowprops={
                "arrowstyle": "-",
                "lw": 0.25,
                "shrinkB": 0,
                "shrinkA": 0,
            },
            zorder=50,
        )

    ax = lf_flatmap_axes[1]
    ax_cbar = ax.inset_axes([0.25, -0.25, 0.5, 0.1], transform=ax.transAxes)

    cbar = plt.colorbar(
        matplotlib.cm.ScalarMappable(cmap=matplotlib.colormaps[PROB_CMAP]),
        cax=ax_cbar, orientation='horizontal')
    cbar.set_label("probability", size=6)
    cbar.outline.set_visible(False)
    ax_cbar.tick_params(labelsize=6)


    gs_l4l5it_examples = gridspec.GridSpecFromSubplotSpec(
        2, 1,
        subplot_spec=gs[2, 0],
    )

    l4l5it_examples = [
        "191812_6431-X3967-Y9193_reg",
        "191812_7148-X3864-Y9658_reg",
    ]
    best_models = best_models[best_models.isin(["surface", "lf", "full"])]

    gs_l4l5it_examples = gridspec.GridSpecFromSubplotSpec(
        2, 2,
        width_ratios=(2.5, 1),
        subplot_spec=gs[2, 0],
    )

    l4l5it_flatmorph_h5f = h5py.File(args["l4l5it_flat_morph_file"], "r")

    pred_structs_with_hemi = best_models.index.tolist()
    pred_structs = [i.split("_")[1] for i in best_models.index]
    print("l4 & l5 it pred structs", pred_structs_with_hemi)
    for i, spec_id in enumerate(l4l5it_examples):
        ax_flat = plt.subplot(gs_l4l5it_examples[i, 0])
        ax_top = plt.subplot(gs_l4l5it_examples[i, 1])
        loo_df = all_loo_df.groupby("specimen_id").get_group(spec_id)
        loo_df = loo_df.loc[loo_df["region"].isin(pred_structs_with_hemi), :]
        pred_values = dict(zip(loo_df["region"], loo_df["pred_prob"]))
        region_names = list(pred_values.keys())
        actual_proj = (wnm_data_df.loc[spec_id, region_names] > 0)
        highlight_regions = actual_proj.index.values[actual_proj].tolist()

        plot_flatmap_probs(ax_flat, pred_values,
            bounds_ipsi=bounds_ipsi, bounds_contra=bounds_contra,
            highlight_regions=highlight_regions)

        plot_top_probs(ax_top, pred_values,  {"CP": top_cp_contours},
            top_whole_background_contours, highlight_regions=highlight_regions)

        flatmap_coords = l4l5it_flatmorph_h5f[spec_id][:]
        ccf_morph = ccfmorph.load_swc_as_dataframe(
            os.path.join(args["ccf_morph_dir"], f"{spec_id}.swc")
        )
        plot_flatmap_morph(ccf_morph, flatmap_coords, ax_flat, zorder=20)
        plot_ccf_morph(ccf_morph, ax_top, zorder=20, scale_factor=10)

    ax = plt.subplot(gs_l4l5it_examples[0, 0])
    xy_text_dict = {
        "ipsi_MOs": (0.2, 1.0),
        "ipsi_ACAd": (0.45, 0.95),
        "ipsi_VISpor": (-0.1, 0),
        "ipsi_RSPagl": (0.35, -0.025),
        "ipsi_RSPd": (0.3, -0.1),
        "ipsi_VISa": (0.45, 0.3),
        "contra_VISp": (0.85, 0),
        "ipsi_VISpl": (0, -0.1),
        "ipsi_VISpm": (0.4, 0.05),
        'ipsi_VISrl': (-0.1, 0.3),
        "ipsi_VISam": (0.45, 0.2),
    }
    xy_dict = {
        "ipsi_RSPd": (800, 1275),
        "ipsi_RSPagl": (820, 1100),
        "ipsi_VISpl": (460, 1200),
    }
    for n in pred_values.keys():
        hemi, struct = n.split("_")
        if hemi == "ipsi" and struct in bounds_ipsi:
            b = bounds_ipsi[struct]
        elif hemi == "contra" and struct in bounds_contra:
            b = bounds_contra[struct]
        else:
            continue
        if n not in xy_dict:
            xy_dict[n] = b.mean(axis=0)
        print(n, xy_dict[n])

        ax.annotate(
            n.split("_")[1],
            xy_dict[n],
            xy_text_dict[n],
            xycoords=ax.transData,
            textcoords=ax.transAxes,
            fontsize=4,
            arrowprops={
                "arrowstyle": "-",
                "lw": 0.25,
                "shrinkB": 0,
                "shrinkA": 0,
            },
            zorder=50,
        )

    ax = plt.subplot(gs_l4l5it_examples[0, 1])
    xy_text_dict = {
        "ipsi_CP": (-0.05, 0.9),
    }
    xy_dict = {}
    for n in pred_values.keys():
        hemi, struct = n.split("_")
        if hemi == "ipsi" and struct in top_contours:
            print(hemi, struct)
            b = top_contours[struct][0]
        else:
            continue
        if n not in xy_dict:
            xy_dict[n] = b.mean(axis=0)
        ax.annotate(
            n.split("_")[1],
            xy_dict[n],
            xy_text_dict[n],
            xycoords=ax.transData,
            textcoords=ax.transAxes,
            fontsize=4,
            arrowprops={
                "arrowstyle": "-",
                "lw": 0.25,
                "shrinkB": 0,
                "shrinkA": 0,
            },
            zorder=50,
        )


    l4l5it_flatmorph_h5f.close()

    # L5 ET
    sc = "L5-ET"

    l5et_data = wnm_data_df.loc[
        wnm_data_df.predicted_met_type.isin(SUBCLASS_MET_TYPES[sc]), :].copy()
    l5et_data.columns = [c.replace("-", "_") for c in l5et_data.columns]
    l5et_lf_cols = l5et_data.columns[l5et_data.columns.str.startswith("LF")]
    l5et_lf_cols = l5et_lf_cols[l5et_lf_cols.str.endswith(sc)]
    surf_cols = l5et_data.columns[l5et_data.columns.str.startswith("surface")]
    best_models = train_df.loc[sc, "best_model_type"]


    gs_l5et_surf = gridspec.GridSpecFromSubplotSpec(
        2, 6,
        hspace=0.2,
        wspace=0,
        width_ratios=(2, 1, 2, 1, 2, 1),
        subplot_spec=gs[1, 1],
    )

    surf_axes = [
        plt.subplot(gs_l5et_surf[1, 0]),
        plt.subplot(gs_l5et_surf[0, 0]),
        plt.subplot(gs_l5et_surf[:, 2]),
        plt.subplot(gs_l5et_surf[1, 4]),
        plt.subplot(gs_l5et_surf[0, 4])
    ]

    subcort_axes = [
        plt.subplot(gs_l5et_surf[1, 1]),
        plt.subplot(gs_l5et_surf[0, 1]),
        plt.subplot(gs_l5et_surf[:, 3]),
        plt.subplot(gs_l5et_surf[1, 5]),
        plt.subplot(gs_l5et_surf[0, 5])
    ]


    best_models_surf = best_models[best_models.isin(["surface", "full"])]
    print(sc, "surface")
    print(best_models_surf)

    surf_structs = [i.split("_")[1] for i in best_models_surf.index]
    subcort_surf_structs = [s for s in surf_structs if s in SUBCORT_STRUCT_IDS.keys()]
    top_struct_contours = {}
    for s in subcort_surf_structs:
        print(s)
        struct_masked = (annot_hemi == SUBCORT_STRUCT_IDS[s]).astype(int)
        top_projection = (struct_masked.sum(axis=1) > 0).astype(int)
        top_struct_contours[s] = find_contours(top_projection.T, level=0.5)


    for i in range(len(VISP_2D_LOCATIONS_FOR_PREDS)):
        pred_values = predict_prob_values(
            best_models_surf,
            all_coef_df.groupby("subclass").get_group(sc),
            VISP_3D_LOCATIONS_FOR_PREDS[i, :],
            l5et_data.loc[:, l5et_lf_cols].median().values,
        )
        ax = surf_axes[i]
        ax_subcort = subcort_axes[i]
        plot_flatmap_probs(ax, pred_values,
            bounds_ipsi=bounds_ipsi, bounds_contra=None)
        ax.scatter(
            [VISP_2D_LOCATIONS_FOR_PREDS[i, 0]],
            [VISP_2D_LOCATIONS_FOR_PREDS[i, 1]],
            s=8,
            marker="x",
            color="firebrick",
            zorder=20,
            linewidths=0.75,
        )
        plot_top_probs(ax_subcort, pred_values,
            top_struct_contours, top_background_contours)

    ax = surf_axes[1]
    xy_text_dict = {
        "ipsi_RSPagl": (0.6, 0),
        "ipsi_VISal": (-0.1, 0.2),
        "ipsi_VISam": (0.65, 0.55),
        "ipsi_VISl": (-0.05, 0.1),
        "ipsi_VISpl": (0, -0.05),
        "ipsi_VISpm": (0.4, -0.1),
        'ipsi_VISrl': (-0.25, 0.3),
    }
    xy_dict = {
        "ipsi_RSPagl": (820, 1100),
        "ipsi_VISpl": (460, 1200),
    }
    for n in best_models_surf.index:
        hemi, struct = n.split("_")
        if hemi == "ipsi" and struct in bounds_ipsi:
            b = bounds_ipsi[struct]
        elif hemi == "contra" and struct in bounds_contra:
            b = bounds_contra[struct]
        else:
            continue
        if n not in xy_dict:
            xy_dict[n] = b.mean(axis=0)
        print(n, xy_dict[n])

        ax.annotate(
            n.split("_")[1],
            xy_dict[n],
            xy_text_dict[n],
            xycoords=ax.transData,
            textcoords=ax.transAxes,
            fontsize=4,
            arrowprops={
                "arrowstyle": "-",
                "lw": 0.25,
                "shrinkB": 0,
                "shrinkA": 0,
            },
            zorder=50,
        )

    ax = subcort_axes[1]
    xy_text_dict = {
        "ipsi_CP": (-0.05, 0.9),
        "ipsi_LD": (1.15, 0.7),
        "ipsi_LGd-ip": (0.2, 0.15),
        "ipsi_LGv": (0.05, 0.3),
        "ipsi_LP": (1.15, 0.5),
        "ipsi_OP": (1.15, 0.4),
        "ipsi_PG": (1.15, 0.3),
    }
    xy_dict = {
    }
    for n in best_models_surf.index:
        hemi, struct = n.split("_")
        if hemi == "ipsi" and struct in top_struct_contours:
            b = top_struct_contours[struct][0]
        else:
            continue
        if n not in xy_dict:
            xy_dict[n] = b.mean(axis=0)
        print(n, xy_dict[n])

        ax.annotate(
            n.split("_")[1],
            xy_dict[n],
            xy_text_dict[n],
            xycoords=ax.transData,
            textcoords=ax.transAxes,
            fontsize=4,
            arrowprops={
                "arrowstyle": "-",
                "lw": 0.25,
                "shrinkB": 0,
                "shrinkA": 0,
            },
            zorder=50,
        )

    gs_l5e_lf = gridspec.GridSpecFromSubplotSpec(
        2, 7,
        hspace=0.05,
        wspace=0.2,
        width_ratios=(1.5, 2, 1, 0.2, 1.5, 2, 1),
        subplot_spec=gs[1, 0],
    )

    l5et_lf_example_values = np.array([
        [0, 0],
        [2, 2],
        [-3, -2],
        [2, -2],
    ])

    lf_scatter_axes = [
        plt.subplot(gs_l5e_lf[0, 0]),
        plt.subplot(gs_l5e_lf[0, 4]),
        plt.subplot(gs_l5e_lf[1, 0]),
        plt.subplot(gs_l5e_lf[1, 4]),
    ]

    lf_flatmap_axes = [
        plt.subplot(gs_l5e_lf[0, 1]),
        plt.subplot(gs_l5e_lf[0, 5]),
        plt.subplot(gs_l5e_lf[1, 1]),
        plt.subplot(gs_l5e_lf[1, 5]),
    ]

    lf_top_axes = [
        plt.subplot(gs_l5e_lf[0, 2]),
        plt.subplot(gs_l5e_lf[0, 6]),
        plt.subplot(gs_l5e_lf[1, 2]),
        plt.subplot(gs_l5e_lf[1, 6]),
    ]

    best_models_lf = best_models[best_models.isin(["lf", "full"])]

    print(sc, "lf")
    print(best_models_lf)

    specimen_ids, genes, w, v = get_sp_rrr_fit_info(
        sc, "morph", h5f
    )
    feature_data = select_and_normalize_feature_data(
        specimen_ids, sc, "morph",
        sp_rrr_features_info, morph_df)
    feature_lf = feature_data.values @ v
    feature_lf_std = feature_lf.std(axis=0)
    print(feature_lf_std)

    lf_structs = [i.split("_")[1] for i in best_models_lf.index]
    subcort_lf_structs = [s for s in lf_structs if s in SUBCORT_STRUCT_IDS.keys()]
    top_struct_contours = {}
    for s in subcort_lf_structs:
        print(s)
        struct_masked = (annot_hemi == SUBCORT_STRUCT_IDS[s]).astype(int)
        top_projection = (struct_masked.sum(axis=1) > 0).astype(int)
        top_struct_contours[s] = find_contours(top_projection.T, level=0.5)

    for i in range(l5et_lf_example_values.shape[0]):
        ax_scatter = lf_scatter_axes[i]
        ax_scatter.scatter(
            feature_lf[:, 0] / feature_lf_std[0],
            feature_lf[:, 1] / feature_lf_std[1],
            c=[MET_TYPE_COLORS[t] for t in inf_met_df.loc[specimen_ids, "inferred_met_type"]],
            s=1,
            edgecolors="white",
            lw=0.25,
        )
        ax_scatter.scatter(
            l5et_data["LF1_L5_ET"] / feature_lf_std[0],
            l5et_data["LF2_L5_ET"] / feature_lf_std[1],
            c=[MET_TYPE_COLORS[t] for t in l5et_data["predicted_met_type"]],
            s=3,
            edgecolors="white",
            lw=0.25,
        )
        ax_scatter.scatter(
            [l5et_lf_example_values[i, 0] / feature_lf_std[0]],
            [l5et_lf_example_values[i, 1] / feature_lf_std[1]],
            s=8,
            marker="x",
            color="firebrick",
            zorder=20,
            linewidths=0.75,
        )
        ax_scatter.set_aspect("equal")
        sns.despine(ax=ax_scatter, left=True, bottom=True)
        ax_scatter.set(xticks=[], yticks=[])
        corr_circle = plt.Circle((0, 0), 3.2, edgecolor="#999999",
            fill=False, linestyle="dotted")
        ax_scatter.add_patch(corr_circle)

        pred_values = predict_prob_values(
            best_models_lf,
            all_coef_df.groupby("subclass").get_group(sc),
            l5et_data.loc[:, surf_cols].mean().values,
            l5et_lf_example_values[i, :],
        )

        ax = lf_flatmap_axes[i]
        plot_flatmap_probs(ax, pred_values,
            bounds_ipsi=bounds_ipsi, bounds_contra=None)

        ax = lf_top_axes[i]
        plot_top_probs(ax, pred_values, top_struct_contours, top_background_contours)

    ax = lf_flatmap_axes[0]
    xy_text_dict = {
        "ipsi_VISal": (-0.1, 0.15),
        "ipsi_VISpl": (0, -0.05),
        'ipsi_VISrl': (-0.25, 0.3),
    }
    xy_dict = {
        "ipsi_VISpl": (460, 1200),
    }

    for n in best_models_lf.index:
        hemi, struct = n.split("_")
        if hemi == "ipsi" and struct in bounds_ipsi:
            b = bounds_ipsi[struct]
        elif hemi == "contra" and struct in bounds_contra:
            b = bounds_contra[struct]
        else:
            continue
        if n not in xy_dict:
            xy_dict[n] = b.mean(axis=0)
        print(n, xy_dict[n])

        ax.annotate(
            n.split("_")[1],
            xy_dict[n],
            xy_text_dict[n],
            xycoords=ax.transData,
            textcoords=ax.transAxes,
            fontsize=4,
            arrowprops={
                "arrowstyle": "-",
                "lw": 0.25,
                "shrinkB": 0,
                "shrinkA": 0,
            },
            zorder=50,
        )

    ax = lf_top_axes[0]
    xy_text_dict = {
        "ipsi_LD": (0.3, 0.7),
        "ipsi_LGd-co": (0.1, 0.15),
        "ipsi_LP": (1.15, 0.5),
        "ipsi_APN": (1.15, 0.3),
        "ipsi_ZI": (1.15, 0.7),
    }
    xy_dict = {
        "ipsi_ZI": (505, 660), # 424.60915105 707.48070562
    }
    for n in best_models_lf.index:
        hemi, struct = n.split("_")
        if hemi == "ipsi" and struct in top_struct_contours:
            b = top_struct_contours[struct][0]
        else:
            continue
        if n not in xy_dict:
            xy_dict[n] = b.mean(axis=0)
        print(n, xy_dict[n])

        ax.annotate(
            n.split("_")[1],
            xy_dict[n],
            xy_text_dict[n],
            xycoords=ax.transData,
            textcoords=ax.transAxes,
            fontsize=4,
            arrowprops={
                "arrowstyle": "-",
                "lw": 0.25,
                "shrinkB": 0,
                "shrinkA": 0,
            },
            zorder=50,
        )

    l5et_examples = [
        "211796_7534-X18739-Y16699_reg",
        "211550_6721-X20655-Y16337_reg",
    ]
    best_models = best_models[best_models.isin(["surface", "lf", "full"])]

    gs_l5et_examples = gridspec.GridSpecFromSubplotSpec(
        2, 2,
        width_ratios=(2, 1),
        subplot_spec=gs[2, 1],
    )

    pred_structs_with_hemi = best_models.index.tolist()
    pred_structs = [i.split("_")[1] for i in best_models.index]
    print("l5 et pred structs", pred_structs_with_hemi)
    subcort_pred_structs = [s for s in pred_structs if s in SUBCORT_STRUCT_IDS.keys()]
    top_struct_contours = {}
    for s in subcort_pred_structs:
        print(s)
        struct_masked = (annot_hemi == SUBCORT_STRUCT_IDS[s]).astype(int)
        top_projection = (struct_masked.sum(axis=1) > 0).astype(int)
        top_struct_contours[s] = find_contours(top_projection.T, level=0.5)

    l5et_flatmorph_h5f = h5py.File(args["l5et_flat_morph_file"], "r")

    for i, spec_id in enumerate(l5et_examples):
        ax_flat = plt.subplot(gs_l5et_examples[i, 0])
        ax_top = plt.subplot(gs_l5et_examples[i, 1])
        loo_df = all_loo_df.groupby("specimen_id").get_group(spec_id)
        loo_df = loo_df.loc[loo_df["region"].isin(pred_structs_with_hemi), :]
        pred_values = dict(zip(loo_df["region"], loo_df["pred_prob"]))
        region_names = list(pred_values.keys())
        actual_proj = (wnm_data_df.loc[spec_id, region_names] > 0)
        highlight_regions = actual_proj.index.values[actual_proj].tolist()

        plot_flatmap_probs(ax_flat, pred_values,
            bounds_ipsi=bounds_ipsi, bounds_contra=None,
            highlight_regions=highlight_regions)

        plot_top_probs(ax_top, pred_values, top_struct_contours,
            top_whole_background_contours, highlight_regions=highlight_regions)

        flatmap_coords = l5et_flatmorph_h5f[spec_id][:]
        ccf_morph = ccfmorph.load_swc_as_dataframe(
            os.path.join(args["ccf_morph_dir"], f"{spec_id}.swc")
        )
        plot_flatmap_morph(ccf_morph, flatmap_coords, ax_flat, zorder=20)
        plot_ccf_morph(ccf_morph, ax_top, zorder=20, scale_factor=10)
    l5et_flatmorph_h5f.close()

    ax = plt.subplot(gs_l5et_examples[0, 0])
    xy_text_dict = {
        "ipsi_VISal": (-0.1, 0.15),
        "ipsi_VISpl": (0, -0.05),
        'ipsi_VISrl': (-0.25, 0.3),
        "ipsi_RSPagl": (0.6, 0),
        "ipsi_VISam": (0.65, 0.55),
        "ipsi_VISl": (-0.05, 0.05),
        "ipsi_VISpm": (0.4, -0.1),
    }
    xy_dict = {
        "ipsi_RSPagl": (820, 1100),
        "ipsi_VISpl": (460, 1200),
    }
    for n in pred_values.keys():
        hemi, struct = n.split("_")
        if hemi == "ipsi" and struct in bounds_ipsi:
            b = bounds_ipsi[struct]
        elif hemi == "contra" and struct in bounds_contra:
            b = bounds_contra[struct]
        else:
            continue
        if n not in xy_dict:
            xy_dict[n] = b.mean(axis=0)
        print(n, xy_dict[n])

        ax.annotate(
            n.split("_")[1],
            xy_dict[n],
            xy_text_dict[n],
            xycoords=ax.transData,
            textcoords=ax.transAxes,
            fontsize=4,
            arrowprops={
                "arrowstyle": "-",
                "lw": 0.25,
                "shrinkB": 0,
                "shrinkA": 0,
            },
            zorder=50,
        )

    ax = plt.subplot(gs_l5et_examples[0, 1])
    xy_text_dict = {
        "ipsi_CP": (-0.05, 0.9),
        "ipsi_ZI": (0.65, 0.65),
        "ipsi_LD": (0.5, 0.7),
        "ipsi_LP": (0.65, 0.5),
        "ipsi_LGd-co": (0, 0.2),
        "ipsi_LGd-ip": (0.2, 0.1),
        "ipsi_LGv": (0, 0.3),
        "ipsi_OP": (0.65, 0.4),
        "ipsi_APN": (0.65, 0.3),
        "ipsi_PG": (0.65, 0.2),
    }
    xy_dict = {
        "ipsi_ZI": (505, 660),
        "ipsi_LGd-co": (323, 750),
        "ipsi_LGv": (290, 745),
    }
    for n in pred_values.keys():
        hemi, struct = n.split("_")
        if hemi == "ipsi" and struct in top_struct_contours:
            b = top_struct_contours[struct][0]
        else:
            continue
        if n not in xy_dict:
            xy_dict[n] = b.mean(axis=0)
        print(n, xy_dict[n])

        ax.annotate(
            n.split("_")[1],
            xy_dict[n],
            xy_text_dict[n],
            xycoords=ax.transData,
            textcoords=ax.transAxes,
            fontsize=4,
            arrowprops={
                "arrowstyle": "-",
                "lw": 0.25,
                "shrinkB": 0,
                "shrinkA": 0,
            },
            zorder=50,
        )

    plt.savefig(args["output_file"], dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=FigGlmPredictionsParameters)
    main(module.args)

