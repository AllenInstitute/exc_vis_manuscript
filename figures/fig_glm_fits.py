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



class FigGlmFitsParameters(ags.ArgSchema):
    glm_results_dir = ags.fields.InputDir()
    wnm_data_file = ags.fields.InputFile()
    projected_atlas_file = ags.fields.InputFile()
    atlas_labels_file = ags.fields.InputFile()
    flatmap_file = ags.fields.InputFile()
    annot_file = ags.fields.InputFile()
    output_file = ags.fields.OutputFile()


REGION_CATEGORIES = {
    "higher visual areas": (
        "ipsi_VISl",
        "ipsi_VISam",
        "ipsi_VISpor",
        "ipsi_VISal",
        "ipsi_VISrl",
        "ipsi_VISli",
        "ipsi_VISpl",
        "ipsi_VISpm",
        "ipsi_VISa",
        "contra_VISp",
    ),
    "other isocortex": (
        "ipsi_RSPd",
        "ipsi_RSPagl",
        "ipsi_MOs",
        "ipsi_ACAd",
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

def plot_aicc_for_subclasses(gs, all_aicc_df, variable_name, label_text):
    gs_idx = 0
    print(variable_name)
    print(all_aicc_df[variable_name].min(), all_aicc_df[variable_name].max())
    grouped = all_aicc_df.reset_index().groupby("subclass_name")
    for n in [SUBCLASS_NAMES[sc] for sc in SUBCLASSES_ORDER]:
        ax = plt.subplot(gs[gs_idx])
        g = grouped.get_group(n)
        sns.stripplot(
            data=g,
            x=variable_name,
            y="region_category",
            hue="region",
            order=REGION_CATEGORY_ORDER,
            palette=REGION_COLORS,
            size=4,
            edgecolor="white",
            linewidth=0.25,
            legend=False,
            ax = ax,
        )
        ax.axvline(0, linestyle=":", color=".5")
        ax.set_title(n, fontsize=7)
        ax.set_xlabel(label_text, fontsize=6)
        ax.set_ylabel("")
        ax.tick_params(axis="both", labelsize=6)
        if gs_idx != 0:
            ax.tick_params(axis="y", left=False, labelleft=False)
            ax.sharex(ax_first)
            ax.sharey(ax_first)
        else:
            ax_first = ax
        sns.despine(ax=ax)
        ax.set_xlim(-30, 20)
        ax.set_aspect(15)

        gs_idx += 1


def plot_pseudo_r2_loo(gs, train_df, loo_df):
    gs_idx = 0
    train_grouped = train_df.loc[train_df["best_model_type"] != "null", :].groupby("subclass_name")
    loo_grouped = loo_df.groupby("subclass")
    yticks = []
    yticklabels = []

    model_type_colors = {
        "full": "#222222",
        "surface": ".5",
        "lf": "#dddddd",
    }

    for n in [SUBCLASS_NAMES[sc] for sc in SUBCLASSES_ORDER]:
        ax = plt.subplot(gs[gs_idx])
        g_train = train_grouped.get_group(n)
        sc = g_train.index.values[0][0]
        g_loo = loo_grouped.get_group(sc)
        train_by_regcat = g_train.groupby("region_category")
        position = 0
        for reg_cat in REGION_CATEGORY_ORDER:
            if reg_cat not in train_by_regcat.groups:
                position -= 2
                continue
            g = train_by_regcat.get_group(reg_cat)

            edgecolors = [model_type_colors[mt] for mt in g["best_model_type"]]
            regions_to_plot = g.reset_index()["region"]
            loo_mean_pR2 = g_loo.groupby("region")["pR2"].mean()
            ax.scatter(
                loo_mean_pR2[regions_to_plot.values],
                np.array([position] * g.shape[0]) +
                np.random.uniform(low=-0.2, high=0.2, size=g.shape[0]),
                s=7,
                edgecolor=edgecolors,
                facecolor=[REGION_COLORS[r] for r in regions_to_plot],
                lw=0.75,
                zorder=20,
            )
            if sc == SUBCLASSES_ORDER[2]:
                # ET has all the categories
                yticks.append(position)
                yticklabels.append(reg_cat)
            position -= 2
        sns.despine(ax=ax)
        ax.set_title(n, fontsize=7)
        ax.set_xlabel("pseudo-$R^2$\n(LOO-CV)", fontsize=6)
        ax.set_xticks([0, 0.5, 1])
        ax.set_aspect(0.17)
        ax.tick_params(axis="both", labelsize=6)
        if gs_idx ==0:
            ax_first = ax
        else:
            ax.sharex(ax_first)
            ax.sharey(ax_first)
            ax.tick_params(axis="y", left=False, labelleft=False)
        gs_idx += 1

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='LF only',
                          markerfacecolor='w', markeredgecolor="#dddddd", markersize=5),
        Line2D([0], [0], marker='o', color='w', label='surface only',
                          markerfacecolor='w', markeredgecolor=".5", markersize=5),
        Line2D([0], [0], marker='o', color='w', label='full',
                          markerfacecolor='w', markeredgecolor="#222222", markersize=5),
    ]
    ax.legend(handles=legend_elements,
        loc='upper left', bbox_to_anchor=(1.05, 1),
        fontsize=6, frameon=False)

    ax_first.set_yticks(yticks)
    ax_first.set_yticklabels(yticklabels)
    ax_first.tick_params(axis="y", labelsize=6)
    ax_first.set_xlim(left=-0.2, right=1)
    ax_first.set_ylim(yticks[-1] - 1, yticks[0] + 1)


def plot_latent_factor_predictions(gs, lf_examples, all_coef_df, wnm_data_df,
        col_wrap=2):
    lf_cols = all_coef_df.columns[all_coef_df.columns.str.startswith("LF")]
    i = 0
    for sc, r, lf_to_plot in lf_examples:
        ax = plt.subplot(gs[i // col_wrap, i % col_wrap])
        lf_coefs = all_coef_df.loc[
            (all_coef_df.region == r) &
            (all_coef_df.subclass == sc) &
            (all_coef_df.model_type == "lf"),
            lf_cols
        ]
        lf_coefs = lf_coefs.dropna(axis=1)

        data_for_estimate = wnm_data_df.loc[
            wnm_data_df.predicted_met_type.isin(SUBCLASS_MET_TYPES[sc]),
            :
        ].copy()
        data_for_estimate.columns = data_for_estimate.columns.str.replace("-", "_")

        lf_to_plot = lf_to_plot + "_" + sc.replace("-", "_")
        lf_coefs_not_plotted = lf_coefs.drop(columns=[lf_to_plot])
        lf_other_cols = lf_coefs_not_plotted.columns
        intercept = all_coef_df.loc[
            (all_coef_df.region == r) &
            (all_coef_df.subclass == sc) &
            (all_coef_df.model_type == "lf"),
            ["(Intercept)"]
        ].values[0, 0]

        lf_plot_values = np.linspace(
            data_for_estimate[lf_to_plot].min(),
            data_for_estimate[lf_to_plot].max(),
            100
        )
        est_log_odds = (intercept + lf_coefs[lf_to_plot].values[0] * lf_plot_values)
        if len(lf_other_cols) > 0:
            est_log_odds += np.dot(
                np.median(data_for_estimate.loc[:, lf_other_cols].values, axis=0),
                lf_coefs_not_plotted.values.T
            )
        est_pred_prob = np.exp(est_log_odds) / (1 + np.exp(est_log_odds))
        proj_values = (data_for_estimate[r.replace("-", "_")] > 0).values
        proj_colors = [MET_TYPE_COLORS[t] for t in data_for_estimate["predicted_met_type"]]

        ax.plot(lf_plot_values, est_pred_prob, c="#333333", linewidth=0.75)
        ax.scatter(data_for_estimate.loc[:, lf_to_plot], proj_values, c=proj_colors,
            s=10, alpha=0.5,
            edgecolors="white",
            linewidth=0.25)
        if i % col_wrap == 0:
            ax.set_ylabel("probability", fontsize=6)
        ax.set_xlabel(f"{SUBCLASS_NAMES[sc]} M-LF-{lf_to_plot[2]}", fontsize=6)
        ax.set_ylim(-0.1, 1.1)
        ax.axhline(0, linestyle="dotted", color="gray", zorder=-1)
        ax.axhline(1, linestyle="dotted", color="gray", zorder=-1)
        ax.tick_params(axis="both", labelsize=6)
        sns.despine(ax=ax)
        region_display = r.replace("ipsi_", "")
        if i == 5:
            display_x = 0.95
            display_ha = "right"
        else:
            display_x = 0.05
            display_ha = "left"
        if i == 1:
            ax.text(1.05, 0, "does not target",
                ha="left", fontsize=6, transform=ax.get_yaxis_transform(), fontstyle="italic")
            ax.text(1.05, 1, "targets",
                ha="left", fontsize=6, transform=ax.get_yaxis_transform(), fontstyle="italic")

        ax.text(display_x, 0.65, region_display,
            ha=display_ha, fontsize=7,
            transform=ax.transAxes)

        i += 1


def plot_surface_predictions(gs, surf_examples, all_coef_df, wnm_data_df,
        flatmap_3d_coords, flatmap_2d_coords, bounds_ipsi, annot_hemi, col_wrap=3):
    i = 0
    for sc, r in surf_examples:
        ax = plt.subplot(gs[i // col_wrap, i % col_wrap])
        surf_coefs = all_coef_df.loc[
            (all_coef_df.region == r) &
            (all_coef_df.model_type == "surface") &
            (all_coef_df.subclass == sc),
            ["surface_ccf_x", "surface_ccf_y", "surface_ccf_z"]].values.T
        log_odds_change = np.dot(flatmap_3d_coords, surf_coefs)

        intercept = all_coef_df.loc[
            (all_coef_df.region == r) &
            (all_coef_df.subclass == sc) &
            (all_coef_df.model_type == "surface"),
            ["(Intercept)"]
        ].values.T
        est_log_odds = log_odds_change + intercept
        est_pred_prob = np.exp(est_log_odds) / (1 + np.exp(est_log_odds))

        targeted_region = r.split("_")[1]
        for k, boundary_coords in bounds_ipsi.items():
            color = "none"
            zorder = 7
            if k == targeted_region and r.split("_")[0] != "contra":
                lw = 1
                edgecolor = "black"
                zorder=10
            else:
                lw = 0.5
                edgecolor = "gray"
            ax.fill(*boundary_coords.T, color=color, edgecolor=edgecolor,
                lw=lw, zorder=zorder)
        ax.scatter(
            flatmap_2d_coords[:, 0],
            flatmap_2d_coords[:, 1], s=1,
            edgecolors="none",
            c=est_pred_prob,
            cmap="PuBu",
            vmin=0, vmax=1,
            zorder=5,
            rasterized=True,
        )
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_title(f"{SUBCLASS_NAMES[sc]} to {r.split('_')[1]}", fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
        sns.despine(ax=ax, left=True, bottom=True)

        if targeted_region not in bounds_ipsi.keys():
            struct_masked = (annot_hemi == SUBCORT_STRUCT_IDS[targeted_region]).astype(int)
            background_projection = (annot_hemi.sum(axis=1) > 0).astype(int)
            struct_projection = (struct_masked.sum(axis=1) > 0).astype(int)
            struct_contours = find_contours(struct_projection.T, level=0.5)
            background_contours = find_contours(background_projection.T, level=0.5)
            ax_target = ax.inset_axes([1.05, 0.167, 0.5, 0.666], transform=ax.transAxes)
            color = REGION_COLORS[r]
            for b in struct_contours:
                ax_target.fill(*b.T, color=color, edgecolor='black',
                    zorder=10, lw=1)


            for b in background_contours:
                ax_target.fill(*b.T, color="#dddddd", edgecolor='gray',
                    zorder=1, lw=0.5)
            ax_target.set_aspect("equal")
            ax_target.invert_yaxis()
            sns.despine(ax=ax_target, left=True, bottom=True)
            ax_target.set(xticks=[], yticks=[])

        if i == 2:
            ax_cbar = ax.inset_axes([1.05, 0.167, 0.1, 0.666], transform=ax.transAxes)
            cbar = plt.colorbar(
                matplotlib.cm.ScalarMappable(cmap=matplotlib.colormaps["PuBu"]),
                cax=ax_cbar, orientation='vertical')
            cbar.set_label("probability", size=6)
            cbar.outline.set_visible(False)
            ax_cbar.tick_params(labelsize=6)
        i += 1


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


    ##### Checking values for examples

    ######### LF ################

    print("values for LF examples")
    lf_examples = (
        ("L4-L5-IT", "ipsi_ACAd", "LF1"),
        ("L4-L5-IT", "ipsi_RSPd", "LF2"),
        ("L4-L5-IT", "ipsi_CP", "LF1"),
        ("L5-ET", "ipsi_LGd-co", "LF2"),
        ("L5-ET", "ipsi_LP", "LF2"),
        ("L6-CT", "ipsi_LP", "LF2"),
    )

    for sc, r, _ in lf_examples:
        print(sc, r)
        print(train_df.loc[
            (train_df.reset_index()["region"].values == r) & (train_df["subclass_name"] == SUBCLASS_NAMES[sc]),
            "best_model_type",
        ])
        print(all_aicc_df.loc[
            (all_aicc_df.reset_index()["region"].values == r) & (all_aicc_df["subclass_name"] == SUBCLASS_NAMES[sc]),
            "delta_lf_null",
        ])
        print(all_loo_df.loc[
            (all_loo_df["region"] == r) & (all_loo_df["subclass"] == sc),
            "pR2",
        ].mean())
        print()

    ######### SURFACE ###########

    print("values for surface examples")
    surf_examples = (
        ("L23-IT", "ipsi_VISrl"),
        ("L4-L5-IT", "ipsi_VISrl"),
        ("L5-ET", "ipsi_VISrl"),
        ("L4-L5-IT", "ipsi_VISpm"),
        ("L5-ET", "ipsi_VISl"),
        ("L5-ET", "ipsi_CP"),
    )

    for sc, r in surf_examples:
        print(sc, r)
        print(train_df.loc[
            (train_df.reset_index()["region"].values == r) & (train_df["subclass_name"] == SUBCLASS_NAMES[sc]),
            "best_model_type",
        ])
        print(all_aicc_df.loc[
            (all_aicc_df.reset_index()["region"].values == r) & (all_aicc_df["subclass_name"] == SUBCLASS_NAMES[sc]),
            "delta_surf_null",
        ])
        print(all_loo_df.loc[
            (all_loo_df["region"] == r) & (all_loo_df["subclass"] == sc),
            "pR2",
        ].mean())
        print()

    ##### PLOTTING ##############
    fig = plt.figure(figsize=(7.5, 8))
    gs = gridspec.GridSpec(
        3, 2,
        hspace=0.4,
        wspace=0.4,
        height_ratios=(1, 2, 1),
    )

    # Latent factor
    gs_lf = gridspec.GridSpecFromSubplotSpec(
        1, 4,
        subplot_spec=gs[0, 0],
        wspace=0.4,
    )

    label_text = "ΔAICc\n(LF - null)"
    variable_name = "delta_lf_null"
    plot_aicc_for_subclasses(gs_lf, all_aicc_df, variable_name, label_text)

    gs_lf_examples = gridspec.GridSpecFromSubplotSpec(
        3, 2,
        subplot_spec=gs[1, 0],
        wspace=0.4,
        hspace=0.75,
    )
    plot_latent_factor_predictions(gs_lf_examples, lf_examples, all_coef_df, wnm_data_df)


    # Surface location
    gs_surf = gridspec.GridSpecFromSubplotSpec(
        1, 4,
        subplot_spec=gs[0, 1],
        wspace=0.4,
    )

    label_text = "ΔAICc\n(surface - null)"
    variable_name = "delta_surf_null"
    plot_aicc_for_subclasses(gs_surf, all_aicc_df, variable_name, label_text)

    gs_surf_examples = gridspec.GridSpecFromSubplotSpec(
        2, 3,
        subplot_spec=gs[1, 1],
        wspace=0.4,
        hspace=0.4,
    )
    plot_surface_predictions(gs_surf_examples, surf_examples, all_coef_df, wnm_data_df,
        flatmap_3d_coords, flatmap_2d_coords, bounds_ipsi, annot_hemi)

    # Full model
    gs_full = gridspec.GridSpecFromSubplotSpec(
        1, 4,
        subplot_spec=gs[2, 0],
        wspace=0.4,
    )

    label_text = "ΔAICc\n(full - null)"
    variable_name = "delta_full_null"
    plot_aicc_for_subclasses(gs_full, all_aicc_df, variable_name, label_text)

    # Pseudo-R2
    gs_pR2 = gridspec.GridSpecFromSubplotSpec(
        1, 4,
        subplot_spec=gs[2, 1],
        wspace=0.4,
    )
    plot_pseudo_r2_loo(gs_pR2, train_df, all_loo_df)

    plt.savefig(args["output_file"], dpi=300, bbox_inches="tight")



if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=FigGlmFitsParameters)
    main(module.args)
