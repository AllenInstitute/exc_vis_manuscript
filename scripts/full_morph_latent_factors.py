import json
import os
import h5py
import warnings
import numpy as np
import pandas as pd
import argschema as ags


class FullMorphLatentFactorsParameters(ags.ArgSchema):
    srrr_fits_file = ags.fields.InputFile()
    srrr_fits_no_diam_file = ags.fields.InputFile()
    srrr_features_file = ags.fields.InputFile()
    patchseq_morph_file = ags.fields.InputFile()
    input_dend_file = ags.fields.InputFile()
    input_axon_file = ags.fields.InputFile()
    input_met_pred_file = ags.fields.InputFile()
    output_file = ags.fields.OutputFile()


MET_GROUP_NAMES = (
    "L23-IT",
    "L4-L5-IT",
    "L6-IT",
    "L5-ET",
    "L6-CT",
    "L6b",
)

MET_PRED_TYPES = {
    "L23-IT": ("L2/3 IT",),
    "L4-L5-IT": ("L4 IT", "L4/L5 IT", "L5 IT-1", "L5 IT-2", "L5 IT-3 Pld5"),
    "L6-IT": ("L6 IT-1", "L6 IT-2", "L6 IT-3"),
    "L5-ET": ("L5 ET-1 Chrna6", "L5 ET-2", "L5 ET-3"),
    "L6-CT": ("L6 CT-1", "L6 CT-2",),
    "L6b": ("L6b",),
}


def sp_rrr_fit_info(filepart, modality, h5f, select_features):
    specimen_ids = h5f[filepart][modality]["specimen_ids"][:]
    features = select_features[filepart][modality]

    w = h5f[filepart][modality]["w"][:].T
    v = h5f[filepart][modality]["v"][:].T

    genes = h5f[filepart][modality]["genes"][:]
    genes = np.array([s.decode() for s in genes])

    return {
        "specimen_ids": specimen_ids,
        "features": features,
        "w": w,
        "v": v,
        "genes": genes,
    }


def main(args):
    ps_morph_df = pd.read_csv(args["patchseq_morph_file"], index_col=0)

    h5f = h5py.File(args["srrr_fits_file"], "r")
    h5f_no_diam = h5py.File(args["srrr_fits_no_diam_file"], "r")
    with open(args["srrr_features_file"], "r") as f:
        select_features = json.load(f)

    print()
    print()

    full_morph_axon_df = pd.read_csv(args["input_axon_file"], index_col=0)
    full_morph_dend_df = pd.read_csv(args["input_dend_file"], index_col=0)
    full_morph_dend_df.set_index("swc_path", inplace=True)
    full_morph_dend_df = pd.merge(
        full_morph_dend_df, full_morph_axon_df[["axon_exit_distance", "axon_exit_theta"]],
        left_index=True, right_index=True, how="left")
    full_morph_dend_df["specimen_id"] = [f"{s}.swc" for s in full_morph_dend_df.index]
    full_morph_dend_df.set_index("specimen_id", inplace=True)

    full_morph_met_pred_df = pd.read_csv(args["input_met_pred_file"], index_col=0)

    modality = "morph"
    df_list = []

    excluded_features = [
        "apical_dendrite_total_surface_area",
        "basal_dendrite_total_surface_area",
        "apical_dendrite_mean_diameter",
        "basal_dendrite_mean_diameter",
    ]

    # Calculate LF values for every cell
    for filepart in MET_GROUP_NAMES:
        print(filepart)

        if filepart in h5f_no_diam.keys():
            fit_info = sp_rrr_fit_info(filepart, modality, h5f_no_diam, select_features)
        else:
            fit_info = sp_rrr_fit_info(filepart, modality, h5f, select_features)
        features = fit_info["features"]
        v = fit_info["v"]

        # Drop radius-dependent features since not captured in fMOST
        feat_drop_mask = np.array([f not in excluded_features for f in features])
        features = [f for f in features if f not in excluded_features]

        ps_means = ps_morph_df.loc[fit_info["specimen_ids"], features].mean(axis=0)
        ps_stdev = ps_morph_df.loc[fit_info["specimen_ids"], features].std(axis=0)

        full_morph_lf = ((full_morph_dend_df[features].values -
            ps_means.values[np.newaxis, :]) / ps_stdev.values[np.newaxis, :]) @ v
        col_names = [f"LF{i + 1}_{filepart}" for i in range(full_morph_lf.shape[1])]
        full_morph_lf_df = pd.DataFrame(full_morph_lf,
            index=full_morph_dend_df.index, columns=col_names)
        df_list.append(full_morph_lf_df)
        print()

    all_lf_df = pd.concat(df_list, axis=1)
    all_lf_df.to_csv(args["output_file"])


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=FullMorphLatentFactorsParameters)
    main(module.args)
