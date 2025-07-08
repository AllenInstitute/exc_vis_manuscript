import os
import h5py
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argschema as ags


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


class ConsolidateSingleFeatureCvFitsParameters(ags.ArgSchema):
    ephys_component_info_file = ags.fields.InputFile()
    single_feature_results_dir = ags.fields.InputDir()
    r2_threshold = ags.fields.Float(
        default=0.1
    )
    output_file = ags.fields.OutputFile()


def main(args):

    with open(args["ephys_component_info_file"], "r") as f:
        ephys_comp_info = json.load(f)
    spca_comp_info_key = {}
    counter = 0
    for d in ephys_comp_info:
        for ind in d["indices"]:
            spca_comp_info_key[counter] = d["key"] + " " + str(ind)
            counter += 1

    ephys_list_of_best = {}
    morph_list_of_best = {}
    for sc in SUBCLASS_INFO:
        filepart = sc['filepart']
        print(filepart)
        f = h5py.File(
            os.path.join(
                args["single_feature_results_dir"],
                f"single_features_elasticnet_cv_{filepart}.h5"),
            "r")

        modality = 'ephys'
        n_features = len(f[modality].keys())
        best_r2 = np.zeros(n_features)
        for feature_index in range(n_features):
            feature_str = str(feature_index + 1)
            c = {}
            if feature_index == 0:
                print("ephys alphas: ", list(f[modality][feature_str].keys()))
            for a in f[modality][feature_str].keys():
                c[a] = f[modality][feature_str][a]['r2'][:]
            best_r2[feature_index] = np.max([np.max(val) for val in c.values()])
        ephys_list_of_best[filepart] = best_r2

        modality = 'morph'
        n_features = n_features = len(f[modality].keys()) - 1
        best_r2 = np.zeros(n_features)
        for feature_index in range(n_features):
            feature_str = str(feature_index + 1)
            c = {}
            if feature_index == 0:
                print("morph alphas: ", list(f[modality][feature_str].keys()))
            for a in f[modality][feature_str].keys():
                c[a] = f[modality][feature_str][a]['r2'][:]
            best_r2[feature_index] = np.max([np.max(val) for val in c.values()])
        morph_list_of_best[filepart] = dict(zip([s.decode() for s in f[modality]['morph_features'][:]],
                                          best_r2))
        f.close()

    sc_best_ephys_df = pd.DataFrame(
        ephys_list_of_best, index = spca_comp_info_key.values())
    sc_best_morph_df = pd.DataFrame(morph_list_of_best)

    threshold = args["r2_threshold"]

    features_for_sprrr = {}
    features_for_sprrr['threshold'] = threshold
    for c in sc_best_morph_df.columns:
        features_for_sprrr[c] = {}
        features_for_sprrr[c]['ephys'] = np.arange(sc_best_ephys_df.shape[0])[sc_best_ephys_df[c] > threshold].tolist()
        features_for_sprrr[c]['ephys_text'] = sc_best_ephys_df.index[sc_best_ephys_df[c] > threshold].tolist()
        features_for_sprrr[c]['morph'] = sc_best_morph_df[c].index[sc_best_morph_df[c] > threshold].tolist()
        print(c, len(features_for_sprrr[c]['ephys']), len(features_for_sprrr[c]['morph']))
    with open(args["output_file"], "w") as f:
        json.dump(features_for_sprrr, f)


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=ConsolidateSingleFeatureCvFitsParameters)
    main(module.args)
