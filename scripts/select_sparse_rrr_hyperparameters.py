import os
import h5py
import json
import numpy as np
import pandas as pd
import argschema as ags

class SelectSparseRrrHyperparamsParameters(ags.ArgSchema):
    input_dir = ags.fields.OutputDir()
    output_file = ags.fields.OutputFile()


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


def main(args):
    input_dir = args["input_dir"]
    output_file = args["output_file"]

    chosen_params = {}
    for si in SUBCLASS_INFO:
        filepart = si["filepart"]
        print(filepart)
        chosen_params[filepart] = {}
        f = h5py.File(os.path.join(input_dir, f"sparse_rrr_cv_{filepart}.h5"), "r")
        tx_pc_f = h5py.File(os.path.join(input_dir, f"tx_pc_elasticnet_cv_{filepart}.h5"), "r")
        # Pick the rank with the highest performance (since the alpha used had the highest
        # performance already)
        for modality in ["ephys", "morph"]:
            chosen_params[filepart][modality] = {}

            alpha = f[modality]["effect_of_rank"].attrs["alpha"][0]
            ranks = f[modality]["effect_of_rank/rank"][:]
            best_rank = -1
            best_rank_r2_relax = -np.inf
            best_lambda = -1
            for r in ranks:
                key = f"rank_{r}"
                r2_relax = np.nanmean(
                    f[modality]["effect_of_rank"][key]["r2_relaxed"][:],
                    axis=(2, 3)).squeeze()
                lambdas = f[modality]["effect_of_rank"][key]["lambda"][:],
                lambdas = lambdas[0].squeeze()
                if best_rank_r2_relax < np.nanmax(r2_relax):
                    best_rank_r2_relax = np.nanmax(r2_relax)
                    best_rank = r
                    best_lambda = lambdas[np.nanargmax(r2_relax)]
            chosen_params[filepart][modality]['sparse_rrr'] = {
                "alpha": float(alpha),
                "rank": int(best_rank),
                "lambda": float(best_lambda),
                "r2_relaxed": float(best_rank_r2_relax),
            }

            tx_pc_alphas = tx_pc_f[modality]['effect_of_alpha/alpha'][:]
            tx_pc_r2 = tx_pc_f[modality]["effect_of_alpha"]
            best_tx_r2 = -np.inf
            best_alpha = -1
            best_lambda = -1
            for i, a in enumerate(tx_pc_alphas):
                tx_pc_r2 = tx_pc_f[modality][f'effect_of_alpha/alpha_ind_{i + 1}/r2'][:]
                lambdas = tx_pc_f[modality][f'effect_of_alpha/alpha_ind_{i + 1}/lambda'][:]
                if np.nanmax(tx_pc_r2) > best_tx_r2:
                    best_tx_r2 = np.nanmax(tx_pc_r2)
                    best_alpha = a
                    best_lambda = lambdas[np.nanargmax(tx_pc_r2)]
            chosen_params[filepart][modality]['tx_pc_elasticnet'] = {
                "alpha": float(best_alpha),
                "r2": float(best_tx_r2),
                "lambda": best_lambda,
            }

    with open(output_file, "w") as out_f:
        json.dump(chosen_params, out_f)



if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=SelectSparseRrrHyperparamsParameters)
    main(module.args)
