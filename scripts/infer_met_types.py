import pandas as pd
import numpy as np
import sklearn.ensemble as ensemble
import argschema as ags

class InferMetTypesParameters(ags.ArgSchema):
    ephys_file = ags.fields.InputFile(
        description="ephys spca file",
    )
    auto_manual_morph_file = ags.fields.InputFile(
        description="morph features from manual and auto reconstructions",
    )
    tx_anno_file = ags.fields.InputFile(
        description="patchseq tx annotation file",
    )
    met_type_file = ags.fields.InputFile(
        description="met type assignments file",
    )
    inferred_met_type_file = ags.fields.OutputFile(
        description="output file",
    )


RANDOM_STATE = 4321

MET_NAME_DICT = {
	0: "L2/3 IT",
	13: "L4 IT",
	6: "L4/L5 IT",
	11: "L5 IT-1",
	12: "L5 IT-2",
	16: "L5 IT-3 Pld5",
	8: "L6 IT-1",
	7: "L6 IT-2",
	15: "L6 IT-3",
	14: "L5/L6 IT Car3",
	10: "L5 ET-1 Chrna6",
	1: "L5 ET-2",
	2: "L5 ET-3",
	9: "L5 NP",
	3: "L6 CT-1",
	4: "L6 CT-2",
	5: "L6b",
}


MET_BY_T_TYPE_DICT = {
    "L2/3 IT VISp Agmat": 0,
    "L2/3 IT VISp Rrad": 0,
    "L2/3 IT VISp Adamts2": 0,
    "L4 IT VISp Rspo1": 13,
    "L5 IT VISp Whrn Tox2": 6,
    "L5 IT VISp Hsd11b1 Endou": 6,
    "L5 IT VISp Batf3": 11,
    "L5 IT VISp Col6a1 Fezf2": 12,
    "L5 IT VISp Col27a1": 16,
    "L6 IT VISp Car3": 14,
    "L5 ET VISp Chrna6": 10,
    "L5 ET VISp C1ql2 Ptgfr": 2,
    "L5 ET VISp C1ql2 Cdh13": 2,
    "L5 NP VISp Trhr Cpne7": 9,
    "L5 NP VISp Trhr Met": 9,
    "L6 CT VISp Ctxn3 Sla": 3,
    "L6 CT VISp Ctxn3 Brinp3": 3,
    "L6 CT VISp Nxph2 Wls": 3,
    "L6 CT VISp Gpr139": 4,
    "L6 CT VISp Krt80 Sla": 4,
    "L6b Col8a1 Rprm": 5,
    "L6b VISp Mup5": 5,
    "L6b VISp Crh": 5,
    "L6b VISp Col8a1 Rxfp1": 5,
    "L6b P2ry12": 5,
}

AUTOTRACE_MORPH_FEATURES = [
    "soma_aligned_dist_from_pia",
    "apical_dendrite_max_euclidean_distance",
    "apical_dendrite_bias_y",
    "apical_dendrite_extent_y",
    "apical_dendrite_emd_with_basal_dendrite",
    "apical_dendrite_soma_percentile_y",
    "apical_dendrite_frac_below_basal_dendrite",
    "apical_dendrite_depth_pc_0",
    "basal_dendrite_frac_above_apical_dendrite",
    "apical_dendrite_depth_pc_1",
    "apical_dendrite_total_length",
    "basal_dendrite_bias_y",
    "apical_dendrite_frac_above_basal_dendrite",
    "basal_dendrite_soma_percentile_y",
    "apical_dendrite_depth_pc_2",
    "apical_dendrite_depth_pc_3",
    "basal_dendrite_max_euclidean_distance",
    "apical_dendrite_frac_intersect_basal_dendrite",
    "basal_dendrite_extent_y",
    "apical_dendrite_num_branches",
    "apical_dendrite_extent_x",
    "basal_dendrite_extent_x",
    "basal_dendrite_total_length",
]


def main(args):
    ephys_df = pd.read_csv(args["ephys_file"], index_col=0)
    morph_df = pd.read_csv(args["auto_manual_morph_file"], index_col=0)
    met_type_df = pd.read_csv(args["met_type_file"], index_col=0)
    anno_df = pd.read_feather(args["tx_anno_file"])
    anno_df["spec_id_label"] = pd.to_numeric(anno_df["spec_id_label"])
    anno_df.set_index("spec_id_label", inplace=True)

    spec_ids = ephys_df.index.values

    # Ephys-based classifier for certain L5 ET t-types
    l5et_met_types_to_classify = [1, 2]
    l5et_t_types_to_classify = ["L5 ET VISp Krt80", "L5 ET VISp Lgr5"]
    l5et_met_df = met_type_df.loc[
        met_type_df["met_type"].isin(l5et_met_types_to_classify), :].copy()

    l5et_rf = ensemble.RandomForestClassifier(n_estimators=200, oob_score=True,
        random_state=RANDOM_STATE)
    l5et_rf.fit(ephys_df.loc[l5et_met_df.index, :].values, l5et_met_df["met_type"].values)
    print("L5 ET ephys classifier OOB score: ", l5et_rf.oob_score_)

    # Ephys & morph-based classifier for L6 IT t-types
    l6it_met_types_to_classify = [8, 7, 15]
    l6it_t_types_to_classify = [
        "L6 IT VISp Penk Col27a1",
        "L6 IT VISp Penk Fst",
        "L6 IT VISp Col18a1",
        "L6 IT VISp Col23a1 Adamts2",
    ]
    l6it_met_df = met_type_df.loc[
        met_type_df["met_type"].isin(l6it_met_types_to_classify), :].copy()

    combo_l6it_data = np.hstack([
        ephys_df.loc[l6it_met_df.index, :].values,
        morph_df.loc[l6it_met_df.index, AUTOTRACE_MORPH_FEATURES].values,
    ])
    l6it_rf = ensemble.RandomForestClassifier(n_estimators=200, oob_score=True,
        random_state=RANDOM_STATE)
    l6it_rf.fit(combo_l6it_data, l6it_met_df["met_type"].values)
    print("L6 IT ephys/morph classifier OOB score: ", l6it_rf.oob_score_)

    df = pd.DataFrame(index=spec_ids)
    df["t_type"] = [t.replace("PT", "ET")
        for t in anno_df.loc[df.index, "Tree_first_cl_label"].values]
    df["met_type"] = ""
    shared_ids = df.index.intersection(met_type_df.index)
    df.loc[shared_ids, "met_type"] = [MET_NAME_DICT[met]
        for met in met_type_df.loc[shared_ids, "met_type"].values]

    inferred_met_types = []
    for spid in df.index:
        if df.at[spid, "met_type"] != "":
            inferred_met_types.append(df.at[spid, "met_type"])
        elif df.at[spid, "t_type"] in MET_BY_T_TYPE_DICT:
            inferred_met_types.append(
                MET_NAME_DICT[MET_BY_T_TYPE_DICT[df.at[spid, "t_type"]]])
        elif df.at[spid, "t_type"] in l5et_t_types_to_classify:
            pred_met = l5et_rf.predict(ephys_df.loc[[spid], :].values)[0]
            inferred_met_types.append(MET_NAME_DICT[pred_met])
        elif df.at[spid, "t_type"] in l6it_t_types_to_classify:
            if spid in morph_df.index:
                data_for_pred = np.hstack([
                    ephys_df.loc[[spid], :].values,
                    morph_df.loc[[spid], AUTOTRACE_MORPH_FEATURES].values,
                ])
                pred_met = l6it_rf.predict(data_for_pred)[0]
                inferred_met_types.append(MET_NAME_DICT[pred_met])
            else:
                # no morph data, so cannot infer met-type
                inferred_met_types.append("")
        else:
            inferred_met_types.append("")

    df['inferred_met_type'] = inferred_met_types
    df.to_csv(args["inferred_met_type_file"])

if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=InferMetTypesParameters)
    main(module.args)