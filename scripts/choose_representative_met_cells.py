import json
import numpy as np
import pandas as pd
import argschema as ags
import scipy.spatial.distance as distance


class ChooseRepMetCellsParameters(ags.ArgSchema):
    met_type_file = ags.fields.InputFile(
        description="csv file with met type text labels",
    )
    ephys_feature_file = ags.fields.InputFile(
        description="csv file with ephys features",
    )
    morph_feature_file = ags.fields.InputFile(
        default="../derived_data/morph_features_mMET_exc_Jan2025_wide_normalized.csv",
        description="csv file with morph features",
    )
    output_selection_file = ags.fields.OutputFile(
        default="../derived_data/rep_met_cells_from_dist_Jan2025.json",
        description="output file with representative cell specimen IDs per met-type",
    )
    rename_pt_to_et = ags.fields.Boolean(
        default=True,
        description="whether to switch names from L5 PT to L5 ET",
    )



def main(args):
    met_type_df = pd.read_csv(args['met_type_file'], index_col=0)
    if args['rename_pt_to_et']:
        met_type_df["met_type"] = [s.replace("PT", "ET") if "PT" in s else s for s in met_type_df["met_type"].tolist()]

    ephys_df = pd.read_csv(args['ephys_feature_file'], index_col=0)
    morph_df = pd.read_csv(args['morph_feature_file'], index_col=0)

    selections = {}

    omitted_cells = [
        701074400, # NP cell - basals are not very representative
    ]

    for met_type, met_group in met_type_df.groupby("met_type"):
        print(met_type)
        met_ids = met_group.index.values
        met_ids = met_ids[~np.in1d(met_ids, omitted_cells)]

        ephys_dist = distance.squareform(distance.pdist(ephys_df.loc[met_ids, :].values))
        morph_dist = distance.squareform(distance.pdist(morph_df.loc[met_ids, :].values))

        # Normalize distances to average cell-cell distance within group
        # to equalize ephys & morph contributions
        ephys_tril = np.tril(ephys_dist)
        morph_tril = np.tril(morph_dist)
        avg_ephys_dist = ephys_tril[ephys_tril > 0].mean()
        avg_morph_dist = morph_tril[morph_tril > 0].mean()

        ephys_dist /= avg_ephys_dist
        morph_dist /= avg_morph_dist

        combo_dist = ephys_dist + morph_dist

        avg_to_others = combo_dist.mean(axis=1)
        min_avg_dist_ind = np.argmin(avg_to_others)

        selections[met_type] = int(met_ids[min_avg_dist_ind])

    with open(args['output_selection_file'], "w") as f:
        json.dump(selections, f)


if __name__ == "__main__":
    module = ags.ArgSchemaParser(schema_type=ChooseRepMetCellsParameters)
    main(module.args)
